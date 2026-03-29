# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:13:58 2017

@author: Calil

GPU Acceleration
----------------
horizontal_pattern() and vertical_pattern() use branchless xp.where
instead of np.where scatter indexing.
"""

import numpy as np
import sys

from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
from sharc.support.backend_handler import xp, backend


class AntennaElementImtF1336(object):
    """
    Implements a single element of an IMT antenna array following ITU-R F.1336-4, item 3.1.1
    using parameters from ITU-R M2292

    Attributes
    ----------
        g_max (float): maximum gain of element
        theta_3db (float): vertical 3dB beamwidth of single element [degrees]
        phi_3db (float): horizontal 3dB beamwidth of single element [degrees]
    """

    def __init__(self, par: ParametersAntennaImt):
        """
        Constructs an AntennaElementImt object.

        Parameters
        ---------
            param (ParametersAntennaImt): antenna IMT parameters
        """
        self.param = par

        self.g_max = par.element_max_g
        self.phi_3db = par.element_phi_3db
        if par.element_theta_3db > 0:
            self.theta_3db = par.element_theta_3db
        else:
            if self.phi_3db > 120.:
                sys.stderr.write(
                    "ERROR\nvertical beamwidth must be givem if horizontal beamwidth > 120 degrees", )
                sys.exit(1)
            # calculate based on F1336
            self.theta_3db = (31000 * 10**(-.1 * self.g_max)) / self.phi_3db

        # antenna paremeters, according to ITU-R M2292
        self.k_a = .7
        self.k_p = .7
        self.k_h = .7
        self.lambda_k_h = 3 * (1 - .5**(-self.k_h))
        self.k_v = .3
        self.incline_factor = \
            10 * np.log10(((180 / self.theta_3db)**1.5 * (4**-1.5 + self.k_v)) / (1 + 8 * self.k_p)) / \
            np.log10(22.5 / self.theta_3db)
        self.x_k = np.sqrt(1 - .36 * self.k_v)
        self.lambda_k_v = 12 - self.incline_factor * \
            np.log10(4) - 10 * np.log10(4**-1.5 + self.k_v)

        self.g_hr_180 = -12. + 10 * \
            np.log10(1 + 8 * self.k_a) - 15 * np.log10(180 / self.theta_3db)
        self.g_hr_0 = 0

    def horizontal_pattern(self, phi: np.array) -> {np.array, float}:
        """
        Calculates the horizontal radiation pattern (branchless GPU version).

        Parameters
        ----------
            phi (np.array): azimuth angle [degrees]

        Returns
        -------
            a_h (np.array): horizontal radiation pattern gain value
        """
        is_scalar = not isinstance(phi, (np.ndarray, list))
        phi_a = xp.asarray(phi if not is_scalar else [phi], dtype=xp.float64)

        x_h = xp.abs(phi_a) / float(self.phi_3db)

        # Branchless: region 0 (x_h < 0.5) vs region 1 (x_h >= 0.5)
        g0 = -12.0 * xp.power(x_h, 2.0)
        g1 = -12.0 * xp.power(x_h, 2.0 - float(self.k_h)) - float(self.lambda_k_h)

        gain = xp.where(x_h < 0.5, g0, g1)
        gain = xp.maximum(gain, float(self.g_hr_180))

        if is_scalar:
            return float(backend.asnumpy(gain)[0])
        return backend.asnumpy(gain)

    def vertical_pattern(self, theta: np.array) -> np.array:
        """
        Calculates the vertical radiation pattern (branchless GPU version).

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]

        Returns
        -------
            a_v (np.array): vertical radiation pattern gain value
        """
        is_scalar = not isinstance(theta, (np.ndarray, list))
        theta_a = xp.asarray(
            (theta if not is_scalar else [theta]), dtype=xp.float64
        ) - 90.0

        x_v = xp.abs(theta_a) / float(self.theta_3db)
        x_k = float(self.x_k)
        x_v_safe = xp.maximum(x_v, 1e-30)

        # Four piecewise regions — branchless
        g0 = -12.0 * xp.power(x_v, 2.0)
        g1 = -12.0 + 10.0 * xp.log10(xp.power(x_v_safe, -1.5) + float(self.k_v))
        g2 = -float(self.lambda_k_v) - float(self.incline_factor) * xp.log10(x_v_safe)
        g3 = xp.full_like(x_v, float(self.g_hr_180))

        gain = xp.where(
            x_v < x_k, g0,
            xp.where(
                x_v < 4.0, g1,
                xp.where(x_v < 90.0 / float(self.theta_3db), g2, g3)
            )
        )

        if is_scalar:
            return float(backend.asnumpy(gain)[0])
        return backend.asnumpy(gain)

    def element_pattern(self, phi: np.array, theta: np.array) -> np.array:
        """
        Calculates the element radiation pattern gain.

        Parameters
        ----------
            theta (np.array): elevation angle [degrees]
            phi (np.array): azimuth angle [degrees]

        Returns
        -------
            gain (np.array): element radiation pattern gain value
        """

        gain_hor = self.horizontal_pattern(phi)
        compression_ratio = (gain_hor - self.g_hr_180) / \
            (self.g_hr_0 - self.g_hr_180)
        gain = self.g_max + gain_hor + \
            compression_ratio * self.vertical_pattern(theta)

        return gain


if __name__ == '__main__':

    from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
    from matplotlib import pyplot as plt

    param = ParametersAntennaImt()

    param.element_max_g = 18
    param.element_phi_3db = 65
    param.element_theta_3db = 0

    antenna = AntennaElementImtF1336(param)

    phi_vec = np.arange(-180, 180, step=1)
    theta_vec = np.arange(0, 180, step=1)

    pattern_hor_0deg = antenna.element_pattern(phi_vec, 0)
    pattern_hor_10deg = antenna.element_pattern(phi_vec, 10)
    pattern_hor_30deg = antenna.element_pattern(phi_vec, 30)
    pattern_hor_60deg = antenna.element_pattern(phi_vec, 60)
    pattern_hor_90deg = antenna.element_pattern(phi_vec, 90)

    pattern_ver_0deg = antenna.element_pattern(0, theta_vec)
    pattern_ver_30deg = antenna.element_pattern(30, theta_vec)
    pattern_ver_60deg = antenna.element_pattern(60, theta_vec)
    pattern_ver_90deg = antenna.element_pattern(90, theta_vec)
    pattern_ver_120deg = antenna.element_pattern(120, theta_vec)

    plt.figure(1)
    plt.plot(phi_vec, pattern_hor_0deg, label='elevation = 0 degrees')
    plt.plot(phi_vec, pattern_hor_10deg, label='elevation = 10 degrees')
    plt.plot(phi_vec, pattern_hor_30deg, label='elevation = 30 degrees')
    plt.plot(phi_vec, pattern_hor_60deg, label='elevation = 60 degrees')
    plt.plot(phi_vec, pattern_hor_90deg, label='elevation = 90 degrees')

    plt.title('horizontal pattern')
    plt.xlabel('azimuth (degrees)')
    plt.ylabel('gain (dBi)')
    plt.legend()

    plt.figure(2)
    plt.plot(theta_vec, pattern_ver_0deg, label='azimuth = 0 degrees')
    plt.plot(theta_vec, pattern_ver_30deg, label='azimuth = 30 degrees')
    plt.plot(theta_vec, pattern_ver_60deg, label='azimuth = 60 degrees')
    plt.plot(theta_vec, pattern_ver_90deg, label='azimuth = 90 degrees')
    plt.plot(theta_vec, pattern_ver_120deg, label='azimuth = 120 degrees')

    plt.title('vertical pattern')
    plt.xlabel('elevation (degrees)')
    plt.ylabel('gain (dBi)')
    plt.legend()

    plt.show()
