# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:29:48 2017

@author: edgar

GPU Acceleration
----------------
This module uses the backend-agnostic xp/backend interface from
sharc.support.backend_handler. When SHARC_USE_GPU=1, all array
operations run on the GPU via CuPy. Otherwise, NumPy is used.
"""

import numpy as np

from sharc.support.enumerations import StationType
from sharc.station import Station
from sharc.antenna.antenna import Antenna
from sharc.mask.spectral_mask import SpectralMask
from sharc.support.backend_handler import xp, backend

_WGS84_A  = 6378137.0                 # semi-major axis [m]
_WGS84_F  = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

class StationManager(object):
    """
    This is the base class that manages an array of stations that will be
    used during a simulation. It acts like a container that vectorizes the
    station properties to speed up calculations.
    """

    def __init__(self, n):
        self.num_stations = n
        self.x = np.empty(n)  # x coordinate
        self.y = np.empty(n)  # y coordinate
        self.z = np.empty(n)  # z coordinate (includes height above ground)
        self.latitude = np.zeros(n, dtype=float)  # Latitude of station
        self.longitude = np.zeros(n, dtype=float)  # Longitude of Base Station
        self.azimuth = np.empty(n)
        self.elevation = np.empty(n)
        self.height = np.empty(n)  # station height above ground
        self.idx_orbit = np.empty(n)
        self.indoor = np.zeros(n, dtype=bool)
        self.active = np.ones(n, dtype=bool)
        self.tx_power = np.empty(n)
        self.rx_power = np.empty(n)
        self.rx_interference = np.empty(n)  # Rx interferece in dBW
        self.ext_interference = np.empty(n)  # External interferece in dBW
        self.antenna = np.empty(n, dtype=Antenna)
        self.bandwidth = np.empty(n)  # Bandwidth in MHz
        self.noise_figure = np.empty(n)
        self.noise_temperature = np.empty(n)
        self.thermal_noise = np.empty(n)
        self.total_interference = np.empty(n)
        self.pfd_external = np.empty(n)  # External PFD in dBW/m²/MHz
        # Aggregated External PFD in dBW/m²/MHz
        self.pfd_external_aggregated = np.empty(n)
        self.snr = np.empty(n)
        self.sinr = np.empty(n)
        self.sinr_ext = np.empty(n)
        self.inr = np.empty(n)  # INR in dBm/MHz
        self.pfd = np.empty(n)  # Powerflux density in dBm/m^2
        self.spectral_mask = np.empty(n, dtype=SpectralMask)
        self.center_freq = np.empty(n)
        self.station_type = StationType.NONE
        self.is_space_station = False
        self.intersite_dist = 0.0

    def get_station_list(self, id=None) -> list:
        """Return a list of Station objects for the given indices.

        Parameters
        ----------
        id : iterable or None, optional
            Indices of stations to retrieve. If None, returns all stations.

        Returns
        -------
        list
            List of Station objects.
        """
        if (id is None):
            id = range(self.num_stations)
        station_list = list()
        for i in id:
            station_list.append(self.get_station(i))
        return station_list

    def get_station(self, id) -> Station:
        """Return a Station object for the given index.

        Parameters
        ----------
        id : int
            Index of the station to retrieve.

        Returns
        -------
        Station
            Station object with properties set from the manager.
        """
        station = Station()
        station.id = id
        station.x = self.x[id]
        station.y = self.y[id]
        station.z = self.z[id]
        station.azimuth = self.azimuth[id]
        station.elevation = self.elevation[id]
        station.height = self.height[id]
        station.indoor = self.indoor[id]
        station.active = self.active[id]
        station.tx_power = self.tx_power[id]
        station.rx_power = self.rx_power[id]
        station.rx_interference = self.rx_interference[id]
        station.ext_interference = self.ext_interference[id]
        station.antenna = self.antenna[id]
        station.bandwidth = self.bandwidth[id]
        station.noise_figure = self.noise_figure[id]
        station.noise_temperature = self.noise_temperature[id]
        station.thermal_noise = self.thermal_noise[id]
        station.total_interference = self.total_interference[id]
        station.snr = self.snr[id]
        station.sinr = self.sinr[id]
        station.sinr_ext = self.sinr_ext[id]
        station.inr = self.inr[id]
        station.station_type = self.station_type
        return station

    def get_distance_to(self, station) -> np.array:
        """Calculate the 2D distance between this manager's stations and another's.

        Parameters
        ----------
        station : StationManager
            StationManager to which the distance is calculated.

        Returns
        -------
        np.array
            2D distance matrix between stations.
        """
        # Vectorized outer subtraction — eliminates Python loop, runs on GPU
        dx = xp.subtract.outer(
            backend.asarray(self.x), backend.asarray(station.x)
        ).astype(xp.float64)
        dy = xp.subtract.outer(
            backend.asarray(self.y), backend.asarray(station.y)
        ).astype(xp.float64)
        return xp.sqrt(dx * dx + dy * dy)

    def get_3d_distance_to(self, station) -> np.array:
        """Calculate the 3D distance between this manager's stations and another's.

        Parameters
        ----------
        station : StationManager
            StationManager to which the distance is calculated.

        Returns
        -------
        np.array
            3D distance matrix between stations.
        """
        dx = xp.subtract.outer(backend.asarray(self.x), backend.asarray(station.x)).astype(xp.float64)
        dy = xp.subtract.outer(backend.asarray(self.y), backend.asarray(station.y)).astype(xp.float64)
        dz = xp.subtract.outer(backend.asarray(self.z), backend.asarray(station.z)).astype(xp.float64)
        xp.square(dx, out=dx)
        xp.square(dy, out=dy)
        xp.square(dz, out=dz)
        xp.sqrt(dx + dy + dz, out=dx)
        return dx

    def get_dist_angles_wrap_around(self, station) -> np.array:
        """Calculate distances and angles using the wrap-around technique.

        Parameters
        ----------
        station : StationManager
            StationManager to which distances and angles are calculated.

        Returns
        -------
        tuple
            distance_2D (np.array): 2D distance between stations
            distance_3D (np.array): 3D distance between stations
            phi (np.array): azimuth of pointing vector to other stations
            theta (np.array): elevation of pointing vector to other stations
        """
        sx = backend.asarray(station.x)
        sy = backend.asarray(station.y)
        sh = backend.asarray(station.height)
        nx = backend.asarray(self.x)
        ny = backend.asarray(self.y)
        nh = backend.asarray(self.height)
        dist_int = float(self.intersite_dist)

        # Initialize on GPU
        distance_3D = xp.empty([self.num_stations, station.num_stations], dtype=xp.float64)
        distance_2D = xp.full_like(distance_3D, xp.inf)
        cluster_num = xp.zeros_like(distance_3D, dtype=int)

        # Cluster coordinates (7 candidate cluster positions)
        sqrt3 = float(np.sqrt(3.0))
        cluster_x = xp.array([
            sx,
            sx + 3.5 * dist_int,
            sx - 0.5 * dist_int,
            sx - 4.0 * dist_int,
            sx - 3.5 * dist_int,
            sx + 0.5 * dist_int,
            sx + 4.0 * dist_int,
        ])

        cluster_y = xp.array([
            sy,
            sy + 1.5 * sqrt3 * dist_int,
            sy + 2.5 * sqrt3 * dist_int,
            sy + 1.0 * sqrt3 * dist_int,
            sy - 1.5 * sqrt3 * dist_int,
            sy - 2.5 * sqrt3 * dist_int,
            sy - 1.0 * sqrt3 * dist_int,
        ])

        # Calculate minimum 2D distance across clusters
        for k, (cx, cy) in enumerate(zip(cluster_x, cluster_y)):
            temp_distance = xp.sqrt(
                xp.power(cx - nx[:, xp.newaxis], 2) +
                xp.power(cy - ny[:, xp.newaxis], 2),
            )
            is_shorter = temp_distance < distance_2D
            distance_2D = xp.where(is_shorter, temp_distance, distance_2D)
            cluster_num = xp.where(is_shorter, k, cluster_num)

        # Calculate 3D distance
        distance_3D = xp.sqrt(
            xp.power(distance_2D, 2) +
            xp.power(sh - nh[:, xp.newaxis], 2),
        )

        # Calculate pointing vector using winning cluster coordinates
        point_vec_x = cluster_x[cluster_num, xp.arange(station.num_stations)] \
            - nx[:, xp.newaxis]
        point_vec_y = cluster_y[cluster_num, xp.arange(station.num_stations)] \
            - ny[:, xp.newaxis]
        point_vec_z = sh - nh[:, xp.newaxis]

        phi = xp.array(
            xp.rad2deg(xp.arctan2(point_vec_y, point_vec_x)),
            ndmin=2,
        )
        theta = xp.rad2deg(xp.arccos(
            xp.clip(point_vec_z / distance_3D, -1.0, 1.0)
        ))

        return distance_2D, distance_3D, phi, theta

    def get_elevation(self, station) -> np.array:
        """Calculate the elevation angle between this manager's stations and another's.

        Parameters
        ----------
        station : StationManager
            StationManager to which the elevation angle is calculated.

        Returns
        -------
        np.array
            Elevation angle matrix (degrees).

        Notes
        -----
        This implementation is essentially the same as get_elevation_angle (free-space elevation),
        despite the different matrix dimensions. Vectorized: no Python loop.
        """
        distance = self.get_distance_to(station)
        sz = backend.asarray(station.z)
        mz = backend.asarray(self.z)
        rel_z = sz - mz[:, xp.newaxis]
        return xp.degrees(xp.arctan2(rel_z, distance))

    def get_pointing_vector_to(self, station) -> tuple:
        """Calculate the pointing vector (angles) with respect to another station.

        Parameters
        ----------
        station : StationManager
            The other StationManager to calculate the pointing vector to.

        Returns
        -------
        tuple
            phi, theta (phi is calculated with respect to x counter-clockwise and
            theta is calculated with respect to z counter-clockwise).
        """
        if (self.latitude[0] != 0):
            # Geodetic (lat/lon) coordinate system path
            # LOS in ECEF, broadcast to (N,M,3)
            bx = backend.asarray(self.x)
            by = backend.asarray(self.y)
            bz = backend.asarray(self.z)
            sx = backend.asarray(station.x)
            sy = backend.asarray(station.y)
            sz = backend.asarray(station.z)

            dx = -(bx[None, :] - sx[:, None])
            dy = -(by[None, :] - sy[:, None])
            dz = -(bz[None, :] - sz[:, None])
            v_ecef = xp.stack([dx, dy, dz], axis=-1)             # (N,M,3)
            dist = xp.linalg.norm(v_ecef, axis=-1)
            dist_safe = xp.where(dist == 0.0, 1.0, dist)

            # Rotation ECEF->ENU at each BS
            R = backend.asarray(_rot_ecef_to_enu(self.latitude, self.longitude))  # (N,3,3)

            # Rotate LOS into ENU of each BS
            v_ecef = xp.swapaxes(v_ecef, 0, 1)
            dist_safe = xp.swapaxes(dist_safe, 0, 1)
            v_enu = xp.einsum('nij,nmj->nmi', R, v_ecef)         # (N,M,3)
            E = v_enu[..., 0]
            N_comp = v_enu[..., 1]
            U = v_enu[..., 2]

            phi = xp.degrees(xp.arctan2(N_comp, E))              # [-180,180]
            cos_th = xp.clip(U / dist_safe, -1.0, 1.0)
            theta = xp.degrees(xp.arccos(cos_th))                # [0,180]
        else:
            # Cartesian coordinate system path
            bx = backend.asarray(self.x)
            by = backend.asarray(self.y)
            bz = backend.asarray(self.z)
            sx = backend.asarray(station.x)
            sy = backend.asarray(station.y)
            sz = backend.asarray(station.z)

            dx = (sx - bx[:, xp.newaxis]).astype(xp.float64)
            dy = (sy - by[:, xp.newaxis]).astype(xp.float64)
            dz = (sz - bz[:, xp.newaxis]).astype(xp.float64)

            dist = self.get_3d_distance_to(station)

            # In-place calculations to minimize memory allocations
            phi = xp.rad2deg(xp.arctan2(dy, dx))
            theta = xp.rad2deg(xp.arccos(xp.clip(dz / dist, -1.0, 1.0)))

        return phi, theta

    def get_off_axis_angle(self, station) -> np.array:
        """Calculate the off-axis angle between this manager's stations and another's.

        Parameters
        ----------
        station : StationManager
            The other StationManager to calculate the off-axis angle to.

        Returns
        -------
        np.array
            Off-axis angle matrix (degrees).
        """
        Az, b = self.get_pointing_vector_to(station)
        Az0 = backend.asarray(self.azimuth)

        a = 90.0 - backend.asarray(self.elevation)[:, xp.newaxis]
        C = Az0[:, xp.newaxis] - Az

        cos_phi = (
            xp.cos(xp.radians(a)) * xp.cos(xp.radians(b))
            + xp.sin(xp.radians(a)) * xp.sin(xp.radians(b)) * xp.cos(xp.radians(C))
        )
        # Numerical imprecision may push values slightly out of [-1, 1]
        phi_deg = xp.degrees(xp.arccos(xp.clip(cos_phi, -1.0, 1.0)))

        return phi_deg

    def is_imt_station(self) -> bool:
        """Return whether this station manager represents IMT stations.

        Returns
        -------
        bool
            True if this station manager is IMT (IMT_BS or IMT_UE), False otherwise.
        """
        if self.station_type is StationType.IMT_BS or self.station_type is StationType.IMT_UE:
            return True
        else:
            return False

def _lla_to_ecef(lat_deg, lon_deg, h_m):
    """Vectorized geodetic (deg,deg,m) -> ECEF XYZ (m) on WGS-84."""
    lat = xp.radians(backend.asarray(lat_deg, dtype=float))
    lon = xp.radians(backend.asarray(lon_deg, dtype=float))
    h = backend.asarray(h_m, dtype=float)

    sl, cl = xp.sin(lat), xp.cos(lat)
    sb, cb = xp.sin(lon), xp.cos(lon)

    N = _WGS84_A / xp.sqrt(1.0 - _WGS84_E2 * sl * sl)
    X = (N + h) * cl * cb
    Y = (N + h) * cl * sb
    Z = (N * (1.0 - _WGS84_E2) + h) * sl
    return X, Y, Z


def _rot_ecef_to_enu(lat_deg, lon_deg):
    """
    Vectorized rotation matrices R (N,3,3) that map v_ecef -> [E,N,U] at each (lat,lon).
    Rows are the ENU basis vectors (always computed on CPU as NumPy for compatibility).
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    sl, cl = np.sin(lat), np.cos(lat)
    sb, cb = np.sin(lon), np.cos(lon)

    # Each R has rows [east; north; up]
    # east  = [-sin(lon),  cos(lon), 0]
    # north = [-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat)]
    # up    = [ cos(lat)cos(lon),  cos(lat)sin(lon), sin(lat)]
    R = np.empty((lat.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = -sb
    R[:, 0, 1] =  cb
    R[:, 0, 2] =  0.0

    R[:, 1, 0] = -sl * cb
    R[:, 1, 1] = -sl * sb
    R[:, 1, 2] =  cl

    R[:, 2, 0] =  cl * cb
    R[:, 2, 1] =  cl * sb
    R[:, 2, 2] =  sl
    return R

def copy_active_stations(stations: StationManager) -> StationManager:
    """Return a new StationManager object containing only the active stations.

    Parameters
    ----------
    stations : StationManager
        StationManager object to copy from.

    Returns
    -------
    StationManager
        A new StationManager object with only the active stations.
    """
    act_sta = StationManager(np.sum(stations.active))
    for idx, active_idx in enumerate(np.where(stations.active)[0]):
        act_sta.x[idx] = stations.x[active_idx]
        act_sta.y[idx] = stations.y[active_idx]
        act_sta.z[idx] = stations.z[active_idx]
        act_sta.azimuth[idx] = stations.azimuth[active_idx]
        act_sta.elevation[idx] = stations.elevation[active_idx]
        act_sta.height[idx] = stations.height[active_idx]
        act_sta.indoor[idx] = stations.indoor[active_idx]
        act_sta.active[idx] = stations.active[active_idx]
        act_sta.tx_power[idx] = stations.tx_power[active_idx]
        act_sta.rx_power[idx] = stations.rx_power[active_idx]
        act_sta.rx_interference[idx] = stations.rx_interference[active_idx]
        act_sta.ext_interference[idx] = stations.ext_interference[active_idx]
        act_sta.antenna[idx] = stations.antenna[active_idx]
        act_sta.bandwidth[idx] = stations.bandwidth[active_idx]
        act_sta.noise_figure[idx] = stations.noise_figure[active_idx]
        act_sta.noise_temperature[idx] = stations.noise_temperature[active_idx]
        act_sta.thermal_noise[idx] = stations.thermal_noise[active_idx]
        act_sta.total_interference[idx] = stations.total_interference[active_idx]
        act_sta.snr[idx] = stations.snr[active_idx]
        act_sta.sinr[idx] = stations.sinr[active_idx]
        act_sta.sinr_ext[idx] = stations.sinr_ext[active_idx]
        act_sta.inr[idx] = stations.inr[active_idx]
        act_sta.pfd[idx] = stations.pfd[active_idx]
        act_sta.spectral_mask = stations.spectral_mask
        act_sta.center_freq[idx] = stations.center_freq[active_idx]
        act_sta.station_type = stations.station_type
        act_sta.is_space_station = stations.is_space_station
        act_sta.intersite_dist = stations.intersite_dist
    return act_sta
