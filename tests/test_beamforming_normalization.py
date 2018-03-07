# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:30:15 2018

@author: Calil
"""

import unittest
import numpy as np
import numpy.testing as npt

from sharc.antenna.beamforming_normalization.beamforming_normalization import BeamformingNormalization
from sharc.support.named_tuples import AntennaPar
from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt

class BeamformingNormalizationTest(unittest.TestCase):
    
    def setUp(self):
        # Test 1
        resolution = 30
        self.norm_1 = BeamformingNormalization(resolution)
        element_pattern = "FIXED"
        element_max_g = 0
        element_phi_deg_3db = 65
        element_theta_deg_3db = 65
        element_am = 30
        element_sla_v = 30
        n_rows = 8
        n_columns = 8
        horiz_spacing = 0.5
        vert_spacing = 0.5
        down_tilt = 0
        self.par_1 = AntennaPar(element_pattern,
                                element_max_g,
                                element_phi_deg_3db,
                                element_theta_deg_3db,
                                element_am,
                                element_sla_v,
                                n_rows,
                                n_columns,
                                horiz_spacing,
                                vert_spacing,
                                down_tilt)
    
    def test_construction(self):
        # Test 1
        self.assertEqual(self.norm_1.res_deg,30)
        self.assertEqual(self.norm_1.phi_min_deg,0)
        self.assertEqual(self.norm_1.phi_max_deg,360)
        self.assertEqual(self.norm_1.theta_min_deg,0)
        self.assertEqual(self.norm_1.theta_max_deg,180)
        self.assertEqual(self.norm_1.phi_min_rad,0)
        self.assertEqual(self.norm_1.phi_max_rad,2*np.pi)
        self.assertEqual(self.norm_1.theta_min_rad,0)
        self.assertEqual(self.norm_1.theta_max_rad,np.pi)
        npt.assert_equal(self.norm_1.phi_vals_deg,np.arange(0,360,30))
        npt.assert_equal(self.norm_1.theta_vals_deg,np.arange(0,180,30))
        
    def test_calculate_correction_factor(self):
        # Test 1
        azi = 0
        ele = 0
        self.norm_1.antenna = AntennaBeamformingImt(self.par_1,azi,ele)
        # Test adjacent channel case: single antenna element
        c_chan = False
        c_fac, err = self.norm_1.calculate_correction_factor(0,0,c_chan)
        self.assertAlmostEqual(c_fac,0.0,delta = 1e-2)
        self.assertLess(np.max(np.abs(err)),1e-3)
        
    def test_generate_correction_matrix(self):
        pass
    
    def test_save_files(self):
        pass
    
if __name__ == '__main__':
    unittest.main()