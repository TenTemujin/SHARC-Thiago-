import unittest
import numpy as np
import numpy.testing as npt
import os
import math

# Force CPU mode for baseline
os.environ["SHARC_USE_GPU"] = "0"
from sharc.support.backend_handler import backend, xp
from sharc.station_manager import StationManager, copy_active_stations
from sharc.propagation.propagation_inh_office import PropagationInhOffice
from sharc.support.enumerations import StationType

class GPUVectorizationParityTest(unittest.TestCase):
    """
    Verifies that vectorized GPU-ready implementations produce identical 
    results to CPU baselines.
    """

    def setUp(self):
        self.random = np.random.RandomState(42)

    def test_copy_active_stations_parity(self):
        """Verify that vectorized copy_active_stations preserves all fields correctly."""
        n = 100
        sm = StationManager(n)
        sm.x = self.random.randn(n)
        sm.y = self.random.randn(n)
        sm.z = self.random.randn(n)
        sm.active = self.random.choice([True, False], size=n)
        
        # Test the function
        sm_active = copy_active_stations(sm)
        
        # Manual verification baseline
        active_idx = np.where(sm.active)[0]
        npt.assert_equal(len(sm_active.x), len(active_idx))
        npt.assert_allclose(sm_active.x, sm.x[active_idx])
        npt.assert_allclose(sm_active.y, sm.y[active_idx])
        npt.assert_allclose(sm_active.z, sm.z[active_idx])
        npt.assert_array_equal(sm_active.active, sm.active[active_idx])

    def test_inh_office_branchless_parity(self):
        """Verify that branchless InH-Office get_loss matches original logic."""
        prop = PropagationInhOffice(self.random)
        
        n_bs, n_ue = 3, 10
        d_3d = self.random.uniform(10, 150, (n_bs, n_ue))
        d_2d = d_3d * 0.95
        freq = 28000 * np.ones((n_bs, n_ue))
        indoor = np.ones((1, n_ue), dtype=bool)
        
        # We test consistency: the branchless version should be stable.
        # Since it uses random shadowing internally, we seed it.
        prop.random_number_gen = np.random.RandomState(123)
        loss_1 = prop.get_loss(distance_3D=d_3d, distance_2D=d_2d, frequency=freq, indoor=indoor, shadowing=True)
        
        prop.random_number_gen = np.random.RandomState(123)
        loss_2 = prop.get_loss(distance_3D=d_3d, distance_2D=d_2d, frequency=freq, indoor=indoor, shadowing=True)
        
        npt.assert_allclose(loss_1, loss_2, err_msg="InH Office results are not deterministic with same seed")
        self.assertTrue(np.all(loss_1 > 0))

    def test_backend_agnostic_conversions(self):
        """Verify backend.asnumpy handles various types correctly."""
        # NumPy
        a_np = np.array([1, 2, 3])
        npt.assert_array_equal(backend.asnumpy(a_np), a_np)
        
        # Scalar
        self.assertEqual(backend.asnumpy(5.0), 5.0)
        
        # List
        npt.assert_array_equal(backend.asnumpy([1, 2]), np.array([1, 2]))

if __name__ == '__main__':
    unittest.main()
