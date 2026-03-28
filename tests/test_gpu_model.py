import unittest
import numpy as np
import numpy.testing as npt

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from sharc.station_manager import StationManager
from sharc.support import backend_handler as bh
import sharc.station_manager as sm

# Original pure-NumPy logic for reliable CPU baseline comparisons
def cpu_get_distance_to(sm1, sm2):
    distance = np.empty([sm1.num_stations, sm2.num_stations])
    for i in range(sm1.num_stations):
        distance[i] = np.sqrt(
            np.power(sm1.x[i] - sm2.x, 2) +
            np.power(sm1.y[i] - sm2.y, 2),
        )
    return distance

def cpu_get_3d_distance_to(sm1, sm2):
    dx = np.subtract.outer(sm1.x, sm2.x).astype(np.float64)
    dy = np.subtract.outer(sm1.y, sm2.y).astype(np.float64)
    dz = np.subtract.outer(sm1.z, sm2.z).astype(np.float64)
    np.square(dx, out=dx)
    np.square(dy, out=dy)
    np.square(dz, out=dz)
    return np.sqrt(dx + dy + dz)

def _cpu_rot_ecef_to_enu(lat_deg, lon_deg):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    sl, cl = np.sin(lat), np.cos(lat)
    sb, cb = np.sin(lon), np.cos(lon)

    R = np.empty((lat.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = -sb; R[:, 0, 1] =  cb; R[:, 0, 2] =  0.0
    R[:, 1, 0] = -sl * cb; R[:, 1, 1] = -sl * sb; R[:, 1, 2] =  cl
    R[:, 2, 0] =  cl * cb; R[:, 2, 1] =  cl * sb; R[:, 2, 2] =  sl
    return R

def cpu_get_pointing_vector_to(sm1, sm2):
    if (sm1.latitude[0] != 0):
        dx = -(sm1.x[None, :] - sm2.x[:, None])
        dy = -(sm1.y[None, :] - sm2.y[:, None])
        dz = -(sm1.z[None, :] - sm2.z[:, None])
        v_ecef = np.stack([dx, dy, dz], axis=-1)
        dist   = np.linalg.norm(v_ecef, axis=-1)
        dist_safe = np.where(dist == 0.0, 1.0, dist)

        R = _cpu_rot_ecef_to_enu(sm1.latitude, sm1.longitude)
        v_ecef = np.swapaxes(v_ecef, 0, 1)
        dist_safe = np.swapaxes(dist_safe, 0, 1)
        v_enu = np.einsum('nij,nmj->nmi', R, v_ecef)
        E = v_enu[..., 0]; N = v_enu[..., 1]; U = v_enu[..., 2]

        phi = np.degrees(np.arctan2(N, E))
        cos_th = np.clip(U / dist_safe, -1.0, 1.0)
        theta  = np.degrees(np.arccos(cos_th))
    else:
        dx = (sm2.x - sm1.x[:, np.newaxis]).astype(np.float64)
        dy = (sm2.y - sm1.y[:, np.newaxis]).astype(np.float64)
        dz = (sm2.z - sm1.z[:, np.newaxis]).astype(np.float64)

        dist = cpu_get_3d_distance_to(sm1, sm2)
        phi = np.rad2deg(np.arctan2(dy, dx))
        theta = np.rad2deg(np.arccos(np.clip(dz / dist, -1.0, 1.0)))
    return phi, theta

def cpu_get_off_axis_angle(sm1, sm2):
    Az, b = cpu_get_pointing_vector_to(sm1, sm2)
    Az0 = sm1.azimuth

    a = 90 - sm1.elevation[:, np.newaxis]
    C = Az0[:, np.newaxis] - Az

    cos_phi = np.cos(np.radians(a)) * np.cos(np.radians(b)) \
        + np.sin(np.radians(a)) * np.sin(np.radians(b)) * np.cos(np.radians(C))
    phi = np.arccos(np.clip(cos_phi, -1., 1.))
    return np.degrees(phi)

class GPUModelTest(unittest.TestCase):
    """
    Unit test to verify that the GPU CuPy implementations in SHARC (particularly inside StationManager) 
    produce the exact identical numerical results as the baseline pure NumPy implementations (explicitly isolated).
    """

    @unittest.skipIf(not HAS_CUPY, "CuPy is not installed on this system. Skipping GPU Parity Test.")
    def test_cpu_gpu_parity(self):
        # Setup Station Managers
        sm1 = StationManager(3)
        sm1.x = np.array([10, 20, 30])
        sm1.y = np.array([15, 25, 35])
        sm1.z = np.array([1, 2, 3])
        sm1.height = np.array([1, 2, 3])
        sm1.latitude = np.array([0, 0, 0])
        sm1.azimuth = np.array([0, 45, 90])
        sm1.elevation = np.array([0, -10, 10])

        sm2 = StationManager(2)
        sm2.x = np.array([100, 200])
        sm2.y = np.array([105, 250])
        sm2.z = np.array([4, 5])
        sm2.height = np.array([4, 5])
        sm2.latitude = np.array([0, 0])
        sm2.azimuth = np.array([180, 225])
        sm2.elevation = np.array([0, 0])

        # Execute pure CPU baseline logic statically isolated in Python tests
        cpu_dist_2d = cpu_get_distance_to(sm1, sm2)
        cpu_dist_3d = cpu_get_3d_distance_to(sm1, sm2)
        cpu_phi, cpu_theta = cpu_get_pointing_vector_to(sm1, sm2)
        cpu_off_axis = cpu_get_off_axis_angle(sm1, sm2)

        # Patch Module attributes to use GPU CuPy
        bh.backend.use_gpu = True
        bh.backend.xp = cp
        sm.xp = cp
        
        gpu_dist_2d = sm1.get_distance_to(sm2)
        gpu_dist_3d = sm1.get_3d_distance_to(sm2)
        gpu_phi, gpu_theta = sm1.get_pointing_vector_to(sm2)
        gpu_off_axis = sm1.get_off_axis_angle(sm2)
        
        # Download GPU arrays to CPU for assertions against baseline output
        gpu_dist_2d_np = bh.backend.asnumpy(gpu_dist_2d)
        gpu_dist_3d_np = bh.backend.asnumpy(gpu_dist_3d)
        gpu_phi_np = bh.backend.asnumpy(gpu_phi)
        gpu_theta_np = bh.backend.asnumpy(gpu_theta)
        gpu_off_axis_np = bh.backend.asnumpy(gpu_off_axis)

        # Assert Exactness within threshold
        npt.assert_allclose(cpu_dist_2d, gpu_dist_2d_np, atol=1e-5, err_msg="2D Distance Parity Failed")
        npt.assert_allclose(cpu_dist_3d, gpu_dist_3d_np, atol=1e-5, err_msg="3D Distance Parity Failed")
        npt.assert_allclose(cpu_phi, gpu_phi_np, atol=1e-5, err_msg="Pointing Azimuth Parity Failed")
        npt.assert_allclose(cpu_theta, gpu_theta_np, atol=1e-5, err_msg="Pointing Elevation Parity Failed")
        npt.assert_allclose(cpu_off_axis, gpu_off_axis_np, atol=1e-5, err_msg="Off-Axis Angle Parity Failed")
        
        # Cleanup
        bh.backend.use_gpu = False
        bh.backend.xp = np
        sm.xp = np

if __name__ == '__main__':
    unittest.main()
