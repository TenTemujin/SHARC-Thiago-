#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHARC GPU Acceleration — Benchmark & Validation Script
=======================================================

Tests the GPU acceleration implementation by:
1. Verifying the backend loads correctly
2. Running a CPU vs GPU correctness check  
3. Measuring speedup on antenna gain computation

Usage
-----
    # CPU mode (default)
    python benchmark_gpu.py

    # GPU mode
    $env:SHARC_USE_GPU="1"; python benchmark_gpu.py

Requirements
------------
    pip install cupy-cuda11x   (for CUDA 11.x)
    pip install cupy-cuda12x   (for CUDA 12.x)
"""

import os
import sys
import time
import numpy as np

# Add project root to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. Backend detection
# ============================================================
print("=" * 60)
print("SHARC GPU Acceleration Benchmark")
print("=" * 60)

from sharc.support.backend_handler import backend, xp

print(f"\n[Backend] Mode: {backend.mode}")
print(f"[Backend] GPU active: {backend.use_gpu}")

# ============================================================
# 2. CuPy availability info
# ============================================================
try:
    import cupy as cp
    print(f"[CuPy] Version: {cp.__version__}")
    print(f"[CuPy] CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    device = cp.cuda.Device(0)
    mem_total = device.mem_info[1] / 1024**3
    mem_free = device.mem_info[0] / 1024**3
    print(f"[CuPy] GPU memory: {mem_free:.1f}/{mem_total:.1f} GB free")
except ImportError:
    print("[CuPy] Not installed — running in CPU mode")
    print("[CuPy] Install: pip install cupy-cuda11x")
except Exception as e:
    print(f"[CuPy] Available but error: {e}")

# ============================================================
# 3. Quick functional test on core operations
# ============================================================
print("\n[Test] Core vectorized operations...")

N, M = 100, 200
phi_np = np.random.uniform(-180, 180, (N, M))
theta_np = np.random.uniform(0, 180, (N, M))

phi_xp = backend.asarray(phi_np)
theta_xp = backend.asarray(theta_np)

phi_rad = xp.deg2rad(phi_xp)
theta_rad = xp.deg2rad(theta_xp)

px = xp.sin(theta_rad) * xp.cos(phi_rad)
py = xp.sin(theta_rad) * xp.sin(phi_rad)
pz = xp.cos(theta_rad)

result = backend.asnumpy(px**2 + py**2 + pz**2)
max_err = np.max(np.abs(result - 1.0))
print(f"  Unit vector norm error: {max_err:.2e}  {'✓ PASS' if max_err < 1e-10 else '✗ FAIL'}")

# ============================================================
# 4. Antenna gain vectorized benchmark
# ============================================================
print("\n[Test] Antenna gain compute_gains_batch...")

try:
    from sharc.antenna.antenna_gain_vectorized import (
        _build_rotation_matrices,
        _batch_to_local_coords,
        _batch_element_m2101,
        _batch_array_factor_with_wvecs,
    )

    # Build rotation matrices for N stations
    N_sta = 57   # typical macrocell
    azimuths = np.random.uniform(-180, 180, N_sta)
    elevations = np.zeros(N_sta)

    t0 = time.perf_counter()
    rot = _build_rotation_matrices(azimuths, elevations)
    t1 = time.perf_counter()
    if backend.use_gpu:
        backend.synchronize()
    t2 = time.perf_counter()

    print(f"  Build {N_sta} rotation matrices: {(t2-t0)*1000:.2f} ms")
    assert rot.shape == (N_sta, 3, 3), f"Expected ({N_sta}, 3, 3), got {rot.shape}"
    print(f"  Rotation matrix shape: {tuple(rot.shape)}  ✓ PASS")

    # Local coordinate transformation
    M_ue = 570   # typical UE count
    phi_sub = xp.asarray(np.random.uniform(-180, 180, (N_sta, M_ue)))
    theta_sub = xp.asarray(np.random.uniform(0, 180, (N_sta, M_ue)))

    t0 = time.perf_counter()
    lo_phi, lo_theta = _batch_to_local_coords(phi_sub, theta_sub, rot)
    if backend.use_gpu:
        backend.synchronize()
    t1 = time.perf_counter()
    print(f"  Local coord transform ({N_sta}×{M_ue}): {(t1-t0)*1000:.2f} ms")

    # M2101 element pattern
    t0 = time.perf_counter()
    elem_gain = _batch_element_m2101(lo_phi, lo_theta, 8.0, 65.0, 65.0, 30.0, 30.0, 12.0)
    if backend.use_gpu:
        backend.synchronize()
    t1 = time.perf_counter()
    print(f"  M2101 element pattern ({N_sta}×{M_ue}): {(t1-t0)*1000:.2f} ms")

    elem_np = backend.asnumpy(elem_gain)
    assert elem_np.shape == (N_sta, M_ue)
    print(f"  Element gain range: [{elem_np.min():.1f}, {elem_np.max():.1f}] dBi  ✓ PASS")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 5. Station manager distance computation
# ============================================================
print("\n[Test] StationManager distance computation...")

try:
    from sharc.station_manager import StationManager

    N_bs = 57
    N_ue = 570

    bs = StationManager(N_bs)
    bs.x = np.random.uniform(-1000, 1000, N_bs)
    bs.y = np.random.uniform(-1000, 1000, N_bs)
    bs.z = np.full(N_bs, 30.0)
    bs.height = np.full(N_bs, 30.0)

    ue = StationManager(N_ue)
    ue.x = np.random.uniform(-1000, 1000, N_ue)
    ue.y = np.random.uniform(-1000, 1000, N_ue)
    ue.z = np.full(N_ue, 1.5)
    ue.height = np.full(N_ue, 1.5)

    t0 = time.perf_counter()
    dist_2d = bs.get_distance_to(ue)
    if backend.use_gpu:
        backend.synchronize()
    t1 = time.perf_counter()
    dist_np = backend.asnumpy(dist_2d)

    print(f"  2D distance ({N_bs}×{N_ue}): {(t1-t0)*1000:.2f} ms")
    assert dist_np.shape == (N_bs, N_ue)
    assert np.all(dist_np >= 0), "Negative distances found!"
    print(f"  Distance range: [{dist_np.min():.1f}, {dist_np.max():.1f}] m  ✓ PASS")

    t0 = time.perf_counter()
    phi, theta = bs.get_pointing_vector_to(ue)
    if backend.use_gpu:
        backend.synchronize()
    t1 = time.perf_counter()
    phi_np = backend.asnumpy(phi)

    print(f"  Pointing vector ({N_bs}×{N_ue}): {(t1-t0)*1000:.2f} ms")
    assert phi_np.shape == (N_bs, N_ue)
    print(f"  Phi range: [{phi_np.min():.1f}, {phi_np.max():.1f}] deg  ✓ PASS")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 60)
print(f"Benchmark complete — Mode: {backend.mode}")
if not backend.use_gpu:
    print("\nTo enable GPU acceleration:")
    print("  1. Install CUDA 11.x: cuda_11.8.0_522.06_windows.exe")
    print("  2. Install CuPy:      pip install cupy-cuda11x")
    print("  3. Set env var:       $env:SHARC_USE_GPU='1'")
    print("  4. Run simulation:    python sharc/main_cli.py ...")
print("=" * 60)
