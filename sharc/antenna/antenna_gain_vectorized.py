# -*- coding: utf-8 -*-
"""
Vectorized (batched) antenna gain computation for AntennaBeamformingImt arrays.
================================================================================

Eliminates per-station and per-direction Python loops by computing all gains
in a single batched operation on the GPU (or CPU fallback via NumPy).

Key insight: for IMT stations of the same type, the antenna array geometry
(n_rows, n_cols, dh, dv) and element pattern parameters are identical.
Only the rotation matrices and beam weights differ per station.

This module provides drop-in replacements operating on (N_stations, N_targets)
shaped arrays, computing all gains simultaneously — yielding linear speedup
proportional to the number of stations on GPU.

Architecture
------------
    1. _build_rotation_matrices      — Ry @ Rz^T for each station (batched)
    2. _batch_to_local_coords        — global → antenna-local coords (batched einsum)
    3. _batch_element_m2101          — M2101 single-element pattern (vectorized)
    4. _batch_element_f1336          — F1336 single-element pattern (vectorized)
    5. _batch_subarray_factor        — Subarray array factor (vectorized)
    6. _batch_array_factor_with_wvecs — Full beamforming array factor (batched)
    7. compute_gains_batch           — Main entry point

Usage
-----
    from sharc.antenna.antenna_gain_vectorized import compute_gains_batch

    gains = compute_gains_batch(
        phi, theta,
        station_1_active, station_2_active,
        antennas, beams_idx,
        co_channel=True,
    )
"""

import numpy as np
from sharc.support.backend_handler import xp, backend


# ---------------------------------------------------------------------------
# Rotation matrix construction
# ---------------------------------------------------------------------------

def _build_rotation_matrices(azimuths, elevations):
    """Build batched rotation matrices from arrays of azimuth/elevation.

    Computes Ry @ Rz^T for each station — matches
    AntennaBeamformingImt._calculate_rotation_matrix().

    Parameters
    ----------
    azimuths : array-like (N,)
        Physical antenna azimuths [degrees].
    elevations : array-like (N,)
        Physical antenna elevations [degrees].

    Returns
    -------
    rot : xp.ndarray (N, 3, 3)
        Batched rotation matrices.
    """
    alpha = xp.deg2rad(backend.asarray(azimuths).astype(xp.float64))
    beta = xp.deg2rad(backend.asarray(elevations).astype(xp.float64))

    ca, sa = xp.cos(alpha), xp.sin(alpha)
    cb, sb = xp.cos(beta), xp.sin(beta)

    N = len(alpha)
    rot = xp.zeros((N, 3, 3), dtype=xp.float64)

    # Ry @ Rz^T:
    # Ry  = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
    # Rz^T= [[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]]
    rot[:, 0, 0] = cb * ca
    rot[:, 0, 1] = cb * sa
    rot[:, 0, 2] = sb
    rot[:, 1, 0] = -sa
    rot[:, 1, 1] = ca
    rot[:, 1, 2] = 0.0
    rot[:, 2, 0] = -sb * ca
    rot[:, 2, 1] = -sb * sa
    rot[:, 2, 2] = cb

    return rot


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def _batch_to_local_coords(phi, theta, rotation_matrices):
    """Transform global (phi, theta) to antenna-local coordinates (batched).

    Parameters
    ----------
    phi : xp.ndarray (N, M)
        Global azimuth angles [degrees].
    theta : xp.ndarray (N, M)
        Global elevation angles [degrees].
    rotation_matrices : xp.ndarray (N, 3, 3)
        Per-station rotation matrices.

    Returns
    -------
    lo_phi : xp.ndarray (N, M)
        Local azimuth angles [degrees].
    lo_theta : xp.ndarray (N, M)
        Local elevation angles [degrees].
    """
    phi_rad = xp.deg2rad(phi)
    theta_rad = xp.deg2rad(theta)

    sin_t = xp.sin(theta_rad)
    cos_t = xp.cos(theta_rad)

    # Cartesian direction vectors: (N, M)
    px = sin_t * xp.cos(phi_rad)
    py = sin_t * xp.sin(phi_rad)
    pz = cos_t

    # Batched matrix-vector product: R(N,3,3) × p(N,M) → (N,M)
    # Expanded row-by-row to avoid explicit einsum (better GPU memory layout)
    R = rotation_matrices  # (N, 3, 3)
    rx = R[:, 0, 0][:, None] * px + R[:, 0, 1][:, None] * py + R[:, 0, 2][:, None] * pz
    ry = R[:, 1, 0][:, None] * px + R[:, 1, 1][:, None] * py + R[:, 1, 2][:, None] * pz
    rz = R[:, 2, 0][:, None] * px + R[:, 2, 1][:, None] * py + R[:, 2, 2][:, None] * pz

    lo_phi = xp.rad2deg(xp.arctan2(ry, rx))
    lo_theta = xp.rad2deg(xp.arccos(xp.clip(rz, -1.0, 1.0)))

    return lo_phi, lo_theta


# ---------------------------------------------------------------------------
# Element patterns (vectorized)
# ---------------------------------------------------------------------------

def _batch_element_m2101(lo_phi, lo_theta, g_max, phi_3db, theta_3db, am, sla_v, mult):
    """Vectorized ITU-R M.2101 element radiation pattern.

    Works for arbitrary shaped arrays (N, M) or any shape.

    Parameters
    ----------
    lo_phi, lo_theta : xp.ndarray
        Local coordinate angles [degrees].
    g_max, phi_3db, theta_3db, am, sla_v, mult : float
        Antenna element parameters.

    Returns
    -------
    xp.ndarray
        Element gain [dBi], same shape as lo_phi.
    """
    a_h = -1.0 * xp.minimum(
        mult * (lo_phi / phi_3db) ** 2,
        float(am)
    )
    a_v = -1.0 * xp.minimum(
        mult * ((lo_theta - 90.0) / theta_3db) ** 2,
        float(sla_v)
    )
    return float(g_max) - xp.minimum(-(a_h + a_v), float(am))


def _batch_element_f1336(lo_phi, lo_theta, elem):
    """Vectorized ITU-R F.1336 element radiation pattern.

    Parameters
    ----------
    lo_phi, lo_theta : xp.ndarray
        Local coordinate angles [degrees] — any shape.
    elem : AntennaElementImtF1336
        Element instance (parameters read from it).

    Returns
    -------
    xp.ndarray
        Element gain [dBi], same shape as lo_phi.
    """
    # --- Horizontal pattern ---
    x_h = xp.abs(lo_phi) / float(elem.phi_3db)
    gain_h = xp.where(
        x_h < 0.5,
        -12.0 * x_h ** 2,
        -12.0 * xp.power(x_h, 2.0 - float(elem.k_h)) - float(elem.lambda_k_h),
    )
    gain_h = xp.maximum(gain_h, float(elem.g_hr_180))

    # --- Vertical pattern ---
    theta_a = lo_theta - 90.0
    x_v = xp.abs(theta_a) / float(elem.theta_3db)
    # Protect against log10(0)
    x_v_safe = xp.maximum(x_v, 1e-30)

    g0 = -12.0 * x_v ** 2
    g1 = -12.0 + 10.0 * xp.log10(xp.power(x_v_safe, -1.5) + float(elem.k_v))
    g2 = -float(elem.lambda_k_v) - float(elem.incline_factor) * xp.log10(x_v_safe)
    g3 = xp.full_like(x_v, float(elem.g_hr_180))

    gain_v = xp.where(x_v < float(elem.x_k), g0,
              xp.where(x_v < 4.0, g1,
              xp.where(x_v < 90.0 / float(elem.theta_3db), g2, g3)))

    compression = (gain_h - float(elem.g_hr_180)) / (float(elem.g_hr_0) - float(elem.g_hr_180))
    return float(elem.g_max) + gain_h + compression * gain_v


def _batch_element_const(lo_phi, g_max):
    """Constant element pattern (isotropic element with fixed gain).

    Parameters
    ----------
    lo_phi : xp.ndarray
        Array of any shape (used only for shape).
    g_max : float
        Constant gain value.

    Returns
    -------
    xp.ndarray
        Constant gain array of same shape as lo_phi.
    """
    return xp.full_like(lo_phi, float(g_max), dtype=xp.float64)


# ---------------------------------------------------------------------------
# Subarray factor (vectorized)
# ---------------------------------------------------------------------------

def _batch_subarray_factor(lo_theta, n_rows_sub, dv_sub, etilt_sub):
    """Vectorized subarray array factor gain.

    Parameters
    ----------
    lo_theta : xp.ndarray
        Local elevation angles [degrees], any shape.
    n_rows_sub : int
        Number of subarray rows.
    dv_sub : float
        Subarray element vertical spacing (d/lambda).
    etilt_sub : float
        Electrical downtilt [degrees].

    Returns
    -------
    xp.ndarray
        Subarray array factor gain [dB], same shape as lo_theta.
    """
    r_theta = xp.deg2rad(lo_theta)
    # Row indices 0..n-1
    m = xp.arange(n_rows_sub, dtype=xp.float64)

    # Super-position vector: v_n = exp(2πj · m · dv · cos(θ))
    cos_t = xp.cos(r_theta)
    # Broadcast: (..., n_rows_sub)
    v_exp = 2.0 * xp.pi * 1j * m * dv_sub * cos_t[..., None]
    v_n = xp.exp(v_exp)

    # Weight vector: w_n = (1/√n) · exp(2πj · m · dv · sin(etilt))
    sin_e = float(np.sin(np.deg2rad(etilt_sub)))
    w_n = (1.0 / np.sqrt(n_rows_sub)) * xp.exp(
        2.0 * xp.pi * 1j * m * dv_sub * sin_e
    )

    inner = xp.sum(v_n * w_n, axis=-1)
    # Protect against log10(0)
    return 10.0 * xp.log10(xp.abs(inner) ** 2 + 1e-30)


# ---------------------------------------------------------------------------
# Beamforming array factor (batched)
# ---------------------------------------------------------------------------

def _batch_array_factor_with_wvecs(lo_phi, lo_theta, w_vecs, n_rows, n_cols, dh, dv):
    """Compute beamforming array factor for all (station, target) pairs.

    This is the main GPU kernel: for each (station i, target j), compute
    the array factor using the pre-selected weight vector.

    Parameters
    ----------
    lo_phi : xp.ndarray (N, M)
        Local azimuth [degrees].
    lo_theta : xp.ndarray (N, M)
        Local elevation [degrees].
    w_vecs : xp.ndarray (N, M, n_rows, n_cols) complex
        Pre-selected weight vectors (one per station-target pair).
    n_rows, n_cols : int
        Antenna array dimensions.
    dh, dv : float
        Element spacing (horizontal, vertical) in wavelengths.

    Returns
    -------
    xp.ndarray (N, M)
        Array factor gain [dB].
    """
    r_phi = xp.deg2rad(lo_phi)
    r_theta = xp.deg2rad(lo_theta)

    # 0-based row/column indices
    nr = xp.arange(n_rows, dtype=xp.float64)   # (n_rows,)
    mc = xp.arange(n_cols, dtype=xp.float64)   # (n_cols,)

    # Pre-compute trig terms: (N, M, 1, 1)
    cos_t = xp.cos(r_theta)[..., None, None]
    st_sp = (xp.sin(r_theta) * xp.sin(r_phi))[..., None, None]

    # Superposition exponent argument: (N, M, n_rows, n_cols)
    exp_arg = (nr[None, None, :, None] * dv * cos_t +
               mc[None, None, None, :] * dh * st_sp)

    v_vec = xp.exp(2.0 * xp.pi * 1j * exp_arg)

    # Element-wise product and sum over array
    inner = xp.sum(v_vec * w_vecs, axis=(-2, -1))   # (N, M)

    # Protect against log10(0) with small epsilon
    return 10.0 * xp.log10(xp.abs(inner) ** 2 + 1e-30)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_gains_batch(
    phi, theta,
    station_1_active, station_2_active,
    antennas, beams_idx,
    co_channel=True,
    expand_bs_beams=False,
    ue_k=1,
):
    """Compute antenna gains for all (station, target) pairs in one GPU kernel.

    Replaces the ``for k in station_1_active`` loop in ``simulation.py::calculate_gains()``.

    Parameters
    ----------
    phi : np.ndarray (N_total, M_total)
        Global azimuth angles [degrees].
    theta : np.ndarray (N_total, M_total)
        Global elevation angles [degrees].
    station_1_active : np.ndarray (N_act,)
        Active source station indices.
    station_2_active : np.ndarray (M_act,)
        Active target station indices.
    antennas : array-like of AntennaBeamformingImt
        One antenna object per source station.
    beams_idx : np.ndarray (M_act,)
        Beam indices per target (for co-channel w/ fixed beam).
    co_channel : bool, optional
        If True, use full array pattern. If False, use element pattern. Default True.
    expand_bs_beams : bool, optional
        If True, phi/theta rows are K-expanded (BS → system case). Default False.
    ue_k : int, optional
        Number of beams per BS (used when expand_bs_beams=True). Default 1.

    Returns
    -------
    gains : np.ndarray (N_total, M_total)
        Antenna gains in original (full) indexing. Non-active cells are 0.0.
    """
    from sharc.antenna.antenna_element_imt_m2101 import AntennaElementImtM2101
    from sharc.antenna.antenna_element_imt_f1336 import AntennaElementImtF1336

    s1a = np.asarray(station_1_active, dtype=int)
    s2a = np.asarray(station_2_active, dtype=int)

    if len(s1a) == 0 or len(s2a) == 0:
        return np.zeros(np.asarray(phi).shape)

    # Read parameters from the first active antenna (all same type/geometry)
    ref_ant = antennas[s1a[0]]
    n_rows = ref_ant.n_rows
    n_cols = ref_ant.n_cols
    dh = ref_ant.dh
    dv = ref_ant.dv
    min_gain = ref_ant.minimum_array_gain
    has_subarray = ref_ant.subarray is not None

    # Determine element type
    is_m2101 = isinstance(ref_ant.element, AntennaElementImtM2101)
    is_f1336 = isinstance(ref_ant.element, AntennaElementImtF1336)

    # Resolve adjacent-channel beamforming model
    effective_co = co_channel
    if not co_channel:
        if ref_ant.adjacent_antenna_model == "BEAMFORMING":
            effective_co = True

    # ------------------------------------------------------------------
    # Build rotation matrices and virtual row indexing
    # ------------------------------------------------------------------
    if expand_bs_beams:
        # Each BS occupies `ue_k` consecutive rows in phi/theta matrix
        parent_bs = np.repeat(s1a, ue_k)
        rot_matrices = _build_rotation_matrices(
            np.array([antennas[k].azimuth for k in parent_bs], dtype=float),
            np.array([antennas[k].elevation for k in parent_bs], dtype=float),
        )
        virtual_rows = np.array([
            k * ue_k + b
            for k in s1a
            for b in range(ue_k)
        ], dtype=int)
    else:
        rot_matrices = _build_rotation_matrices(
            np.array([antennas[k].azimuth for k in s1a], dtype=float),
            np.array([antennas[k].elevation for k in s1a], dtype=float),
        )
        virtual_rows = s1a.copy()

    N = len(virtual_rows)
    M = len(s2a)

    # ------------------------------------------------------------------
    # Extract relevant sub-matrix and move to GPU
    # ------------------------------------------------------------------
    phi_gpu = backend.asarray(np.asarray(phi, dtype=float))
    theta_gpu = backend.asarray(np.asarray(theta, dtype=float))

    phi_sub = phi_gpu[virtual_rows][:, s2a]      # (N, M)
    theta_sub = theta_gpu[virtual_rows][:, s2a]  # (N, M)

    # ------------------------------------------------------------------
    # Transform to antenna-local coordinates
    # ------------------------------------------------------------------
    lo_phi, lo_theta = _batch_to_local_coords(phi_sub, theta_sub, rot_matrices)

    # ------------------------------------------------------------------
    # Element pattern
    # ------------------------------------------------------------------
    if is_m2101:
        elem = ref_ant.element
        elem_gain = _batch_element_m2101(
            lo_phi, lo_theta,
            elem.g_max, elem.phi_3db, elem.theta_3db,
            elem.am, elem.sla_v, elem.multiplication_factor,
        )
    elif is_f1336:
        elem_gain = _batch_element_f1336(lo_phi, lo_theta, ref_ant.element)
    else:
        # Constant element (AntennaElementImtConst) or unknown
        g = float(ref_ant.element.g_max) if hasattr(ref_ant.element, 'g_max') else 0.0
        elem_gain = _batch_element_const(lo_phi, g)

    # ------------------------------------------------------------------
    # Subarray factor (if applicable)
    # ------------------------------------------------------------------
    if has_subarray:
        sub = ref_ant.subarray
        sub_factor = _batch_subarray_factor(
            lo_theta, sub.n_rows, sub.dv_sub, sub.eletrical_downtilt
        )
        elem_gain = elem_gain + sub_factor

    # ------------------------------------------------------------------
    # Array factor
    # ------------------------------------------------------------------
    if effective_co:
        # ------------------------------------------------------------------
        # Vectorized weight-vector assembly
        # ------------------------------------------------------------------
        # FAST PATH: build a compact weight library (N, n_beams_max, nr, nc)
        # and perform beam selection on the GPU with a single fancy-index op.
        # This replaces the O(N × M) Python loop with an O(N × K) loop
        # (N_stations × n_beams_per_station — typically 50-100× smaller).
        # PCIe upload: (N, K, nr, nc) complex128 instead of (N, M, nr, nc).
        # Example: 57 BS × 3 beams × 8×8 = ~168 KB  vs  57×171×8×8 = ~10 MB.
        # ------------------------------------------------------------------

        if expand_bs_beams:
            # Each virtual row corresponds to a single, fixed beam from its parent BS.
            # Build one weight per virtual row, then broadcast across M targets on GPU.
            all_wl_exp = np.zeros((N, n_rows, n_cols), dtype=np.complex128)
            for v_idx in range(N):
                parent_k = s1a[v_idx // ue_k]
                beam_b = v_idx % ue_k
                wl = antennas[parent_k].w_vec_list
                if beam_b < len(wl):
                    all_wl_exp[v_idx] = wl[beam_b]
            # Upload (N, nr, nc) and broadcast across M targets — zero extra PCIe
            wl_gpu_exp = backend.asarray(all_wl_exp)          # (N, nr, nc)
            w_vecs_gpu = xp.ascontiguousarray(
                xp.broadcast_to(wl_gpu_exp[:, None, :, :], (N, M, n_rows, n_cols))
            )
        else:
            # Main path: beams_idx selects one of K beams per target (same for all BSs).
            # Build compact library (N, K, nr, nc) → index on GPU → (N, M, nr, nc).
            bi = np.asarray(beams_idx, dtype=int)

            n_beams_per = [len(antennas[k].w_vec_list) for k in s1a]
            n_beams_max = max(n_beams_per) if n_beams_per else 1

            # O(N × K) Python loop — e.g. 57 BSs × 3 beams = 171 iterations
            all_wl_np = np.zeros((N, n_beams_max, n_rows, n_cols), dtype=np.complex128)
            for s_idx, k in enumerate(s1a):
                wl = antennas[k].w_vec_list
                nb = len(wl)
                if nb > 0:
                    # np.stack: (nb, nr, nc) — one NumPy call per station
                    all_wl_np[s_idx, :nb] = np.stack(wl, axis=0)

            # Clamp beam indices (handles -1 / out-of-range gracefully)
            bi_safe = np.clip(bi, 0, n_beams_max - 1)  # (M,)

            # Upload compact library — then index once on GPU
            all_wl_gpu = backend.asarray(all_wl_np)      # (N, K, nr, nc)
            bi_safe_gpu = backend.asarray(bi_safe)         # (M,)
            w_vecs_gpu = all_wl_gpu[:, bi_safe_gpu, :, :] # (N, M, nr, nc) — GPU fancy-index

        array_g = _batch_array_factor_with_wvecs(
            lo_phi, lo_theta, w_vecs_gpu, n_rows, n_cols, dh, dv,
        )

        # ------------------------------------------------------------------
        # Normalization correction factors — same vectorized strategy
        # ------------------------------------------------------------------
        corr = xp.zeros((N, M), dtype=xp.float64)
        if ref_ant.normalize:
            if expand_bs_beams:
                # One scalar per virtual row, broadcast to M targets
                cf_per_row = np.zeros(N, dtype=np.float64)
                for v_idx in range(N):
                    parent_k = s1a[v_idx // ue_k]
                    beam_b = v_idx % ue_k
                    cflist = antennas[parent_k].co_correction_factor_list
                    if beam_b < len(cflist):
                        cf_per_row[v_idx] = float(cflist[beam_b])
                cf_gpu = backend.asarray(cf_per_row)       # (N,)
                corr = xp.broadcast_to(cf_gpu[:, None], (N, M)).copy()
            else:
                # O(N × K) compact build, then GPU fancy-index → (N, M)
                all_cf_np = np.zeros((N, n_beams_max), dtype=np.float64)
                for s_idx, k in enumerate(s1a):
                    cflist = antennas[k].co_correction_factor_list
                    for b_idx, cf in enumerate(cflist):
                        if b_idx < n_beams_max:
                            all_cf_np[s_idx, b_idx] = float(cf)
                all_cf_gpu = backend.asarray(all_cf_np)    # (N, K)
                corr = all_cf_gpu[:, bi_safe_gpu]           # (N, M) — GPU fancy-index

        gains_sub = elem_gain + array_g + corr

    else:
        # Adjacent channel: single element pattern + correction
        adj_corr = float(ref_ant.adj_correction_factor)
        gains_sub = elem_gain + float(adj_corr)

    # ------------------------------------------------------------------
    # Apply minimum array gain
    # ------------------------------------------------------------------
    gains_sub = xp.maximum(gains_sub, float(min_gain))

    # ------------------------------------------------------------------
    # Write results back into full-size output array (CPU)
    # Scatter only the active (N_act × M_act) sub-block — avoids
    # downloading a full phi-sized GPU matrix when most stations are inactive.
    # ------------------------------------------------------------------
    gains_sub_np = backend.asnumpy(gains_sub)   # (N_act, M_act) — compact download
    gains_full = np.zeros(np.asarray(phi).shape, dtype=np.float64)
    gains_full[np.ix_(virtual_rows, s2a)] = gains_sub_np
    return gains_full
