# -*- coding: utf-8 -*-
"""
Vectorized (batched) antenna gain computation for AntennaBeamformingImt arrays.

Eliminates per-station and per-direction Python loops by computing all gains
in a single batched operation on the GPU (or CPU via NumPy).

The key insight: for IMT stations of the same type, the antenna array geometry
(n_rows, n_cols, dh, dv) and element pattern parameters are identical.
Only the rotation matrices and beam weights differ per station.

This module provides drop-in replacements that operate on (N_stations, N_targets)
shaped arrays, computing all gains simultaneously.
"""

import numpy as np
from sharc.support.backend_handler import xp, backend


def _build_rotation_matrices(azimuths, elevations):
    """Build batched rotation matrices from arrays of azimuth/elevation.

    Parameters
    ----------
    azimuths : array (N,) — physical antenna azimuths [degrees]
    elevations : array (N,) — physical antenna elevations [degrees]

    Returns
    -------
    rot : array (N, 3, 3)
    """
    alpha = xp.deg2rad(backend.asarray(azimuths))
    beta = xp.deg2rad(backend.asarray(elevations))

    ca, sa = xp.cos(alpha), xp.sin(alpha)
    cb, sb = xp.cos(beta), xp.sin(beta)

    N = len(alpha)
    rot = xp.zeros((N, 3, 3), dtype=xp.float64)

    # Ry @ Rz^T  (matches AntennaBeamformingImt._calculate_rotation_matrix)
    # Ry = [[cb, 0, sb],[0,1,0],[-sb,0,cb]]
    # Rz^T = [[ca, sa, 0],[-sa, ca, 0],[0,0,1]]
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


def _batch_to_local_coords(phi, theta, rotation_matrices):
    """Transform global (phi, theta) to antenna-local coordinates for all antennas.

    Parameters
    ----------
    phi : array (N, M) — global azimuth angles [degrees]
    theta : array (N, M) — global elevation angles [degrees]
    rotation_matrices : array (N, 3, 3)

    Returns
    -------
    lo_phi, lo_theta : arrays (N, M) — local coordinates [degrees]
    """
    phi_rad = xp.deg2rad(phi)
    theta_rad = xp.deg2rad(theta)

    sin_t = xp.sin(theta_rad)
    cos_t = xp.cos(theta_rad)

    # Direction vectors: (N, M, 3)
    px = sin_t * xp.cos(phi_rad)
    py = sin_t * xp.sin(phi_rad)
    pz = cos_t

    # Batched rotation: R(N,3,3) @ p(N,M,3) -> (N,M,3) via einsum
    rx = rotation_matrices[:, 0, 0][:, None] * px + \
         rotation_matrices[:, 0, 1][:, None] * py + \
         rotation_matrices[:, 0, 2][:, None] * pz
    ry = rotation_matrices[:, 1, 0][:, None] * px + \
         rotation_matrices[:, 1, 1][:, None] * py + \
         rotation_matrices[:, 1, 2][:, None] * pz
    rz = rotation_matrices[:, 2, 0][:, None] * px + \
         rotation_matrices[:, 2, 1][:, None] * py + \
         rotation_matrices[:, 2, 2][:, None] * pz

    lo_phi = xp.rad2deg(xp.arctan2(ry, rx))
    lo_theta = xp.rad2deg(xp.arccos(xp.clip(rz, -1.0, 1.0)))

    return lo_phi, lo_theta


def _batch_element_m2101(lo_phi, lo_theta, g_max, phi_3db, theta_3db, am, sla_v, mult):
    """Vectorized M2101 element pattern for arbitrary shaped arrays.

    Returns element gain with same shape as lo_phi.
    """
    a_h = -1.0 * xp.minimum(mult * (lo_phi / phi_3db) ** 2, float(am))
    a_v = -1.0 * xp.minimum(mult * ((lo_theta - 90.0) / theta_3db) ** 2, float(sla_v))
    return g_max - xp.minimum(-(a_h + a_v), float(am))


def _batch_element_f1336(lo_phi, lo_theta, elem):
    """Vectorized F1336 element pattern.

    Parameters
    ----------
    lo_phi, lo_theta : arrays (any shape) — local coordinates [degrees]
    elem : AntennaElementImtF1336 instance
    """
    # Horizontal
    x_h = xp.abs(lo_phi) / elem.phi_3db
    gain_h = xp.where(
        x_h < 0.5,
        -12.0 * x_h ** 2,
        -12.0 * xp.power(x_h, 2.0 - elem.k_h) - elem.lambda_k_h,
    )
    gain_h = xp.maximum(gain_h, elem.g_hr_180)

    # Vertical
    theta_a = lo_theta - 90.0
    x_v = xp.abs(theta_a) / elem.theta_3db
    x_v_safe = xp.maximum(x_v, 1e-30)

    g0 = -12.0 * x_v ** 2
    g1 = -12.0 + 10.0 * xp.log10(xp.power(x_v_safe, -1.5) + elem.k_v)
    g2 = -elem.lambda_k_v - elem.incline_factor * xp.log10(x_v_safe)
    g3 = xp.full_like(x_v, elem.g_hr_180)

    gain_v = xp.where(x_v < elem.x_k, g0,
             xp.where(x_v < 4.0, g1,
             xp.where(x_v < 90.0 / elem.theta_3db, g2, g3)))

    compression = (gain_h - elem.g_hr_180) / (elem.g_hr_0 - elem.g_hr_180)
    return elem.g_max + gain_h + compression * gain_v


def _batch_subarray_factor(lo_theta, n_rows_sub, dv_sub, etilt_sub):
    """Compute subarray array factor gain.

    Parameters
    ----------
    lo_theta : array (any shape) — local elevation [degrees]
    n_rows_sub, dv_sub, etilt_sub : subarray parameters

    Returns
    -------
    array_g : array same shape as lo_theta — array factor gain [dB]
    """
    r_theta = xp.deg2rad(lo_theta)
    m = xp.arange(n_rows_sub, dtype=xp.float64)  # 0..n-1

    # Super-position: v_n = exp(2πj * m * dv * cos(θ))
    cos_t = xp.cos(r_theta)
    v_exp = 2.0 * xp.pi * 1j * m * dv_sub * cos_t[..., None]
    v_n = xp.exp(v_exp)  # (..., n_rows_sub)

    # Weight: w_n = (1/√n) * exp(2πj * m * dv * sin(etilt))
    sin_e = np.sin(np.deg2rad(etilt_sub))
    w_n = (1.0 / np.sqrt(n_rows_sub)) * xp.exp(
        2.0 * xp.pi * 1j * m * dv_sub * sin_e
    )

    inner = xp.sum(v_n * w_n, axis=-1)
    return 10.0 * xp.log10(xp.abs(inner) ** 2 + 1e-30)


def _batch_array_factor_with_wvecs(lo_phi, lo_theta, w_vecs, n_rows, n_cols, dh, dv):
    """Compute beamforming array factor for batched inputs with pre-selected weights.

    Parameters
    ----------
    lo_phi : array (N, M) — local azimuth [deg]
    lo_theta : array (N, M) — local elevation [deg]
    w_vecs : array (N, M, n_rows, n_cols) complex — weight vectors
    n_rows, n_cols, dh, dv : antenna config

    Returns
    -------
    array_g : (N, M) — array gain [dB]
    """
    r_phi = xp.deg2rad(lo_phi)
    r_theta = xp.deg2rad(lo_theta)

    # Index arrays (0-based like original code: (n-1) where n starts at 1)
    nr = xp.arange(n_rows, dtype=xp.float64)
    mc = xp.arange(n_cols, dtype=xp.float64)

    cos_t = xp.cos(r_theta)[..., None, None]          # (N, M, 1, 1)
    st_sp = (xp.sin(r_theta) * xp.sin(r_phi))[..., None, None]

    # exp_arg: (N, M, n_rows, n_cols)
    exp_arg = nr[None, None, :, None] * dv * cos_t + \
              mc[None, None, None, :] * dh * st_sp

    v_vec = xp.exp(2.0 * xp.pi * 1j * exp_arg)

    inner = xp.sum(v_vec * w_vecs, axis=(-2, -1))
    return 10.0 * xp.log10(xp.abs(inner) ** 2 + 1e-30)


def compute_gains_batch(
    phi, theta,
    station_1_active, station_2_active,
    antennas, beams_idx,
    co_channel=True,
    expand_bs_beams=False,
    ue_k=1,
):
    """Main entry point: compute antenna gains for all (station, target) pairs.

    Parameters
    ----------
    phi, theta : array (N_total, M_total) — global angles
    station_1_active : 1D int array — active source station indices
    station_2_active : 1D int array — active target station indices
    antennas : array of AntennaBeamformingImt — one per source station
    beams_idx : 1D int array — beam indices (per target for normal, per virtual row for expanded)
    co_channel : bool
    expand_bs_beams : bool — if True, phi/theta rows are K-expanded (BS→system case)
    ue_k : int — beams per BS (only used when expand_bs_beams=True)

    Returns
    -------
    gains : array (N_total, M_total) — gains in original indexing
    """
    from sharc.antenna.antenna_element_imt_m2101 import AntennaElementImtM2101
    from sharc.antenna.antenna_element_imt_f1336 import AntennaElementImtF1336

    ref_ant = antennas[station_1_active[0]]
    n_rows = ref_ant.n_rows
    n_cols = ref_ant.n_cols
    dh = ref_ant.dh
    dv = ref_ant.dv
    min_gain = ref_ant.minimum_array_gain
    has_subarray = ref_ant.subarray is not None

    # Determine element type
    is_m2101 = isinstance(ref_ant.element, AntennaElementImtM2101)
    is_f1336 = isinstance(ref_ant.element, AntennaElementImtF1336)

    # Determine co_channel behaviour for adjacent
    effective_co = co_channel
    if not co_channel:
        if ref_ant.adjacent_antenna_model == "BEAMFORMING":
            effective_co = True

    s1a = station_1_active
    s2a = station_2_active

    # --- Build rotation matrices ---
    if expand_bs_beams:
        # Virtual rows: each BS has K rows, same rotation matrix
        parent_bs = np.repeat(s1a, ue_k)
        rot_matrices = _build_rotation_matrices(
            np.array([antennas[k].azimuth for k in parent_bs]),
            np.array([antennas[k].elevation for k in parent_bs]),
        )
        # Virtual row indices in the phi/theta array
        virtual_rows = []
        for k in s1a:
            for b in range(ue_k):
                virtual_rows.append(k * ue_k + b)
        virtual_rows = np.array(virtual_rows)
        N = len(virtual_rows)
    else:
        rot_matrices = _build_rotation_matrices(
            np.array([antennas[k].azimuth for k in s1a]),
            np.array([antennas[k].elevation for k in s1a]),
        )
        virtual_rows = s1a
        N = len(s1a)

    M = len(s2a)

    # --- Extract sub-matrix of phi/theta ---
    phi_gpu = backend.asarray(phi)
    theta_gpu = backend.asarray(theta)
    phi_sub = phi_gpu[virtual_rows][:, s2a]      # (N, M)
    theta_sub = theta_gpu[virtual_rows][:, s2a]  # (N, M)

    # --- Transform to local coordinates ---
    lo_phi, lo_theta = _batch_to_local_coords(phi_sub, theta_sub, rot_matrices)

    # --- Element pattern ---
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
        # Constant element or unknown — use g_max
        elem_gain = xp.full_like(lo_phi, ref_ant.element.g_max if hasattr(ref_ant.element, 'g_max') else 0.0)

    # --- Subarray factor ---
    if has_subarray:
        sub = ref_ant.subarray
        sub_factor = _batch_subarray_factor(lo_theta, sub.n_rows, sub.dv_sub, sub.eletrical_downtilt)
        elem_gain = elem_gain + sub_factor

    # --- Array factor (co-channel or beamforming adjacent) ---
    if effective_co:
        # Build w_vecs tensor: (N, M, n_rows, n_cols)
        w_vecs_np = np.zeros((N, M, n_rows, n_cols), dtype=np.complex128)

        if expand_bs_beams:
            for v_idx, v_row in enumerate(virtual_rows):
                parent_k = s1a[v_idx // ue_k]
                beam_b = v_idx % ue_k
                ant = antennas[parent_k]
                if beam_b < len(ant.w_vec_list):
                    w = ant.w_vec_list[beam_b]
                    w_vecs_np[v_idx, :] = w[None, :, :]  # broadcast to all targets
        else:
            bi = np.asarray(beams_idx, dtype=int)
            for s_idx, k in enumerate(s1a):
                ant = antennas[k]
                wl = ant.w_vec_list
                for t_idx in range(M):
                    b = int(bi[t_idx]) if t_idx < len(bi) else 0
                    if b >= 0 and b < len(wl):
                        w_vecs_np[s_idx, t_idx] = wl[b]
                    elif len(wl) > 0:
                        w_vecs_np[s_idx, t_idx] = wl[0]

        w_vecs_gpu = backend.asarray(w_vecs_np)
        array_g = _batch_array_factor_with_wvecs(
            lo_phi, lo_theta, w_vecs_gpu, n_rows, n_cols, dh, dv,
        )

        # Correction factors
        corr = xp.zeros((N, M), dtype=xp.float64)
        if ref_ant.normalize:
            if expand_bs_beams:
                for v_idx in range(N):
                    parent_k = s1a[v_idx // ue_k]
                    beam_b = v_idx % ue_k
                    ant = antennas[parent_k]
                    if beam_b < len(ant.co_correction_factor_list):
                        corr[v_idx, :] = float(ant.co_correction_factor_list[beam_b])
            else:
                for s_idx, k in enumerate(s1a):
                    ant = antennas[k]
                    for t_idx in range(M):
                        b = int(beams_idx[t_idx]) if t_idx < len(beams_idx) else 0
                        if b >= 0 and b < len(ant.co_correction_factor_list):
                            corr[s_idx, t_idx] = float(ant.co_correction_factor_list[b])

        gains_sub = elem_gain + array_g + corr
    else:
        # Adjacent channel, SINGLE_ELEMENT: just element pattern + adj correction
        adj_corr = ref_ant.adj_correction_factor
        gains_sub = elem_gain + adj_corr

    gains_sub = xp.maximum(gains_sub, min_gain)

    # --- Write back to full-size output ---
    gains = xp.zeros(phi_gpu.shape, dtype=xp.float64)
    # Use loop-free indexing
    row_idx = backend.asarray(virtual_rows)
    col_idx = backend.asarray(s2a)
    gains[xp.ix_(row_idx, col_idx)] = gains_sub

    return backend.asnumpy(gains)
