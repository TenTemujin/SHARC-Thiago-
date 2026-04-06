# -*- coding: utf-8 -*-
"""
GPU-Accelerated Propagation Models — Drop-in backend for SHARC
=============================================================

Wraps all propagation model computations to run on GPU (CuPy) when
SHARC_USE_GPU=1, falling back transparently to NumPy.

Key improvements over original:
- All numpy array ops replaced with xp (CuPy/NumPy)
- `np.where` masked scatter eliminated for UMa/UMi LOS split
  → replaced by `xp.where` (branchless, GPU-friendly)
- LOS probability fully vectorized (no scatter indexing)
- `random_number_gen` sample generated on CPU then moved to GPU
- Free-space loss: pure elementwise, no change needed (already fast)

Usage
-----
    from sharc.propagation.propagation_gpu import PropagationUMaGPU

These classes fully replace PropagationUMa/UMi/ABG/FreeSpace
when SHARC_USE_GPU=1.  They share the same interface so they can
be used as drop-in replacements in propagation_factory.py.
"""

import numpy as np
from sharc.support.backend_handler import xp, backend


# ---------------------------------------------------------------------------
# Helper: branchless LOS/NLOS merge
# ---------------------------------------------------------------------------

def _where_split(condition, val_true, val_false):
    """GPU-friendly branchless merge of two value arrays.

    Replaces the `loss[i_los] = ...` / `loss[i_nlos] = ...` scatter pattern
    with a single `xp.where` — avoids index synchronisation between CPU and GPU.

    Parameters
    ----------
    condition : xp.ndarray (bool)
    val_true, val_false : xp.ndarray
        Same shape as condition.

    Returns
    -------
    xp.ndarray
    """
    return xp.where(condition, val_true, val_false)


# ---------------------------------------------------------------------------
# Free Space (GPU)
# ---------------------------------------------------------------------------

def free_space_loss_gpu(frequency, distance):
    """Vectorized free-space path loss on GPU.

    Parameters
    ----------
    frequency : float or xp.ndarray
        Frequency [MHz].
    distance : xp.ndarray
        3D distance [m], any shape.

    Returns
    -------
    xp.ndarray
        Path loss [dB], same shape as distance.
    """
    dist = backend.asarray(distance)
    freq = backend.asarray(frequency) if not np.isscalar(frequency) else float(frequency)
    return 20 * xp.log10(xp.maximum(dist, 1e-6)) + 20 * xp.log10(freq) - 27.55


# ---------------------------------------------------------------------------
# UMa — GPU-accelerated
# ---------------------------------------------------------------------------

def uma_get_los_probability_gpu(distance_2d, h_ue):
    """Vectorized UMa LOS probability (ITU-R 3GPP TR 38.900).

    Replaces numpy scatter pattern with branchless xp.where.

    Parameters
    ----------
    distance_2d : xp.ndarray (N_bs, N_ue)
        2D distances [m].
    h_ue : xp.ndarray (N_ue,)
        UE antenna heights [m].

    Returns
    -------
    xp.ndarray (N_bs, N_ue)
        LOS probability values.
    """
    d2d = backend.asarray(distance_2d)
    h = backend.asarray(h_ue, dtype=xp.float64)

    # c_prime: (N_ue,) → broadcast to (N_bs, N_ue)
    c_prime_ue = xp.where(
        h > 13.0,
        xp.power(xp.abs((h - 13.0) / 10.0), 1.5),
        xp.zeros_like(h),
    )
    c_prime = c_prime_ue[xp.newaxis, :]  # (1, N_ue) broadcasts

    # Far-distance: (18/d + exp(-d/63) * (1 - 18/d)) * (1 + 1.25 * c' * (d/100)^3 * exp(-d/150))
    d_safe = xp.maximum(d2d, 1e-6)
    p_los_far = (
        (18.0 / d_safe + xp.exp(-d_safe / 63.0) * (1.0 - 18.0 / d_safe))
        * (1.0 + 1.25 * c_prime * xp.power(d_safe / 100.0, 3) * xp.exp(-d_safe / 150.0))
    )

    return xp.where(d2d > 18.0, p_los_far, xp.ones_like(d2d))


def uma_get_breakpoint_distance_gpu(frequency, h_bs, h_ue, h_e):
    """UMa breakpoint distance — GPU vectorized.

    Parameters
    ----------
    frequency : float
    h_bs : xp.ndarray (N_bs,)
    h_ue : xp.ndarray (N_ue,)
    h_e  : xp.ndarray (N_bs, N_ue)  — effective environment height
    """
    h_bs_a = backend.asarray(h_bs, dtype=xp.float64)
    h_ue_a = backend.asarray(h_ue, dtype=xp.float64)
    h_e_a = backend.asarray(h_e, dtype=xp.float64)   # (N_bs, N_ue)
    freq = float(frequency)

    # h_e is 2D: subtract per-row (BS) and per-column (UE)
    h_bs_eff = h_bs_a[:, xp.newaxis] - h_e_a          # (N_bs, N_ue)
    h_ue_eff = h_ue_a[xp.newaxis, :] - h_e_a          # (N_bs, N_ue)
    return 4.0 * h_bs_eff * h_ue_eff * (freq * 1e6) / 3e8


def uma_get_loss_los_gpu(distance_2d, distance_3d, frequency,
                          h_bs, h_ue, h_e, shadowing_std, rng):
    """UMa LOS path loss — GPU vectorized, branchless.

    Uses xp.where to avoid scatter indexing:
    - Near region (d < d_bp): free-space formula
    - Far region (d >= d_bp): extended formula
    """
    d2d = backend.asarray(distance_2d, dtype=xp.float64)
    d3d = backend.asarray(distance_3d, dtype=xp.float64)
    freq = backend.asarray(frequency, dtype=xp.float64)
    h_bs_a = backend.asarray(h_bs, dtype=xp.float64)
    h_ue_a = backend.asarray(h_ue, dtype=xp.float64)
    h_e_a = backend.asarray(h_e, dtype=xp.float64)

    d_bp = uma_get_breakpoint_distance_gpu(
        backend.asnumpy(freq).flat[0] if hasattr(freq, 'flat') else float(freq),
        h_bs_a, h_ue_a, h_e_a,
    )

    # Near formula: 20*log10(d3d) + 20*log10(f) - 27.55
    loss_near = (
        20.0 * xp.log10(xp.maximum(d3d, 1e-6))
        + 20.0 * xp.log10(xp.maximum(freq, 1.0))
        - 27.55
    )

    # Far formula: 40*log10(d3d) + 20*log10(f) - 27.55 + fitting_term
    fitting_term = -10.0 * xp.log10(
        d_bp ** 2 + (h_bs_a[:, xp.newaxis] - h_ue_a[xp.newaxis, :]) ** 2 + 1e-30
    )
    loss_far = (
        40.0 * xp.log10(xp.maximum(d3d, 1e-6))
        + 20.0 * xp.log10(xp.maximum(freq, 1.0))
        - 27.55
        + fitting_term
    )

    loss = xp.where(d2d < d_bp, loss_near, loss_far)

    if shadowing_std > 0:
        shad_np = rng.normal(0, shadowing_std, distance_2d.shape)
        loss = loss + backend.asarray(shad_np)

    return loss


def uma_get_loss_nlos_gpu(distance_2d, distance_3d, frequency,
                           h_bs, h_ue, h_e, shadowing_std, rng,
                           loss_los_ref=None):
    """UMa NLOS path loss — GPU vectorized.

    Parameters
    ----------
    loss_los_ref : xp.ndarray, optional
        Pre-computed LOS loss (no shadowing). If provided, avoids recomputing LOS.
    """
    d3d = backend.asarray(distance_3d, dtype=xp.float64)
    d2d = backend.asarray(distance_2d, dtype=xp.float64)
    freq = backend.asarray(frequency, dtype=xp.float64)
    h_ue_a = backend.asarray(h_ue, dtype=xp.float64)

    loss_nlos = (
        -46.46
        + 39.08 * xp.log10(xp.maximum(d3d, 1e-6))
        + 20.0 * xp.log10(xp.maximum(freq, 1.0))
        - 0.6 * (h_ue_a[xp.newaxis, :] - 1.5)
    )

    # UMa NLOS >= UMa LOS (for d2d < 5000) — reuse pre-computed LOS if available
    if loss_los_ref is None:
        loss_los_ref = uma_get_loss_los_gpu(
            distance_2d, distance_3d, frequency, h_bs, h_ue, h_e, 0, rng,
        )
    d2d_mask = d2d < 5000.0
    loss_nlos = xp.where(d2d_mask, xp.maximum(loss_los_ref, loss_nlos), loss_nlos)

    if shadowing_std > 0:
        # .shape on a CuPy array does NOT require GPU sync — no roundtrip needed
        shad_np = rng.normal(0, shadowing_std, d3d.shape)
        loss_nlos = loss_nlos + backend.asarray(shad_np)

    return loss_nlos


def uma_get_loss_gpu(distance_3d, distance_2d, frequency,
                      h_bs, h_ue, shadowing, rng):
    """UMa combined LOS+NLOS path loss — GPU vectorized, no scatter.

    This is the main entry point for GPU-accelerated UMa loss.
    Replaces the `i_los`/`i_nlos` scatter pattern with branchless merge.

    Parameters
    ----------
    distance_3d, distance_2d : xp.ndarray (N_bs, N_ue)
    frequency : xp.ndarray (N_bs, N_ue)
    h_bs : xp.ndarray (N_bs,)
    h_ue : xp.ndarray (N_ue,)
    shadowing : bool
    rng : np.random.RandomState

    Returns
    -------
    xp.ndarray (N_bs, N_ue) — path loss [dB]
    """
    shadowing_los = 4 if shadowing else 0
    shadowing_nlos = 6 if shadowing else 0

    # .shape on CuPy array is zero-copy — no GPU sync required
    h_e = xp.ones(distance_2d.shape)

    # LOS probability and condition — .shape is zero-copy on CuPy
    los_prob = uma_get_los_probability_gpu(distance_2d, h_ue)
    rand_sample = backend.asarray(rng.random_sample(los_prob.shape))
    los_condition = rand_sample < los_prob  # (N_bs, N_ue) bool on GPU

    # Compute LOS loss once (no shadowing) — reused by NLOS for max(LOS, NLOS)
    loss_los_bare = uma_get_loss_los_gpu(
        distance_2d, distance_3d,
        frequency, h_bs, h_ue, h_e, 0, rng,
    )
    # Apply shadowing on top of bare LOS
    if shadowing_los > 0:
        shad = backend.asarray(rng.normal(0, shadowing_los, loss_los_bare.shape))
        loss_los = loss_los_bare + shad
    else:
        loss_los = loss_los_bare

    # Pass bare LOS to NLOS to avoid recomputing it
    loss_nlos = uma_get_loss_nlos_gpu(
        distance_2d, distance_3d,
        frequency, h_bs, h_ue, h_e, shadowing_nlos, rng,
        loss_los_ref=loss_los_bare,
    )

    # Branchless merge — one kernel instead of scatter + two separate kernels
    return _where_split(los_condition, loss_los, loss_nlos)


# ---------------------------------------------------------------------------
# UMi — GPU-accelerated
# ---------------------------------------------------------------------------

def umi_get_los_probability_gpu(distance_2d, los_adjustment_factor):
    """UMi LOS probability — GPU vectorized, branchless."""
    d2d = backend.asarray(distance_2d, dtype=xp.float64)
    alpha = float(los_adjustment_factor)

    d_safe = xp.maximum(d2d, 1e-6)
    p_far = alpha / d_safe + xp.exp(-d_safe / 36.0) * (1.0 - alpha / d_safe)

    return xp.where(d2d > alpha, p_far, xp.ones_like(d2d))


def umi_get_breakpoint_distance_gpu(frequency, h_bs, h_ue, h_e):
    """UMi breakpoint distance — GPU vectorized.

    Parameters
    ----------
    frequency : float
    h_bs : xp.ndarray (N_bs,)
    h_ue : xp.ndarray (N_ue,)
    h_e  : xp.ndarray (N_bs, N_ue)  — effective environment height
    """
    h_bs_a = backend.asarray(h_bs, dtype=xp.float64)
    h_ue_a = backend.asarray(h_ue, dtype=xp.float64)
    h_e_a = backend.asarray(h_e, dtype=xp.float64)  # (N_bs, N_ue)
    freq = float(frequency)

    h_bs_eff = h_bs_a[:, xp.newaxis] - h_e_a         # (N_bs, N_ue)
    h_ue_eff = h_ue_a[xp.newaxis, :] - h_e_a         # (N_bs, N_ue)
    return 4.0 * h_bs_eff * h_ue_eff * (freq * 1e6) / 3e8


def umi_get_loss_los_gpu(distance_2d, distance_3d, frequency,
                          h_bs, h_ue, h_e, shadowing_std, rng):
    """UMi LOS path loss — GPU vectorized, branchless."""
    d2d = backend.asarray(distance_2d, dtype=xp.float64)
    d3d = backend.asarray(distance_3d, dtype=xp.float64)
    freq = backend.asarray(frequency, dtype=xp.float64)
    h_bs_a = backend.asarray(h_bs, dtype=xp.float64)
    h_ue_a = backend.asarray(h_ue, dtype=xp.float64)
    h_e_a = backend.asarray(h_e, dtype=xp.float64)

    d_bp = umi_get_breakpoint_distance_gpu(
        backend.asnumpy(freq).flat[0] if hasattr(freq, 'flat') else float(freq),
        h_bs_a, h_ue_a, h_e_a,
    )

    loss_near = (
        21.0 * xp.log10(xp.maximum(d3d, 1e-6))
        + 20.0 * xp.log10(xp.maximum(freq, 1.0))
        - 27.55
    )
    fitting_term = -9.5 * xp.log10(
        d_bp ** 2 + (h_bs_a[:, xp.newaxis] - h_ue_a[xp.newaxis, :]) ** 2 + 1e-30
    )
    loss_far = (
        40.0 * xp.log10(xp.maximum(d3d, 1e-6))
        + 20.0 * xp.log10(xp.maximum(freq, 1.0))
        - 27.55
        + fitting_term
    )

    loss = xp.where(d2d < d_bp, loss_near, loss_far)

    if shadowing_std > 0:
        shad_np = rng.normal(0, shadowing_std, distance_2d.shape)
        loss = loss + backend.asarray(shad_np)

    return loss


def umi_get_loss_nlos_gpu(distance_2d, distance_3d, frequency,
                           h_bs, h_ue, h_e, shadowing_std, rng,
                           loss_los_ref=None):
    """UMi NLOS path loss — GPU vectorized.

    Parameters
    ----------
    loss_los_ref : xp.ndarray, optional
        Pre-computed LOS loss (no shadowing). Avoids recomputing LOS when provided.
    """
    d3d = backend.asarray(distance_3d, dtype=xp.float64)
    freq = backend.asarray(frequency, dtype=xp.float64)
    h_ue_a = backend.asarray(h_ue, dtype=xp.float64)

    loss_nlos = (
        -37.55
        + 35.3 * xp.log10(xp.maximum(d3d, 1e-6))
        + 21.3 * xp.log10(xp.maximum(freq, 1.0))
        - 0.3 * (h_ue_a[xp.newaxis, :] - 1.5)
    )

    # Reuse pre-computed LOS to avoid a full second LOS computation
    if loss_los_ref is None:
        loss_los_ref = umi_get_loss_los_gpu(
            distance_2d, distance_3d, frequency, h_bs, h_ue, h_e, 0, rng,
        )
    loss_nlos = xp.maximum(loss_los_ref, loss_nlos)

    if shadowing_std > 0:
        # .shape is zero-cost on CuPy — no GPU sync
        shad_np = rng.normal(0, shadowing_std, d3d.shape)
        loss_nlos = loss_nlos + backend.asarray(shad_np)

    return loss_nlos


def umi_get_loss_gpu(distance_3d, distance_2d, frequency,
                      h_bs, h_ue, shadowing, los_adjustment_factor, rng):
    """UMi combined LOS+NLOS path loss — GPU vectorized, branchless.

    Parameters
    ----------
    distance_3d, distance_2d : xp.ndarray (N_bs, N_ue)
    frequency : xp.ndarray (N_bs, N_ue)
    h_bs : xp.ndarray (N_bs,)
    h_ue : xp.ndarray (N_ue,)
    shadowing : bool
    los_adjustment_factor : float
    rng : np.random.RandomState

    Returns
    -------
    xp.ndarray (N_bs, N_ue) — path loss [dB]
    """
    shadowing_los = 4 if shadowing else 0
    shadowing_nlos = 7.82 if shadowing else 0

    # .shape on CuPy array is zero-copy — no GPU sync
    h_e = xp.ones(distance_2d.shape)

    los_prob = umi_get_los_probability_gpu(distance_2d, los_adjustment_factor)
    rand_sample = backend.asarray(rng.random_sample(los_prob.shape))
    los_condition = rand_sample < los_prob

    # Compute bare LOS once — reused as the max(LOS,NLOS) baseline in NLOS
    loss_los_bare = umi_get_loss_los_gpu(
        distance_2d, distance_3d,
        frequency, h_bs, h_ue, h_e, 0, rng,
    )
    if shadowing_los > 0:
        shad = backend.asarray(rng.normal(0, shadowing_los, loss_los_bare.shape))
        loss_los = loss_los_bare + shad
    else:
        loss_los = loss_los_bare

    loss_nlos = umi_get_loss_nlos_gpu(
        distance_2d, distance_3d,
        frequency, h_bs, h_ue, h_e, shadowing_nlos, rng,
        loss_los_ref=loss_los_bare,
    )

    return _where_split(los_condition, loss_los, loss_nlos)


# ---------------------------------------------------------------------------
# ABG — GPU-accelerated
# ---------------------------------------------------------------------------

def abg_get_loss_gpu(distance, frequency, indoor_stations,
                      alpha, beta, gamma, building_loss_db,
                      shadowing_sigma, shadowing, rng):
    """ABG path loss model — GPU vectorized.

    Parameters
    ----------
    distance : xp.ndarray (N, M) — 3D distances [m]
    frequency : xp.ndarray (N, M) — frequency [MHz]
    indoor_stations : np.ndarray (N,) bool
    alpha, beta, gamma, building_loss_db, shadowing_sigma : float
    shadowing : bool
    rng : np.random.RandomState

    Returns
    -------
    xp.ndarray (N, M)
    """
    d = backend.asarray(distance, dtype=xp.float64)
    f = backend.asarray(frequency, dtype=xp.float64)

    if shadowing:
        # d.shape is zero-copy on CuPy — no GPU sync needed
        shad_np = rng.normal(0, shadowing_sigma, d.shape)
        shadow = backend.asarray(shad_np)
    else:
        shadow = 0.0

    indoor = np.asarray(indoor_stations, dtype=float)
    build_loss = backend.asarray(
        building_loss_db * np.tile(indoor, (d.shape[1], 1)).T
    )

    loss = (
        10.0 * alpha * xp.log10(xp.maximum(d, 1e-6))
        + beta
        + 10.0 * gamma * xp.log10(xp.maximum(f * 1e-3, 1e-6))
        + shadow
        + build_loss
    )

    return loss


# ---------------------------------------------------------------------------
# SINR computation utilities
# ---------------------------------------------------------------------------

def sinr_intra_imt_gpu(
    bs_active, ue_active, tx_power_ue_active, coupling_loss_imt,
):
    """GPU-vectorized intra-IMT interference accumulation.

    Parameters
    ----------
    bs_active : np.ndarray (N_bs,)
        Indices of active base stations.
    ue_active : np.ndarray (N_ue,)
        Indices of active UEs (globally indexed).
    tx_power_ue_active : np.ndarray (N_ue,)
        Transmit power of each active UE [dBm] — flat array, same order as ue_active.
    coupling_loss_imt : np.ndarray (N_bs_total, N_ue_total)
        Full coupling loss matrix. Active subset is extracted here.

    Returns
    -------
    rx_power : np.ndarray (N_bs, K)  — dBm
    rx_interference : np.ndarray (N_bs, K)  — dBm
        Both indexed as (bs_idx_within_active, ue_within_bs) — i.e. flattened
        across bs_active × K beams.
    """
    if len(bs_active) == 0 or len(ue_active) == 0:
        return np.array([]), np.array([])

    N_bs = len(bs_active)
    K = len(ue_active) // N_bs         # UEs per BS

    # ── Move to GPU ──────────────────────────────────────────────────────────
    cpl = backend.asarray(coupling_loss_imt)
    tx_pw = backend.asarray(tx_power_ue_active, dtype=xp.float64)  # (N_ue,)

    # Active subset of coupling loss: (N_bs, N_ue)
    cpl_active = cpl[xp.array(bs_active, dtype=xp.intp), :][:, xp.array(ue_active, dtype=xp.intp)]

    # Linear received power from every UE at every BS: (N_bs, N_ue)
    tx_lin = xp.power(10.0, 0.1 * tx_pw)                     # (N_ue,)
    cpl_lin = xp.power(10.0, -0.1 * cpl_active)              # (N_bs, N_ue)
    rx_lin_all = cpl_lin * tx_lin[xp.newaxis, :]             # (N_bs, N_ue)

    # Total received power per BS (from all UEs)
    total_rx_lin = xp.sum(rx_lin_all, axis=1)                # (N_bs,)

    # Own signal: diagonal block — BS i serves UEs [i*K : (i+1)*K]
    # Reshape to expose per-BS blocks: (N_bs, K)
    rx_blocks = rx_lin_all.reshape(N_bs, N_bs, K)
    own_lin = rx_blocks[xp.arange(N_bs), xp.arange(N_bs), :]  # (N_bs, K)

    # Received power [dBm] of serving BS for its K UEs
    rx_power_dbm = 10.0 * xp.log10(xp.maximum(own_lin, 1e-50))  # (N_bs, K)

    # Interference = total - own signal, per beam
    total_expanded = total_rx_lin[:, xp.newaxis]             # (N_bs, 1) → broadcast
    interf_lin = total_expanded - own_lin                    # (N_bs, K)
    interf_lin = xp.maximum(interf_lin, 1e-50)
    rx_interference_dbm = 10.0 * xp.log10(interf_lin)       # (N_bs, K)

    # Return as flat NumPy arrays (one transfer each) — callers index by ue_active
    return backend.asnumpy(rx_power_dbm), backend.asnumpy(rx_interference_dbm)
