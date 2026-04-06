#!/usr/bin/env python3
"""
SHARC GPU Profiler
==================
Runs one snapshot with cProfile to identify the hottest functions.

Usage:
    $env:SHARC_USE_GPU="1"
    python profile_sharc.py path/to/params.yaml

    $env:SHARC_USE_GPU="0"
    python profile_sharc.py path/to/params.yaml
"""

import sys
import os
import cProfile
import pstats
import io
import time

# ── time patch: wrap key functions with wall-clock timers ──────────────────────
_TIMERS = {}

def _make_timer(module_path, class_name, method_name):
    """Monkey-patch a method to measure cumulative wall-clock time."""
    parts = module_path.split(".")
    mod = __import__(module_path, fromlist=[parts[-1]])
    cls = getattr(mod, class_name)
    orig = getattr(cls, method_name)

    key = f"{class_name}.{method_name}"
    _TIMERS[key] = {"calls": 0, "total": 0.0}

    def wrapper(self_, *args, **kwargs):
        t0 = time.perf_counter()
        result = orig(self_, *args, **kwargs)
        dt = time.perf_counter() - t0
        _TIMERS[key]["calls"] += 1
        _TIMERS[key]["total"] += dt
        return result

    setattr(cls, method_name, wrapper)


def _make_func_timer(module_path, func_name):
    """Monkey-patch a module-level function."""
    parts = module_path.split(".")
    mod = __import__(module_path, fromlist=[parts[-1]])
    orig = getattr(mod, func_name)

    key = f"{module_path}.{func_name}"
    _TIMERS[key] = {"calls": 0, "total": 0.0}

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig(*args, **kwargs)
        dt = time.perf_counter() - t0
        _TIMERS[key]["calls"] += 1
        _TIMERS[key]["total"] += dt
        return result

    setattr(mod, func_name, wrapper)


# ── Patch BEFORE importing simulation ─────────────────────────────────────────
_make_timer("sharc.antenna.antenna_beamforming_imt", "AntennaBeamformingImt", "add_beam")
_make_timer("sharc.antenna.antenna_beamforming_imt", "AntennaBeamformingImt", "to_local_coord")
_make_timer("sharc.antenna.antenna_beamforming_imt", "AntennaBeamformingImt", "_weight_vector")
_make_func_timer("sharc.antenna.antenna_gain_vectorized", "compute_gains_batch")
_make_func_timer("sharc.antenna.antenna_gain_vectorized", "_batch_array_factor_with_wvecs")

# ── Run simulation ─────────────────────────────────────────────────────────────
from sharc.main_cli import main

param_file = sys.argv[1] if len(sys.argv) > 1 else "parameters/parameters_downlink.yaml"

print(f"\n{'='*70}")
print(f"  SHARC Profiler — backend: {os.environ.get('SHARC_USE_GPU','0')=='1' and 'GPU' or 'CPU'}")
print(f"  Parameter file: {param_file}")
print(f"{'='*70}\n")

t_total_start = time.perf_counter()

pr = cProfile.Profile()
pr.enable()
main([param_file])
pr.disable()

t_total = time.perf_counter() - t_total_start

# ── Print cProfile top-20 ─────────────────────────────────────────────────────
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())

# ── Print manual timers ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Manual Timers (wall-clock)")
print(f"{'='*70}")
print(f"  {'Function':<55} {'Calls':>8} {'Total(s)':>10} {'Per-call(ms)':>14}")
print(f"  {'-'*55} {'-'*8} {'-'*10} {'-'*14}")

sorted_timers = sorted(_TIMERS.items(), key=lambda kv: kv[1]["total"], reverse=True)
for key, data in sorted_timers:
    calls = data["calls"]
    total = data["total"]
    per_call_ms = (total / calls * 1000) if calls > 0 else 0
    print(f"  {key:<55} {calls:>8} {total:>10.3f} {per_call_ms:>14.3f}")

print(f"\n  TOTAL wall-clock: {t_total:.1f}s")
print(f"{'='*70}\n")
