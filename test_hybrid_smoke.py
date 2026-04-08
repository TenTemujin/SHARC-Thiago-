"""Quick smoke test for hybrid acceleration changes — lightweight."""
import os
os.environ["SHARC_USE_GPU"] = "0"
# Prevent matplotlib from loading (slow)
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

# Test 1: Results staging
print("=" * 50)
print("Test 1: Results GPU staging")
from sharc.results import Results, SampleList
r = Results()
assert hasattr(r, '_gpu_staging'), "Missing _gpu_staging"
assert hasattr(r, 'stage_gpu'), "Missing stage_gpu method"
assert hasattr(r, 'flush_gpu_staged'), "Missing flush_gpu_staged"

# Test stage_gpu with numpy arrays (CPU mode)
arr = np.array([1.0, 2.0, 3.0])
r.stage_gpu('imt_dl_sinr', arr)
assert len(r.imt_dl_sinr) == 3, f"Expected 3, got {len(r.imt_dl_sinr)}"
assert r.imt_dl_sinr[0] == 1.0
print("  CPU staging: PASS")

# Test 2D arrays
arr2d = np.array([[4.0, 5.0], [6.0, 7.0]])
r.stage_gpu('imt_dl_snr', arr2d)
assert len(r.imt_dl_snr) == 4, f"Expected 4, got {len(r.imt_dl_snr)}"
print("  2D flatten staging: PASS")

# Multiple stages + flush
r2 = Results()
r2.stage_gpu('imt_dl_sinr', np.array([10.0, 20.0]))
r2.stage_gpu('imt_dl_sinr', np.array([30.0, 40.0]))
assert len(r2.imt_dl_sinr) == 4  # All appended immediately in CPU mode
print("  Multiple CPU staging: PASS")

# Flush does nothing with no staged GPU data
r.flush_gpu_staged()
print("  Empty flush: PASS")
print("Test 1: ALL PASS\n")

# Test 2: Backend handler
print("=" * 50)
print("Test 2: Backend handler")
from sharc.support.backend_handler import backend, xp
print(f"  Mode: {backend.mode}")
assert backend.mode == "CPU (NumPy)"

sm = backend.stream_manager
print(f"  StreamManager: {type(sm).__name__}")
assert type(sm).__name__ == "_NoOpStreamManager"

bs = backend.suggest_batch_size(10, 100)
print(f"  suggest_batch_size(10, 100): {bs}")
assert bs == 1  # CPU mode

cpu_arr, evt = sm.async_to_cpu(np.array([1, 2, 3]))
assert np.array_equal(cpu_arr, [1, 2, 3])
assert evt is None
print("  async_to_cpu (CPU stub): PASS")

sm.sync_download()
sm.sync_all()
print("  sync operations (no-op): PASS")
print("Test 2: ALL PASS\n")

# Test 3: Pipeline manager import
print("=" * 50)
print("Test 3: Pipeline manager")
from sharc.support.pipeline_manager import PipelineManager, SequentialRunner
print("  Import: PASS")

# Verify interfaces exist
assert hasattr(PipelineManager, 'run')
assert hasattr(SequentialRunner, 'run')
assert hasattr(PipelineManager, 'get_timing_summary')
print("  API surface: PASS")
print("Test 3: ALL PASS\n")

print("=" * 50)
print("ALL SMOKE TESTS PASSED!")
