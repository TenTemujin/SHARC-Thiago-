# -*- coding: utf-8 -*-
"""
SHARC GPU Backend Handler
=========================
Singleton that manages GPU/CPU compute backend for the SHARC simulator.

Injects the correct numerical library (CuPy or NumPy) globally,
so all computation modules remain backend-agnostic.

Usage
-----
    from sharc.support.backend_handler import xp, backend

    # Use xp exactly like numpy — runs on GPU when SHARC_USE_GPU=1
    arr = xp.zeros((100, 100))

Activation
----------
    Set environment variable before running:
        $env:SHARC_USE_GPU="1"  (PowerShell)
        export SHARC_USE_GPU=1  (bash/Linux)

    Or set it programmatically before importing this module:
        import os
        os.environ["SHARC_USE_GPU"] = "1"

Notes
-----
- Falls back to NumPy transparently if CuPy is not installed or GPU unavailable.
- CuPy requires CUDA Toolkit installed. Compatible with CUDA 11.x and 12.x.
  Install: pip install cupy-cuda11x  OR  pip install cupy-cuda12x

Hybrid Acceleration (Phase 4)
-----------------------------
StreamManager provides three CUDA streams for overlapped execution:
  - upload   : CPU → GPU data transfer
  - compute  : GPU kernel execution
  - download : GPU → CPU result transfer (async, non-blocking)
"""

import os


class StreamManager:
    """Manages three CUDA streams for pipelined GPU execution.

    Streams allow overlapping data upload/download with GPU compute,
    eliminating idle time where either the GPU or the bus is waiting.

    Only instantiated when GPU mode is active. CPU-only mode returns
    a no-op stub from ``ComputeBackend.stream_manager``.

    Parameters
    ----------
    cupy_module : module
        The CuPy module (not NumPy).
    """

    def __init__(self, cupy_module):
        cp = cupy_module
        self.upload = cp.cuda.Stream(non_blocking=True)
        self.compute = cp.cuda.Stream(non_blocking=True)
        self.download = cp.cuda.Stream(non_blocking=True)
        self._cp = cp

    def async_to_cpu(self, gpu_array):
        """Begin non-blocking GPU → CPU transfer.

        Returns
        -------
        cpu_array : numpy.ndarray
            The destination array (data may not be ready yet).
        event : cupy.cuda.Event
            CUDA event. Call ``event.synchronize()`` before
            reading ``cpu_array``.

        Notes
        -----
        Requires CuPy ≥ 11 for ``stream`` kwarg in ``asnumpy``.
        Falls back to synchronous transfer on older versions.
        """
        try:
            with self.download:
                cpu_out = self._cp.asnumpy(gpu_array)
            event = self.download.record()
            return cpu_out, event
        except TypeError:
            # CuPy < 11 — synchronous fallback
            return self._cp.asnumpy(gpu_array), None

    def sync_download(self):
        """Wait for all async downloads to complete."""
        self.download.synchronize()

    def sync_all(self):
        """Wait for all streams to complete."""
        self.upload.synchronize()
        self.compute.synchronize()
        self.download.synchronize()


class _NoOpStreamManager:
    """CPU-mode stub — all operations are immediate / no-op."""

    def async_to_cpu(self, array):
        import numpy as np
        return np.asarray(array), None

    def sync_download(self):
        pass

    def sync_all(self):
        pass


class ComputeBackend:
    """
    Singleton that manages the GPU/CPU state for the SHARC simulator.

    Injects the correct library (CuPy or NumPy) globally so equations
    remain agnostic to the underlying hardware.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComputeBackend, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the backend based on environment configuration."""
        # Can be set via environment variable: export SHARC_USE_GPU=1
        self.use_gpu = os.environ.get("SHARC_USE_GPU", "0") == "1"
        self._stream_manager = None  # Lazy init

        if self.use_gpu:
            try:
                import cupy as cp
                # Verify GPU is actually accessible
                cp.cuda.Device(0).use()
                self.xp = cp
                self._cupy = cp
                print(f"[SHARC-GPU] CuPy Acceleration Activated. "
                      f"GPU: {cp.cuda.Device(0).attributes.get('DeviceName', 'Unknown')}"
                      if hasattr(cp.cuda.Device(0), 'attributes') else
                      "[SHARC-GPU] CuPy Acceleration Activated.")
            except ImportError:
                print("[SHARC-GPU] Warning: CuPy not installed. "
                      "Install with: pip install cupy-cuda11x\n"
                      "[SHARC-GPU] Falling back to NumPy (CPU mode).")
                import numpy as cp
                self.xp = cp
                self._cupy = None
                self.use_gpu = False
            except Exception as e:
                print(f"[SHARC-GPU] Warning: GPU initialization failed ({e}). "
                      f"Falling back to NumPy.")
                import numpy as cp
                self.xp = cp
                self._cupy = None
                self.use_gpu = False
        else:
            import numpy as cp
            self.xp = cp
            self._cupy = None

    def asnumpy(self, array):
        """Transfer array from GPU (VRAM) to CPU (RAM).

        Short-circuits if the array is already a NumPy ndarray.
        """
        import numpy as np
        if isinstance(array, np.ndarray):
            return array  # already on CPU — zero-cost
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        if hasattr(array, '__array__'):
            return np.asarray(array)
        return array

    def asarray(self, array, dtype=None):
        """Load array into GPU memory (VRAM) if GPU is active.

        Short-circuits if the array is already the correct backend type and dtype.
        """
        if dtype is not None:
            # If already the right type+dtype, avoid the copy
            if hasattr(array, 'dtype') and isinstance(array, type(self.xp.empty(0))):
                if array.dtype == dtype:
                    return array
            return self.xp.asarray(array, dtype=dtype)
        # Fast path: already the right array type (CuPy or NumPy)
        if isinstance(array, type(self.xp.empty(0))):
            return array
        return self.xp.asarray(array)

    def synchronize(self):
        """Synchronize GPU computation (wait for all kernels to complete).

        Only has effect in GPU mode. Use for accurate profiling.
        """
        if self.use_gpu and self._cupy is not None:
            self._cupy.cuda.Stream.null.synchronize()

    @property
    def stream_manager(self):
        """Lazy-initialized StreamManager (GPU) or no-op stub (CPU).

        Returns
        -------
        StreamManager or _NoOpStreamManager
        """
        if self._stream_manager is None:
            if self.use_gpu and self._cupy is not None:
                self._stream_manager = StreamManager(self._cupy)
                print("[SHARC-GPU] StreamManager initialized (3 CUDA streams).")
            else:
                self._stream_manager = _NoOpStreamManager()
        return self._stream_manager

    def suggest_batch_size(self, n_bs: int, n_ue: int) -> int:
        """Heuristic batch size for snapshot batching (Phase 3).

        Estimates how many snapshots can fit in ~60% of free VRAM.
        Returns 1 in CPU mode.

        Parameters
        ----------
        n_bs : int
            Number of base stations.
        n_ue : int
            Number of user equipments.

        Returns
        -------
        int
            Suggested number of snapshots per batch.
        """
        if not self.use_gpu or self._cupy is None:
            return 1
        try:
            free_bytes, _ = self._cupy.cuda.runtime.memGetInfo()
            bytes_per_snapshot = n_bs * n_ue * 8  # float64
            if bytes_per_snapshot == 0:
                return 1
            batch = int(0.6 * free_bytes / bytes_per_snapshot)
            return max(1, min(batch, 512))  # Cap at 512
        except Exception:
            return 1

    @property
    def mode(self) -> str:
        """Return current compute mode as string."""
        return "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"


# ---------------------------------------------------------------------------
# Global singletons — import these directly in any module
# ---------------------------------------------------------------------------

#: Singleton backend manager instance
backend = ComputeBackend()

#: Drop-in replacement for numpy — use as `xp.zeros(...)`, `xp.sin(...)`, etc.
xp = backend.xp
