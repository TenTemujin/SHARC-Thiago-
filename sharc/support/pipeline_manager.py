# -*- coding: utf-8 -*-
"""
SHARC Pipeline Manager — Hybrid CPU/GPU Acceleration
=====================================================

Implements a 3-stage pipelined snapshot executor that overlaps:

  Stage A (CPU Thread):  Prepare topology, stations, beams, power control
  Stage B (GPU Thread):  Coupling loss, propagation, SINR calculation
  Stage C (CPU Thread):  Collect results, write to disk

The key insight is that CuPy GPU kernel launches **release the Python
GIL**, meaning Stages A and B genuinely run in parallel even with the
GIL. Stage C downloads are also GIL-free during the actual DMA transfer.

Usage
-----
    from sharc.support.pipeline_manager import PipelineManager

    # In model.py, instead of the manual snapshot loop:
    mgr = PipelineManager(simulation, seeds, write_interval=10)
    mgr.run()

Thread Safety
-------------
- Each stage communicates via bounded queues (no shared mutable state).
- np.random.RandomState instances are per-snapshot (seed-based), NOT shared.
- The simulation object is shared, but stages access different fields:
    - Prepare: writes topology/bs/ue/link
    - Compute: reads topology/bs/ue, writes coupling_loss/sinr
    - Collect: reads results, writes to disk
  This works because stages are serialized per-snapshot (S1 prep finishes
  before S1 compute begins). The overlap is between *different* snapshots.
"""

import time
import queue
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np

from sharc.support.backend_handler import backend


logger = logging.getLogger("sharc.pipeline")


class PipelineManager:
    """3-stage pipelined snapshot executor for hybrid CPU/GPU acceleration.

    Parameters
    ----------
    simulation : SimulationDownlink or SimulationUplink
        The simulation object. Must have ``snapshot()`` method.
    seeds : list[int]
        Pre-generated seeds, one per snapshot.
    write_interval : int
        How often to write results to disk (every N snapshots).
    notify_callback : callable or None
        Function(snapshot_number, message) for progress notifications.
    stop_flag : threading.Event or None
        External stop signal (from GUI thread, etc.).
    """

    SENTINEL = object()  # End-of-stream marker

    def __init__(
        self,
        simulation,
        seeds: list,
        write_interval: int = 10,
        notify_callback=None,
        stop_flag=None,
    ):
        self.simulation = simulation
        self.seeds = seeds
        self.num_snapshots = len(seeds)
        self.write_interval = write_interval
        self.notify_callback = notify_callback
        self.stop_flag = stop_flag

        # Bounded queues prevent memory blowup if one stage is faster
        # maxsize=2 allows 1 item being processed + 1 item ready
        self._prep_to_compute: queue.Queue = queue.Queue(maxsize=2)
        self._compute_to_collect: queue.Queue = queue.Queue(maxsize=2)

        # Performance counters
        self._timing = {
            "prepare": [],
            "compute": [],
            "collect": [],
        }

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute all snapshots through the 3-stage pipeline.

        Returns
        -------
        dict
            Timing statistics per stage (lists of seconds per snapshot).

        Raises
        ------
        Exception
            Re-raises exceptions from any worker thread.
        """
        with ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="sharc-pipeline",
        ) as pool:
            f_prep = pool.submit(self._prepare_stage)
            f_compute = pool.submit(self._compute_stage)
            f_collect = pool.submit(self._collect_stage)

            # Wait for all stages and propagate exceptions
            exceptions = []
            for f in [f_prep, f_compute, f_collect]:
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Pipeline stage failed: {e}")
                    exceptions.append(e)

            if exceptions:
                # Drain queues to unblock other stages
                for q in [self._prep_to_compute, self._compute_to_collect]:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                raise exceptions[0]

        return self._timing

    # ── Stage A: Preparation (CPU-bound) ──────────────────────────────────

    def _prepare_stage(self):
        """Prepare snapshot state: topology, stations, beams, power control.

        Runs on CPU thread. Produces prepared snapshot state and puts
        it into the prep→compute queue.
        """
        sim = self.simulation
        try:
            for i, seed in enumerate(self.seeds):
                if self.stop_flag and self.stop_flag.is_set():
                    break

                snapshot_num = i + 1
                t0 = time.perf_counter()

                random_number_gen = np.random.RandomState(seed)

                # Topology recalculation
                num_stations_before = sim.topology.num_base_stations
                sim.topology.calculate_coordinates(random_number_gen)
                if num_stations_before != sim.topology.num_base_stations:
                    sim.initialize_topology_dependant_variables()

                # Station creation (CPU-heavy: Python object construction)
                from sharc.station_factory import StationFactory

                sim.bs = StationFactory.generate_imt_base_stations(
                    sim.parameters.imt,
                    sim.parameters.imt.bs.antenna.array,
                    sim.topology, random_number_gen,
                )
                sim.system = StationFactory.generate_system(
                    sim.parameters, sim.topology, random_number_gen,
                    coordinate_system=sim.coordinate_system,
                )
                sim.ue = StationFactory.generate_imt_ue(
                    sim.parameters.imt,
                    sim.parameters.imt.ue.antenna.array,
                    sim.topology, random_number_gen,
                )

                # Connect, select, schedule, power control
                sim.connect_ue_to_bs()
                sim.select_ue(random_number_gen)
                sim.scheduler()
                sim.power_control()

                dt = time.perf_counter() - t0
                self._timing["prepare"].append(dt)

                # Pass snapshot to compute stage
                write_to_file = (snapshot_num % self.write_interval == 0)
                self._prep_to_compute.put((snapshot_num, seed, write_to_file))

        except Exception as e:
            logger.error(f"Prepare stage error: {traceback.format_exc()}")
            raise
        finally:
            self._prep_to_compute.put(self.SENTINEL)

    # ── Stage B: GPU Computation ──────────────────────────────────────────

    def _compute_stage(self):
        """Run GPU-heavy computations: coupling loss, SINR, interference.

        CuPy kernel launches release the GIL, so this genuinely
        overlaps with CPU prep of the next snapshot.
        """
        sim = self.simulation
        try:
            while True:
                item = self._prep_to_compute.get()
                if item is self.SENTINEL:
                    break

                snapshot_num, seed, write_to_file = item

                if self.stop_flag and self.stop_flag.is_set():
                    self._compute_to_collect.put(self.SENTINEL)
                    return

                t0 = time.perf_counter()

                # ── Coupling loss + SINR ──────────────────────────────
                if sim.parameters.imt.interfered_with:
                    sim.coupling_loss_imt = sim.calculate_intra_imt_coupling_loss(
                        sim.ue, sim.bs,
                    )
                    sim.calculate_sinr()
                    sim.calculate_sinr_ext()
                else:
                    if not getattr(sim.parameters.imt, 'imt_dl_intra_sinr_calculation_disabled', False):
                        sim.coupling_loss_imt = sim.calculate_intra_imt_coupling_loss(
                            sim.ue, sim.bs,
                        )
                        sim.calculate_sinr()
                    sim.calculate_external_interference()

                dt = time.perf_counter() - t0
                self._timing["compute"].append(dt)

                # Synchronize GPU before handing to collector
                backend.synchronize()

                self._compute_to_collect.put((snapshot_num, write_to_file))

        except Exception as e:
            logger.error(f"Compute stage error: {traceback.format_exc()}")
            raise
        finally:
            self._compute_to_collect.put(self.SENTINEL)

    # ── Stage C: Result Collection (CPU-bound) ────────────────────────────

    def _collect_stage(self):
        """Collect results and write to disk at write boundaries.

        GPU→CPU transfers happen here via ``flush_gpu_staged()``,
        batched over multiple snapshots.
        """
        sim = self.simulation
        try:
            while True:
                item = self._compute_to_collect.get()
                if item is self.SENTINEL:
                    break

                snapshot_num, write_to_file = item

                if self.stop_flag and self.stop_flag.is_set():
                    return

                t0 = time.perf_counter()

                sim.collect_results(write_to_file, snapshot_num)

                if self.notify_callback and (snapshot_num % self.write_interval == 0):
                    self.notify_callback(snapshot_num, f"Snapshot #{snapshot_num}")

                dt = time.perf_counter() - t0
                self._timing["collect"].append(dt)

        except Exception as e:
            logger.error(f"Collect stage error: {traceback.format_exc()}")
            raise

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_timing_summary(self) -> str:
        """Return a formatted summary of per-stage timing.

        Returns
        -------
        str
            Multi-line timing report.
        """
        lines = ["[SHARC Pipeline] Timing Summary"]
        lines.append("=" * 50)
        for stage, times in self._timing.items():
            if times:
                total = sum(times)
                avg = total / len(times)
                lines.append(
                    f"  {stage:10s}: total={total:8.2f}s  "
                    f"avg={avg*1000:7.1f}ms/snap  "
                    f"n={len(times)}"
                )
        lines.append("=" * 50)
        return "\n".join(lines)


class SequentialRunner:
    """Fallback runner that uses the original sequential snapshot loop.

    Used when pipelining is disabled or for debugging. Has the same
    interface as PipelineManager.run().

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    seeds : list[int]
        Pre-generated seeds.
    write_interval : int
        How often to write.
    notify_callback : callable or None
        Progress callback.
    stop_flag : threading.Event or None
        External stop signal.
    """

    def __init__(self, simulation, seeds, write_interval=10,
                 notify_callback=None, stop_flag=None):
        self.simulation = simulation
        self.seeds = seeds
        self.write_interval = write_interval
        self.notify_callback = notify_callback
        self.stop_flag = stop_flag

    def run(self) -> dict:
        """Run all snapshots sequentially (original behavior)."""
        for i, seed in enumerate(self.seeds):
            if self.stop_flag and self.stop_flag.is_set():
                break

            snapshot_num = i + 1
            write_to_file = (snapshot_num % self.write_interval == 0)

            self.simulation.snapshot(
                write_to_file=write_to_file,
                snapshot_number=snapshot_num,
                seed=seed,
            )

            if self.notify_callback and (snapshot_num % self.write_interval == 0):
                self.notify_callback(snapshot_num, f"Snapshot #{snapshot_num}")

        return {}
