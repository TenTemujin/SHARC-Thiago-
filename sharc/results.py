# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:47:46 2017

@author: edgar

GPU Acceleration — Deferred Result Collection
----------------------------------------------
This module includes a GPU-staging mechanism that eliminates
per-snapshot synchronous GPU→CPU transfers (.tolist() calls).
Results remain as GPU arrays in a staging dict until a write
boundary (every N snapshots), when a single batched transfer
is performed per field.
"""

import glob
import os
import datetime
import re
import pathlib
import numpy as np
import pandas as pd
from shutil import copy
from sharc.support.sharc_logger import SimulationLogger


class SampleList(list):
    """
    This class only exists so that no list property can be confused with a SampleList
    """


class Results(object):
    """Handle the output of the simulator"""

    # This should always be true for 1st samples flush
    overwrite_sample_files = True

    def __init__(self):
        # ── GPU Staging Buffer ─────────────────────────────────────────────
        # field_name → list of GPU (CuPy) arrays, flushed in batches.
        self._gpu_staging: dict[str, list] = {}

        # Transmit power density [dBm/Hz]
        self.imt_ul_tx_power_density = SampleList()
        self.imt_ul_tx_power = SampleList()
        # SINR [dB]
        self.imt_ul_sinr_ext = SampleList()
        # SINR [dB]
        self.imt_ul_sinr = SampleList()
        # SNR [dB]
        self.imt_ul_snr = SampleList()
        self.imt_ul_inr = SampleList()
        # Throughput [bits/s/Hz]
        self.imt_ul_tput_ext = SampleList()
        # Throughput [bits/s/Hz]
        self.imt_ul_tput = SampleList()

        self.imt_path_loss = SampleList()
        self.imt_coupling_loss = SampleList()
        # Antenna gain [dBi]
        self.imt_bs_antenna_gain = SampleList()
        # Antenna gain [dBi]
        self.imt_ue_antenna_gain = SampleList()

        # Antenna gain [dBi]
        self.system_imt_antenna_gain = SampleList()
        # Antenna gain [dBi]
        self.imt_system_antenna_gain = SampleList()
        # Antenna gain [dBi]
        self.imt_system_antenna_gain_adjacent = SampleList()

        # Path Loss [dB]
        self.imt_system_path_loss = SampleList()
        # Building entry loss [dB]
        self.imt_system_build_entry_loss = SampleList()
        # System diffraction loss [dB]
        self.imt_system_diffraction_loss = SampleList()
        # System to IMT coupling loss
        self.sys_to_imt_coupling_loss = SampleList()

        self.imt_dl_tx_power_density = SampleList()
        # Transmit power [dBm]
        self.imt_dl_tx_power = SampleList()
        # SINR [dB]
        self.imt_dl_sinr_ext = SampleList()
        # SINR [dB]
        self.imt_dl_sinr = SampleList()
        # SNR [dB]
        self.imt_dl_snr = SampleList()
        # I/N [dB]
        self.imt_dl_inr = SampleList()
        # Throughput [bits/s/Hz]
        self.imt_dl_tput_ext = SampleList()
        # Throughput [bits/s/Hz]
        self.imt_dl_tput = SampleList()

        # PFD
        self.imt_dl_pfd_external = SampleList()
        self.imt_dl_pfd_external_aggregated = SampleList()

        self.system_ul_coupling_loss = SampleList()
        self.system_ul_interf_power = SampleList()
        # Interference Power [dBm]

        self.system_dl_coupling_loss = SampleList()
        self.system_dl_interf_power = SampleList()
        # Interference Power [dBm/MHz]
        # NOTE: this may not be what you want for a correct
        # protection criteria analysis since it is
        # a mean value. If you have both cochannel
        # and adjacent channel, the adjacent channel interference
        # will always drag the mean down
        self.system_dl_interf_power_per_mhz = SampleList()
        self.system_ul_interf_power_per_mhz = SampleList()

        self.system_inr = SampleList()
        self.system_pfd = SampleList()
        self.system_rx_interf = SampleList()

        self.__sharc_dir = pathlib.Path(__file__).parent.resolve()

    def prepare_to_write(
        self,
        parameters_filename: str,
        overwrite_output: bool,
        output_dir="output",
        output_dir_prefix="output",
    ):
        """Prepare the output directory and filenames for writing results.

        Parameters
        ----------
        parameters_filename : str
            Name of the parameters file.
        overwrite_output : bool
            Whether to overwrite existing output.
        output_dir : str, optional
            Parent output directory. Default is "output".
        output_dir_prefix : str, optional
            Prefix for output directory. Default is "output".
        """
        self.output_dir_parent = output_dir

        if not overwrite_output:
            today = datetime.date.today()

            results_number = 1
            results_dir_head = (
                output_dir_prefix + "_" + today.isoformat() + "_" + "{:02n}"
            )
            self.create_dir(results_number, results_dir_head)
            copy(parameters_filename, self.output_directory)

        else:
            self.output_directory = self.__sharc_dir / self.output_dir_parent
            try:
                os.makedirs(self.output_directory)
            except FileExistsError:
                pass
        SimulationLogger.set_output_dir(self.output_directory)

        return self

    def create_dir(self, results_number: int, dir_head: str):
        """Creates the output directory if it doesn't exist.

        Parameters
        ----------
        results_number : int
            Increment used in directory name
        dir_head : str
            Directory name prefix

        Returns
        -------
        str
            output directory name
        """

        dir_head_complete = (
            self.__sharc_dir /
            self.output_dir_parent /
            dir_head.format(results_number))

        try:
            os.makedirs(dir_head_complete)
            self.output_directory = dir_head_complete
        except FileExistsError:
            self.create_dir(results_number + 1, dir_head)

    def get_relevant_attributes(self):
        """
        Returns the attributes that are used for storing samples
        """
        self_dict = self.__dict__

        results_relevant_attr_names = list(
            filter(
                lambda x: isinstance(
                    getattr(
                        self,
                        x),
                    SampleList),
                self_dict))

        return results_relevant_attr_names

    # ── GPU Staging API ────────────────────────────────────────────────────

    def stage_gpu(self, field: str, gpu_array):
        """Stage a GPU array for deferred transfer — NO synchronization.

        If the array is already a NumPy ndarray (CPU mode), it is
        appended directly to the result list. Otherwise it is kept
        as a CuPy array in the staging buffer until :meth:`flush_gpu_staged`
        is called (typically every 10 snapshots at write boundaries).

        Parameters
        ----------
        field : str
            Name of the SampleList attribute on this Results object.
        gpu_array : array-like
            1-D or 2-D GPU/CPU array of results.
        """
        if isinstance(gpu_array, np.ndarray):
            # Already on CPU — bypass staging
            getattr(self, field).extend(gpu_array.ravel().tolist())
        else:
            self._gpu_staging.setdefault(field, []).append(gpu_array.ravel())

    def flush_gpu_staged(self):
        """Transfer ALL staged GPU arrays to CPU in one batch.

        For each field, the staged 1-D CuPy arrays are concatenated
        on-device and then transferred with a single ``asnumpy()``
        call, resulting in **one** GPU synchronization per field
        instead of one per snapshot.
        """
        if not self._gpu_staging:
            return

        from sharc.support.backend_handler import backend

        for field, arrays in self._gpu_staging.items():
            if not arrays:
                continue
            # Import here to avoid circular imports / CPU-only envs
            try:
                from sharc.support.backend_handler import xp as _xp
                concatenated = _xp.concatenate(arrays)
                cpu_flat = backend.asnumpy(concatenated)
            except Exception:
                # Fallback: convert individually (shouldn't happen)
                cpu_flat = np.concatenate(
                    [backend.asnumpy(a) for a in arrays],
                )
            getattr(self, field).extend(cpu_flat.tolist())

        self._gpu_staging.clear()

    # ── File I/O ──────────────────────────────────────────────────────────

    def write_files(self, snapshot_number: int):
        """Writes the sample data to the output file.

        Automatically flushes any pending GPU-staged results before
        writing to ensure no data is lost.

        Parameters
        ----------
        snapshot_number : int
            Current snapshot number
        """
        # Safety net: flush any un-flushed GPU staging
        self.flush_gpu_staged()

        results_relevant_attr_names = self.get_relevant_attributes()
        for attr_name in results_relevant_attr_names:
            file_path = os.path.join(
                self.output_directory,
                attr_name + ".csv",
            )
            samples = getattr(self, attr_name)
            if len(samples) == 0:
                continue
            df = pd.DataFrame({"samples": samples})
            if self.overwrite_sample_files:
                df.to_csv(file_path, mode="w", index=False)
            else:
                df.to_csv(file_path, mode="a", index=False, header=False)
            setattr(self, attr_name, SampleList())

        if self.overwrite_sample_files:
            self.overwrite_sample_files = False

    @staticmethod
    def load_many_from_dir(
        root_dir: str,
        *,
        only_latest=True,
        only_samples: list[str] = None,
        filter_fn=None,
    ) -> list["Results"]:
        """
        Load multiple Results objects from a directory containing output folders.

        Args:
            root_dir (str): The root directory to search for output folders.
            only_latest (bool, optional): If True, only load the most recent output for each prefix. Defaults to True.
            only_samples (list[str], optional): List of sample names to load. If None, load all samples. Defaults to None.
            filter_fn (callable, optional): Function to filter output directories. Defaults to None.

        Returns:
            list[Results]: A list of loaded Results objects.
        """
        output_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))

        if len(output_dirs) == 0:
            print("[WARNING]: Results.load_many_from_dir did not find any results")

        if only_latest:
            output_dirs = Results.get_most_recent_outputs_for_each_prefix(
                output_dirs)

        if filter_fn:
            output_dirs = filter(filter_fn, output_dirs)

        all_res = []
        for output_dir in output_dirs:
            res = Results()
            res.load_from_dir(output_dir, only_samples=only_samples)
            all_res.append(res)

        return all_res

    def load_from_dir(
            self,
            abs_path: str,
            *,
            only_samples: list[str] = None) -> "Results":
        """
        Load results from a specified directory, optionally loading only specified samples.

        Args:
            abs_path (str): Absolute path to the output directory.
            only_samples (list[str], optional): List of sample names to load. If None, load all samples. Defaults to None.

        Returns:
            Results: The Results object with loaded data.
        """
        self.output_directory = abs_path

        self_dict = self.__dict__
        if only_samples is not None:
            results_relevant_attr_names = only_samples
        else:
            results_relevant_attr_names = filter(
                lambda x: isinstance(getattr(self, x), SampleList), self_dict
            )

        for attr_name in results_relevant_attr_names:
            file_path = os.path.join(abs_path, f"{attr_name}.csv")
            if os.path.exists(file_path):
                try:
                    # Try reading the .csv file using pandas with different
                    # delimiters
                    try:
                        data = pd.read_csv(file_path, delimiter=",")
                    except pd.errors.ParserError:
                        data = pd.read_csv(file_path, delimiter=";")

                    # Ensure the data has exactly one column
                    if data.shape[1] != 1:
                        raise Exception(
                            f"The file with samples of {attr_name} should have a single column.", )

                    # Remove rows that do not contain valid numeric values
                    data = data.apply(pd.to_numeric, errors="coerce").dropna()

                    # Ignore if there is no data
                    if data.empty:
                        continue
                    # Check if there is enough data to load results from.

                    setattr(self, attr_name, SampleList(data.to_numpy()[:, 0]))

                except Exception as e:
                    print(e)
                    raise Exception(
                        f"Error processing the sample file ({attr_name}.csv) for {attr_name}: {e}")

        return self

    @staticmethod
    def get_most_recent_outputs_for_each_prefix(
            dirnames: list[str]) -> list[str]:
        """
        Get the most recent output directory for each output prefix.

        Args:
            dirnames (list[str]): A list of output directory names.

        Returns:
            list[str]: A list containing the most recent output dirname for each output_prefix.
        """
        res = {}

        for dirname in dirnames:
            prefix, date, id = Results.get_prefix_date_and_id(dirname)
            res.setdefault(
                prefix, {
                    "date": date, "id": id, "dirname": dirname})
            if date > res[prefix]["date"]:
                res[prefix]["date"] = date
                res[prefix]["id"] = id
                res[prefix]["dirname"] = dirname
            if date == res[prefix]["date"] and id > res[prefix]["id"]:
                res[prefix]["id"] = id
                res[prefix]["dirname"] = dirname

        return list(map(lambda x: x["dirname"], res.values()))

    @staticmethod
    def get_prefix_date_and_id(dirname: str) -> tuple[str, str, str]:
        """
        Extract the prefix, date, and id from a directory name.

        Args:
            dirname (str): The directory name to parse.

        Returns:
            tuple: A tuple containing the prefix, date, and id as strings.
        """
        mtch = re.search(
            "(.*)(20[2-9][0-9]-[0-1][0-9]-[0-3][0-9])_([0-9]{2})",
            dirname)
        prefix, date, id = mtch.group(1), mtch.group(2), mtch.group(3)
        return prefix, date, id
