# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:06:41 2017

@author: edgar
"""

import numpy as np
import math
import warnings

from sharc.simulation import Simulation
from sharc.parameters.parameters import Parameters
from sharc.station_factory import StationFactory
from sharc.parameters.constants import BOLTZMANN_CONSTANT
from sharc.support.backend_handler import backend, xp

warn = warnings.warn


class SimulationDownlink(Simulation):
    """
    Implements the flowchart of simulation downlink method
    """

    def __init__(self, parameters: Parameters, parameter_file: str):
        """Initialize the SimulationDownlink with parameters and parameter file.

        Parameters
        ----------
        parameters : Parameters
            Simulation parameters object.
        parameter_file : str
            Path to the parameter file.
        """
        super().__init__(parameters, parameter_file)

    def snapshot(self, *args, **kwargs):
        """Run a simulation snapshot for the downlink scenario.

        Parameters
        ----------
        *args : tuple
            Positional arguments (unused).
        **kwargs : dict
            Keyword arguments, must include 'write_to_file', 'snapshot_number', and 'seed'.
        """
        write_to_file = kwargs["write_to_file"]
        snapshot_number = kwargs["snapshot_number"]
        seed = kwargs["seed"]

        random_number_gen = np.random.RandomState(seed)

        # In case of hotspots, base stations coordinates have to be calculated
        # on every snapshot. Anyway, let topology decide whether to calculate
        # or not
        num_stations_before = self.topology.num_base_stations

        self.topology.calculate_coordinates(random_number_gen)

        if num_stations_before != self.topology.num_base_stations:
            self.initialize_topology_dependant_variables()

        # Create the base stations (remember that it takes into account the
        # network load factor)
        self.bs = StationFactory.generate_imt_base_stations(
            self.parameters.imt,
            # TODO: remove this:
            self.parameters.imt.bs.antenna.array,
            self.topology, random_number_gen,
        )

        # Create the other system (FSS, HAPS, etc...)
        self.system = StationFactory.generate_system(
            self.parameters, self.topology, random_number_gen,
            coordinate_system=self.coordinate_system
        )

        # Create IMT user equipments
        self.ue = StationFactory.generate_imt_ue(
            self.parameters.imt,
            # TODO: remove this:
            self.parameters.imt.ue.antenna.array,
            self.topology, random_number_gen,
        )

        # self.plot_scenario()

        self.connect_ue_to_bs()
        self.select_ue(random_number_gen)
        self.scheduler()
        self.power_control()

        if self.parameters.imt.interfered_with:
            # Execute this piece of code if the other system generates
            # interference into IMT

            # Calculate coupling loss after beams are created
            self.coupling_loss_imt = self.calculate_intra_imt_coupling_loss(
                self.ue, self.bs,
            )
            self.calculate_sinr()
            self.calculate_sinr_ext()
        else:
            # Execute this piece of code if IMT generates interference into
            # the other system

            # If the intra SINR calculation is disabled, we do not calculate
            # the SINR for the IMT UEs, but we still calculate the external
            # interference.
            if not self.parameters.imt.imt_dl_intra_sinr_calculation_disabled:
                self.coupling_loss_imt = self.calculate_intra_imt_coupling_loss(
                    self.ue, self.bs,
                )
                self.calculate_sinr()
            self.calculate_external_interference()

        self.collect_results(write_to_file, snapshot_number)

    def finalize(self, *args, **kwargs):
        """
        Finalize the simulation and notify observers with the results.

        Parameters
        ----------
        *args : tuple
            Positional arguments (unused).
        **kwargs : dict
            Keyword arguments (unused).
        """
        self.notify_observers(source=__name__, results=self.results)

    def power_control(self):
        """
        Apply downlink power control algorithm to distribute power among selected UEs.
        """
        # Currently, the maximum transmit power of the base station is equaly
        # divided among the selected UEs
        total_power = self.parameters.imt.bs.conducted_power \
            + self.bs_power_gain
        tx_power = total_power - 10 * math.log10(self.parameters.imt.ue.k)
        # calculate transmit powers to have a structure such as
        # {bs_1: [pwr_1, pwr_2,...], ...}, where bs_1 is the base station id,
        # pwr_1 is the transmit power from bs_1 to ue_1, pwr_2 is the transmit
        # power from bs_1 to ue_2, etc
        bs_active = np.where(self.bs.active)[0]
        self.bs.tx_power = dict(
            [(bs, tx_power * xp.ones(self.parameters.imt.ue.k)) for bs in bs_active])

        # Update the spectral mask
        if self.adjacent_channel:
            self.bs.spectral_mask.set_mask(p_tx=total_power)

    def calculate_sinr(self):
        """
        Calculates the downlink SINR for each UE.

        Fully vectorized — zero Python-level loops:
          1. Stack TX power dict into a matrix.
          2. Build received-power matrix (N_BS, N_UE) via broadcasting.
          3. Use block-diagonal view to extract serving-BS signal.
          4. Interference = total - own signal.
        """
        bs_active = np.where(self.bs.active)[0]
        ue_active = np.where(self.ue.active)[0]
        if len(bs_active) == 0 or len(ue_active) == 0:
            return

        K = self.parameters.imt.ue.k
        N_bs = len(bs_active)

        # Ensure coupling_loss_imt is a plain NumPy array (never CuPy)
        cpl = np.asarray(self.coupling_loss_imt)   # (N_BS_total, N_UE_total)

        # ── Stack TX-power dict into a (N_BS_active, K) matrix ─────────────
        tx_pwr_mat = np.stack(
            [self.bs.tx_power[bs] for bs in bs_active], axis=0
        )  # shape: (N_BS_active, K) — one entry per bs in dict

        # ── Coupling loss active subset: (N_BS_active, N_UE_active) ─────────
        cpl_active = cpl[np.ix_(bs_active, ue_active)]

        # ── Sum linear TX power across beams per BS: (N_BS_active,) ─────────
        tx_lin_per_bs = np.sum(np.power(10.0, 0.1 * tx_pwr_mat), axis=1)

        # ── Received linear power from each BS at each UE: (N_A, N_U) ───────
        cpl_lin = np.power(10.0, -0.1 * cpl_active)
        rx_lin_all = tx_lin_per_bs[:, np.newaxis] * cpl_lin

        # ── Total received linear power per UE (all BSs): (N_U,) ────────────
        total_rx_lin = np.sum(rx_lin_all, axis=0)

        # ── Own signal: BS i serves UEs [i*K : (i+1)*K] (block diagonal) ────
        rx_blocks = rx_lin_all.reshape(N_bs, N_bs, K)  # (N_bs, N_bs, K)
        own_lin = rx_blocks[np.arange(N_bs), np.arange(N_bs), :]   # (N_bs, K)

        # ── Signal power per UE: rx_power[ue] = tx - coupling ───────────────
        # Use dB form directly: TX (per beam) - CPL to its UE
        bs_indices = np.repeat(bs_active, K)
        tx_pwr_flat = tx_pwr_mat.flatten()              # (N_UE_active,)
        cpl_serving = cpl[bs_indices, ue_active]        # (N_UE_active,)
        self.ue.rx_power[ue_active] = tx_pwr_flat - cpl_serving

        # ── Interference: total - own signal (vectorized block diagonal) ─────
        # own_lin is (N_bs, K) — place each row into position [i*K:(i+1)*K]
        own_lin_flat = own_lin.flatten()                        # (N_UE_active,)
        interf_lin = total_rx_lin - own_lin_flat
        interf_lin = np.maximum(interf_lin, 1e-30)
        self.ue.rx_interference[ue_active] = 10.0 * np.log10(interf_lin)

        # ── Thermal noise, total interference, SNR, SINR ────────────────────
        # Use xp to stay on GPU when active
        thermal_noise = \
            10 * math.log10(BOLTZMANN_CONSTANT * self.parameters.imt.noise_temperature * 1e3) + \
            10 * xp.log10(self.ue.bandwidth * 1e6) + \
            self.ue.noise_figure

        total_interference = \
            10 * xp.log10(
                xp.power(10.0, 0.1 * self.ue.rx_interference) +
                xp.power(10.0, 0.1 * thermal_noise),
            )

        # Assign back — __setattr__ will promote to GPU if active
        self.ue.thermal_noise = thermal_noise
        self.ue.total_interference = total_interference
        self.ue.sinr = self.ue.rx_power - total_interference
        self.ue.snr = self.ue.rx_power - thermal_noise


    def calculate_sinr_ext(self):
        """
        Calculates the downlink SINR and INR for each UE taking into account the
        interference that is generated by the other system into IMT system.
        """
        if self.co_channel or (
            self.adjacent_channel and self.param_system.adjacent_ch_emissions != "OFF"
        ):
            self.coupling_loss_imt_system = self.calculate_coupling_loss_system_imt(
                self.system,
                self.ue,
                is_co_channel=True,
            )
        if self.adjacent_channel:
            self.coupling_loss_imt_system_adjacent = \
                self.calculate_coupling_loss_system_imt(
                    self.system,
                    self.ue,
                    is_co_channel=False,
                )

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the satellite's bandwidth
        active_sys = np.where(self.system.active)[0]

        # All UEs are active on an active BS
        ue_active = np.where(self.ue.active)[0]

        if len(ue_active) > 0 and len(active_sys) > 0:
            # Get the weight factor for the system overlaping bandwidth in each UE band.
            weights = self.calculate_bw_weights(
                self.ue.bandwidth[ue_active],
                self.ue.center_freq[ue_active],
                float(self.param_system.bandwidth),
                float(self.param_system.frequency),
            )

            # Move to active backend (GPU when active)
            weights = backend.asarray(weights)

            in_band_interf_power = -500.
            if self.co_channel:
                if self.overlapping_bandwidth > 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning,
                            message="divide by zero encountered in log10",
                        )
                        in_band_interf_power = \
                            self.param_system.tx_power_density + 10 * xp.log10(
                                backend.asarray(self.ue.bandwidth[ue_active, np.newaxis]) * 1e6
                            ) + 10 * xp.log10(weights)[:, np.newaxis] - \
                            backend.asarray(self.coupling_loss_imt_system[ue_active, :][:, active_sys])

            oob_power = xp.resize(backend.asarray(-500.0), (len(ue_active), 1))
            if self.adjacent_channel:
                tx_oob = xp.resize(backend.asarray(-500.0), len(ue_active))
                rx_oob = xp.resize(backend.asarray(-500.0), len(ue_active))

                if self.parameters.imt.adjacent_ch_reception == "ACS":
                    if self.overlapping_bandwidth:
                        if not getattr(self, "_acs_warned", False):
                            warn(
                                "You're trying to use ACS on a partially overlapping band "
                                "with UEs.\n\tVerify the code implements the behavior you expect!!"
                            )
                            self._acs_warned = True
                    non_overlap_sys_bw = self.param_system.bandwidth - self.overlapping_bandwidth
                    acs_dB = self.parameters.imt.ue.adjacent_ch_selectivity
                    rx_oob[::] = self.param_system.tx_power_density + 10 * xp.log10(non_overlap_sys_bw * 1e6) - acs_dB
                elif self.parameters.imt.adjacent_ch_reception == "OFF":
                    pass
                else:
                    raise ValueError(
                        f"No implementation for parameters.imt.adjacent_ch_reception == {self.parameters.imt.adjacent_ch_reception}")

                if self.param_system.adjacent_ch_emissions == "SPECTRAL_MASK":
                    ue_bws = self.ue.bandwidth[ue_active]
                    center_freqs = self.ue.center_freq[ue_active]

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")
                        for i, center_freq, bw in zip(range(len(center_freqs)), center_freqs, ue_bws):
                            tx_oob[i] = self.system.spectral_mask.power_calc(center_freq, bw) - 30
                elif self.param_system.adjacent_ch_emissions == "ACLR":
                    non_overlap_imt_bw = self.ue.bandwidth[ue_active] * (1. - weights)
                    measurement_bw = self.param_system.bandwidth
                    aclr_dB = self.param_system.adjacent_ch_leak_ratio

                    if self.parameters.imt.bandwidth - self.overlapping_bandwidth > measurement_bw:
                        if not getattr(self, "_aclr_warned", False):
                            warn(
                                "Using System ACLR into IMT, but ACLR measurement bw is "
                                f"{measurement_bw} while the IMT bw is bigger ({self.parameters.imt.bandwidth}).\n"
                                "Are you sure you intend to apply the same ACLR to the entire IMT bw?"
                            )
                            self._aclr_warned = True
                    tx_oob[::] = self.param_system.tx_power_density + \
                        10 * xp.log10(1e6) - aclr_dB + 10 * xp.log10(non_overlap_imt_bw)
                elif self.param_system.adjacent_ch_emissions == "OFF":
                    pass
                else:
                    raise ValueError(f"No implementation for param_system.adjacent_ch_emissions == {self.param_system.adjacent_ch_emissions}")

                if self.param_system.adjacent_ch_emissions != "OFF":
                    tx_oob = tx_oob[:, np.newaxis] - backend.asarray(self.coupling_loss_imt_system[ue_active, :][:, active_sys])

                if self.param_system.adjacent_ch_reception != "OFF":
                    rx_oob = rx_oob[:, np.newaxis] - backend.asarray(self.coupling_loss_imt_system_adjacent[ue_active, :][:, active_sys])

                oob_power = 10 * xp.log10(10 ** (0.1 * tx_oob) + 10 ** (0.1 * rx_oob))

            ue_ext_int = 10 * xp.log10(xp.power(10, 0.1 * in_band_interf_power) + xp.power(10, 0.1 * oob_power))

            self.ue.ext_interference[ue_active] = 10 * xp.log10(xp.sum(xp.power(10, 0.1 * ue_ext_int), axis=1)) + 30

            self.ue.sinr_ext[ue_active] = \
                self.ue.rx_power[ue_active] - (10 * xp.log10(xp.power(10, 0.1 * self.ue.total_interference[ue_active]) +
                                                             xp.power(10, 0.1 * (self.ue.ext_interference[ue_active]))))

            self.ue.thermal_noise[ue_active] = \
                10 * xp.log10(BOLTZMANN_CONSTANT * self.parameters.imt.noise_temperature * 1e3) + \
                10 * xp.log10(self.ue.bandwidth[ue_active] * 1e6) + self.parameters.imt.ue.noise_figure

            self.ue.inr[ue_active] = self.ue.ext_interference[ue_active] - self.ue.thermal_noise[ue_active]

        # Calculate PFD at the UE

        # Distance from each system transmitter to each UE receiver (in meters)
        dist_sys_to_imt = self.system.get_3d_distance_to(
            self.ue)  # shape: [n_tx, n_ue]

        # EIRP in dBW/MHz per transmitter
        eirp_dBW_MHz = self.param_system.tx_power_density + \
            60 + self.system_imt_antenna_gain

        # PFD formula (dBW/m²/MHz)
        # PFD = EIRP - 10log10(4π) - 20log10(distance)
        # Store the PFD for each transmitter and each UE
        self.ue.pfd_external = eirp_dBW_MHz - \
            10.992098640220963 - 20 * xp.log10(backend.asarray(dist_sys_to_imt))

        # Total PFD per UE (sum of PFDs from each transmitter)
        # Convert PFD from dB to linear scale (W/m²/MHz)
        pfd_linear = 10 ** (self.ue.pfd_external / 10)
        # Sum PFDs from all transmitters for each UE (axis=0 assumes shape
        # [n_tx, n_ue])
        pfd_agg_linear = xp.sum(pfd_linear[active_sys], axis=0)
        # Convert back to dBW
        self.ue.pfd_external_aggregated = 10 * xp.log10(pfd_agg_linear)

    def calculate_external_interference(self):
        """
        Calculates interference that IMT system generates on other system
        """
        if self.co_channel or (
            self.adjacent_channel and self.param_system.adjacent_ch_reception != "OFF"
        ):
            self.coupling_loss_imt_system = self.calculate_coupling_loss_system_imt(
                self.system, self.bs, is_co_channel=True, )
        if self.adjacent_channel:
            self.coupling_loss_imt_system_adjacent = \
                self.calculate_coupling_loss_system_imt(
                    self.system,
                    self.bs,
                    is_co_channel=False,
                )

        # applying a bandwidth scaling factor since UE transmits on a portion
        # of the interfered systems bandwidth
        # calculate interference only from active UE's
        pow_coch = -np.inf
        # These are in dB. Turn to zero linear.
        tx_oob = -np.inf
        rx_oob = -np.inf

        bs_active = np.where(self.bs.active)[0]
        # this implm assumes some parameters will be same for all interferring BS's
        frst_bs = bs_active[0]
        if self.co_channel:
            ue = self.link[frst_bs]
            weights = self.calculate_bw_weights(
                self.ue.bandwidth[ue],
                self.ue.center_freq[ue],
                self.param_system.bandwidth,
                self.param_system.frequency,
            )

            interference = self.bs.tx_power[frst_bs]
            pow_coch = 10 * xp.log10(
                backend.asarray(weights) * xp.power(
                    10.0,
                    0.1 * backend.asarray(interference),
                ),
            )

        if self.adjacent_channel:
            # Calculate how much power is emitted in the adjacent channel:
            if self.parameters.imt.adjacent_ch_emissions == "SPECTRAL_MASK":
                # The unwanted emission is calculated in terms of TRP (after
                # antenna). In SHARC implementation, ohmic losses are already
                # included in coupling loss. Then, care has to be taken;
                # otherwise ohmic loss will be included twice.
                tx_oob = self.bs.spectral_mask.power_calc(
                    self.param_system.frequency,
                    self.system.bandwidth) + self.parameters.imt.bs.ohmic_loss

            elif self.parameters.imt.adjacent_ch_emissions == "ACLR":
                non_overlap_sys_bw = self.param_system.bandwidth - self.overlapping_bandwidth
                # NOTE: approximated equal to IMT bw
                measurement_bw = self.parameters.imt.bandwidth
                aclr_dB = self.parameters.imt.bs.adjacent_ch_leak_ratio

                if non_overlap_sys_bw > measurement_bw:
                    # NOTE: ACLR defines total leaked power over a fixed measurement bandwidth.
                    # If the victim bandwidth is wider, you’re assuming the same leakage
                    # profile extends beyond the ACLR-defined region, which may overestimate interference
                    # FIXME: if the victim bw fully contains tx bw, then
                    # EACH region should be <= measurement_bw
                    warn(
                        "Using IMT ACLR into system, but ACLR measurement bw is "
                        f"{measurement_bw} while the system bw is bigger ({non_overlap_sys_bw}).\n"
                        "Are you sure you intend to apply ACLR to the entire system bw?"
                    )

                # tx_oob_in_measurement = (tx_pow_lin / aclr)
                # => approx. PSD = (tx_pow_lin / aclr) / measurement_bw
                # approximated received tx_oob = PSD * non_overlap_sys_bw
                # NOTE: we don't get total power, but power per beam
                # because later broadcast will sum this tx_oob `k` times
                tx_oob = self.bs.tx_power[frst_bs] - aclr_dB + 10 * np.log10(
                    non_overlap_sys_bw / measurement_bw
                )
            elif self.parameters.imt.adjacent_ch_emissions == "OFF":
                pass
            else:
                raise ValueError(
                    f"No implementation for self.parameters.imt.adjacent_ch_emissions == {
                        self.parameters.imt.adjacent_ch_emissions}")

            # Calculate how much power is received in the adjacent channel
            if self.param_system.adjacent_ch_reception == "ACS":
                non_overlap_imt_bw = self.parameters.imt.bandwidth - self.overlapping_bandwidth
                tx_bw = self.parameters.imt.bandwidth
                acs_dB = self.param_system.adjacent_ch_selectivity

                # NOTE: only the power not overlapping is attenuated by ACS
                # PSD = tx_pow_lin / tx_bw
                # tx_pow_adj_lin = PSD * non_overlap_imt_bw
                # rx_oob = tx_pow_adj_lin / acs
                rx_oob = self.bs.tx_power[frst_bs] + 10 * np.log10(
                    non_overlap_imt_bw / tx_bw
                ) - acs_dB
            elif self.param_system.adjacent_ch_reception == "OFF":
                if self.parameters.imt.adjacent_ch_emissions == "OFF":
                    raise ValueError(
                        "parameters.imt.adjacent_ch_emissions and parameters.imt.adjacent_ch_reception"
                        " cannot be both set to \"OFF\"")
            else:
                raise ValueError(
                    f"No implementation for self.param_system.adjacent_ch_reception == {
                        self.param_system.adjacent_ch_reception}")

        sys_active = np.where(self.system.active)[0]
        if len(sys_active) > 1:
            raise NotImplementedError(
                "Implementation does not support victim system with more than 1 active station"
            )

        rx_interference = 0
        if len(bs_active) > 0 and len(sys_active) > 0:
            active_beams = np.concatenate([
                np.arange(bs * self.parameters.imt.ue.k, (bs + 1) * self.parameters.imt.ue.k)
                for bs in bs_active
            ]).astype(int)

            if len(active_beams) > 0:
                if self.co_channel:
                    rx_interference += xp.sum(
                        10 ** (0.1 * (pow_coch - backend.asarray(self.coupling_loss_imt_system)[active_beams, :][:, sys_active]))
                    )

                if self.adjacent_channel:
                    adj_loss = self.coupling_loss_imt_system_adjacent[np.ix_(active_beams, sys_active)]

                    assert np.all(adj_loss == adj_loss.flat[0])

                    tx_oob_s = tx_oob - adj_loss[0, :]
                    if self.param_system.adjacent_ch_reception != "OFF":
                        rx_oob_s = rx_oob - backend.asarray(self.coupling_loss_imt_system)[active_beams, :][:, sys_active]
                    else:
                        rx_oob_s = -xp.inf

                    oob_power = 10 * xp.log10(
                        10 ** (0.1 * tx_oob_s) + 10 ** (0.1 * rx_oob_s)
                    )

                    rx_interference += xp.sum(
                        xp.power(10, 0.1 * oob_power)
                    )

        self.system.rx_interference = 10 * xp.log10(xp.maximum(rx_interference, 1e-50))
        # calculate N
        self.system.thermal_noise = \
            10 * math.log10(BOLTZMANN_CONSTANT * self.system.noise_temperature * 1e3) + \
            10 * math.log10(self.param_system.bandwidth * 1e6)

        # Calculate INR at the system - dBm
        self.system.inr = xp.array(
            [self.system.rx_interference - self.system.thermal_noise],
        )

        # Calculate PFD at the system
        # TODO: generalize this a bit more if needed
        if hasattr(
                self.system.antenna[0],
                "effective_area") and self.system.num_stations == 1:
            self.system.pfd = 10 * \
                xp.log10(
                    10**(self.system.rx_interference / 10) /
                    self.system.antenna[0].effective_area,
                )

    def collect_results(self, write_to_file: bool, snapshot_number: int):
        """
        Collect and store results for the current downlink simulation snapshot.

        Args:
            write_to_file (bool): Whether to write results to file.
            snapshot_number (int): The current snapshot number.
        """
        if not self.parameters.imt.interfered_with and np.any(self.bs.active):
            self.results.system_inr.extend(self.system.inr.flatten())
            self.results.system_dl_interf_power.extend(
                self.system.rx_interference.flatten(),
            )
            self.results.system_dl_interf_power_per_mhz.extend(
                self.system.rx_interference.flatten() - 10 * math.log10(self.system.bandwidth),
            )
            # TODO: generalize this a bit more if needed (same conditional as
            # above)
            if hasattr(
                    self.system.antenna[0],
                    "effective_area") and self.system.num_stations == 1:
                self.results.system_pfd.extend([self.system.pfd])

        bs_active = np.where(self.bs.active)[0]
        sys_active = np.where(self.system.active)[0]
        ue_active = np.where(self.ue.active)[0]
        
        if len(bs_active) > 0 and len(ue_active) > 0:
            bs_indices = np.repeat(bs_active, len(ue_active) // len(bs_active))
            
            if not self.parameters.imt.imt_dl_intra_sinr_calculation_disabled:
                self.results.imt_path_loss.extend(self.path_loss_imt[bs_indices, ue_active].tolist())
                self.results.imt_coupling_loss.extend(self.coupling_loss_imt[bs_indices, ue_active].tolist())

                self.results.imt_bs_antenna_gain.extend(self.imt_bs_antenna_gain[bs_indices, ue_active].tolist())
                self.results.imt_ue_antenna_gain.extend(self.imt_ue_antenna_gain[bs_indices, ue_active].tolist())

                tput = self.calculate_imt_tput(
                    self.ue.sinr[ue_active],
                    self.parameters.imt.downlink.sinr_min,
                    self.parameters.imt.downlink.sinr_max,
                    self.parameters.imt.downlink.attenuation_factor,
                )
                self.results.imt_dl_tput.extend(tput.tolist())

            # Results for IMT-SYSTEM
            if self.parameters.imt.interfered_with:  # IMT suffers interference
                tput_ext = self.calculate_imt_tput(
                    self.ue.sinr_ext[ue_active],
                    self.parameters.imt.downlink.sinr_min,
                    self.parameters.imt.downlink.sinr_max,
                    self.parameters.imt.downlink.attenuation_factor,
                )
                self.results.imt_dl_tput_ext.extend(tput_ext.tolist())
                self.results.imt_dl_sinr_ext.extend(self.ue.sinr_ext[ue_active].tolist())
                self.results.imt_dl_inr.extend(self.ue.inr[ue_active].tolist())

                self.results.imt_dl_pfd_external.extend(
                    self.ue.pfd_external[sys_active[:, np.newaxis], ue_active].flatten().tolist())

                self.results.imt_dl_pfd_external_aggregated.extend(
                    self.ue.pfd_external_aggregated[ue_active].tolist())

                self.results.system_imt_antenna_gain.extend(
                    self.system_imt_antenna_gain[sys_active[:, np.newaxis], ue_active].flatten().tolist())
                
                if len(self.imt_system_antenna_gain):
                    self.results.imt_system_antenna_gain.extend(
                        self.imt_system_antenna_gain[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                    )
                if len(self.imt_system_antenna_gain_adjacent):
                    self.results.imt_system_antenna_gain_adjacent.extend(
                        self.imt_system_antenna_gain_adjacent[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                    )
                self.results.imt_system_path_loss.extend(
                    self.imt_system_path_loss[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                )
                if self.param_system.channel_model == "HDFSS":
                    self.results.imt_system_build_entry_loss.extend(
                        self.imt_system_build_entry_loss[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                    )
                    self.results.imt_system_diffraction_loss.extend(
                        self.imt_system_diffraction_loss[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                    )
                self.results.sys_to_imt_coupling_loss.extend(
                    self.coupling_loss_imt_system[ue_active[:, np.newaxis], sys_active].flatten().tolist())
            else:  # IMT is the interferer
                self.results.system_imt_antenna_gain.extend(
                    self.system_imt_antenna_gain[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                )
                if len(self.imt_system_antenna_gain):
                    self.results.imt_system_antenna_gain.extend(
                        self.imt_system_antenna_gain[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                    )
                if len(self.imt_system_antenna_gain_adjacent):
                    self.results.imt_system_antenna_gain_adjacent.extend(
                        self.imt_system_antenna_gain_adjacent[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                    )
                self.results.imt_system_path_loss.extend(
                    self.imt_system_path_loss[sys_active[:, np.newaxis], ue_active].flatten().tolist()
                )
                if self.param_system.channel_model == "HDFSS":
                    self.results.imt_system_build_entry_loss.extend(
                        self.imt_system_build_entry_loss[:, bs_active].flatten().tolist()
                    )
                    self.results.imt_system_diffraction_loss.extend(
                        self.imt_system_diffraction_loss[:, bs_active].flatten().tolist()
                    )

            tx_pwr_flat = np.concatenate([self.bs.tx_power[bs] for bs in bs_active])
            self.results.imt_dl_tx_power.extend(tx_pwr_flat.tolist())

            if not self.parameters.imt.imt_dl_intra_sinr_calculation_disabled:
                self.results.imt_dl_sinr.extend(self.ue.sinr[ue_active].tolist())
                self.results.imt_dl_snr.extend(self.ue.snr[ue_active].tolist())

        if write_to_file:
            self.results.write_files(snapshot_number)
            self.notify_observers(source=__name__, results=self.results)
