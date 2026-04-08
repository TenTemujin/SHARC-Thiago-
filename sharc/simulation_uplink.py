# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:02:35 2017

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


class SimulationUplink(Simulation):
    """
    Implements the flowchart of simulation downlink method
    """

    def __init__(self, parameters: Parameters, parameter_file: str):
        super().__init__(parameters, parameter_file)

    def snapshot(self, *args, **kwargs):
        """
        Execute a simulation snapshot for the uplink scenario.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Should include 'write_to_file', 'snapshot_number', and 'seed'.
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

        # Calculate coupling loss after beams are created
        self.coupling_loss_imt = self.calculate_intra_imt_coupling_loss(
            self.ue,
            self.bs,
        )
        self.scheduler()
        self.power_control()

        if self.parameters.imt.interfered_with:
            # Execute this piece of code if the other system generates
            # interference into IMT
            self.calculate_sinr()
            self.calculate_sinr_ext()
        else:
            # Execute this piece of code if IMT generates interference into
            # the other system
            self.calculate_sinr()
            self.calculate_external_interference()

        self.collect_results(write_to_file, snapshot_number)

    def power_control(self):
        """
        Apply uplink power control algorithm
        """
        if self.parameters.imt.ue.tx_power_control == "OFF":
            ue_active = np.where(self.ue.active)[0]
            self.ue.tx_power[ue_active] = self.parameters.imt.ue.p_cmax * \
                xp.ones(len(ue_active))
        else:
            bs_active = np.where(self.bs.active)[0]
            ue_active = np.where(self.ue.active)[0]
            if len(bs_active) > 0 and len(ue_active) > 0:
                bs_indices = np.repeat(bs_active, len(ue_active) // len(bs_active))

                p_cmax = self.parameters.imt.ue.p_cmax
                m_pusch = self.num_rb_per_ue
                p_o_pusch = self.parameters.imt.ue.p_o_pusch
                alpha = self.parameters.imt.ue.alpha
                ue_power_dynamic_range = self.parameters.imt.ue.power_dynamic_range

                cl = self.coupling_loss_imt[bs_indices, ue_active]
                self.ue.tx_power[ue_active] = backend.asarray(np.minimum(
                    p_cmax, 10 * np.log10(m_pusch) + p_o_pusch + alpha * cl,
                ))
                # apply the power dymanic range
                self.ue.tx_power[ue_active] = xp.maximum(
                    self.ue.tx_power[ue_active], backend.asarray(p_cmax - ue_power_dynamic_range),
                )
        if self.adjacent_channel:
            self.ue_power_diff = self.parameters.imt.ue.p_cmax - self.ue.tx_power

    def calculate_sinr(self):
        """
        Calculates the uplink SINR for each BS.
        """
        bs_active = np.where(self.bs.active)[0]
        ue_active = np.where(self.ue.active)[0]
        if len(bs_active) == 0 or len(ue_active) == 0:
            return

        K = len(ue_active) // len(bs_active)
        bs_indices = np.repeat(bs_active, K)

        # Phase 3: single transfer of potentially-GPU arrays to CPU
        # coupling_loss_imt is already NumPy (forced in Phase 3 of simulation.py)
        _np = lambda a: backend.asnumpy(a) if backend.use_gpu else np.asarray(a)
        ue_tx_power_np = _np(self.ue.tx_power)
        bs_bandwidth_np = _np(self.bs.bandwidth)
        bs_nf_np = _np(self.bs.noise_figure)

        # 1. Received power at BS from its connected UEs
        rx_power_lin = ue_tx_power_np[ue_active] - self.coupling_loss_imt[bs_indices, ue_active]
        rx_power_reshaped = rx_power_lin.reshape(-1, K)

        # 2. Interference from OTHER UEs (intra-system)
        ue_tx_lin = np.power(10.0, 0.1 * ue_tx_power_np[ue_active])
        cpl_all_lin = np.power(10.0, -0.1 * self.coupling_loss_imt[np.ix_(bs_active, ue_active)])
        rx_all_lin = cpl_all_lin * ue_tx_lin[np.newaxis, :]
        
        # Reshape to (N_bs, N_bs, K) to sum correctly per resource
        rx_all_reshaped = rx_all_lin.reshape(len(bs_active), len(bs_active), K)
        total_rx_k = np.sum(rx_all_reshaped, axis=1)
        own_rx_k = rx_all_reshaped[np.arange(len(bs_active)), np.arange(len(bs_active)), :]
        
        interf_lin = total_rx_k - own_rx_k
        interf_lin = np.maximum(interf_lin, 1e-50)

        # calculate N
        # thermal noise in dBm
        thermal_noise_bs = \
            10 * np.log10(BOLTZMANN_CONSTANT * self.parameters.imt.noise_temperature * 1e3) + \
            10 * np.log10(bs_bandwidth_np[bs_active] * 1e6) + \
            bs_nf_np[bs_active]
        # Assign back to GPU array
        self.bs.thermal_noise[bs_active] = thermal_noise_bs

        # 3. Results stored using bs_active indexing — no per-BS loop needed
        rx_interf_db = 10.0 * xp.log10(backend.asarray(interf_lin))          # (N_bs, K)
        total_interf_db = 10.0 * xp.log10(
            xp.power(10.0, 0.1 * rx_interf_db)
            + xp.power(10.0, 0.1 * backend.asarray(thermal_noise_bs)[:, np.newaxis]),
        )                                                     # (N_bs, K)

        self.bs.rx_power[bs_active] = backend.asarray(rx_power_reshaped)      # (N_bs, K)
        self.bs.rx_interference[bs_active] = rx_interf_db    # (N_bs, K)
        self.bs.total_interference[bs_active] = total_interf_db  # (N_bs, K)
        self.bs.sinr[bs_active] = backend.asarray(rx_power_reshaped) - total_interf_db  # (N_bs, K)
        self.bs.snr[bs_active] = backend.asarray(rx_power_reshaped) \
            - backend.asarray(thermal_noise_bs)[:, np.newaxis]                 # (N_bs, K)

    def calculate_sinr_ext(self):
        """
        Calculates the uplink SINR for each BS taking into account the
        interference that is generated by the other system into IMT system.
        """

        if self.co_channel or (
            self.adjacent_channel and self.param_system.adjacent_ch_emissions != "OFF"
        ):
            self.coupling_loss_imt_system = self.calculate_coupling_loss_system_imt(
                self.system,
                self.bs,
                is_co_channel=True,
            )

        if self.adjacent_channel:
            self.coupling_loss_imt_system_adjacent = \
                self.calculate_coupling_loss_system_imt(
                    self.system,
                    self.bs,
                    is_co_channel=False,
                )

        bs_active = np.where(self.bs.active)[0]
        sys_active = np.where(self.system.active)[0]
        ue_active = np.where(self.ue.active)[0]

        if len(bs_active) > 0 and len(sys_active) > 0 and len(ue_active) > 0:
            K = len(ue_active) // len(bs_active)
            active_beams = ue_active
            
            beams_bw = self.ue.bandwidth[ue_active]
            bs_center_freqs_expanded = np.repeat(self.bs.center_freq[bs_active], K)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        category=RuntimeWarning,
                                        message="divide by zero encountered in log10")
                weights = self.calculate_bw_weights(
                    beams_bw,
                    bs_center_freqs_expanded,
                    float(self.param_system.bandwidth),
                    float(self.param_system.frequency),)

            in_band_interf_lin = np.zeros(len(active_beams))
            if self.co_channel:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            category=RuntimeWarning,
                                            message="divide by zero encountered in log10")
                    in_band_interf = self.param_system.tx_power_density + \
                        10 * xp.log10(backend.asarray(beams_bw)[:, np.newaxis] * 1e6) + \
                        10 * xp.log10(backend.asarray(weights))[:, np.newaxis] - \
                        backend.asarray(self.coupling_loss_imt_system[active_beams, :][:, sys_active])
                    in_band_interf_lin = 10 ** (in_band_interf / 10)

            oob_interf_lin = 0
            if self.adjacent_channel:
                tx_oob = np.resize(-500., len(active_beams))
                rx_oob = np.resize(-500., len(active_beams))

                if self.parameters.imt.adjacent_ch_reception == "ACS":
                    non_overlap_sys_bw = self.param_system.bandwidth - self.overlapping_bandwidth
                    if self.overlapping_bandwidth > 0:
                        if not hasattr(self, "_acs_warned"):
                            warn(
                                "You're trying to use ACS on a partially overlapping band "
                                "with UEs.\n\tVerify the code implements the behavior you expect!!"
                            )
                            self._acs_warned = True
                    acs_dB = self.parameters.imt.bs.adjacent_ch_selectivity
                    rx_oob[::] = self.param_system.tx_power_density + 10 * np.log10(non_overlap_sys_bw * 1e6) - acs_dB
                elif self.parameters.imt.adjacent_ch_reception == "OFF":
                    pass
                elif self.parameters.imt.adjacent_ch_reception is False:
                    pass
                else:
                    raise ValueError(
                        f"No implementation for parameters.imt.adjacent_ch_reception == {self.parameters.imt.adjacent_ch_reception}")

                if self.param_system.adjacent_ch_emissions == "SPECTRAL_MASK":
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                category=RuntimeWarning,
                                                message="divide by zero encountered in log10")
                        for i, center_freq, bw in zip(
                                range(len(bs_center_freqs_expanded)), bs_center_freqs_expanded, beams_bw):
                            tx_oob[i] = self.system.spectral_mask.power_calc(
                                center_freq,
                                bw
                            ) - 30
                elif self.param_system.adjacent_ch_emissions == "ACLR":
                    non_overlap_imt_bw = beams_bw * (1. - weights)
                    measurement_bw = self.param_system.bandwidth
                    aclr_dB = self.param_system.adjacent_ch_leak_ratio
                    if self.parameters.imt.bandwidth - self.overlapping_bandwidth > measurement_bw:
                        if not hasattr(self, "_aclr_warned"):
                            warn(
                                "Using System ACLR into IMT, but ACLR measurement bw is "
                                f"{measurement_bw} while the IMT bw is bigger ({self.parameters.imt.bandwidth}).\n"
                                "Are you sure you intend to apply the same ACLR to the entire IMT bw?"
                            )
                            self._aclr_warned = True

                    tx_oob[::] = self.param_system.tx_power_density + \
                        10 * np.log10(1e6) -  \
                        aclr_dB + 10 * np.log10(
                            non_overlap_imt_bw)
                elif self.param_system.adjacent_ch_emissions == "OFF":
                    pass
                else:
                    raise ValueError(
                        f"No implementation for param_system.adjacent_ch_emissions == {self.param_system.adjacent_ch_emissions}")

                if self.param_system.adjacent_ch_emissions != "OFF":
                    tx_oob = tx_oob[:, np.newaxis] - self.coupling_loss_imt_system[active_beams, :][:, sys_active]

                rx_oob = rx_oob[:, np.newaxis] - self.coupling_loss_imt_system_adjacent[active_beams, :][:, sys_active]

                oob_interf_lin = 10 ** (0.1 * tx_oob) + 10 ** (0.1 * rx_oob)

            ext_interference = 10 * xp.log10(backend.asarray(in_band_interf_lin) + backend.asarray(oob_interf_lin)) + 30

            ext_interf_lin_total = xp.sum(xp.power(10, 0.1 * ext_interference), axis=1)
            ext_interf_dbm = 10 * xp.log10(ext_interf_lin_total)
            ext_interf_reshaped = ext_interf_dbm.reshape(-1, K)

            # ── Vectorized assignment — no per-BS loop ──────────────────────
            self.bs.ext_interference[bs_active] = ext_interf_reshaped   # (N_bs, K)
            self.bs.sinr_ext[bs_active] = self.bs.rx_power[bs_active] \
                - 10.0 * xp.log10(
                    xp.power(10.0, 0.1 * self.bs.total_interference[bs_active])
                    + xp.power(10.0, 0.1 * ext_interf_reshaped),
                )
            self.bs.inr[bs_active] = (
                self.bs.ext_interference[bs_active]
                - self.bs.thermal_noise[bs_active, np.newaxis]
            )

    def calculate_external_interference(self):
        """
        Calculates interference that IMT system generates on other system
        """

        if self.co_channel or (
            # then rx receives emission inside the tx band, so it is co-channel with IMT
            self.adjacent_channel and self.param_system.adjacent_ch_reception != "OFF"
        ):
            self.coupling_loss_imt_system = self.calculate_coupling_loss_system_imt(
                self.system, self.ue, is_co_channel=True, )
        if self.adjacent_channel:
            self.coupling_loss_imt_system_adjacent = \
                self.calculate_coupling_loss_system_imt(
                    self.system,
                    self.ue,
                    is_co_channel=False,
                )

        rx_interference = 0

        bs_active = np.where(self.bs.active)[0]
        sys_active = np.where(self.system.active)[0]
        ue_active = np.where(self.ue.active)[0]

        if len(bs_active) > 0 and len(sys_active) > 0 and len(ue_active) > 0:
            if self.co_channel:
                weights = self.calculate_bw_weights(
                    self.ue.bandwidth[ue_active],
                    self.ue.center_freq[ue_active],
                    self.param_system.bandwidth,
                    self.param_system.frequency,
                )

                interference_ue = backend.asarray(self.ue.tx_power[ue_active])[:, np.newaxis] - \
                    backend.asarray(self.coupling_loss_imt_system[ue_active, :][:, sys_active])
                
                rx_interference += xp.sum(
                    backend.asarray(weights)[:, np.newaxis] * xp.power(
                        10.0,
                        0.1 * interference_ue,
                    ),
                )

            if self.adjacent_channel:
                tx_oob = -np.inf
                rx_oob = -np.inf
                
                if self.parameters.imt.adjacent_ch_emissions == "SPECTRAL_MASK":
                    tx_oob = self.ue.spectral_mask.power_calc(self.param_system.frequency, self.system.bandwidth) \
                        - self.ue_power_diff[ue_active] \
                        + self.parameters.imt.ue.ohmic_loss

                elif self.parameters.imt.adjacent_ch_emissions == "ACLR":
                    non_overlap_sys_bw = self.param_system.bandwidth - self.overlapping_bandwidth
                    measurement_bw = self.parameters.imt.bandwidth
                    aclr_dB = self.parameters.imt.ue.adjacent_ch_leak_ratio

                    if non_overlap_sys_bw > measurement_bw:
                        if not hasattr(self, "_aclr_warned"):
                            warn(
                                "Using IMT ACLR into system, but ACLR measurement bw is "
                                f"{measurement_bw} while the system bw is bigger ({non_overlap_sys_bw}).\n"
                                "Are you sure you intend to apply ACLR to the entire system bw?"
                            )
                            self._aclr_warned = True

                    tx_oob = self.ue.tx_power[ue_active] - aclr_dB + 10 * np.log10(
                        non_overlap_sys_bw / measurement_bw
                    )
                elif self.parameters.imt.adjacent_ch_emissions == "OFF":
                    pass
                else:
                    raise ValueError(
                        f"No implementation for self.parameters.imt.adjacent_ch_emissions == {self.parameters.imt.adjacent_ch_emissions}"
                    )

                if self.param_system.adjacent_ch_reception == "ACS":
                    non_overlap_imt_bw = self.parameters.imt.bandwidth - self.overlapping_bandwidth
                    tx_bw = self.parameters.imt.bandwidth
                    acs_dB = self.param_system.adjacent_ch_selectivity

                    rx_oob = self.ue.tx_power[ue_active] + 10 * np.log10(
                        non_overlap_imt_bw / tx_bw
                    ) - acs_dB
                elif self.param_system.adjacent_ch_reception == "OFF":
                    if self.parameters.imt.adjacent_ch_emissions == "OFF":
                        raise ValueError("parameters.imt.adjacent_ch_emissions and parameters.imt.adjacent_ch_reception"
                                         " cannot be both set to \"OFF\"")
                    pass
                else:
                    raise ValueError(
                        f"No implementation for self.param_system.adjacent_ch_reception == {self.param_system.adjacent_ch_reception}"
                    )

                tx_oob_expanded = np.resize(tx_oob, len(ue_active))[:, np.newaxis] - \
                    self.coupling_loss_imt_system_adjacent[ue_active, :][:, sys_active]

                if self.param_system.adjacent_ch_reception != "OFF":
                    rx_oob_expanded = np.resize(rx_oob, len(ue_active))[:, np.newaxis] - \
                        self.coupling_loss_imt_system[ue_active, :][:, sys_active]
                else:
                    rx_oob_expanded = np.full((len(ue_active), len(sys_active)), -np.inf)

                oob_power_lin = 10 ** (0.1 * backend.asarray(tx_oob_expanded)) + 10 ** (0.1 * backend.asarray(rx_oob_expanded))

                rx_interference += xp.sum(oob_power_lin)

        self.system.rx_interference = 10 * xp.log10(xp.maximum(backend.asarray(rx_interference), 1e-50))
        # calculate N
        self.system.thermal_noise = \
            10 * np.log10(
                BOLTZMANN_CONSTANT *
                self.system.noise_temperature * 1e3,
            ) + \
            10 * math.log10(self.param_system.bandwidth * 1e6)

        # calculate INR at the system
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
        Collect and store results for the current uplink simulation snapshot.

        Uses ``stage_gpu()`` for GPU-resident arrays so that the GPU
        pipeline is NOT flushed on every snapshot. The staged data is
        transferred in batch when ``write_files`` is called.

        Args:
            write_to_file (bool): Whether to write results to file.
            snapshot_number (int): The current snapshot number.
        """
        # Helper: stage GPU arrays without sync; direct extend for CPU scalars
        _sg = self.results.stage_gpu

        if not self.parameters.imt.interfered_with and np.any(self.bs.active):
            _sg('system_inr', self.system.inr.flatten())
            _sg('system_ul_interf_power',
                xp.atleast_1d(self.system.rx_interference))
            _sg('system_ul_interf_power_per_mhz',
                xp.atleast_1d(self.system.rx_interference - 10 * math.log10(self.system.bandwidth)))
            if hasattr(
                    self.system.antenna[0],
                    "effective_area") and self.system.num_stations == 1:
                _sg('system_pfd', xp.atleast_1d(self.system.pfd))

        sys_active = np.where(self.system.active)[0]
        bs_active = np.where(self.bs.active)[0]
        ue_active = np.where(self.ue.active)[0]

        if len(bs_active) > 0 and len(ue_active) > 0:
            bs_indices = np.repeat(bs_active, len(ue_active) // len(bs_active))

            _sg('imt_path_loss', self.path_loss_imt[bs_indices, ue_active])
            _sg('imt_coupling_loss', self.coupling_loss_imt[bs_indices, ue_active])
            _sg('imt_bs_antenna_gain', self.imt_bs_antenna_gain[bs_indices, ue_active])
            _sg('imt_ue_antenna_gain', self.imt_ue_antenna_gain[bs_indices, ue_active])

            # bs.sinr[bs_active] is (N_bs, K) since calculate_sinr vectorization
            sinr_flat = self.bs.sinr[bs_active].flatten()
            tput = self.calculate_imt_tput(
                sinr_flat,
                self.parameters.imt.uplink.sinr_min,
                self.parameters.imt.uplink.sinr_max,
                self.parameters.imt.uplink.attenuation_factor,
            )
            _sg('imt_ul_tput', tput)

            if self.parameters.imt.interfered_with:
                sinr_ext_flat = self.bs.sinr_ext[bs_active].flatten()
                inr_flat = self.bs.inr[bs_active].flatten()

                tput_ext = self.calculate_imt_tput(
                    sinr_ext_flat,
                    self.parameters.imt.uplink.sinr_min,
                    self.parameters.imt.uplink.sinr_max,
                    self.parameters.imt.uplink.attenuation_factor,
                )
                _sg('imt_ul_tput_ext', tput_ext)
                _sg('imt_ul_sinr_ext', sinr_ext_flat)
                _sg('imt_ul_inr', inr_flat)

                _sg('system_imt_antenna_gain',
                    self.system_imt_antenna_gain[np.ix_(sys_active, ue_active)].flatten())
                if len(self.imt_system_antenna_gain):
                    _sg('imt_system_antenna_gain',
                        self.imt_system_antenna_gain[np.ix_(sys_active, ue_active)].flatten())
                if len(self.imt_system_antenna_gain_adjacent):
                    _sg('imt_system_antenna_gain_adjacent',
                        self.imt_system_antenna_gain_adjacent[np.ix_(sys_active, ue_active)].flatten())
                _sg('imt_system_path_loss',
                    self.imt_system_path_loss[np.ix_(sys_active, ue_active)].flatten())
                if self.param_system.channel_model == "HDFSS":
                    _sg('imt_system_build_entry_loss',
                        self.imt_system_build_entry_loss[np.ix_(sys_active, ue_active)].flatten())
                    _sg('imt_system_diffraction_loss',
                        self.imt_system_diffraction_loss[np.ix_(sys_active, ue_active)].flatten())
            else:  # IMT is the interferer
                _sg('system_imt_antenna_gain',
                    self.system_imt_antenna_gain[np.ix_(sys_active, ue_active)].flatten())
                if len(self.imt_system_antenna_gain):
                    _sg('imt_system_antenna_gain',
                        self.imt_system_antenna_gain[np.ix_(sys_active, ue_active)].flatten())
                if len(self.imt_system_antenna_gain_adjacent):
                    _sg('imt_system_antenna_gain_adjacent',
                        self.imt_system_antenna_gain_adjacent[np.ix_(sys_active, ue_active)].flatten())
                _sg('imt_system_path_loss',
                    self.imt_system_path_loss[np.ix_(sys_active, ue_active)].flatten())
                if self.param_system.channel_model == "HDFSS":
                    _sg('imt_system_build_entry_loss',
                        self.imt_system_build_entry_loss[np.ix_(sys_active, ue_active)].flatten())
                    _sg('imt_system_diffraction_loss',
                        self.imt_system_diffraction_loss[np.ix_(sys_active, ue_active)].flatten())

            # Record UE tx power properties
            tx_pow_active = self.ue.tx_power[ue_active]
            _sg('imt_ul_tx_power', tx_pow_active)
            imt_ul_tx_power_density = 10 * xp.log10(
                xp.power(10, 0.1 * backend.asarray(tx_pow_active)) / (
                    self.num_rb_per_ue * self.parameters.imt.rb_bandwidth * 1e6
                )
            )
            _sg('imt_ul_tx_power_density', imt_ul_tx_power_density)

            _sg('imt_ul_sinr', sinr_flat)

            snr_flat = self.bs.snr[bs_active].flatten()
            _sg('imt_ul_snr', snr_flat)

        if write_to_file:
            self.results.write_files(snapshot_number)
            self.notify_observers(source=__name__, results=self.results)
