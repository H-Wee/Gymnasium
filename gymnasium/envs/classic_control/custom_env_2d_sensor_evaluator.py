""" gym != gymnasium """
# import gym
# from gym import spaces

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import datetime
import pygame
from pygame.locals import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab
import copy
import torch
import xarray as xr
import os
from tabulate import tabulate

from typing import Union, List, Dict, Callable, Any, Optional

import tuning_toolkit
import tuning_toolkit.framework as ttf
# from tuning_toolkit.framework.autorunner_basic_functions import *
from tuning_toolkit.framework.autorunner_ana_1d import Ana1D
# from tuning_toolkit.framework.autorunner_ana_1d_backup import Ana1D   # this is exactly the same as old ana

# from tuning_toolkit.framework.autorunner_ana_2d import *
# from tuning_toolkit.framework.autorunner_utils import *

# from tuning_toolkit.framework.lead_transition_simulation import *
# from tuning_toolkit.framework.lead_transition_simulation import gaussian_dist
from tuning_toolkit.framework.autorunner_sensor_sim import *

from tuning_toolkit.framework.autorunner_ana_1d import *  # Ana1D

# ========= logging
# Configure basic logging settings
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
# logging.disable(logging.CRITICAL)

""" logger name colide with lead_transition_simulation """
# Create a logger
env_logger = logging.getLogger(__name__)

# Create a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the level for this handler to INFO

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)  # Set the formatter for the handler

# Add the handler to the logger
env_logger.addHandler(console_handler)

"""
    with mu
"""


class SensorEnv2DEval(gym.Env, ttf.skeleton.Evaluator, ttf.skeleton.Measurement,  # do we need those?
                        ):

    def __init__(self,
                 thresholds: Dict[str, Any],
                 device_parameter: Dict[str, Any],
                 max_del_v: Dict[str, Any],
                 measurement,
                 evaluator,
                 # sweep_gate_name: str = None,   # i guess str is enough
                 # max_del_v: List[float] = None,
                 # at_bounds_strategy: str = 'push',
                 del_thresholds: Dict[str, Any] = None,
                 ana_pars: Dict[str, Any] = None,
                 show: bool = False,
                 show_ana: bool = False,
                 ana_raw: bool = True,
                 # use_seed: bool = False,
                 # size: int = 100,
                 # raw: bool = True,  # use bool by default
                 # physical_units : bool = False,
                 show_only_True: bool = False,
                 # save_path: str = os.getcwd(),
                 **kwargs,
                 ):

        # ==========
        #  Exp stuff
        # ==========
        self.max_del_v_dict = max_del_v
        self.max_del_v = np.array(list(max_del_v.values()), dtype=np.float64)

        self.device_parameter = device_parameter
        assert list(self.device_parameter.keys()) == list(max_del_v.keys())

        self.measurement = measurement
        try:
            resolution = self.measurement.parameter['x_points']  # fixme: is this right? only x points
        except:
            resolution = self.measurement.parameter['points'][0]
        self.resolution = resolution

        # self.sweep_gate_name = sweep_gate_name  # Need to differentiate which gate is for sweeping
        # assert sweep_gate_name in list(
        #     self.device_parameter.keys())  # list(measurement.device_parameter.keys())  # recheck if it is the same and also sweep gate is contained

        # save
        self.init_gate_voltages = {g: v.value() for g, v in self.device_parameter.items()}
        self.get_current_gate_voltages(show=False, return_value=False)

        # super(SensorEnv2DSimple, self).__init__(
        #         name='',
        #         # parameter=parameter,
        #         device_parameter=self.device_parameter,
        #         cachelogic=True,
        #         auto_save=True,
        #         save_path=None,  # hardcoded
        #         **kwargs
        # )

        self.thresholds = thresholds
        self.init_thresholds = copy.deepcopy(thresholds)
        self.del_thresholds = del_thresholds

        if ana_pars is None:
            ana_pars = {'peak_wheel_pars': None, 'smooth_wheel_pars': None}   # TODO: add more pars
        self.ana_pars = ana_pars

        # ==========
        #  RL stuff
        # ==========
        self.step_count = 0
        # action space is different from device parameter as only choose to tune non zero max del v
        action_space = spaces.Box(low=-1.,
                                  high=1.,
                                  shape=(len(self.max_del_v),), dtype=np.float64)
        # shape=(len(self.device_parameter.keys()),), dtype=np.float64)  # dtype=np.float32)

        self.action_space = action_space
        self.state = None
        self.observation_space = spaces.Box(low=np.full((3, self.resolution), -np.inf),
                                            high=np.full((3, self.resolution), np.inf),
                                            shape=(3, self.resolution,),
                                            dtype=np.float64)  # self.measurement.data   1e100 -> 1.

        # data
        self.show_only_True = show_only_True

        # TODO: this does not do anything
        if self.show_only_True:
            self.show = False
            self.show_ana = show_ana   # False
        else:
            self.show = show
            self.show_ana = show_ana

        self.ana_raw = ana_raw
        self.data_x = None
        self.dataitem = None
        # self.raw = raw

        # ===========
        #  Ana stuff
        # ===========
        self.evaluator = evaluator
        # self.peak_wheel_pars =
        # self.smooth_wheel_pars =


        self.ana = None
        # self.tot_del_gauss_stds = None

        self.avg_dyn = None
        self.sum_dyn = None

        self.avg_n_peaks = None
        self.n_peaks = None

        self.avg_steepest_slope = None
        self.sum_steepest_slope = None

        self.avg_peaks_inc_std = None
        self.sum_peaks_inc_std = None

        self.noise_lvls = None

        self.has_coulombs = None
        self.curr_statuses = None

        self.dist_passed_rewards = {}

        # =======================
        #  action, reward, state
        # =======================
        self.action = None
        self.reward = 0.
        self.ana_reward = 0.
        self.termination_reward = 0.
        self.reward_scale = None

        # ===========
        #  Check cuda
        # ===========
        # FIXME: putting here does not really help --> why it does not use gpu?
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device('cpu')
            print("Using CPU")

    def get_current_gate_voltages(self, show=True, return_value=True):
        self.current_gate_voltages = {k: v.value() for k, v in self.device_parameter.items()}
        if show:
            print_dict_pretty(self.current_gate_voltages)

        if return_value:
            return self.current_gate_voltages

    def _normalize_action(self, action):
        # Perform min-max scaling
        """
        normalize based on
        have to take - into account!
        """
        # lower_b = - np.array(list(self.max_del_v_dict.values()), dtype=np.float64)
        # high_b = np.array(list(self.max_del_v_dict.values()), dtype=np.float64)

        low_bound_v = - np.array(list(self.max_del_v_dict.values()), dtype=np.float64)
        high_bound_v = np.array(list(self.max_del_v_dict.values()), dtype=np.float64)
        lower_b = -1.  # 0.
        high_b = 1.     # this will make the action always cut the its boundary... as it changes quite a lot in the action space

        norm_factor = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        real_action = norm_factor * (high_b - lower_b) + lower_b

        # NOTE: made change here! and undo the change in limit gate voltage! basically the same though
        # Clip real-world action within bounds
        clipped_action = np.clip(real_action, low_bound_v, high_bound_v)
        return clipped_action

    def _normalize_obs(self, state):
        # Perform min-max scaling
        """
        normalize based on
        """
        lower_b = 0.  # -1.  # makes the gauss fit par a become nonsense
        high_b = 1.

        norm_factor = (state - np.min(state)) / (
                np.max(state) - np.min(state) + 1e-10)  # + 1e-10 # avoid dividing by 0 -> otherwise gives nan
        normalized_obs = norm_factor * (high_b - lower_b) + lower_b
        return normalized_obs

    # def reset(self):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        info = {}

        if self.show_only_True:
            self.show = False
            # self.show_ana = False

        # SEt init gate
        if self.show:
            print(f"[info] Reset all gates to initial values.")
        for gate_i, (gate_name, dv) in enumerate(self.device_parameter.items()):
            # print(f"[info] Reset action {self.init_gate_voltages[gate_name]:3f} V will be applied to {gate_name}")  # too wordy
            # print(f"[info] Reset {gate_name}: {self.init_gate_voltages[gate_name]:3f} V")
            dv.value(self.init_gate_voltages[gate_name] - 1e-5)  # can throw some errors
        # self.get_current_gate_voltages(show=self.show, return_value=False)  # update


        # TODO: remove this later --> manually setting barrier gates
        if self.show:
            print(f"Setting TBL={self.device_parameter['TBL'].bounds[0]:.3f}, BBL={self.device_parameter['BBL'].bounds[0]:.3f}.")
        self.device_parameter['TBL'].value(self.device_parameter['TBL'].bounds[0])
        self.device_parameter['BBL'].value(self.device_parameter['BBL'].bounds[0])
        if self.show:
            print("Resetting Done ================================================================================================")
        self.get_current_gate_voltages(show=self.show, return_value=False)  # update

        # Then measure
        self.measurement.measure(show=self.show)  # use_seed = False by default
        """ disabled the normalization as it has to take care of the physical value of steepest slope """
        # state = self._normalize_obs(self.measurement.data.data)  # has additional dim, final shape : (3, resolution)

        # print('IS it here?????????')
        state = self.measurement.data[0].data  # returns a tuple of 1D [0] and 2D [1]
        self.state = state
        # state = self.normalize_obs(self.measurement.data.data.flatten())  # self.measurement.data.data.flatten()
        self.data_x = self.measurement.data[0].x.data  # the same shape: (3, resolution)
        # self.state = state
        self.dataitem = self.measurement.data[0]

        """ !!! The thing is simulation resolution and line cut resolution can be different """
        if len(self.state[0]) != self.resolution:
            state = np.array([down_sample_1d(tm_s, self.resolution) for tm_s in state])
            self.state = state

        if not self.observation_space.contains(state):
            print(
                f"[warning] The init observation {state} is not within the observation space {self.observation_space}. We clip it.")
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        # reset reward needed? maybe not

        # # TODO: remove this later
        # print(f"Setting TBL={self.device_parameter['TBL'].bounds[0]:.3f}, BBL={self.device_parameter['BBL'].bounds[0]:.3f}.")
        # self.device_parameter['TBL'].value(self.device_parameter['TBL'].bounds[0])
        # self.device_parameter['BBL'].value(self.device_parameter['BBL'].bounds[0])
        # print("Resetting Done ================================================================================================")
        #


        return state, info

    def step(self, action, use_seed=False, **kwargs):  # debug True for now  stepsizes ---> give error
        if self.show_only_True:  # turn off temporarily
            show = False
            show_ana = False
        else:
            show = self.show
            show_ana = self.show_ana

        #     self.show = False
        #     self.show_ana = False

        self.step_count += 1
        # ============================
        #   Try different thresholds
        # ============================
        if self.del_thresholds is not None:
            # Change the threshold every 400 steps
            if self.step_count % 400 == 0:
                # self.thresholds = np.random.uniform(0.5, 1.5)  # Change to a new value
                self.thresholds = {n: np.random.uniform(self.init_thresholds[n] - th, self.init_thresholds[n] + th) for
                                   n, th in self.del_thresholds.items()}
                print(f"New threshold: {self.thresholds}")

        # ==================
        #   Process action
        # ==================
        valid_gates_ = {g: dp for g, dp in self.device_parameter.items() if
                        g in self.max_del_v_dict.keys()}  # need to get only valid gates based on max del v non zero elements
        # valid_gates = dict(sorted(valid_gates_.items()))  # sorted
        valid_gates = valid_gates_  # why sorted?

        action_ = action
        # print(action_)
        action = self._normalize_action(action_)
        # print(action)

        if not self.action_space.contains(action):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = action
        # print(action)
        self.action = action

        # ============================
        #   change gate voltages
        # ============================
        """ order: always stick to device parameter - ground truth
        """
        if show:   # self.show:
            # print("Current gate voltages:")
            self.get_current_gate_voltages(show=False, return_value=False)  # self.show
            curr_gate_voltages = self.get_current_gate_voltages(show=False, return_value=True)

        for gate_i, (gate_name, dv) in enumerate(valid_gates.items()):
            action_i = action[gate_i]
            self.change_gate_voltages(gate_name, action_i)

        if show:  # self.show:
            new_gate_voltages = self.get_current_gate_voltages(show=False, return_value=True)
            row_list = []
            for (gn, c_gv), act_gv, n_gv in zip(curr_gate_voltages.items(), action, new_gate_voltages.values()):
                row_list.append([gn, act_gv, c_gv, n_gv])

            print(tabulate(row_list,
                           headers=['         action', '    current voltages', '   new voltages'],
                           floatfmt=(".5f", ".5f", ".5f", ".5f")
                           ))  # only header here
        # ====================
        #   evaluate
        # ====================
        # print(f"{self.evaluator.ana_pars=}")
        # print("How many times is it called?")    # the simulator might show two times of show_verbose as it sets pars prior to measurement
        # print(f"{self.show=}, {self.show_ana=}")
        self.evaluator.evaluate(
                                # data=self.dataitem,  # self.state,
                                # data_x=self.data_x,
                                peak_wheel_pars=self.ana_pars['peak_wheel_pars'],  #  self.evaluator.default_peak_wheel_pars,
                                smooth_wheel_pars=self.ana_pars['smooth_wheel_pars'],   # self.evaluator.default_smooth_wheel_pars,
                                show=show_ana,   # self.show_ana,
                                auto_plot=show_ana,   # self.show_ana,
                                **kwargs)  # <--- **kwargs not working then TODO !!!!

        # ====================
        #   measure
        # ====================
        # self.measurement.measure(show=self.show)  # use_seed = False by default

        """ disabled the normalization as it has to take care of the physical value of steepest slope """
        # state = self._normalize_obs(self.measurement.data.data)  # has additional dim, final shape : (3, resolution)
        state = self.measurement.data[0].data
        data_x = self.measurement.data[0].x.data  # the same shape: (3, resolution)
        self.state = state
        self.data_x = data_x
        self.dataitem = self.measurement.data[0]

        """ !!! The thing is simulation resolution and line cut resolution can be different """
        if len(self.state[0]) != self.resolution:
            state = np.array([down_sample_1d(tm_s, self.resolution) for tm_s in state])
            data_x = np.array([down_sample_1d(tm_x, self.resolution) for tm_x in data_x])
            self.state = state
            self.data_x = data_x

        if not self.observation_space.contains(state):
            print(
                f"[warning] The init observation {state} is not within the observation space {self.observation_space}. We clip it.")
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        # ====================
        #   reward
        # ====================
        # reset extra reward
        extra_reward = 0.

        """ now change to average across 3 different sweeps """
        # ===========
        #  slope
        # ===========
        self.avg_steepest_slope = self.evaluator.ana_results['slope']['avg']
        assert self.avg_steepest_slope is not None
        slope_passed = 'X'
        if abs(self.avg_steepest_slope) < self.thresholds['steepest_slope']:  # self.ana.steepest_slope
            slope_dist = (self.thresholds['steepest_slope'] - abs(self.avg_steepest_slope))  # minus!!
            slope_reward = abs(self.avg_steepest_slope) / self.thresholds['steepest_slope'] - 1.
            slope_reward *= 10  # maybe try it?
        else:
            # slope_reward = abs(abs(self.ana.steepest_slope) - self.thresholds['steepest_slope']) *2
            slope_dist = 0.
            slope_reward = 10  # 50.
            slope_passed = 'O'

        # ===============
        #  dynamic range
        # ===============
        self.avg_dyn = self.evaluator.ana_results['dyn']['avg']
        assert self.avg_dyn is not None
        dyn_passed = 'X'
        if self.avg_dyn < self.thresholds['dynamic_range']:  # self.ana.dynamic_range
            dyn_dist = self.thresholds['dynamic_range'] - self.avg_dyn
            dyn_reward = abs(self.avg_dyn) / self.thresholds['dynamic_range'] - 1.
            dyn_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            dyn_dist = 0.
            dyn_reward = 10  # 50.
            dyn_passed = 'O'

        # ============================================================
        #  n_peaks (no meaningful for BBP sweeps) --> NOTE it is!
        # ============================================================
        self.avg_n_peaks = self.evaluator.ana_results['n_peaks']['avg']
        assert self.avg_n_peaks is not None
        n_peak_passed = 'X'
        peaks_reward = 0.
        if self.avg_n_peaks < self.thresholds['peaks']:
            peaks_dist = self.thresholds['peaks'] - self.avg_n_peaks
            # peaks_reward = abs(self.avg_n_peaks) / self.thresholds['peaks'] - 1.
            # peaks_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            peaks_dist = 0.
            # peaks_reward = 10  # 50.
            n_peak_passed = 'O'

        # ==============================
        #  peak increasing fitting std
        # ==============================
        self.sum_peaks_inc_std = self.evaluator.ana_results['peaks_inc']['sum']
        # print(f"{self.sum_peaks_inc_std=}")
        peak_inc_fit_passed = 'X'
        if self.sum_peaks_inc_std > self.thresholds['sum_peaks_inc_std']:  # the opposite sign!
            del_pinc_stds_dist = self.sum_peaks_inc_std - self.thresholds['sum_peaks_inc_std']
            del_pinc_stds_reward = self.thresholds['sum_peaks_inc_std'] / self.sum_peaks_inc_std - 1.
            del_pinc_stds_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            del_pinc_stds_dist = 0.
            del_pinc_stds_reward = 10  # 50.
            peak_inc_fit_passed = 'O'

        """
           intermediate rewards:
        """
        # ===============
        #  noise level
        # ===============
        noise_lvl_passed = []
        noise_lvl_passed_bool = []
        noise_lvl_reward = 0
        self.noise_lvls = self.evaluator.ana_results['noise_level']['results']
        if 'noise_level' in self.thresholds:
            for noise_lvl in self.noise_lvls:
                # for each line cut
                if noise_lvl < self.thresholds['noise_level']:
                    noise_lvl_reward -= 50
                    noise_lvl_passed.append('X')
                    noise_lvl_passed_bool.append(False)
                else:
                    noise_lvl_passed.append('O')
                    noise_lvl_passed_bool.append(True)
        else:
            print("[warning] No noise level defined. This is make the barrier calibration less robust.")
        # extra_reward += noise_lvl_reward

        # TODO: does not really help for the out of range but maybe?

        # ========================
        #  out of window 2: peaks
        # ========================
        oow_n_peaks_passed = []
        oow_n_peaks_passed_bool = []
        oow_n_peaks_reward = 0
        self.n_peaks = self.evaluator.ana_results['n_peaks']['results']
        for oow_n_peak in self.n_peaks:
            # for each line cut
            if oow_n_peak < self.thresholds['peaks']:
                oow_n_peaks_reward -= 25  # 50
                oow_n_peaks_passed.append('X')
                oow_n_peaks_passed_bool.append(False)
            else:
                oow_n_peaks_passed.append('O')
                oow_n_peaks_passed_bool.append(True)

        # =============================
        #   has_coulomb & curr status
        # =============================
        has_coulomb_passed = []
        has_coulomb_passed_bool = []
        has_coulomb_reward = 0
        self.has_coulombs = self.evaluator.ana_results['has_coulomb']['results']  # a list of bools
        for has_coulomb in self.has_coulombs:   # for each line cut
            if not has_coulomb:
                has_coulomb_reward -= 50  # 50
                has_coulomb_passed.append('X')
                has_coulomb_passed_bool.append(False)
            else:
                has_coulomb_passed.append('O')
                has_coulomb_passed_bool.append(True)
        has_coulomb_list = [hc for hc in self.has_coulombs]  # a list

        # TODO: curr status as well? or is it overlapping with noise_level there
        self.curr_statuses = self.evaluator.ana_results['curr_status']['results']  # a list of str
        curr_status_list = [cs for cs in self.curr_statuses]  # a list


        # -----------------------------
        # add up all ana rewards
        # -----------------------------

        # extra_reward += peaks_reward  # was missing? --> not included

        # extra_reward += slope_reward
        # extra_reward += dyn_reward
        # extra_reward += del_pinc_stds_reward
        # extra_reward += noise_lvl_reward
        # extra_reward += oow_n_peaks_reward
        extra_reward += has_coulomb_reward

        self.ana_reward = extra_reward

        """ 
        =============================================================================
            terminate condition:
                above threshold

            thresholds = {'steepest_slope': 0.5,   # 10 too low   0.5 too high --->
                          'dynamic_range': 35,     # 40 seems never reachable
                          'peaks': 1,
                         }
            UPDATE: removed gauss fitting threshold
            UPDATE (13.01.2025)
            1) Avg Slope >
            2) Avg Dyn >
            3) N peaks > all
            
            these need some thoughts ... 
            4) has_coulomb all?
            5) peak increasing ----> this fitting indicate how irregular the peaks are so maybe necessary ... 
            
            
        =============================================================================
        """
        terminated = False
        # if abs(self.avg_steepest_slope) >= self.thresholds['steepest_slope'] and \
        #         self.avg_dyn >= self.thresholds['dynamic_range'] and \
        #         all(oow_n_peaks_passed_bool) and \
        #         self.sum_peaks_inc_std <= self.thresholds['sum_peaks_inc_std']:  # added n_peaks!
        """ HERE it has terminate conditions --------------------------------------------------------------------- """
        if all(self.has_coulombs):

                # TODO: add condition like sufficient current!

                # self.avg_n_peaks >= self.thresholds['peaks'] and \

            terminated = True
            if self.show_only_True:
                # print(f"{self.show_ana=}")
                self.evaluator.evaluate(
                                        peak_wheel_pars=self.ana_pars['peak_wheel_pars'],  #  self.evaluator.default_peak_wheel_pars,
                                        smooth_wheel_pars=self.ana_pars['smooth_wheel_pars'],   # self.evaluator.default_smooth_wheel_pars,
                                        show=self.show_ana,   # True, make it possible to see only measurement
                                        auto_plot=True,  # self.show_ana, #  True,  <--- always show measurement
                                        **kwargs)  # <--- **kwargs not working then TODO !!!!


        if terminated:
            termination_reward = 10  # 100.  # too much? ^^"
        else:
            termination_reward = -10  # -100.  # How to scale this? T-T
        self.termination_reward = termination_reward

        tot_reward_ = extra_reward + termination_reward

        reward_scale = 1e2
        tot_reward = tot_reward_ / reward_scale  #  # 1e1  # 1e3  # scaling
        self.reward = tot_reward
        self.reward_scale = reward_scale

        dist_passed_rewards = {'slope': [slope_dist, slope_passed, slope_reward],
                               'dyn': [dyn_dist, dyn_passed, dyn_reward],
                               'n_peaks': [peaks_dist, n_peak_passed, peaks_reward],    # note: neglected
                               'oof_n_peaks': [oow_n_peaks_passed, oow_n_peaks_passed_bool, oow_n_peaks_reward],  # [0, oow_n_peaks_passed_bool, oow_n_peaks_reward],
                               'peak_increasing': [del_pinc_stds_dist, peak_inc_fit_passed, del_pinc_stds_reward],
                               'noise_level': [noise_lvl_passed_bool, noise_lvl_passed, noise_lvl_reward],  # [0, noise_lvl_passed, noise_lvl_reward],

                                # has_coulomb_passed : str  has_coulomb_list: bool
                               'has_coulomb': [0, has_coulomb_passed, has_coulomb_reward],  # a list, exception # fixme: maybe not needed?
                               'curr_status': [0, curr_status_list, 0.],

                               # FIxme: shouldn't we put the dist to the target value for the ind n_peaks and noise level too?
                               }

        self.dist_passed_rewards = dist_passed_rewards

        if show:   # self.show:
            self.print_tables(termination_reward=termination_reward, extra_reward=extra_reward,
                              tot_reward=tot_reward, tot_reward_=tot_reward_, terminated=terminated)
            print(
                f'============================================================================================='
                f'=============================================================================================')
        else:
            if self.show_only_True and terminated:
                self.print_tables(termination_reward=termination_reward, extra_reward=extra_reward,
                                  tot_reward=tot_reward, tot_reward_=tot_reward_, terminated=terminated)
                print(
                    f'============================================================================================='
                    f'=============================================================================================')
        info = {}

        return self._get_obs(), tot_reward, terminated, False, info

    """
    # ====================
    #     tables
    # ====================
    """
    def print_tables(self, termination_reward, extra_reward, tot_reward, tot_reward_, terminated):

        # peak pars
        print(tabulate([
            [i, f"{self.evaluator.peak_pars_all[f'data_{i}']['prominence']:.3e}",
             f"{self.evaluator.peak_pars_all[f'data_{i}']['width']:.3f}",
             f"{self.evaluator.peak_wheel_pars['window_ratio']:.3f}",
             f"{self.evaluator.peak_wheel_pars['n_sigma']:.3f}",
             f"{self.evaluator.peak_wheel_pars['prom_amp']:.3f}",
             ] for i in range(3)
        ],
            headers=['Peak pars', 'prominence', 'width', 'window ratio (wheel)', 'n sigma (wheel)', 'prom amp (wheel)'],
            tablefmt='orgtbl'))  # only header here

        # coulomb peak pars
        print(tabulate([
            [i, f"{self.evaluator.coulomb_status_ana['results'][i]['ana_status']['peak_pars']['prominence']:.3e}",
             f"{self.evaluator.coulomb_status_ana['results'][i]['ana_status']['peak_pars']['width']:.3e}",
             f"{self.evaluator.coulomb_status_ana['results'][i]['coulomb_conditions']['peak_wheel_pars']['window_ratio']:.3e}",
             f"{self.evaluator.coulomb_status_ana['results'][i]['coulomb_conditions']['peak_wheel_pars']['n_sigma']:.3e}",
             f"{self.evaluator.coulomb_status_ana['results'][i]['coulomb_conditions']['peak_wheel_pars']['prom_amp']:.3e}",
             ] for i in range(3)
        ],
            # headers=['ana', 'actual', 'threshold', 'distance', 'passed', 'reward'],
            headers=['Peak pars', 'prominence', 'width', 'window ratio (wheel)', 'n sigma (wheel)', 'prom amp (wheel)'],
            tablefmt='orgtbl'))  # only header here

        # smooth pars should be the same for all data
        print(tabulate([
            [0, f"{self.evaluator.smooth_pars_all['data_0']['window_size']:.3f}",
             f"{self.evaluator.smooth_pars_all['data_0']['filter_order']:.3f}",
             f"{self.evaluator.smooth_wheel_pars['window_ratio']:.3f}",
             f"{self.evaluator.smooth_wheel_pars['filter_order']:.3f}",
             ],
        ],
            # headers=['ana', 'actual', 'threshold', 'distance', 'passed', 'reward'],
            headers=['Smooth pars', 'window size', 'filter order', 'window ratio (wheel)', 'filter order (wheel)'],
            tablefmt='orgtbl'))  # only header here

        print(tabulate([
            # coulomb pars
            [0, f"{self.evaluator.coulomb_conditions['steepest_slope']:.3e}",
             f"{self.evaluator.coulomb_conditions['n_peaks']}",
             [f'{nl:.3e}' for nl in self.evaluator.coulomb_conditions['avg_curr_level']],
             f"{self.evaluator.coulomb_conditions['gauss_fit_std']:.3f}",
             ]
        ],
            # headers=['ana', 'actual', 'threshold', 'distance', 'passed', 'reward'],
            headers=['coulomb pars'] + [' '.join(cc.split('_')) for cc in self.evaluator.coulomb_conditions.keys()][
                                       :-2],
            tablefmt='orgtbl'))  # only header here

        print(tabulate([
            ['Steepest slope', f"{self.avg_steepest_slope:.3e}", f"{self.thresholds['steepest_slope']:.3e}",
             f"{self.dist_passed_rewards['slope'][0]:.4e}", self.dist_passed_rewards['slope'][1], f"{self.dist_passed_rewards['slope'][2]:.4f}"],

            ['Dynamic range', f'{self.avg_dyn:.4f}', f"{self.thresholds['dynamic_range']:.4f}",
             f"{self.dist_passed_rewards['dyn'][0]:.4f}", self.dist_passed_rewards['dyn'][1], f"{self.dist_passed_rewards['dyn'][2]:.4f}"],

            ['N peaks (indiv)', [f'{oow_p}' for oow_p in self.n_peaks], f"{self.thresholds['peaks']:.3f}",
             # oow_n_peaks_passed_bool, oow_n_peaks_passed, f"{oow_n_peaks_reward:.4f}"], # NOTE: different order passed: str, passed_bool: bool
             f"{self.dist_passed_rewards['oof_n_peaks'][1]}", f"{self.dist_passed_rewards['oof_n_peaks'][0]}", f"{self.dist_passed_rewards['dyn'][2]:.4f}"],

            ['Peak increasing (sum. std)',
             f"{[f'{pic:.3e}' for pic in self.evaluator.ana_results['peaks_inc']['results']]} | sum = {self.sum_peaks_inc_std:.3e}", f"{self.thresholds['sum_peaks_inc_std']:.3e}",
            f"{self.dist_passed_rewards['peak_increasing'][0]:.3e}", self.dist_passed_rewards['peak_increasing'][1], f"{self.dist_passed_rewards['peak_increasing'][2]:.4f}"],

            ## noise
            ['Current level', [f'{nl:.3e}' for nl in self.noise_lvls], f"{self.thresholds['noise_level']:.3e}",
             # noise_lvl_passed_bool, noise_lvl_passed, f"{noise_lvl_reward:.4f}"],
            f"{self.dist_passed_rewards['noise_level'][0]}", f"{self.dist_passed_rewards['noise_level'][1]}", f"{self.dist_passed_rewards['noise_level'][2]:.4f}"],

            # coulomb
            ['has_coulomb', self.has_coulombs, [True, True, True], self.has_coulombs,
             # has_coulomb_passed, has_coulomb_reward],
             f"{self.dist_passed_rewards['has_coulomb'][1]}", f"{self.dist_passed_rewards['has_coulomb'][2]:.4f}"],

            ['current status', self.curr_statuses, "", "", "", ""],

            ['coulomb info',
             [r['coulomb_status']['info'] for r in self.evaluator.coulomb_status_ana['results']],
             "",
             "",
             "", ""
             ],

            ['', '', '', '', '', ''],
            ['[Reward]', '', '', '', '', ''],
            ['Termination', '', '', '', '', f'{termination_reward:.5f}'],
            ['Extra (ana)', '', '', '', '', f'{extra_reward:.5f}'],
            ['Total', '', '', '', '', f'{tot_reward_:.5f}'],
            ['Scaled Total', '', '', '', '', f'{tot_reward:.5f}'],
            ['', '', '', '', '', f''],
            ['Terminated', '', '', '', '', f'{terminated}'],
        ],
            # headers=['ana', 'actual', 'threshold', 'distance', 'passed', 'reward'],
            headers=['ana', 'actual', 'threshold', 'distance', 'passed', 'reward'],
            tablefmt='orgtbl'))  # only header here

    """
    # ====================
    #     Obs
    # ====================
    """

    def _get_obs(self):  # --> returns state
        state = self.state
        assert state.shape == (3, self.resolution)  # hardcoded

        if not self.observation_space.contains(state):
            # Adjust the initial observation to ensure it falls within the observation space
            env_logger.warning(
                f"[warning] The observation {state} is not within the observation space {self.observation_space}. We clip it.")
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def get_observation(self):
        # Public method to access the observation result
        return self._get_obs()

    """
    # ====================
    #     GATEs (always stick to device parameter order)
    # ====================
    """

    # TODO: check if already this function exists somewhere  --> private?
    def get_current_gate_voltages(self, show=True, return_value=True):
        self.current_gate_voltages = {k: v.value() for k, v in self.device_parameter.items()}
        if show:
            print_dict_pretty(self.current_gate_voltages)

        if return_value:
            return self.current_gate_voltages

    def _limit_gate_voltages(self, action, del_v, precision=1e-7):
        """ clip to Maximum change for each gate (different from setting gate voltages) """

        assert isinstance(action, np.ndarray) and isinstance(del_v, np.ndarray)
        assert len(action) == len(del_v)
        action_clip = copy.deepcopy(action)

        for i, (act, dv) in enumerate(zip(action_clip, del_v)):
            if abs(act) > dv:
                # if self.show:
                print(
                    f"[warning] ({list(self.device_parameter.keys())[i]}) {act:.3f} V exceeds gate voltage change limit. We clip it to its boundary. {dv:3f}")
                # logger.warning(
                # f"[warning] ({list(self.device_parameter.keys())[i]}) {act:.3f} V exceeds gate voltage change limit. We clip it to its boundary. {dv:3f}")
                if np.sign(act) < 0:
                    act = -dv
                else:
                    act = dv
            else:
                act = act
            action_clip[i] = act
        # if self.show:
        #     print(f"[info] New action: {[round(x, 3) for x in action_clip]}")  # pretty printing

        return action_clip

    def _set_gate_voltages(self, gate_name, value, precision=1e-7):  # strategy=None,
        """ Set gate voltages within bounds """
        low_bound = self.device_parameter[gate_name].bounds[0]
        high_bound = self.device_parameter[gate_name].bounds[1]

        if value > high_bound:
            value = high_bound - 1e-5
            if self.show:
                print(
                    f'[warning] The gate {gate_name} has reached its boundary {self.device_parameter[gate_name].bounds}. '
                    f'We clip the value to {value} to {high_bound - 1e-5}')
            self.device_parameter[gate_name].value(value, soft_fail=False)
            # at_bounds = True

        elif value < low_bound:  # add more constraint  --> No..
            value = low_bound + 1e-5
            if self.show:
                print(
                    f'[warning] The gate {gate_name} has reached its boundary {self.device_parameter[gate_name].bounds}. '
                    f'We clip the value to {value} to {low_bound + 1e-5}')
            self.device_parameter[gate_name].value(value, soft_fail=False)
            # at_bounds = True

        else:
            self.device_parameter[gate_name].value(value, soft_fail=False)
            # print(value)
            # at_bounds = False

    def change_gate_voltages(self, gate_name, action):
        if not isinstance(action, float):
            action = np.float64(action)

        self.get_current_gate_voltages(show=False, return_value=False)  # --> make sure it is up to date

        at_bounds_hard = self._set_gate_voltages(gate_name, self.current_gate_voltages[gate_name] + action)
        # strategy=self.at_bounds_strategy)  # hardcoded


