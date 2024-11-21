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
import os
from tabulate import tabulate

from typing import Union, List, Dict, Callable, Any, Optional

import tuning_toolkit.framework as ttf
# from tuning_toolkit.framework.autorunner_basic_functions import *
from tuning_toolkit.framework.autorunner_ana_1d import Ana1D
# from tuning_toolkit.framework.autorunner_ana_1d_backup import Ana1D   # this is exactly the same as old ana

# from tuning_toolkit.framework.autorunner_ana_2d import *
# from tuning_toolkit.framework.autorunner_utils import *

# from tuning_toolkit.framework.lead_transition_simulation import *
# from tuning_toolkit.framework.lead_transition_simulation import gaussian_dist
from tuning_toolkit.framework.autorunner_sensor_sim import *

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


class SensorEnv2DSimple(gym.Env, ttf.skeleton.Evaluator, ttf.skeleton.Measurement,  # do we need those?
                        ):

    def __init__(self,
                 thresholds: Dict[str, Any],
                 device_parameter: Dict[str, Any],
                 max_del_v: Dict[str, Any],
                 measurement,
                 # sweep_gate_name: str = None,   # i guess str is enough
                 # max_del_v: List[float] = None,
                 # at_bounds_strategy: str = 'push',
                 del_thresholds: Dict[str, Any] = None,
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
        if self.show_only_True:
            self.show = False
            self.show_ana = False
        else:
            self.show = show
            self.show_ana = show_ana

        self.ana_raw = ana_raw
        self.data_x = None
        self.dataxarr = None
        # self.raw = raw

        # ===========
        #  Ana stuff
        # ===========
        self.ana = None
        # self.tot_del_gauss_stds = None

        self.avg_dyn = None
        self.sum_dyn = None

        self.avg_n_peaks = None

        self.avg_steepest_slope = None
        self.sum_steepest_slope = None

        self.avg_peaks_inc_std = None
        self.sum_peaks_inc_std = None

        self.dist_passed_rewards = {}

        # ===========
        #  Check cuda
        # ===========
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
        lower_b = - np.array(list(self.max_del_v_dict.values()), dtype=np.float64)
        high_b = np.array(list(self.max_del_v_dict.values()), dtype=np.float64)
        # lower_b = -1.  # 0.
        # high_b = 1.     # this will make the action always cut the its boundary... as it changes quite a lot in the action space

        norm_factor = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        real_action = norm_factor * (high_b - lower_b) + lower_b

        # NOTE: made change here! and undo the change in limit gate voltage! basically the same though
        # Clip real-world action within bounds
        clipped_action = np.clip(real_action, lower_b, high_b)
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
            self.show_ana = False

        # SEt init gate
        if self.show:
            print(f"[info] Reset all gates to initial values.")
        for gate_i, (gate_name, dv) in enumerate(self.device_parameter.items()):
            # print(f"[info] Reset action {self.init_gate_voltages[gate_name]:3f} V will be applied to {gate_name}")  # too wordy
            # print(f"[info] Reset {gate_name}: {self.init_gate_voltages[gate_name]:3f} V")
            dv.value(self.init_gate_voltages[gate_name] - 1e-5)  # can throw some errors
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
        self.dataxarr = self.measurement.data[0]

        """ !!! The thing is simulation resolution and line cut resolution can be different """
        if len(self.state[0]) != self.resolution:
            state = np.array([down_sample_1d(tm_s, self.resolution) for tm_s in state])
            self.state = state

        if not self.observation_space.contains(state):
            print(
                f"[warning] The init observation {state} is not within the observation space {self.observation_space}. We clip it.")
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        return state, info

    def step(self, action, use_seed=False, **kwargs):  # debug True for now  stepsizes ---> give error

        if self.show_only_True:
            self.show = False
            self.show_ana = False

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
            # print(f"Action {action} is not within action space. We clip the action to {np.clip(action, self.action_space.low, self.action_space.high)}.")
            # env_logger.warning(f"[warning] Action {action} is not within action space {self.action_space}. We clip the action.")
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = action

        # ============================
        #   change gate voltages
        # ============================
        """ order: always stick to device parameter - ground truth
        """
        if self.show:
            # print("Current gate voltages:")
            self.get_current_gate_voltages(show=False, return_value=False)  # self.show
            curr_gate_voltages = self.get_current_gate_voltages(show=False, return_value=True)
        # action = self._limit_gate_voltages(action, self.max_del_v)   # dv being a list of float
        # print(valid_gates)
        # print(action)

        for gate_i, (gate_name, dv) in enumerate(valid_gates.items()):
            action_i = action[gate_i]
            # print(f"[info] Action {action_i:3f} V will be applied to {gate_name}")  # make it prettier
            # if self.show:
            #     print(f"[info] Action {gate_name}: {action_i:3f} V")  # make it prettier
            # logger.info(f"[info] Action {action_i:3f} V will be applied to {gate_name}")

            # at_bounds = self.change_gate_voltages(gate_name, action_i)
            self.change_gate_voltages(gate_name, action_i)

        if self.show:
            # print("New gate voltages:")
            # self.get_current_gate_voltages(show=self.show, return_value=False)
            new_gate_voltages = self.get_current_gate_voltages(show=False, return_value=True)
            row_list = []
            for (gn, c_gv), act_gv, n_gv in zip(curr_gate_voltages.items(), action, new_gate_voltages.values()):
                row_list.append([gn, act_gv, c_gv, n_gv])

            # print('\n')
            print(tabulate(row_list,
                           headers=['         action', '    current voltages', '   new voltages'],
                           floatfmt=(".5f", ".5f", ".5f", ".5f")
                           ))  # only header here
            # ugly!
            # c_str = 'current_gate_voltages'
            # n_str = 'new_gate_voltages'
            # action_str = 'action'
            # print(f"{'': <20} {c_str: <20} {action_str: <15} {n_str: <20}\n")
            # for (gn, c_gv), act_gv, n_gv in zip(curr_gate_voltages.items(), action, new_gate_voltages.values()):
            #     print(f"{gn: <20} {c_gv: <20} {act_gv: <15} {n_gv: <20}")

        # ====================
        #   measure
        # ====================
        self.measurement.measure(show=self.show)  # use_seed = False by default

        """ disabled the normalization as it has to take care of the physical value of steepest slope """
        # state = self._normalize_obs(self.measurement.data.data)  # has additional dim, final shape : (3, resolution)
        state = self.measurement.data[0].data
        data_x = self.measurement.data[0].x.data  # the same shape: (3, resolution)
        self.state = state
        self.data_x = data_x
        self.dataxarr = self.measurement.data[0]

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
        #   evaluate
        # ====================
        self.evaluate(**kwargs)  # <--- **kwargs not working then TODO !!!!
        # ====================
        #   reward
        # ====================

        extra_reward = 0.

        """ now change to average across 3 different sweeps """
        # ===========
        #  slope
        # ===========
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

        # ===============
        #  n_peaks (no meaningful for BBP sweeps) --> NOTE it is!
        # ===============
        n_peak_passed = 'X'
        # if len(self.ana.peaks) < self.thresholds['peaks']:
        #     peaks_dist = self.thresholds['peaks'] - len(self.ana.peaks)
        #     peaks_reward = abs(len(self.ana.peaks)) / self.thresholds['peaks'] - 1.
        #     peaks_reward *= 10  # maybe try it?
        if self.avg_n_peaks < self.thresholds['peaks']:
            peaks_dist = self.thresholds['peaks'] - self.avg_n_peaks
            peaks_reward = abs(self.avg_n_peaks) / self.thresholds['peaks'] - 1.
            peaks_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            peaks_dist = 0.
            peaks_reward = 10  # 50.
            n_peak_passed = 'O'

        # ===============
        #  peak increasing fitting std
        # ===============
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

        # ===============
        #  gaussian fitting
        #  need abs()?
        #  -> Removed it as it is unstable and can make the trainng worse
        # ===============
        # if self.tot_del_gauss_stds > self.thresholds['tot_del_gauss_stds']:   # the opposite sign!
        #     del_gauss_stds_dist = self.tot_del_gauss_stds - self.thresholds['tot_del_gauss_stds']
        #     del_gauss_stds_reward = self.thresholds['tot_del_gauss_stds'] / self.tot_del_gauss_stds - 1.
        #     del_gauss_stds_reward *= 10  # maybe try it?
        # else:
        #     # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
        #     del_gauss_stds_dist = 0.
        #     del_gauss_stds_reward = 50.
        # if self.show:
        #     print(f" tot_del_gauss_stds : {self.tot_del_gauss_stds:.3f} vs. {self.thresholds['tot_del_gauss_stds']:.3f}. Distance: {del_gauss_stds_dist:.4f}, Extra Reward: {del_gauss_stds_reward:.4f}")
        # extra_reward += del_gauss_stds_reward

        extra_reward += slope_reward
        extra_reward += dyn_reward
        extra_reward += peaks_reward  # was missing?
        extra_reward += del_pinc_stds_reward

        # ===============
        #  out of window
        # ===============
        state_out_of_window = False

        if state_out_of_window:
            extra_reward -= 50

        """
            terminate condition:
                above threshold

            thresholds = {'steepest_slope': 0.5,   # 10 too low   0.5 too high --->
                          'dynamic_range': 35,     # 40 seems never reachable
                          'peaks': 1,
                         }
            UPDATE: removed gauss fitting threshold
        """

        terminated = False
        # if abs(self.avg_steepest_slope) >= self.thresholds['steepest_slope'] and self.tot_del_gauss_stds <= self.thresholds['tot_del_gauss_stds'] and len(self.ana.peaks) >= self.thresholds['peaks'] and self.sum_peaks_inc_std <= self.thresholds['sum_peaks_inc_std']:  # added n_peaks!
        # if abs(self.avg_steepest_slope) >= self.thresholds['steepest_slope'] and \
        #     len(self.ana.peaks) >= self.thresholds['peaks'] and \
        #     self.sum_peaks_inc_std <= self.thresholds['sum_peaks_inc_std']:  # added n_peaks!
        if abs(self.avg_steepest_slope) >= self.thresholds['steepest_slope'] and \
                self.avg_n_peaks >= self.thresholds['peaks'] and \
                self.sum_peaks_inc_std <= self.thresholds['sum_peaks_inc_std']:  # added n_peaks!

            terminated = True

        if terminated:
            reward = 10  # 100.  # too much? ^^"
        else:
            reward = -10  # -100.  # How to scale this? T-T

        tot_reward_ = extra_reward + reward
        tot_reward = tot_reward_ / 1e1  # 1e2  # 1e3  # scaling

        dist_passed_rewards = {'slope': [slope_dist, slope_passed, slope_reward],
                               'dyn': [dyn_dist, dyn_passed, dyn_reward],
                               'n_peaks': [peaks_dist, n_peak_passed, peaks_reward],
                               'peak_increasing': [del_pinc_stds_dist, peak_inc_fit_passed, del_pinc_stds_reward]}

        self.dist_passed_rewards = dist_passed_rewards

        if self.show:
            # print(f'----------------------------------------------------------')
            # print(f'[Reward]')
            print(tabulate([
                ['Steepest slope', f"{self.avg_steepest_slope:.3e}", f"{self.thresholds['steepest_slope']:.3e}",
                 f"{slope_dist:.4e}", slope_passed, f"{slope_reward:.4f}"],
                ['Dynamic range', f'{self.avg_dyn:.4f}', f"{self.thresholds['dynamic_range']:.4f}", f"{dyn_dist:.4f}",
                 dyn_passed, f"{dyn_reward:.4f}"],
                ['N peaks', f'{self.avg_n_peaks}', f"{self.thresholds['peaks']:.3f}", f"{peaks_dist:.4f}",
                 n_peak_passed, f"{peaks_reward:.4f}"],
                ['Peak increasing (std)', f'{self.sum_peaks_inc_std:.3e}',
                 f"{self.thresholds['sum_peaks_inc_std']:.3e}", f"{del_pinc_stds_dist:.4f}", peak_inc_fit_passed,
                 f"{del_pinc_stds_reward:.4f}"],
                ['', '', '', '', '', ''],
                ['[Reward]', '', '', '', '', ''],
                ['Termination', '', '', '', '', f'{reward:.5f}'],
                ['Extra (ana)', '', '', '', '', f'{extra_reward:.5f}'],
                ['Total', '', '', '', '', f'{tot_reward_:.5f}'],
                ['Scaled Total', '', '', '', '', f'{tot_reward:.5f}'],
            ],
                headers=['ana', 'actual', 'threshold', 'distance', 'passed', 'reward'],
                tablefmt='orgtbl'))  # only header here

        info = {}
        # tot_reward, terminated = 0., 0.  # ????????? what the hell loool
        if self.show:
            print(
                f'========================================================================================================================== DONE: {terminated}')
            # print('End -------------------------------------------------------')

        return self._get_obs(), tot_reward, terminated, False, info

    """
    # ====================
    #     Analyze
    # ====================
    """

    def get_ana_values(self, ana=None):
        if ana is None:
            ana = self.ana
        assert ana is not None
        ana_attrs = [k[1:] for k in ana.__dict__.keys() if k.startswith('_')]
        ana_results = {atr: ana.__dict__['_' + atr] for atr in ana_attrs if atr in self.thresholds.keys()}

        self.ana_value_state = np.array(list(ana_results.values()))

        return np.array(list(ana_results.values()))  # returns array

    def evaluate(self, **kwargs):  # show not needed but fine   show=True,

        dyns = []
        # del_dyn = []
        steepest_slopes = []
        n_peaks = []

        peaks = []
        proms = []
        # del_proms = []   # Fixme: do I need this?

        peak_inc_std = []
        # del_steepest_slope = []
        for data, data_x in zip(self.state, self.data_x):
            data_ana1d = Ana1D(data=data, data_x=data_x, **kwargs)  # sweep_gate_name='any',

            # _ = data_ana1d.gauss_fit(show=self.show, raw=False)  # Do not use raw
            # avg_gauss_stds.append(data_ana1d.gauss_avg_std)
            # del_gauss_stds.append(np.max(data_ana1d.gauss_stds) - np.min(data_ana1d.gauss_stds))

            """ dyn: always the steepest slope's dyn """
            # dyn
            data_ana1d.get_dynamic_range(show=self.show_ana, raw=False, **kwargs)
            # if self.show_ana:
            #     plt.show()
            dyns.append(data_ana1d.dynamic_range)

            # steepest slope
            data_ana1d.get_steepest_slope(show=self.show_ana, raw=False, **kwargs)
            # if self.show_ana:
            # plt.show()
            steepest_slopes.append(data_ana1d.steepest_slope)

            # peaks
            data_ana1d.get_peaks(show=self.show_ana, **kwargs)
            # if self.show_ana:
            # plt.show()
            peaks.append(data_ana1d.peaks)
            proms.append(data_ana1d.prominences)
            n_peaks.append(len(data_ana1d.peaks))
            # del_proms.append(np.max(data_ana1d.prominences) - np.min(data_ana1d.prominences))

            # peak increasing fit

            data_ana1d.peaks_increasing_fit(show=self.show_ana, raw=False, normalize=True)  # this one gets normalized?
            # if self.show_ana:
            # plt.show()

            if len(data_ana1d.peaks_increasing_std) == 0:  # it means perfectly fitted
                if len(data_ana1d.peaks) <= self.thresholds['peaks']:  # usual reason
                    res = [1e1]  # too large?
                    peak_inc_std.append(res)
                else:
                    raise Exception(
                        f"Unknown error with res={data_ana1d.peaks_increasing_std:.4f} and n_peaks={len(data_ana1d.peaks)}")
            else:
                peak_inc_std.append(data_ana1d.peaks_increasing_std)
            # print(data_ana1d.peaks_increasing_std)

        # assign before handing over
        # data_ana1d.tot_gauss_std = tot_gauss_std
        # data_ana1d.gauss_std = data_ana1d.gauss_stds
        # print(f'avg: {np.mean(avg_gauss_stds)}')

        # self.tot_del_gauss_stds = np.sum(del_gauss_stds)

        self.avg_dyn = np.mean(dyns)
        self.sum_dyn = np.sum(dyns)

        self.avg_n_peaks = np.mean(n_peaks)

        self.avg_steepest_slope = np.mean(steepest_slopes)
        self.sum_steepest_slope = np.sum(steepest_slopes)

        self.avg_peaks_inc_std = np.mean(peak_inc_std)
        self.sum_peaks_inc_std = np.sum(peak_inc_std)

        self.ana = data_ana1d  # at the end so that it stores all the results

    """
    # ====================
    #     Obs
    # ====================
    """

    def _get_obs(self):  # --> returns state
        # state = np.array(list(self.get_current_gate_voltages(show=False, return_value=True).values()) + list(self.ana_value_state) + [self.ana_state], dtype=np.float32)
        # state = np.array(list(self.get_current_gate_voltages(show=False, return_value=True).values()) + list(self.ana_value_state), dtype=np.float32)
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




