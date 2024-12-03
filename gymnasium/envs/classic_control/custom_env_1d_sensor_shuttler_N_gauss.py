""" gym != gymnasium """
# import gym
# from gym import spaces

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import pygame
from pygame.locals import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab
import copy
from tabulate import tabulate

from typing import Union, List, Dict, Callable, Any, Optional

import tuning_toolkit.framework as ttf
# from tuning_toolkit.framework.autorunner_basic_functions import *
from tuning_toolkit.framework.autorunner_ana_1d import Ana1D
from tuning_toolkit.framework.autorunner_utils_old import print_dict_pretty

# from tuning_toolkit.framework.autorunner_ana_1d_backup import Ana1D   # this is exactly the same as old ana

# from tuning_toolkit.framework.autorunner_ana_2d import *
# from tuning_toolkit.framework.autorunner_utils import *

# from tuning_toolkit.framework.lead_transition_simulation import *
from tuning_toolkit.framework.autorunner_sensor_sim import gaussian_dist

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


class Sensor1DEnvShuttlerNGauss(gym.Env, ttf.skeleton.Evaluator, ttf.skeleton.Measurement,  # do we need those?
                                ):

    def __init__(self,
                 device_parameter: Dict[str, Any],  # needs to be connected with simulation ...
                 sweep_gate_name: str, # = None,  # i guess str is enough
                 max_del_v: Dict[str, Any],
                 thresholds: Dict[str, Any],
                 measurement,  # : ttf.skeleton.MeasureFunction,
                 # resolution defined by measurement
                 resolution: int = None,
                 show: bool = False,
                 show_ana: bool = False,
                 raw: bool = False,
                 **kwargs,
                 ):

        # ==========
        #  Exp stuff
        # ==========
        self.max_del_v_dict = max_del_v
        self.max_del_v = np.array(list(max_del_v.values()), dtype=np.float64)

        self.device_parameter = device_parameter
        self.measurement = measurement
        if len(self.measurement.parameter) != 0:
            try:
                resolution = self.measurement.parameter['x_points']
            except:
                resolution = self.measurement.parameter['points'][0]
        else:
            resolution = resolution   # FIXME: this is not pretty, fix it. But this is not experiment tho.
        self.resolution = resolution


        self.sweep_gate_name = sweep_gate_name  # Need to differentiate which gate is for sweeping
        assert sweep_gate_name in list(
            self.device_parameter.keys())  # list(measurement.device_parameter.keys())  # recheck if it is the same and also sweep gate is contained

        # save
        self.init_gate_voltages = {g: v.value() for g, v in self.device_parameter.items()}
        self.get_current_gate_voltages(show=False, return_value=False)

        super(Sensor1DEnvShuttlerNGauss, self).__init__(
                                                    name='',
                                                    # parameter=parameter,
                                                    device_parameter=self.device_parameter,
                                                    cachelogic=True,
                                                    auto_save=True,
                                                    save_path=None,  # hardcoded
                                                    **kwargs
                                                )

        self.thresholds = thresholds

        # ==========
        #  RL stuff
        # ==========
        # action space is different from device parameter as only choose to tune non zero max del v
        action_space = spaces.Box(low=-1.,
                                  high=1.,
                                  shape=(len(self.max_del_v),), dtype=np.float64)
        # shape=(len(self.device_parameter.keys()),), dtype=np.float64)  # dtype=np.float32)

        self.action_space = action_space
        self.state = None
        self.observation_space = spaces.Box(low=-1e5, high=1e7, shape=(self.resolution,),
                                            dtype=np.float64)  # self.measurement.data   1e100 -> 1.

        # data
        self.show = show
        self.show_ana = show_ana
        self.data_x = None
        self.dataxarr = None
        self.raw = raw

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
        lower_b = -1.
        high_b = 1.

        norm_factor = (state - np.min(state)) / (
                    np.max(state) - np.min(state) + 1e-10)  # + 1e-10 # avoid dividing by 0 -> otherwise gives nan
        normalized_obs = norm_factor * (high_b - lower_b) + lower_b
        return normalized_obs

    # def reset(self):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        info = {}

        # SEt init gate
        for gate_i, (gate_name, dv) in enumerate(self.device_parameter.items()):
            # print(f"[info] Reset action {self.init_gate_voltages[gate_name]:3f} V will be applied to {gate_name}")  # too wordy
            # print(f"[info] Reset {gate_name}: {self.init_gate_voltages[gate_name]:3f} V")
            dv.value(self.init_gate_voltages[gate_name] - 1e-5)   # can throw some errors
        self.get_current_gate_voltages(show=self.show, return_value=False)  # update

        # Then measure
        self.measurement.measure(auto_plot=self.show)  # use_seed = False by default
        # state = self._normalize_obs(self.measurement.data.data.flatten())  # self.measurement.data.data.flatten()
        state = self.measurement.data.data.flatten()  # self.measurement.data.data.flatten()
        self.data_x = self.measurement.data.coords[self.sweep_gate_name].data
        self.state = state
        self.dataxarr = self.measurement.data

        if not self.observation_space.contains(state):
            print(
                f"[warning] The init observation {state} is not within the observation space {self.observation_space}. We clip it.")
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        return state, info

    def step(self, action, show=None, use_seed=False, **kwargs):  # debug True for now  stepsizes ---> give error
        """
          Action space: [a, b] with a, b within [-1, 1] for amp, sigma

          increase/decrease

        """
        if show is None:
            show = False
        else:
            show = self.show

        # ==================
        #   Process action
        # ==================
        valid_gates_ = {g: dp for g, dp in self.device_parameter.items() if g in self.max_del_v_dict.keys()} # need to get only valid gates based on max del v non zero elements
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

        # ====================
        #   measure
        # ====================
        self.measurement.measure(auto_plot=self.show)  # use_seed = False by default
        # if self.measurement.data
        """ Be careful when directly editing dataxarr instead of defining a new one, it gives error """
        # data_tmp = self.measurement.data.data                       # should be normalized!
        # self.measurement.data.data = self._normalize_obs(data_tmp)  # should be normalized!
        # state = self._normalize_obs(self.measurement.data.data.flatten())   #  change to public function from _normalize_obs # self.measurement.data.data.flatten()
        state = self.measurement.data.data.flatten()
        self.data_x = self.measurement.data.coords[self.sweep_gate_name].data
        self.state = state
        self.dataxarr = self.measurement.data

        # ====================
        #   evaluate
        # ====================
        self.evaluate(**kwargs)  # <--- **kwargs not working then TODO !!!!

        """ 
        Reward shaping : give distance to the target as reward

        if above, then the same
        """
        extra_reward = 0.

        slope_passed = 'X'
        if abs(self.ana.steepest_slope) < self.thresholds['steepest_slope']:
            slope_dist = (self.thresholds['steepest_slope'] - abs(self.ana.steepest_slope))  # minus!!
            slope_reward = abs(self.ana.steepest_slope) / self.thresholds['steepest_slope'] - 1.
            slope_reward *= 10  # maybe try it?
        else:
            # slope_reward = abs(abs(self.ana.steepest_slope) - self.thresholds['steepest_slope']) *2
            slope_dist = 0.
            slope_reward = 10  #  50.
            slope_passed = 'O'
        # print(f"Steepest slope : {self.ana.steepest_slope:.3f} vs. {self.thresholds['steepest_slope']:.3f}. Distance: {slope_dist:.4f}, Extra Reward: {slope_reward:.4f}")
        # env_logger.info(f"Steepest slope : {abs(self.ana.steepest_slope):.3f} vs. {self.thresholds['steepest_slope']:.3f}. Extra Reward: {slope_reward}")

        dyn_passed = 'X'
        if self.ana.dynamic_range < self.thresholds['dynamic_range']:
            dyn_dist = self.thresholds['dynamic_range'] - self.ana.dynamic_range
            dyn_reward = abs(self.ana.dynamic_range) / self.thresholds['dynamic_range'] - 1.
            dyn_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            dyn_dist = 0.
            dyn_reward = 10  # 50.
            dyn_passed = 'O'
        # print(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Distance: {dyn_dist:.4f}, Extra Reward: {dyn_reward:.4f}")
        # env_logger.info(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Extra Reward: {dyn_reward}")

        extra_reward += slope_reward
        extra_reward += dyn_reward
        # print(f'Shaped reward: {extra_reward}')
        # env_logger.info(f'Shaped reward: {extra_reward}')

        """ 
            [ ] need n peaks reward? 

        """
        n_peak_passed = 'X'
        if len(self.ana.peaks) < self.thresholds['peaks']:
            peaks_dist = self.thresholds['peaks'] - len(self.ana.peaks)
            peaks_reward = abs(len(self.ana.peaks)) / self.thresholds['peaks'] - 1.
            peaks_reward *= 50  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            peaks_dist = 0.
            peaks_reward = 10.
            n_peak_passed = 'O'
        # print(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Distance: {dyn_dist:.4f}, Extra Reward: {dyn_reward:.4f}")
        # env_logger.info(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Extra Reward: {dyn_reward}")

        extra_reward += peaks_reward

        # if abs(self.ana.steepest_slope) >= self.thresholds['steepest_slope']:
        #     extra_reward += 50.
        # if self.ana.dynamic_range <= self.thresholds['dynamic_range']:
        #     extra_reward += 50.

        """
            terminate condition:
                above threshold

            thresholds = {'steepest_slope': 0.5,   # 10 too low   0.5 too high ---> 
                          'dynamic_range': 35,   # 40 seems never reachable
                          'peaks': 1,
                         }

            For mu, we need peak detection!

            and give distance to the target as reward
        """

        terminated = False
        if abs(self.ana.steepest_slope) >= self.thresholds['steepest_slope'] and \
                self.ana.dynamic_range >= self.thresholds['dynamic_range'] and \
                len(self.ana.peaks) >= self.thresholds['peaks']:
            terminated = True

        if terminated:
            reward = 10   # 100.  # too much? ^^"
        else:
            reward = -10   # -100.  # How to scale this? T-T

        # print(f'Reward without shaping: {reward}')
        # env_logger.info(f'Reward without shaping: {reward}')

        tot_reward_ = extra_reward + reward
        tot_reward = tot_reward_ / 1e2  #  1e3  # scaling
        # print(tot_reward)
        # print(f'Total Reward: {tot_reward}')
        # env_logger.info(f'Total Reward: {tot_reward}')

        if self.show:
            # print(f'----------------------------------------------------------')
            # print(f'[Reward]')
            print(tabulate([
                ['Steepest slope', f"{abs(self.ana.steepest_slope):.3e}", f"{self.thresholds['steepest_slope']:.3e}",
                 f"{slope_dist:.4e}", slope_passed, f"{slope_reward:.4f}"],
                ['Dynamic range', f'{self.ana.dynamic_range:.4f}', f"{self.thresholds['dynamic_range']:.4f}", f"{dyn_dist:.4f}",
                 dyn_passed, f"{dyn_reward:.4f}"],
                ['N peaks', f'{len(self.ana.peaks)}', f"{self.thresholds['peaks']:.3f}", f"{peaks_dist:.4f}",
                 n_peak_passed, f"{peaks_reward:.4f}"],
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

    def evaluate(self, show=True, **kwargs):  # show not needed but fine

        data = self.state
        assert len(data.shape) == 1
        data_ana1d = Ana1D(data=data, data_x=self.data_x, **kwargs)  # sweep_gate_name='any',
        # except KeyError:

        # ANa: Slope, Prom, Dyn
        data_ana1d.get_dynamic_range(show=self.show_ana, raw=self.raw, **kwargs)  # show No....?
        data_ana1d.get_steepest_slope(show=self.show_ana, raw=self.raw, **kwargs)

        # For mu
        data_ana1d.get_peaks(show=self.show_ana, **kwargs)

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
        assert state.shape == (self.resolution,)  # hardcoded

        if not self.observation_space.contains(state):
            # Adjust the initial observation to ensure it falls within the observation space
            # env_logger.warning(
            #     f"[warning] The observation {state} is not within the observation space {self.observation_space}. We clip it.")
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def get_observation(self):
        # Public method to access the observation result
        return self._get_obs()

    """
    # ====================
    #     GATEs
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
        """ clip to Maximum change for each gate """

        assert isinstance(action, np.ndarray) and isinstance(del_v, np.ndarray)
        assert len(action) == len(del_v)
        action_clip = copy.deepcopy(action)

        for i, (act, dv) in enumerate(zip(action_clip, del_v)):
            if abs(act) > dv:
                # print(f"[warning] ({list(self.device_parameter.keys())[i]}) {act:.3f} V exceeds gate voltage change limit. We clip it to its boundary. {dv:3f}")
                # logger.warning(
                    # f"[warning] ({list(self.device_parameter.keys())[i]}) {act:.3f} V exceeds gate voltage change limit. We clip it to its boundary. {dv:3f}")
                if np.sign(act) < 0:
                    act = -dv
                else:
                    act = dv
            else:
                act = act
            action_clip[i] = act

        # print(f"[info] New action: {[round(x, 3) for x in action_clip]}")  # pretty printing
        # logger.info(f"[info] New action: {[round(x, 3) for x in action_clip]}")

        """ 
        Have to add that if the sweeping gate is at the bound, then have to put the lowest point where the sweep is enabled 
        """
        if self.device_parameter[self.sweep_gate_name].value() + self.measurement.parameter['x_range']/2 >= self.device_parameter[self.sweep_gate_name].bounds[1]:
            new_sweep_gv = self.device_parameter[self.sweep_gate_name].bounds[1] - self.measurement.parameter['x_range']/2 - 1e-5
            print(f"[warning] Next sweep will make the sweeping gate voltage exceed the limit: "
                    f"{self.device_parameter[self.sweep_gate_name].value() + self.measurement.parameter['x_range']/2} >= "
                    f"{self.device_parameter[self.sweep_gate_name].bounds[1]}")
            print(f"[warning] We set the sweep gate voltage to the highest bound: {new_sweep_gv}")

            self.device_parameter[self.sweep_gate_name].value(new_sweep_gv)

        if self.device_parameter[self.sweep_gate_name].value() - self.measurement.parameter['x_range']/2 <= self.device_parameter[self.sweep_gate_name].bounds[0]:
            new_sweep_gv = self.device_parameter[self.sweep_gate_name].bounds[0] + self.measurement.parameter['x_range']/2 + 1e-5
            print(f"[warning] Next sweep will make the sweeping gate voltage exceed the limit: "
                    f"{self.device_parameter[self.sweep_gate_name].value() - self.measurement.parameter['x_range']/2} <= "
                    f"{self.device_parameter[self.sweep_gate_name].bounds[0]}")
            print(f"[warning] We set the sweep gate voltage to the highest bound: {new_sweep_gv}")

            self.device_parameter[self.sweep_gate_name].value(new_sweep_gv)


        return action_clip


    def _at_bounds(self, gate_name, value, precision=1e-7):   # strategy=None,
        high_bound = self.device_parameter[gate_name].bounds[1]
        low_bound = self.device_parameter[gate_name].bounds[0]

        if value > high_bound:
            value = high_bound - 1e-4  # 1e-5 lloooollll
            # print(f'[warning] The gate {gate_name} has reached its boundary {self.device_parameter[gate_name].bounds}. '
            #         f'We clip the value to {value} to {high_bound - 1e-4}')   # 1e-5
            self.device_parameter[gate_name].value(value, soft_fail=False)
            # at_bounds = True

        elif value < low_bound:
            value = low_bound + 1e-4   # 1e-5
            # print(f'[warning] The gate {gate_name} has reached its boundary {self.device_parameter[gate_name].bounds}. '
            #         f'We clip the value to {value} to {low_bound + 1e-4}')   # 1e-5
            self.device_parameter[gate_name].value(value, soft_fail=False)
            # at_bounds = True

        else:
            self.device_parameter[gate_name].value(value, soft_fail=False)
            # at_bounds = False


    def change_gate_voltages(self, gate_name, action):
        if not isinstance(action, float):
            action = np.float64(action)

        self.get_current_gate_voltages(show=False, return_value=False)  # --> make sure it is up to date

        at_bounds_hard = self._at_bounds(gate_name, self.current_gate_voltages[gate_name] + action,
                                         )
                                         # strategy=self.at_bounds_strategy)  # hardcoded

    """
    # ====================
    #     Render
    # ====================
    """

    def render(self, save=False, mode='human'):

        """
            TODO: make my own visualization live window
        """
        if self.state is None:
            raise Exception("The observation is empty! Reset first! ")

        matplotlib.use("Agg")

        # matplotlib
        fig = pylab.figure(figsize=[6, 4],  # Inches
                           dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca()
        ax.plot(self.data_x, self.state)
        ax.set_title("1D sensor")
        ax.set_xlabel(f"x")  # {self.sweep_gate_name}
        ax.set_ylabel(f"I (arb.)")
        ax.grid(color='blue', linestyle='-.', linewidth=1, alpha=0.2)

        # canvas, renderer setting
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        pygame.init()
        pygame.display.init()

        # Window dimensions
        width, height = 800, 600
        window = pygame.display.set_mode((width, height), DOUBLEBUF)
        screen = pygame.display.get_surface()

        # size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        screen.fill(WHITE)

        screen.blit(surf, (0, 0))  # x, y position on screen

        # clock?
        clock = pygame.time.Clock()

        # crashed = False
        # while not crashed:
        # while True:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             crashed = True

        clock.tick(1000000)
        pygame.display.flip()

        # pygame.quit()  # <---- close function is somewhere else

    def close_render(self):
        pygame.quit()

