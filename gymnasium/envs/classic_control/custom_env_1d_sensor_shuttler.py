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

from typing import Union, List, Dict, Callable, Any, Optional

import tuning_toolkit.framework as ttf
# from tuning_toolkit.framework.autorunner_basic_functions import *
from tuning_toolkit.framework.autorunner_ana_1d import Ana1D
# from tuning_toolkit.framework.autorunner_ana_1d_backup import Ana1D   # this is exactly the same as old ana

# from tuning_toolkit.framework.autorunner_ana_2d import *
# from tuning_toolkit.framework.autorunner_utils import *

# from tuning_toolkit.framework.lead_transition_simulation import *
from tuning_toolkit.framework.lead_transition_simulation import gaussian_dist

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


class Sensor1DEnvSimpleShuttler(gym.Env, ttf.skeleton.Evaluator, ttf.skeleton.Measurement,  # do we need those?
                                ):

    def __init__(self,
                 thresholds: Dict[str, Any],
                 device_parameter: Dict[str, Any],
                 max_del_v: Dict[str, Any],

                 # sweep_gate_name: str = None,   # i guess str is enough
                 # max_del_v: List[float] = None,
                 # at_bounds_strategy: str = 'push',
                 show: bool = False,
                 show_ana: bool = False,
                 # use_seed: bool = False,
                 size: int = 100,
                 raw: bool = True,  # use bool by default
                 **kwargs,
                 ):

        self.thresholds = thresholds
        self.device_parameter = device_parameter
        self.max_del_v = max_del_v
        if len(set(self.device_parameter.keys()) ^ set(self.max_del_v.keys())) != 0:
            raise Exception(
                f"Gates from device parameter are not the same as max_del_v: {set(self.device_parameter.keys()) ^ set(self.max_del_v.keys())}")

        self.size = size

        # ==========
        #  RL stuff
        # ==========
        action_space = spaces.Box(low=-1.,
                                  high=1.,
                                  shape=(len(self.device_parameter.keys()),), dtype=np.float64)  # dtype=np.float32)

        self.action_space = action_space
        self.state = None
        # self.observation_space = spaces.Box(low=-1., high=1., shape=(self.size,), dtype=np.float64)    # self.measurement.data   1e100 -> 1.
        self.observation_space = spaces.Box(low=0., high=1e7, shape=(self.size,),
                                            dtype=np.float64)  # self.measurement.data   1e100 -> 1.

        self._amp = None
        self._sigma = None
        self._mu = None

        # data
        self.show = show
        self.show_ana = show_ana
        self.raw = raw
        # self.data = None    # it is a state anyway
        self.data_x = None

    """ no setter! """

    #     @amp.setter
    #     def amp(self):
    #         self._amp = amp

    #     @sigma.setter
    #     def sigma(self):
    #         self._sigma = sigma

    @property
    def amp(self):
        return self._amp

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu(self):
        return self._mu

    def _normalize_action(self, action):
        # Perform min-max scaling
        """
        normalize based on

        have to take - into account!
        """
        lower_b = - np.array(list(self.max_del_v.values()), dtype=np.float64)
        high_b = np.array(list(self.max_del_v.values()), dtype=np.float64)
        # lower_b = -1.  # 0.
        # high_b = 1.

        norm_factor = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        normed_action = norm_factor * (high_b - lower_b) + lower_b
        return normed_action

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

        """
            Only sigma and amp change
        """
        data_x = np.linspace(0, self.size - 1, self.size)
        slope_changer = 0.5  # np.random.rand()
        dyn_changer = 0.35  # np.random.rand()
        mu_changer = 2.  # init value is located at the middle

        amp = 1 * slope_changer
        sigma = 10 * dyn_changer
        mu = self.size / 2

        self._amp = amp
        self._sigma = sigma
        self._mu = mu

        # print(f'amp: {self._amp}, sigma = {self._sigma}')
        env_logger.info(f'amp: {self._amp}, sigma = {self._sigma}')
        data = gaussian_dist(x=data_x, mu=mu, sigma=sigma, amp=amp, size=self.size)
        self.state = data  # self._normalize_obs(data)
        state = self.state
        self.data_x = data_x

        if not self.observation_space.contains(state):
            # Adjust the initial observation to ensure it falls within the observation space
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        if self.show:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(state)

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

        #         action_ = action
        #         if not self.action_space.contains(action_):
        #             print(f"Action {action_} is not within action space. We clip the action to {np.clip(action_, self.action_space.low, self.action_space.high)}.")
        #             # env_logger.warning(f"[warning] Action {action} is not within action space {self.action_space}. We clip the action.")
        #             clip_action = np.clip(action_, self.action_space.low, self.action_space.high)
        #         else:
        #             clip_action = action_

        #         action = self._normalize_action(clip_action)
        #         print(f"[info] Action {clip_action} is normalized to {action}.")

        action_ = action
        action = self._normalize_action(action_)
        if self.show:
            # print(f"[info] Action {action_} is normalized to {action}.")
            for act, (gn, v) in zip(action, self.device_parameter.items()):
                print(f"[info] {gn} : {act}.")
            print('\n')
        if not self.action_space.contains(action):
            # print(f"Action {action} is not within action space. We clip the action to {np.clip(action, self.action_space.low, self.action_space.high)}.")
            # env_logger.warning(f"[warning] Action {action} is not within action space {self.action_space}. We clip the action.")
            clip_action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            clip_action = action

        """ Need to clip amp, sigma if negative """
        # next_amp = self._amp + action[0]
        # next_sigma = self._sigma + action[1]

        next_amp = self.change_amp(action)
        next_sigma = self.change_sigma(action)
        next_mu = self.change_mu(action)

        if next_amp <= 0:
            next_amp = 0. + 1e-5
            # print(f'Next amp is negative: {next_amp:.4f}. We clip it to 0.')
        if next_sigma <= 0:
            next_sigma = 0. + 1e-5
            # print(f'Next amp is negative: {next_sigma:.4f}. We clip it to 0.')
        # any constraints on mu?

        # measure
        # data_x  = np.linspace(0, self.size-1, self.size)   # this is always the same
        next_data = gaussian_dist(x=self.data_x, mu=next_mu, sigma=next_sigma, amp=next_amp, size=self.size)
        # next_data = self._normalize_obs(next_data)
        self.state = next_data
        state = self.state
        self._amp = next_amp
        self._sigma = next_sigma
        self._mu = next_mu

        if self.show:
            # print(f"Amulator amp: {self._amp:.4f}, sigma: {self._sigma:.4f}")
            # env_logger.info(f"Amulator amp: {self._amp:.5f}, sigma: {self._sigma:.5f}")
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(state)
            plt.show()
        self.state = state

        # ====================
        #   evaluate
        # ====================
        self.evaluate(**kwargs)  # <--- **kwargs not working then TODO !!!!

        """ 
        Reward shaping : give distance to the target as reward

        if above, then the same
        """
        extra_reward = 0.

        if abs(self.ana.steepest_slope) < self.thresholds['steepest_slope']:
            slope_dist = (self.thresholds['steepest_slope'] - abs(self.ana.steepest_slope))  # minus!!
            slope_reward = abs(self.ana.steepest_slope) / self.thresholds['steepest_slope'] - 1.
            slope_reward *= 10  # maybe try it?
        else:
            # slope_reward = abs(abs(self.ana.steepest_slope) - self.thresholds['steepest_slope']) *2
            slope_dist = 0.
            slope_reward = 50.
        # print(f"Steepest slope : {self.ana.steepest_slope:.3f} vs. {self.thresholds['steepest_slope']:.3f}. Distance: {slope_dist:.4f}, Extra Reward: {slope_reward:.4f}")
        # env_logger.info(f"Steepest slope : {abs(self.ana.steepest_slope):.3f} vs. {self.thresholds['steepest_slope']:.3f}. Extra Reward: {slope_reward}")

        if self.ana.dynamic_range < self.thresholds['dynamic_range']:
            dyn_dist = self.thresholds['dynamic_range'] - self.ana.dynamic_range
            dyn_reward = abs(self.ana.dynamic_range) / self.thresholds['dynamic_range'] - 1.
            dyn_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            dyn_dist = 0.
            dyn_reward = 50.
        # print(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Distance: {dyn_dist:.4f}, Extra Reward: {dyn_reward:.4f}")
        # env_logger.info(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Extra Reward: {dyn_reward}")

        extra_reward += slope_reward
        extra_reward += dyn_reward
        # print(f'Shaped reward: {extra_reward}')
        # env_logger.info(f'Shaped reward: {extra_reward}')

        """ 
            [ ] need n peaks reward? 

        """
        if len(self.ana.peaks) < self.thresholds['peaks']:
            peaks_dist = self.thresholds['peaks'] - len(self.ana.peaks)
            peaks_reward = abs(len(self.ana.peaks)) / self.thresholds['peaks'] - 1.
            peaks_reward *= 10  # maybe try it?
        else:
            # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
            peaks_dist = 0.
            peaks_reward = 50.
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
        if self.show:
            print(f'------------------------------------------------------------------ Terminated: {terminated}')
            # print(f'Gauss \n\t amp: {self.amp} \n\t sigma: {self.sigma} \n\t mu: {self.mu}\n')
            print(
                f'Ana \n\t steepest_slope: {self.ana.steepest_slope} \n\t dynamic_range: {self.ana.dynamic_range} \n\t n_peaks: {len(self.ana.peaks)}\n')

        if terminated:
            reward = 100.  # too much? ^^"
        else:
            reward = -100.  # How to scale this? T-T

        # print(f'Reward without shaping: {reward}')
        # env_logger.info(f'Reward without shaping: {reward}')

        tot_reward = extra_reward + reward

        tot_reward /= 1e3  # scaling
        # print(f'Total Reward: {tot_reward}')
        # env_logger.info(f'Total Reward: {tot_reward}')

        info = {}

        return self._get_obs(), tot_reward, terminated, False, info

    """
    # =======================
    #    Amp, sigma changer
    # =======================
    """

    def change_amp(self, action):
        """
                                   scale
            AG2       +: increase ?     0.8 (1?)
            CBL       +: decrease     - 0.3     (Claviature Barrier L)    <-- [ ] it might behave same as PSG ????
            LPG:      +: decrease ?   - 0.05    (Left Pinchoff G0)        <-- [ ] is it similar to left PSG?
            TBL/BBL   +: decrease ?   - 0.15
            PL        +: decrease ?   - 0.2

        """
        # changes_gates = {'AG2': 0.8, 'PSG': -0.3, 'LB1': -0.15, 'LB2': -0.15, 'LP1': -0.2}
        # changes_gates = {'AG2': 80., 'PSG': -30., 'LB1': -15, 'LB2': -15, 'LP1': -20.}
        # changes_gates = {'AG2': 120., 'PSG': -30., 'LB1': -10, 'LB2': -10, 'LP1': -20.}    # working one  21.05 at least before

        changes_gates = {'AG2': 120., 'CBL': -30., 'LPG': -5., 'TBL': -10, 'BBL': -10, 'PL': -20.}  # working one

        next_amp_ = copy.deepcopy(self._amp)
        next_amp = next_amp_

        if 'AG2' in self.device_parameter.keys():
            AG2_idx = np.where(np.array(list(self.device_parameter.keys())) == 'AG2')[0][0]
            next_amp += action[AG2_idx] * changes_gates['AG2']
            if self.show:
                print(f"AG2 added {action[AG2_idx] * changes_gates['AG2']} change to amp: {next_amp}")

        if 'CBL' in self.device_parameter.keys():
            PSG_idx = np.where(np.array(list(self.device_parameter.keys())) == 'CBL')[0][0]
            next_amp += action[PSG_idx] * changes_gates['CBL']
            if self.show:
                print(f"PSG added {action[PSG_idx] * changes_gates['CBL']} change to amp: {next_amp}")

        if 'LPG' in self.device_parameter.keys():
            PSG_idx = np.where(np.array(list(self.device_parameter.keys())) == 'LPG')[0][0]
            next_amp += action[PSG_idx] * changes_gates['LPG']
            if self.show:
                print(f"PSG added {action[PSG_idx] * changes_gates['LPG']} change to amp: {next_amp}")

        if 'TBL' in self.device_parameter.keys():
            LB1_idx = np.where(np.array(list(self.device_parameter.keys())) == 'TBL')[0][0]
            next_amp += action[LB1_idx] * changes_gates['TBL']
            if self.show:
                print(f"LB1 added {action[LB1_idx] * changes_gates['TBL']} change to amp: {next_amp}")

        if 'BBL' in self.device_parameter.keys():
            LB2_idx = np.where(np.array(list(self.device_parameter.keys())) == 'BBL')[0][0]
            next_amp += action[LB2_idx] * changes_gates['BBL']
            if self.show:
                print(f"LB1 added {action[LB2_idx] * changes_gates['BBL']} change to amp: {next_amp}")

        if 'PL' in self.device_parameter.keys():
            LP1_idx = np.where(np.array(list(self.device_parameter.keys())) == 'PL')[0][0]
            next_amp += action[LP1_idx] * changes_gates['PL']
            if self.show:
                print(f"LP1 added {action[LP1_idx] * changes_gates['PL']} change to amp: {next_amp}")

        if self.show:
            print(f'Final next amp: {next_amp}\n')

        return next_amp

    def change_sigma(self, action):
        """
                                   scale
            AG2   +: decrease ?   + 0.3 (1?)
            CBL   +: increase     - 0.5
            LPG   +: increase     - 0.08
            TBL/BBL +: decrease ?   - 0.3
            PL   +: decrease ?   - 0.3

        """
        # changes_gates = {'AG2': 0.3, 'PSG': -0.5, 'LB1': -0.3, 'LB2': -0.3, 'LP1': -0.3}
        # changes_gates = {'AG2': 100., 'PSG': -50, 'LB1': -30, 'LB2': -30, 'LP1': -30}
        # changes_gates = {'AG2': 150., 'PSG': -50, 'LB1': -15, 'LB2': -15, 'LP1': -15}     # working one

        changes_gates = {'AG2': 150., 'CBL': -50, 'LPG': -8, 'TBL': -15, 'BBL': -15, 'PL': -15}  # working one

        next_sigma_ = copy.deepcopy(self._sigma)
        next_sigma = next_sigma_

        if 'AG2' in self.device_parameter.keys():
            AG2_idx = np.where(np.array(list(self.device_parameter.keys())) == 'AG2')[0][0]
            next_sigma += action[AG2_idx] * changes_gates['AG2']
            if self.show:
                print(f"AG2 added {action[AG2_idx] * changes_gates['AG2']} change to sigma: {next_sigma}.")

        if 'CBL' in self.device_parameter.keys():
            PSG_idx = np.where(np.array(list(self.device_parameter.keys())) == 'CBL')[0][0]
            next_sigma += action[PSG_idx] * changes_gates['CBL']
            if self.show:
                print(f"PSG added {action[PSG_idx] * changes_gates['CBL']} change to sigma: {next_sigma}")

        if 'LPG' in self.device_parameter.keys():
            PSG_idx = np.where(np.array(list(self.device_parameter.keys())) == 'LPG')[0][0]
            next_sigma += action[PSG_idx] * changes_gates['LPG']
            if self.show:
                print(f"PSG added {action[PSG_idx] * changes_gates['LPG']} change to sigma: {next_sigma}")

        if 'TBL' in self.device_parameter.keys():
            LB1_idx = np.where(np.array(list(self.device_parameter.keys())) == 'TBL')[0][0]
            next_sigma += action[LB1_idx] * changes_gates['TBL']
            if self.show:
                print(f"LB1 added {action[LB1_idx] * changes_gates['TBL']} change to sigma: {next_sigma}.")

        if 'BBL' in self.device_parameter.keys():
            LB2_idx = np.where(np.array(list(self.device_parameter.keys())) == 'BBL')[0][0]
            next_sigma += action[LB2_idx] * changes_gates['BBL']
            if self.show:
                print(f"LB1 added {action[LB2_idx] * changes_gates['BBL']} change to sigma: {next_sigma}")

        if 'PL' in self.device_parameter.keys():
            LP1_idx = np.where(np.array(list(self.device_parameter.keys())) == 'PL')[0][0]
            next_sigma += action[LP1_idx] * changes_gates['PL']
            if self.show:
                print(f"LP1 added {action[LP1_idx] * changes_gates['PL']} change to sigma: {next_sigma}.")

        if self.show:
            print(f'Final next sigma: {next_sigma}\n')

        return next_sigma

    def change_mu(self, action):
        """
                                   scale
            AG2   +: go positive ?   50
            PSG   +: go negative ?
            LB1/2 +: decrease ?      [ ] LB1 vs LB2 should be opposing?
            LP1   +: decrease ?      [ ] what about LP1....?

        """
        # changes_gates = {'AG2': 0.3, 'PSG': -0.5, 'LB1': -0.3, 'LB2': -0.3, 'LP1': -0.3}
        # changes_gates = {'AG2': 100., 'PSG': -50, 'LB1': -30, 'LB2': -30, 'LP1': -30}

        # changes_gates = {'AG2': 20000., 'PSG': -500, 'LB1': -10, 'LB2': 10, 'LP1': -0}    # working one before 21.05

        changes_gates = {'AG2': 20000., 'CBL': -500, 'LPG': -500, 'TBL': -10, 'BBL': 10,
                         'PL': -0}  # working one before 21.05
        changes_gates = {k: v * 0.01 for k, v in changes_gates.items()}  # reduced by 2

        next_mu_ = copy.deepcopy(self._mu)
        next_mu = next_mu_

        if 'AG2' in self.device_parameter.keys():
            AG2_idx = np.where(np.array(list(self.device_parameter.keys())) == 'AG2')[0][0]
            next_mu += action[AG2_idx] * changes_gates['AG2']
            if self.show:
                print(f"AG2 added {action[AG2_idx] * changes_gates['AG2']} change to mu: {next_mu}.")

        if 'CBL' in self.device_parameter.keys():
            PSG_idx = np.where(np.array(list(self.device_parameter.keys())) == 'CBL')[0][0]
            next_mu += action[PSG_idx] * changes_gates['CBL']
            if self.show:
                print(f"PSG added {action[PSG_idx] * changes_gates['CBL']} change to mu: {next_mu}")

        if 'LPG' in self.device_parameter.keys():
            PSG_idx = np.where(np.array(list(self.device_parameter.keys())) == 'LPG')[0][0]
            next_mu += action[PSG_idx] * changes_gates['LPG']
            if self.show:
                print(f"PSG added {action[PSG_idx] * changes_gates['LPG']} change to mu: {next_mu}")

        if 'TBL' in self.device_parameter.keys():
            LB1_idx = np.where(np.array(list(self.device_parameter.keys())) == 'TBL')[0][0]
            next_mu += action[LB1_idx] * changes_gates['TBL']
            if self.show:
                print(
                    f"LB1 added {action[LB1_idx]}, {changes_gates['TBL']}, {action[LB1_idx] * changes_gates['TBL']} change to mu: {next_mu}.")

        if 'BBL' in self.device_parameter.keys():
            LB2_idx = np.where(np.array(list(self.device_parameter.keys())) == 'BBL')[0][0]
            next_mu += action[LB2_idx] * changes_gates['BBL']
            if self.show:
                print(f"LB1 added {action[LB2_idx] * changes_gates['BBL']} change to mu: {next_mu}")

        if 'PL' in self.device_parameter.keys():
            LP1_idx = np.where(np.array(list(self.device_parameter.keys())) == 'PL')[0][0]
            next_mu += action[LP1_idx] * changes_gates['PL']
            if self.show:
                print(f"LP1 added {action[LP1_idx] * changes_gates['PL']} change to mu: {next_mu}.")

        if self.show:
            print(f'Final next mu: {next_mu}\n')

        return next_mu

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
        assert state.shape == (self.size,)  # hardcoded

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

