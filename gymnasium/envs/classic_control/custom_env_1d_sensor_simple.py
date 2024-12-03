""" gym != gymnasium """
# import gym
# from gym import spaces

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import pygame
from pygame.locals import *
import matplotlib.backends.backend_agg as agg
import pylab


from typing import Union, List, Dict, Callable, Any, Optional

import tuning_toolkit.framework as ttf
from tuning_toolkit.framework.autorunner_basic_functions import *
from tuning_toolkit.framework.autorunner_ana_1d import *
# from tuning_toolkit.framework.autorunner_ana_2d import *
from tuning_toolkit.framework.autorunner_utils_old import *

from tuning_toolkit.simulation.lead_transition_simulation import *
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


class Sensor1DEnvSimple(gym.Env, ttf.skeleton.Evaluator, ttf.skeleton.Measurement,   # do we need those?
            ):

    def __init__(self,
                 thresholds: Dict[str, Any],
                 sweep_gate_name: str = None,   # i guess str is enough
                 # continuous : bool = True,
                 max_del_v: List[float] = None,
                 at_bounds_strategy: str = 'push',
                 show: bool = False,
                 use_seed: bool = False,
                 size: int = 100,
                 raw: bool = True,   # use bool by default
                 **kwargs,
                 ):
        # super(Sensor1DEnvSimple, self).__init__()   # should not call twice
        # super(Sensor1DEnvSimple, self).__init__(
        #                                         max_del_v=max_del_v,
        #                                         # parameter=parameter,
        #                                         at_bounds_strategy=at_bounds_strategy,
        #                                         cachelogic=True,
        #                                         auto_save=True,
        #                                         save_path=None,    # hardcoded
        #                                         **kwargs
        #                                         )

        """

        THoughts:
        - [ ] make env specific to a sample like shuttler or short loop
              -> which means it should have different set of gates
              1. Short loop
                  - AG2
                  - PSG
                  - Barriers
              2. Shuttler

            [State]: ndarray

            - ana_state:
            | Num | State  (failing)             | Min               | Max             |
            |-----|------------------------------|-------------------|-----------------|
            | 0   | 0 failed -> passed           | 0. (float32)      | 1. (float32)    |
            | 1   | 1 failed -> passed           | 0. (float32)      | 1. (float32)    |
            | 2   | 2 failed -> passed           | 0. (float32)      | 1. (float32)    |
            | 3   | 3 failed -> passed           | 0. (float32)      | 1. (float32)    |
            low=0, high=3

            Options:
            - [ ] Maybe one can consider the 1D trace as the entire state together with the analysis result?
            - [ ] add numerous values for analysis results instead of binaries
            - [ ] check if gate voltages are actually related..

            [Action space]:
              1) Discrete
            | Num | Action                                | Unit |
            |-----|---------------------------------------|------|
            | 0   | decrease (with stepsize)              |   V  |
            | 1   | increase (with stepsize)              |   V  |
            | 2   | hold (with stepsize)                  |   V  |
            | ... | repetition (x n_gates)                |   V  |

            [0, 2, 1, 2] .... len(gate_space)

              2) Continous
            | Num | Action                                | Unit |
            |-----|---------------------------------------|------|
            |  -  | ndarray with shape (n_gates, 1)       |   V  |

            [0.1, 0.2, 0.1, -0.2] .... len(gate_space)

            [Observation space] criteria based on thresholds:
              - Slope
              - Dynamic range (linear range)
              - Prominent

            * Edited: Typically states --> ndarray *

            thresholds must have identical atrribute values as in ana

            It is an one hot code: [0, 0, 1] for example. [-1, -1, -1] for initial state (failed if True)
                | Num | Observation  (failing)       | Min                 | Max               |
                |-----|------------------------------|---------------------|-------------------|
                | 0   | Slope, Dyn, Prom             | -1 (False)           | 1 (True)          |
                | 1   | Slope, Dyn                   | -1 (False)           | 1 (True)          |
                | 2   | Slope, Prom                  | -1 (False)           | 1 (True)          |
                | 3   | Dyn, Prom                    | -1 (False)           | 1 (True)          |
                | 4   | Slope                        | -1 (float32)      | arb. (float32)    |
                | 5   | Dyn                          | arb. (float32)      | arb. (float32)    |
                | 6   | Prom                         | arb. (float32)      | arb. (float32)    |
                | 7   | None                         | arb. (float32)      | arb. (float32)    |
                | 7   | Empty [-1, -1, -1]           | arb. (float32)      | arb. (float32)    |


            -------------
            State vs. Obs
            -------------
            -> Observation: ndarray of float
            -> State: ndarray of float of [0., 1.]
        """
#         thresholds = {'steepest_slope': 70.,   # 10 too low
#                       'dynamic_range': 0.04,  # 0.045 0.05 is too high  0.03 to little
#                       'prominence': 70.}       # 10 too low

        self.thresholds = thresholds
        self.use_seed = use_seed
        self.size = size

        # ==========
        #  RL stuff
        # ==========
        action_space = spaces.Box(low=-1.,
                                   high=1.,
                                   shape=(2,), dtype=np.float64)  # dtype=np.float32)

        self.action_space = action_space
        self.state = None
        # self.observation_space = spaces.Box(low=-1., high=1., shape=(self.size,), dtype=np.float64)    # self.measurement.data   1e100 -> 1.
        self.observation_space = spaces.Box(low=0., high=1e7, shape=(self.size,), dtype=np.float64)    # self.measurement.data   1e100 -> 1.


        self._amp = None
        self._sigma = None

        # data
        self.show = show
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


    def _normalize_action(self, action):
        # Perform min-max scaling
        """
        normalize based on

        have to take - into account!
        """
        # lower_b = np.array(-self.max_del_v, dtype=np.float64)
        # high_b = np.array(+self.max_del_v, dtype=np.float64)
        lower_b = -1.  # 0.
        high_b = 1.

        norm_factor = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        normed_action = norm_factor*(high_b - lower_b) + lower_b
        return normed_action


    def _normalize_obs(self, state):
        # Perform min-max scaling
        """
        normalize based on
        """
        lower_b = -1.
        high_b = 1.

        norm_factor = (state - np.min(state)) / (np.max(state) - np.min(state) + 1e-10)   # + 1e-10 # avoid dividing by 0 -> otherwise gives nan
        normalized_obs = norm_factor*(high_b - lower_b) + lower_b
        return normalized_obs


    # def reset(self):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        info = {}

        """
            Only sigma and amp change
        """
        data_x  = np.linspace(0, self.size-1, self.size)
        slope_changer = 0.5  # np.random.rand()
        dyn_changer = 0.35  # np.random.rand()

        amp = 1*slope_changer
        sigma = 10*dyn_changer

        self._amp = amp
        self._sigma = sigma

        # print(f'amp: {self._amp}, sigma = {self._sigma}')
        # env_logger.warnings(f'amp: {self._amp}, sigma = {self._sigma}')
        data = gaussian_dist(x=data_x, mu=self.size/2, sigma=sigma, amp=amp, size=self.size)
        self.state = data  # self._normalize_obs(data)
        state = self.state
        self.data_x = data_x

        if not self.observation_space.contains(state):
            # Adjust the initial observation to ensure it falls within the observation space
            state = np.clip(state, self.observation_space.low, self.observation_space.high)

        if self.show:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
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

        action_ = action
        if not self.action_space.contains(action_):
            # print(f"Action {action_} is not within action space. We clip the action to {np.clip(action_, self.action_space.low, self.action_space.high)}.")
            # env_logger.warning(f"[warning] Action {action} is not within action space {self.action_space}. We clip the action.")
            clip_action = np.clip(action_, self.action_space.low, self.action_space.high)
        else:
            clip_action = action_

        action = self._normalize_action(clip_action)
        # print(f"[info] Action {clip_action} is normalized to {action}.")

        """ Need to clip amp, sigma if negative """
        next_amp = self._amp + action[0]
        next_sigma = self._sigma + action[1]
        if next_amp <= 0:
            next_amp = 0. + 1e-5
            # print(f'Next amp is negative: {next_amp:.4f}. We clip it to 0.')
        if next_sigma <= 0:
            next_sigma = 0. + 1e-5
            # print(f'Next amp is negative: {next_sigma:.4f}. We clip it to 0.')

        # measure
        # data_x  = np.linspace(0, self.size-1, self.size)   # this is always the same
        next_data = gaussian_dist(x=self.data_x, mu=self.size/2, sigma=next_sigma, amp=next_amp, size=self.size)
        # next_data = self._normalize_obs(next_data)
        self.state = next_data
        state = self.state
        self._amp = next_amp
        self._sigma = next_sigma

        if self.show:
            # print(f"Amulator amp: {self._amp:.4f}, sigma: {self._sigma:.4f}")
            # env_logger.info(f"Amulator amp: {self._amp:.5f}, sigma: {self._sigma:.5f}")
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
            ax.plot(state)
        self.state = state

        # ====================
        #   evaluate
        # ====================
        self.evaluate(**kwargs)   # <--- **kwargs not working then TODO !!!!

        """ 
        Reward shaping : give distance to the target as reward
        
        if above, then the same
        """
        extra_reward = 0.

        if abs(self.ana.steepest_slope) < self.thresholds['steepest_slope']:
            slope_dist = (self.thresholds['steepest_slope'] - abs(self.ana.steepest_slope))     # minus!!
            slope_reward = abs(self.ana.steepest_slope)/self.thresholds['steepest_slope'] - 1.
            slope_reward *= 10  # maybe try it?
        else:
            # slope_reward = abs(abs(self.ana.steepest_slope) - self.thresholds['steepest_slope']) *2
            slope_dist = 0.
            slope_reward = 50.
        # print(f"Steepest slope : {self.ana.steepest_slope:.3f} vs. {self.thresholds['steepest_slope']:.3f}. Distance: {slope_dist:.4f}, Extra Reward: {slope_reward:.4f}")
        # env_logger.info(f"Steepest slope : {abs(self.ana.steepest_slope):.3f} vs. {self.thresholds['steepest_slope']:.3f}. Extra Reward: {slope_reward}")

        if self.ana.dynamic_range < self.thresholds['dynamic_range']:
            dyn_dist = self.thresholds['dynamic_range'] - self.ana.dynamic_range
            dyn_reward = abs(self.ana.dynamic_range)/self.thresholds['dynamic_range'] - 1.
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


        # if abs(self.ana.steepest_slope) >= self.thresholds['steepest_slope']:
        #     extra_reward += 50.
        # if self.ana.dynamic_range <= self.thresholds['dynamic_range']:
        #     extra_reward += 50.

        """
            terminate condition:
                above threshold
                
            and give distance to the target as reward
        """

        terminated = False
        if abs(self.ana.steepest_slope) >= self.thresholds['steepest_slope'] and \
           self.ana.dynamic_range >= self.thresholds['dynamic_range']:
            terminated = True

        if terminated:
            reward = 100.   # too much? ^^"
        else:
            reward = -100.  # How to scale this? T-T

        # print(f'Reward without shaping: {reward}')
        # env_logger.info(f'Reward without shaping: {reward}')

        tot_reward = extra_reward + reward

        tot_reward /= 1e3   # scaling
        # print(f'Total Reward: {tot_reward}')
        # env_logger.info(f'Total Reward: {tot_reward}')

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
        data_ana1d.get_dynamic_range(show=self.show, raw=self.raw, **kwargs)   # show No....?
        data_ana1d.get_steepest_slope(show=self.show, raw=self.raw, **kwargs)
        # data_ana1d.get_peaks(show=self.show, **kwargs)

        """ not critical ana results but contribute to the reward 
            By hand!
        """

        # reward = data_ana1d.steepest_slope + data_ana1d.dynamic_range



#         non_crit_thresholds = {'gauss_avg_std': 1.0,  'peaks_increasing': 1}
#         data_ana1d.gauss_fit(show=self.show, **kwargs)           # gauss_avg_std
#         data_ana1d.check_peak_increasing(**kwargs)

#         if self.ana is not None:
#             old_ana_values = self.get_ana_values(self.ana)   # <--- it is before updating to new ana so it is an old values
#         else:
#             old_ana_values = np.array([0., 0., 0.])

#         new_ana_values = self.get_ana_values(data_ana1d)

        """ log ana with length of 2 for reward shaping HARDCODED
            - 1) if better than old value -> give binary reward?
        """
        # shaped_reward = sum(np.sign(new_ana_values - old_ana_values))  # give as a reward
        # # shaped_reward = sum(np.sign(new_ana_values - old_ana_values) == 1)  # give as a reward
        # # shaped_reward = -sum(np.sign(new_ana_values - old_ana_values) == -1)  # give as a reward

        """ - 2) give reward prop. distance to the threshold as additional shaped rewarddddddd 
                - it should be normalized i think,.. 
                ex. array([ 0.23512847, -0.47503059,  1.05017752])
                
                with weights: I guess the width of the peak is more important than the other two as the width is the hard thing to achieve. --> domain knowledge
        
        """
        # # distance_reward = sum(new_ana_values - np.array(list(self.thresholds.values())))
        # # distance_reward = sum(new_ana_values/np.array(list(self.thresholds.values())) - 1.)   # add distance if lower, minus, else, larger than 0.
        # distance_reward = new_ana_values/np.array(list(self.thresholds.values())) - 1.   # add distance if lower, minus, else, larger than 0.
        # distance_reward = sum([d/10 for i, d in enumerate(distance_reward) if i==0 or i==2])
        # shaped_reward += distance_reward

#         logger.info(f"[info] Shaped reward: old values = {[round(x, 2) for x in old_ana_values]}, new values = {[round(x, 2) for x in new_ana_values]}. Total reward {shaped_reward} will be added")
#         self.shaped_reward = shaped_reward

#         # non crit ana reward
#         non_crit_ana_reward = 0.
#         if data_ana1d.gauss_avg_std <= non_crit_thresholds['gauss_avg_std']:
#             non_crit_ana_reward += 1.
#             logger.info(f"[info] Shaped reward: gauss fitting above threshold: {data_ana1d.gauss_avg_std:3f} >= {non_crit_thresholds['gauss_avg_std']:3f}. Total reward {non_crit_ana_reward} will be added")
#         if data_ana1d.peaks_increasing == non_crit_thresholds['peaks_increasing']:
#             non_crit_ana_reward += 1.
#             logger.info(f"[info] Shaped reward: peak increasing above threshold: {data_ana1d.peaks_increasing} >= {non_crit_thresholds['peaks_increasing']}. Total reward {non_crit_ana_reward} will be added")
#         shaped_reward += non_crit_ana_reward


        # TOTAL SHAPED REWARD
        # self.shaped_reward = shaped_reward

        # update
        self.ana = data_ana1d  # at the end so that it stores all the results
#         self.ana_value_state = self.get_ana_values(self.ana)

#         # Compare
#         compa_results = check_thresholds_simple(self.ana, self.thresholds)   # True if passed
#         self.compa_results = compa_results

#         self.ana_state = sum(self.compa_results.values())
#         # self.state = np.array(list(self.get_current_gate_voltages(show=False).values()) + [self.ana_state], dtype=np.float32)
#         self.state = self._normalize_obs(self.dataxarr.data.flatten())  # self.dataxarr.coords[self.sweep_gate_name].data

    """
    # ====================
    #     Obs
    # ====================
    """

    def _get_obs(self):   # --> returns state
        # state = np.array(list(self.get_current_gate_voltages(show=False, return_value=True).values()) + list(self.ana_value_state) + [self.ana_state], dtype=np.float32)
        # state = np.array(list(self.get_current_gate_voltages(show=False, return_value=True).values()) + list(self.ana_value_state), dtype=np.float32)
        state = self.state
        assert state.shape == (self.size,)   # hardcoded

        if not self.observation_space.contains(state):
            # Adjust the initial observation to ensure it falls within the observation space
            env_logger.warning(f"[warning] The observation {state} is not within the observation space {self.observation_space}. We clip it.")
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
#     def define_simulator(self):
#         # Reset simu
#         # simple_sensor_1d = SimpleSensor1d(device_parameter=self.device_parameter, sweep_range=[0.1, 0.5], sweep_gate='LB1')
#         # simple_sensor_1d.init_gate_effect()

#         if self.state is None:
#             raise Exception("The observation is empty! Reset first! ")
#         self.simulator = self.state

    # FOr visualization
    def render(self, mode='human'):

        """
            TODO: make my own visualization live window
        """
        if self.state is None:
            raise Exception("The observation is empty! Reset first! ")

        matplotlib.use("Agg")

        # matplotlib
        fig = pylab.figure(figsize=[6, 4], # Inches
                           dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca()
        ax.plot(self.data_x, self.state)
        ax.set_title("1D sensor")
        ax.set_xlabel(f"x")   # {self.sweep_gate_name}
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

        screen.blit(surf, (0, 0))   # x, y position on screen

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



"""
This one does not work, This is only for backing up, 30-04-2024 for debugging what is wrong with the ana 1d
"""
""" gym != gymnasium """
# # import gym
# # from gym import spaces
#
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import logging
# import pygame
# from pygame.locals import *
# import matplotlib.backends.backend_agg as agg
# import pylab
#
# from typing import Union, List, Dict, Callable, Any, Optional
#
# import tuning_toolkit.framework as ttf
# from tuning_toolkit.framework.autorunner_basic_functions import *
# from tuning_toolkit.framework.autorunner_ana_1d import *
# # from tuning_toolkit.framework.autorunner_ana_2d import *
# from tuning_toolkit.framework.autorunner_utils import *
#
# from tuning_toolkit.framework.lead_transition_simulation import *
#
# # ========= logging
# # Configure basic logging settings
# logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
# # logging.disable(logging.CRITICAL)
#
# """ logger name colide with lead_transition_simulation """
# # Create a logger
# env_logger = logging.getLogger(__name__)
#
# # Create a handler for console output
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)  # Set the level for this handler to INFO
#
# # Create a formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)  # Set the formatter for the handler
#
# # Add the handler to the logger
# env_logger.addHandler(console_handler)
#
#
# class Sensor1DEnvSimple(gym.Env, ttf.skeleton.Evaluator, ttf.skeleton.Measurement,  # do we need those?
#                         ):
#
#     def __init__(self,
#                  thresholds: Dict[str, Any],
#                  sweep_gate_name: str = None,  # i guess str is enough
#                  # continuous : bool = True,
#                  max_del_v: List[float] = None,
#                  at_bounds_strategy: str = 'push',
#                  show: bool = False,
#                  use_seed: bool = False,
#                  size: int = 100,
#                  raw: bool = True,  # use bool by default
#                  **kwargs,
#                  ):
#
#         self.thresholds = thresholds
#         self.use_seed = use_seed
#         self.size = size
#
#         # ==========
#         #  RL stuff
#         # ==========
#         action_space = spaces.Box(low=-1.,
#                                   high=1.,
#                                   shape=(2,), dtype=np.float64)  # dtype=np.float32)
#
#         self.action_space = action_space
#         self.state = None
#         # self.observation_space = spaces.Box(low=-1., high=1., shape=(self.size,), dtype=np.float64)    # self.measurement.data   1e100 -> 1.
#         self.observation_space = spaces.Box(low=0., high=1e7, shape=(self.size,),
#                                             dtype=np.float64)  # self.measurement.data   1e100 -> 1.
#
#         self._amp = None
#         self._sigma = None
#
#         # data
#         self.show = show
#         self.raw = raw
#         # self.data = None    # it is a state anyway
#         self.data_x = None
#
#     """ no setter! """
#
#     #     @amp.setter
#     #     def amp(self):
#     #         self._amp = amp
#
#     #     @sigma.setter
#     #     def sigma(self):
#     #         self._sigma = sigma
#
#     @property
#     def amp(self):
#         return self._amp
#
#     @property
#     def sigma(self):
#         return self._sigma
#
#     def _normalize_action(self, action):
#         # Perform min-max scaling
#         """
#         normalize based on
#
#         have to take - into account!
#         """
#         # lower_b = np.array(-self.max_del_v, dtype=np.float64)
#         # high_b = np.array(+self.max_del_v, dtype=np.float64)
#         lower_b = -1.  # 0.
#         high_b = 1.
#
#         norm_factor = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
#         normed_action = norm_factor * (high_b - lower_b) + lower_b
#         return normed_action
#
#     def _normalize_obs(self, state):
#         # Perform min-max scaling
#         """
#         normalize based on
#         """
#         lower_b = -1.
#         high_b = 1.
#
#         norm_factor = (state - np.min(state)) / (
#                     np.max(state) - np.min(state) + 1e-10)  # + 1e-10 # avoid dividing by 0 -> otherwise gives nan
#         normalized_obs = norm_factor * (high_b - lower_b) + lower_b
#         return normalized_obs
#
#     # def reset(self):
#     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
#
#         super().reset(seed=seed)
#         info = {}
#
#         """
#             Only sigma and amp change
#         """
#         data_x = np.linspace(0, self.size - 1, self.size)
#         slope_changer = 0.5  # np.random.rand()
#         dyn_changer = 0.35  # np.random.rand()
#
#         amp = 1 * slope_changer
#         sigma = 10 * dyn_changer
#
#         self._amp = amp
#         self._sigma = sigma
#
#         # print(f'amp: {self._amp}, sigma = {self._sigma}')
#         env_logger.info(f'amp: {self._amp}, sigma = {self._sigma}')
#         data = gaussian_dist(x=data_x, mu=self.size / 2, sigma=sigma, amp=amp, size=self.size)
#         self.state = data  # self._normalize_obs(data)
#         state = self.state
#         self.data_x = data_x
#
#         if not self.observation_space.contains(state):
#             # Adjust the initial observation to ensure it falls within the observation space
#             state = np.clip(state, self.observation_space.low, self.observation_space.high)
#
#         if self.show:
#             fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#             ax.plot(state)
#
#         return state, info
#
#     def step(self, action, show=None, use_seed=False, **kwargs):  # debug True for now  stepsizes ---> give error
#         """
#           Action space: [a, b] with a, b within [-1, 1] for amp, sigma
#
#           increase/decrease
#
#         """
#         if show is None:
#             show = False
#         else:
#             show = self.show
#
#         action_ = action
#         if not self.action_space.contains(action_):
#             # print(f"Action {action_} is not within action space. We clip the action to {np.clip(action_, self.action_space.low, self.action_space.high)}.")
#             # env_logger.warning(f"[warning] Action {action} is not within action space {self.action_space}. We clip the action.")
#             clip_action = np.clip(action_, self.action_space.low, self.action_space.high)
#         else:
#             clip_action = action_
#
#         action = self._normalize_action(clip_action)
#         # print(f"[info] Action {clip_action} is normalized to {action}.")
#
#         """ Need to clip amp, sigma if negative """
#         next_amp = self._amp + action[0]
#         next_sigma = self._sigma + action[1]
#         if next_amp <= 0:
#             next_amp = 0. + 1e-5
#             # print(f'Next amp is negative: {next_amp:.4f}. We clip it to 0.')
#         if next_sigma <= 0:
#             next_sigma = 0. + 1e-5
#             # print(f'Next amp is negative: {next_sigma:.4f}. We clip it to 0.')
#
#         # measure
#         # data_x  = np.linspace(0, self.size-1, self.size)   # this is always the same
#         next_data = gaussian_dist(x=self.data_x, mu=self.size / 2, sigma=next_sigma, amp=next_amp, size=self.size)
#         # next_data = self._normalize_obs(next_data)
#         self.state = next_data
#         state = self.state
#         self._amp = next_amp
#         self._sigma = next_sigma
#
#         if self.show:
#             # print(f"Amulator amp: {self._amp:.4f}, sigma: {self._sigma:.4f}")
#             # env_logger.info(f"Amulator amp: {self._amp:.5f}, sigma: {self._sigma:.5f}")
#             fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#             ax.plot(state)
#         self.state = state
#
#         # ====================
#         #   evaluate
#         # ====================
#         self.evaluate(**kwargs)  # <--- **kwargs not working then TODO !!!!
#
#         """
#         Reward shaping : give distance to the target as reward
#
#         if above, then the same
#         """
#         extra_reward = 0.
#
#         if abs(self.ana.steepest_slope) < self.thresholds['steepest_slope']:
#             slope_dist = (self.thresholds['steepest_slope'] - abs(self.ana.steepest_slope))  # minus!!
#             slope_reward = abs(self.ana.steepest_slope) / self.thresholds['steepest_slope'] - 1.
#             slope_reward *= 10  # maybe try it?
#         else:
#             # slope_reward = abs(abs(self.ana.steepest_slope) - self.thresholds['steepest_slope']) *2
#             slope_dist = 0.
#             slope_reward = 50.
#         # print(f"Steepest slope : {self.ana.steepest_slope:.3f} vs. {self.thresholds['steepest_slope']:.3f}. Distance: {slope_dist:.4f}, Extra Reward: {slope_reward:.4f}")
#         # env_logger.info(f"Steepest slope : {abs(self.ana.steepest_slope):.3f} vs. {self.thresholds['steepest_slope']:.3f}. Extra Reward: {slope_reward}")
#
#         if self.ana.dynamic_range < self.thresholds['dynamic_range']:
#             dyn_dist = self.thresholds['dynamic_range'] - self.ana.dynamic_range
#             dyn_reward = abs(self.ana.dynamic_range) / self.thresholds['dynamic_range'] - 1.
#             dyn_reward *= 10  # maybe try it?
#         else:
#             # dyn_reward  = abs(self.ana.dynamic_range - self.thresholds['dynamic_range']) *2
#             dyn_dist = 0.
#             dyn_reward = 50.
#         # print(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Distance: {dyn_dist:.4f}, Extra Reward: {dyn_reward:.4f}")
#         # env_logger.info(f"Dynamic range : {self.ana.dynamic_range:.3f} vs. {self.thresholds['dynamic_range']:.3f}. Extra Reward: {dyn_reward}")
#
#         extra_reward += slope_reward
#         extra_reward += dyn_reward
#         # print(f'Shaped reward: {extra_reward}')
#         # env_logger.info(f'Shaped reward: {extra_reward}')
#
#         # if abs(self.ana.steepest_slope) >= self.thresholds['steepest_slope']:
#         #     extra_reward += 50.
#         # if self.ana.dynamic_range <= self.thresholds['dynamic_range']:
#         #     extra_reward += 50.
#
#         """
#             terminate condition:
#                 above threshold
#
#             and give distance to the target as reward
#         """
#
#         terminated = False
#         if abs(self.ana.steepest_slope) >= self.thresholds['steepest_slope'] and \
#                 self.ana.dynamic_range >= self.thresholds['dynamic_range']:
#             terminated = True
#
#         if terminated:
#             reward = 100.  # too much? ^^"
#         else:
#             reward = -100.  # How to scale this? T-T
#
#         # print(f'Reward without shaping: {reward}')
#         # env_logger.info(f'Reward without shaping: {reward}')
#
#         tot_reward = extra_reward + reward
#
#         tot_reward /= 1e3  # scaling
#         # print(f'Total Reward: {tot_reward}')
#         # env_logger.info(f'Total Reward: {tot_reward}')
#
#         info = {}
#
#         return self._get_obs(), tot_reward, terminated, False, info
#
#     """
#     # ====================
#     #     Analyze
#     # ====================
#     """
#
#     def get_ana_values(self, ana=None):
#         if ana is None:
#             ana = self.ana
#         assert ana is not None
#         ana_attrs = [k[1:] for k in ana.__dict__.keys() if k.startswith('_')]
#         ana_results = {atr: ana.__dict__['_' + atr] for atr in ana_attrs if atr in self.thresholds.keys()}
#
#         self.ana_value_state = np.array(list(ana_results.values()))
#
#         return np.array(list(ana_results.values()))  # returns array
#
#     def evaluate(self, show=True, **kwargs):  # show not needed but fine
#
#         data = self.state
#         assert len(data.shape) == 1
#         data_ana1d = Ana1D(data=data, data_x=self.data_x, **kwargs)  # sweep_gate_name='any',
#         # except KeyError:
#
#         # ANa: Slope, Prom, Dyn
#         data_ana1d.get_dynamic_range(show=self.show, raw=self.raw, **kwargs)  # show No....?
#         data_ana1d.get_steepest_slope(show=self.show, raw=self.raw, **kwargs)
#         # data_ana1d.get_peaks(show=self.show, **kwargs)
#
#         """ not critical ana results but contribute to the reward
#             By hand!
#         """
#
#         # reward = data_ana1d.steepest_slope + data_ana1d.dynamic_range
#
#         #         non_crit_thresholds = {'gauss_avg_std': 1.0,  'peaks_increasing': 1}
#         #         data_ana1d.gauss_fit(show=self.show, **kwargs)           # gauss_avg_std
#         #         data_ana1d.check_peak_increasing(**kwargs)
#
#         #         if self.ana is not None:
#         #             old_ana_values = self.get_ana_values(self.ana)   # <--- it is before updating to new ana so it is an old values
#         #         else:
#         #             old_ana_values = np.array([0., 0., 0.])
#
#         #         new_ana_values = self.get_ana_values(data_ana1d)
#
#         """ log ana with length of 2 for reward shaping HARDCODED
#             - 1) if better than old value -> give binary reward?
#         """
#         # shaped_reward = sum(np.sign(new_ana_values - old_ana_values))  # give as a reward
#         # # shaped_reward = sum(np.sign(new_ana_values - old_ana_values) == 1)  # give as a reward
#         # # shaped_reward = -sum(np.sign(new_ana_values - old_ana_values) == -1)  # give as a reward
#
#         """ - 2) give reward prop. distance to the threshold as additional shaped rewarddddddd
#                 - it should be normalized i think,..
#                 ex. array([ 0.23512847, -0.47503059,  1.05017752])
#
#                 with weights: I guess the width of the peak is more important than the other two as the width is the hard thing to achieve. --> domain knowledge
#
#         """
#         # # distance_reward = sum(new_ana_values - np.array(list(self.thresholds.values())))
#         # # distance_reward = sum(new_ana_values/np.array(list(self.thresholds.values())) - 1.)   # add distance if lower, minus, else, larger than 0.
#         # distance_reward = new_ana_values/np.array(list(self.thresholds.values())) - 1.   # add distance if lower, minus, else, larger than 0.
#         # distance_reward = sum([d/10 for i, d in enumerate(distance_reward) if i==0 or i==2])
#         # shaped_reward += distance_reward
#
#         #         logger.info(f"[info] Shaped reward: old values = {[round(x, 2) for x in old_ana_values]}, new values = {[round(x, 2) for x in new_ana_values]}. Total reward {shaped_reward} will be added")
#         #         self.shaped_reward = shaped_reward
#
#         #         # non crit ana reward
#         #         non_crit_ana_reward = 0.
#         #         if data_ana1d.gauss_avg_std <= non_crit_thresholds['gauss_avg_std']:
#         #             non_crit_ana_reward += 1.
#         #             logger.info(f"[info] Shaped reward: gauss fitting above threshold: {data_ana1d.gauss_avg_std:3f} >= {non_crit_thresholds['gauss_avg_std']:3f}. Total reward {non_crit_ana_reward} will be added")
#         #         if data_ana1d.peaks_increasing == non_crit_thresholds['peaks_increasing']:
#         #             non_crit_ana_reward += 1.
#         #             logger.info(f"[info] Shaped reward: peak increasing above threshold: {data_ana1d.peaks_increasing} >= {non_crit_thresholds['peaks_increasing']}. Total reward {non_crit_ana_reward} will be added")
#         #         shaped_reward += non_crit_ana_reward
#
#         # TOTAL SHAPED REWARD
#         # self.shaped_reward = shaped_reward
#
#         # update
#         self.ana = data_ana1d  # at the end so that it stores all the results
#
#     #         self.ana_value_state = self.get_ana_values(self.ana)
#
#     #         # Compare
#     #         compa_results = check_thresholds_simple(self.ana, self.thresholds)   # True if passed
#     #         self.compa_results = compa_results
#
#     #         self.ana_state = sum(self.compa_results.values())
#     #         # self.state = np.array(list(self.get_current_gate_voltages(show=False).values()) + [self.ana_state], dtype=np.float32)
#     #         self.state = self._normalize_obs(self.dataxarr.data.flatten())  # self.dataxarr.coords[self.sweep_gate_name].data
#
#     """
#     # ====================
#     #     Obs
#     # ====================
#     """
#
#     def _get_obs(self):  # --> returns state
#         # state = np.array(list(self.get_current_gate_voltages(show=False, return_value=True).values()) + list(self.ana_value_state) + [self.ana_state], dtype=np.float32)
#         # state = np.array(list(self.get_current_gate_voltages(show=False, return_value=True).values()) + list(self.ana_value_state), dtype=np.float32)
#         state = self.state
#         assert state.shape == (self.size,)  # hardcoded
#
#         if not self.observation_space.contains(state):
#             # Adjust the initial observation to ensure it falls within the observation space
#             env_logger.warning(
#                 f"[warning] The observation {state} is not within the observation space {self.observation_space}. We clip it.")
#             state = np.clip(state, self.observation_space.low, self.observation_space.high)
#
#         return state
#
#     def get_observation(self):
#         # Public method to access the observation result
#         return self._get_obs()
#
#     """
#     # ====================
#     #     Render
#     # ====================
#     """
#
#     #     def define_simulator(self):
#     #         # Reset simu
#     #         # simple_sensor_1d = SimpleSensor1d(device_parameter=self.device_parameter, sweep_range=[0.1, 0.5], sweep_gate='LB1')
#     #         # simple_sensor_1d.init_gate_effect()
#
#     #         if self.state is None:
#     #             raise Exception("The observation is empty! Reset first! ")
#     #         self.simulator = self.state
#
#     # FOr visualization
#     def render(self, mode='human'):
#
#         """
#             TODO: make my own visualization live window
#         """
#         if self.state is None:
#             raise Exception("The observation is empty! Reset first! ")
#
#         matplotlib.use("Agg")
#
#         # matplotlib
#         fig = pylab.figure(figsize=[6, 4],  # Inches
#                            dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
#                            )
#         ax = fig.gca()
#         ax.plot(self.data_x, self.state)
#         ax.set_title("1D sensor")
#         ax.set_xlabel(f"x")  # {self.sweep_gate_name}
#         ax.set_ylabel(f"I (arb.)")
#         ax.grid(color='blue', linestyle='-.', linewidth=1, alpha=0.2)
#
#         # canvas, renderer setting
#         canvas = agg.FigureCanvasAgg(fig)
#         canvas.draw()
#         renderer = canvas.get_renderer()
#         raw_data = renderer.tostring_rgb()
#
#         pygame.init()
#         pygame.display.init()
#
#         # Window dimensions
#         width, height = 800, 600
#         window = pygame.display.set_mode((width, height), DOUBLEBUF)
#         screen = pygame.display.get_surface()
#
#         # size = canvas.get_width_height()
#         surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
#
#         WHITE = (255, 255, 255)
#         BLACK = (0, 0, 0)
#         RED = (255, 0, 0)
#         screen.fill(WHITE)
#
#         screen.blit(surf, (0, 0))  # x, y position on screen
#
#         # clock?
#         clock = pygame.time.Clock()
#
#         # crashed = False
#         # while not crashed:
#         # while True:
#         #     for event in pygame.event.get():
#         #         if event.type == pygame.QUIT:
#         #             crashed = True
#
#         clock.tick(1000000)
#         pygame.display.flip()
#
#         # pygame.quit()  # <---- close function is somewhere else
#
#     def close_render(self):
#         pygame.quit()

