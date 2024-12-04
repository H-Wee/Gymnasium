import gymnasium as gym
from gymnasium import spaces

import numpy as np
from gymnasium import spaces

class PatternTrackingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.window_position = np.array([0.5, 0.5])  # Window position (x, y)
        self.window_size = 0.1  # Window size (radius)
        self.pattern_position = np.random.rand(2)  # Pattern position (x, y)
        self.pattern_velocity = np.random.uniform(-0.05, 0.05, size=2)  # Pattern velocity

        # Observation: [window_position_x, window_position_y, pattern_position_x, pattern_position_y]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Actions: [move_x, move_y]
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(2,), dtype=np.float32)

    def reset(self):
        self.window_position = np.array([0.5, 0.5])
        self.pattern_position = np.random.rand(2)
        self.pattern_velocity = np.random.uniform(-0.05, 0.05, size=2)
        return self._get_observation()

    def step(self, action):
        # Update window position
        self.window_position += action
        self.window_position = np.clip(self.window_position, 0, 1)  # Keep within bounds

        # Update pattern position
        self.pattern_position += self.pattern_velocity
        self.pattern_position = np.clip(self.pattern_position, 0, 1)  # Keep within bounds

        # Calculate reward
        distance = np.linalg.norm(self.pattern_position - self.window_position)
        if distance <= self.window_size:
            reward = 1.0  # Pattern is within the window
        else:
            reward = -1.0  # Pattern is outside the window

        # Check termination
        done = False

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return np.concatenate([self.window_position, self.pattern_position])

    def render(self, mode="human"):
        print(f"Window: {self.window_position}, Pattern: {self.pattern_position}")