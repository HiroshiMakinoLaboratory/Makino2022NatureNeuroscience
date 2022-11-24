"""
Arena environment for continuous state and action space.
"""

import numpy as np
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class object_manipulation_task_task3Env(gym.Env):

    def __init__(self):
        self.reward_zone_size = 0.4
        self.reward_zone_x_size = 0.05
        self.reward_zone_y_size = 0.0031
        self.min_x_pos = -1.0
        self.max_x_pos = 1.0
        self.min_y_pos = -1.0
        self.max_y_pos = 1.0
        self.min_x_speed = -0.1
        self.max_x_speed = 0.1
        self.min_y_speed = -0.1
        self.max_y_speed = 0.1
        self.min_action_x_speed = -0.1
        self.max_action_x_speed = 0.1
        self.min_action_y_speed = -0.1
        self.max_action_y_speed = 0.1
        self.min_action_lick = 0
        self.max_action_lick = 0.1

        self.low_state = np.array([self.min_x_pos, self.min_y_pos, self.min_x_speed, self.min_y_speed])
        self.high_state = np.array([self.max_x_pos, self.max_y_pos, self.max_x_speed, self.max_y_speed])

        self.low_action = np.array([self.min_action_x_speed, self.min_action_y_speed, self.min_action_lick])
        self.high_action = np.array([self.max_action_x_speed, self.max_action_y_speed, self.max_action_lick])

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x_pos = self.state[0]
        y_pos = self.state[1]
        x_speed = self.state[2]
        y_speed = self.state[3]

        action_x_speed = action[0]
        action_y_speed = action[1]
        action_lick = action[2]

        # Once reaching the target, it does not move.
        if x_pos <= self.reward_zone_x_size and x_pos >= -self.reward_zone_x_size and y_pos <= self.min_y_pos + self.reward_zone_y_size and y_pos >= self.min_y_pos:
            x_pos = x_pos
            y_pos = y_pos
        else:
            x_pos += action_x_speed
            y_pos += action_y_speed

        if x_pos <= self.min_x_pos:
          x_pos = self.min_x_pos
        if x_pos >= self.max_x_pos:
          x_pos = self.max_x_pos
        if y_pos <= self.min_y_pos:
          y_pos = self.min_y_pos
        if y_pos >= self.max_y_pos:
          y_pos = self.max_y_pos

        x_speed = action_x_speed
        y_speed = action_y_speed

        # Finish if the agent reaches the target and licks.
        done = bool(x_pos <= self.reward_zone_x_size and x_pos >= -self.reward_zone_x_size and y_pos <= self.min_y_pos + self.reward_zone_y_size and y_pos >= self.min_y_pos and action_lick > 0.08)

        reward = 0
        if done:
          reward = 1.0

        self.state = np.array([x_pos, y_pos, x_speed, y_speed], dtype=np.float32)
        return self.state, reward, done, {}

    def reset(self):
        # Start position.
        x_pos_start = np.random.uniform(low=self.min_x_pos, high=self.max_x_pos)
        y_pos_start = np.random.uniform(low=self.min_y_pos, high=self.max_y_pos)

        # Set the start position to be outside the goal state.
        while x_pos_start <= self.reward_zone_x_size and x_pos_start >= -self.reward_zone_x_size and y_pos_start <= self.min_y_pos + self.reward_zone_y_size and y_pos_start >= self.min_y_pos:
            x_pos_start = np.random.uniform(low=self.min_x_pos, high=self.max_x_pos)
            y_pos_start = np.random.uniform(low=self.min_y_pos, high=self.max_y_pos)

        self.state = np.array([x_pos_start, y_pos_start, 0, 0])
        return self.state
