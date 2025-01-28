import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np

class Maze:

    def __init__(self):

        self.structure = [[1,1,1,1,1,1,1,1,1,1],
                          [1,2,0,0,0,0,1,1,0,1],
                          [1,1,1,1,1,0,0,0,0,1],
                          [1,0,1,0,1,1,1,0,1,1],
                          [1,0,0,0,1,0,1,0,0,1],
                          [1,0,1,1,1,0,1,1,0,1],
                          [1,0,0,0,1,0,0,0,0,1],
                          [1,1,1,0,1,1,0,1,1,1],
                          [1,0,0,0,0,0,0,0,0,1],
                          [1,1,1,1,1,1,1,1,1,1]
                         ]
        
        self.start_r: int = 8 #4 or 8
        self.start_c: int = 8 #4 or 8

        self.current_r: int = self.start_r
        self.current_c: int = self.start_c

        self.max_timesteps: int = 1_000
        self.current_timesteps: int = 0

        self.action_space = spaces.Discrete(4)

        self.type = "float64"


    def step(self, action: int):

        delta_r: int = 0
        delta_c: int = 0

        if action == 0:
            delta_r = -1
        elif action == 1:
            delta_c = 1
        elif action == 2:
            delta_r = 1
        elif action == 3:
            delta_c = -1

        #Determine new position
        if self.structure[self.current_r + delta_r][self.current_c + delta_c] != 1:
            self.current_r += delta_r
            self.current_c += delta_c

        self.current_timesteps += 1

        reward: float = 0

        if self.structure[self.current_r][self.current_c] == 2:
            reward = 1 - (float(self.current_timesteps)/self.max_timesteps)
            reward = 1

        done: bool = (self.current_timesteps == self.max_timesteps) or self.structure[self.current_r][self.current_c] == 2

        obs = self.get_obs()

        return obs, reward, done, None, None


    def reset(self):
        self.current_r = self.start_r
        self.current_c = self.start_c

        self.current_timesteps = 0

        return self.get_obs(), None

    def get_obs(self):

        obs = np.zeros((12), dtype=self.type)

        index = self.structure[self.current_r - 1][self.current_c]
        obs[index] = 1

        index = self.structure[self.current_r][self.current_c + 1]
        obs[3 + index] = 1

        index = self.structure[self.current_r + 1][self.current_c]
        obs[6 + index] = 1

        index = self.structure[self.current_r][self.current_c - 1]
        obs[9 + index] = 1

        return obs