import gymnasium as gym
import numpy as np
from combo import Game
from typing import List, Any
from combo import basic_actions
from gymnasium.envs.registration import register

class ComboGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR"):
        self._game = Game(rows, columns, problem)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self._game.get_actions()))
        self.n_steps = 0
        self.program_stack = None
        self.option_index = None

    def get_observation(self):
        return self._game.get_observation()
    
    def setup_options(self, options:List[Any]=None):
        """
        Enables the corresponding agents to choose from both actions and options
        """
        self.option_index = len(self._game.get_actions())
        self.program_stack = [basic_actions(i) for i in range(self.option_index)] + options
        self.action_space = gym.spaces.Discrete(len(self.program_stack)) 
    
    def reset(self, seed=0, options=None):
        self._game.reset()
        self.n_steps = 0
        return self.get_observation(), {}
    
    def step(self, action:int):
        truncated = False
        def process_action(action: int):
            nonlocal truncated
            self._game.apply_action(action)
            self.n_steps += 1
            terminated = self._game.is_over()
            reward = 0 if terminated else -1 
            if self.n_steps == 500:
                truncated = True
            return self.get_observation(), reward, terminated, truncated, {}
        
        if self.option_index and action >= self.option_index:
            reward_sum = 0
            for _ in range(self.option_sizes[action - self.option_index]):
                action = self.program_stack[action].get_action_with_mask(self.get_observation())
                obs, reward, terminated, truncated, _ = process_action(action)
                reward_sum += reward
                if terminated or truncated:
                    return obs, reward_sum, terminated, truncated, {}
            return obs, reward_sum, terminated, truncated, {}
        else:
            return process_action(action)
    
    def is_over(self, loc=None):
        if loc:
            return loc == self._game.problem.goal
        return self._game.is_over()
        
    def get_observation_space(self):
        return self._rows * self._columns * 2 + 9
    
    def get_action_space(self):
        return 3
    
    def represent_options(self, options):
        return self._game.represent_options(options)
    
register(
     id="ComboGridWorld-v0",
     entry_point=ComboGym
)