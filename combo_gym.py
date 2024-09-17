import gymnasium as gym
import numpy as np
from combo import Game

class ComboGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR", options=None, partial_observability=True):
        self._game = Game(rows, columns, problem, options, partial_observability)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self._game.get_actions()))
        self.n_steps = 0

    def _get_obs(self):
        return self._game.get_observation()
    
    def reset(self,seed=0, **kwargs):
        self._game.reset()
        self.n_steps = 0
        return self._get_obs(), {}
    
    def step(self, action:int):
        trunctuated = False
        self._game.apply_action(int(action))
        self.n_steps += 1
        terminated = self._game.is_over()
        #reward is 0 in goal state and -1 everywhere else
        reward = -1 
        if terminated:
            reward = 0
        #info about each step, Not being used 
        info = self._game.__repr__()

        #Max steps each episode, will probably remove it
        if self.n_steps == 500:
            trunctuated = True

        
        return self._get_obs(), reward, terminated, trunctuated, {}
    
