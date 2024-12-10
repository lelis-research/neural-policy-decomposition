import gymnasium as gym
import numpy as np
from combo import Game

class ComboGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR", options=None, partial_observability=True, random_initial=False, episode_length=None, visitation_bonus=True):
        self._game = Game(rows, columns, problem, partial_observability, random_initial, visitation_bonus)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self._options = options
        self._visitation_bonus = visitation_bonus
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self._game.get_actions()))
        self.n_steps = 0
        self.ep_len = 500
        if episode_length is not None:
            self.ep_len = episode_length

    def _get_obs(self):
        return self._game.get_observation()
    
    def reset(self,seed=0, **kwargs):
        self._game.reset()
        self.n_steps = 0
        return self._get_obs(), {}
    
    def step(self, action:int):
        trunctuated = False
        action = int(action)
        if action > 2:
            is_applicable, actions = self._options[action - len(self._game.get_actions())].transition(self._game, apply_actions=True)
        else:
            self._game.apply_action(action)
        self.n_steps += 1
        terminated = self._game.is_over()
        #reward is 0 in goal state and -1 everywhere else
        reward = -1
        #add exploration bonus to reward to encourage agent to visit less visited states
        if self._visitation_bonus:
            reward += self._game.get_exploration_bonus()
        if terminated:
            reward = 0
        #info about each step, Not being used 
        info = self._game.__repr__()

        #Max steps each episode, will probably remove it
        if self.n_steps == self.ep_len:
            trunctuated = True

        
        return self._get_obs(), reward, terminated, trunctuated, {}
    
