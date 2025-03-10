import gymnasium as gym
import numpy as np
from envs.combogrid import ComboGridEnv

class ComboGridGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR", options=[], partial_observability=True, random_initial=False, episode_length=None, visitation_bonus=1):
        self._game = ComboGridEnv(rows, columns, problem, partial_observability, random_initial, visitation_bonus)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self._options = options
        self._visitation_bonus = visitation_bonus
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self._game.get_actions()) + len(options))
        self.n_steps = 0
        self.ep_len = 500
        if episode_length is not None:
            self.ep_len = episode_length
        self.reset()

    def observation(self):
        return self._game.get_observation()
    
    def reset(self,seed=0, **kwargs):
        self._game.reset()
        self.n_steps = 0
        self.goals_reached = 0
        self.truncated = False
        self.terminated = False
        return self.observation(), {}
    
    def _calculate_reward(self, action):
        reward = 0
        reach_goal = self._game.apply_action(action)
        self.n_steps += 1

        if self._visitation_bonus:
            if not reach_goal:
                reward += -1 + self._game.get_exploration_bonus()
            else:
                reward += 1 + self._game.get_exploration_bonus()
                self.goals_reached += 1
        else:
            if not reach_goal:
                reward += -1
            else:
                reward += 1
                self.goals_reached += 1

        if self._game.is_over(): 
            self.terminated = True
            return reward
        if self.n_steps == self.ep_len:
            self.truncated = True
            return reward
        return reward
    
    def step(self, action:int):
        self.truncated = False
        action = int(action)
        reward = 0
        #reward is 0 in goal state and -1 everywhere else
        #add exploration bonus to reward to encourage agent to visit less visited states
        if action > 2:
            # is_applicable, actions = self._options[action - len(self._game.get_actions())].transition(self._game, apply_actions=False)
            is_applicable = True
            if action == 3:
                actions = [0, 0, 1] #UP
            elif action == 4:
                actions = [0, 1, 2] #DOWN
            elif action == 5:
                actions = [2, 1, 0] #LEFT
            elif action == 6:
                actions = [1, 0, 2] #RIGHT
            if is_applicable:
                for a in actions:
                    reward += self._calculate_reward(a)
                    if self.truncated or self.terminated:
                        break
        else:
            reward += self._calculate_reward(action)

        #info about each step, Not being used 
        info = self._game.__repr__()

        #Max steps each episode, will probably remove it
        if self.n_steps == self.ep_len:
            self.truncated = True

        
        return self.observation(), reward, self.terminated, self.truncated, {"l":self.n_steps, "r":reward, "g":self.goals_reached}
    
