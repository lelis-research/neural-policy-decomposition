import gymnasium as gym
from minigrid.core.world_object import Wall
from envs.combogrid_gym import ComboGridGym
from envs.minigrid_env import MiniGridWrap
from minigrid.envs.crossing import CrossingEnv


def make_combogrid_env(problem, episode_length=None, width=3, visitation_bonus=1, options=[]):
    def thunk():
        if len(options) > 0:
            env = ComboGridGym(rows=width, columns=width, problem=problem, random_initial=False, episode_length=episode_length, options=options, visitation_bonus=visitation_bonus)
        else:
            env = ComboGridGym(rows=width, columns=width, problem=problem, random_initial=True, episode_length=episode_length, visitation_bonus=visitation_bonus)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def make_env_simple_crossing(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                CrossingEnv(obstacle_type=Wall, max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps']),
                seed=kwargs['seed'],
                n_discrete_actions=3,
                view_size=kwargs['view_size'],
                step_reward=-1, 
                options=None if 'options' not in kwargs else kwargs['options'])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk