import gymnasium as gym
import gymnasium
import copy
import torch
import numpy as np
from minigrid.wrappers import ViewSizeWrapper
from minigrid.core.world_object import Goal


class MiniGridWrap(gym.Env):
    def __init__(self, env, seed=None, n_discrete_actions=3,
                 view_size=5, max_episode_steps=500, step_reward=0, options=None,
                 show_direction=True):
        super(MiniGridWrap, self).__init__()
        # Define action and observation space
        self.seed_ = seed
        self.show_direction = show_direction
        self.step_reward = step_reward
        self.n_discrete_actions = n_discrete_actions
        self.env = ViewSizeWrapper(env, agent_view_size=view_size)
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.reset()
        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        if options:
            self.setup_options(options)
        else:
            self.options = None
        
        shape = (len(self.get_observation()), )

        self.observation_space = gym.spaces.Box(low=0,
                                            high=100,
                                            shape=shape, dtype=np.float64)

        self.spec=self.env.spec

    def setup_options(self, options):
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(options))
        self.options = copy.deepcopy(options)

    def one_hot_encode(self, observation):
        OBJECT_TO_ONEHOT = {
            0: [0,0,0,0],  # Unseen
            1: [1,0,0,0],  # Empty space
            2: [0,1,0,0],  # Wall
            8: [0,0,1,0],  # Goal
            10: [0,0,0,1], # Agent
        }
        one_hot = [OBJECT_TO_ONEHOT[int(x)] for x in observation]
        return np.array(one_hot).flatten()

    def get_observation(self):
        obs = self.env.unwrapped.gen_obs()
        image = self.one_hot_encode(self.env.observation(obs)['image'][:,:,0].flatten())
        if self.show_direction:
            return np.concatenate((
                image,
                [self.env.observation(obs)['direction']],
                [self.agent_pos[0] - self.goal_position[0], self.agent_pos[1] - self.goal_position[1]]
            ))
        return np.concatenate((image, [self.env.observation(obs)['direction']]))

    def take_basic_action(self, action):
        _, reward, terminal, truncated, _ = self.env.step(action)
        self.agent_pos = self.env.unwrapped.agent_pos
        self.steps += 1
        if terminal:
            reward = 1
        if self.steps == 500:
            truncated = True
        if self.steps >= self.max_episode_steps:
            terminal = True
        if terminal:
            self.reset()
        return (terminal, truncated, reward)

    def _dir_to_numeric(self, direction: str):
        return {"R":0, "D":1, "L":2, "U":3}[direction.upper()]

    def step(self, action: int): 
        reward_sum = 0
        if self.options and action >= self.n_discrete_actions:
            option = self.options[action - self.n_discrete_actions]
            for _ in range(option.option_size):
                option_action, _ = option.get_action_with_mask(torch.tensor(self.get_observation(), dtype=torch.float32).view(1, -1))
                terminal, truncated, reward = self.take_basic_action(option_action)
                reward_sum += reward + self.step_reward
                if terminal or truncated:
                    break
            reward = reward_sum
        else:
            terminal, truncated, reward = self.take_basic_action(action)
        return (self.get_observation(), reward + self.step_reward, bool(terminal), bool(truncated), {})

    def reset(self, init_loc=None, init_dir:str=None, seed=None, options=None):
        self.steps = 0
        if seed is not None:
            self.seed_ = seed
        self.env.reset(seed=self.seed_)
        self.goal_position = [
            x for x, y in enumerate(self.env.unwrapped.grid.grid) if isinstance(y, Goal)
        ]
        self.goal_position = (
            int(self.goal_position[0] / self.env.unwrapped.height),
            self.goal_position[0] % self.env.unwrapped.height,
        )
        if init_loc and init_dir:
            self.env.unwrapped.agent_pos = np.array(init_loc)
            self.env.unwrapped.dir_init = self._dir_to_numeric(init_dir)
        self.agent_pos = self.env.unwrapped.agent_pos
        return self.get_observation(), {}

    def is_over(self, loc: tuple=None):
        if loc is None:
            loc = tuple(self.env.unwrapped.agent_pos)
        return loc == self.goal_position

    def render(self):
        return self.env.render()

    def seed(self, seed):
        self.seed_ = seed
        self.env.reset(seed=seed)

    def get_observation_space(self):
        return self.get_observation().size
    
    def get_action_space(self):
        return self.action_space.n

    def represent_options(self, options):
        str_map = ""
        for i in range(self.env.agent_view_size):
            for j in range(self.env.agent_view_size):
                if (i,j) in options:
                    str_map += ",".join([str(action) for action in options[(i,j)]]) + " "
                else:
                    str_map += "-,-,-"
            str_map += "\n"
        return str_map


def get_training_tasks_simplecross(view_size=7, seed=0):
    return MiniGridWrap(
                gymnasium.make("MiniGrid-SimpleCrossingS9N1-v0"),
                seed=seed, max_episode_steps=1000, n_discrete_actions=3,
                view_size=view_size, step_reward=-1)


def get_test_tasks_fourrooms(view_size=7, seed=0):
    return MiniGridWrap(
            gymnasium.make("MiniGrid-FourRooms-v0"),
            max_episode_steps=19*19, n_discrete_actions=3, view_size=view_size, seed=seed,
    )


def get_test_tasks_fourrooms2(view_size=7, seed=0):
    return MiniGridWrap(
            gymnasium.make("MiniGrid-FourRooms-v0"),
            max_episode_steps=19*19, n_discrete_actions=3, view_size=view_size, seed=51,
    )


def get_test_tasks_fourrooms3(view_size=7, seed=0):
    return MiniGridWrap(
            gymnasium.make("MiniGrid-FourRooms-v0"),
            max_episode_steps=19*19, n_discrete_actions=3, view_size=view_size, seed=41,
    )


def make_env_simple_crossing(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                gymnasium.make("MiniGrid-SimpleCrossingS9N1-v0"),
                seed=kwargs['seed'], max_episode_steps=1000, n_discrete_actions=3,
                view_size=kwargs['view_size'], step_reward=-1, 
                options=None if 'options' not in kwargs else kwargs['options'])
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_env_four_rooms(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                gymnasium.make("MiniGrid-FourRooms-v0"),
                seed=kwargs['seed'], max_episode_steps=19*19, n_discrete_actions=3,
                view_size=kwargs['view_size'],
                options=None if 'options' not in kwargs else kwargs['options'])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk