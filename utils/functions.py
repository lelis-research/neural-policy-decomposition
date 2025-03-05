import gymnasium as gym
import torch
import copy
import os
import pickle
import numpy as np
import random
from minigrid.core.world_object import Wall
from envs.combogrid_gym import ComboGridGym
from envs.minigrid_env import MiniGridWrap
from minigrid.envs.crossing import CrossingEnv
from minigrid.wrappers import PositionBonus
from models.model_recurrent import GruAgent
from envs.combogrid import ComboGridEnv
from constants import *
import tyro
from args import Args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence


def set_seed(seed: int):
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility


def get_model_path(env_name, model_index, base_dir=MODEL_DIR):
    return os.path.join(base_dir, f"{env_name}_model_{model_index}.pt")

def get_trajectory_path(env_name, model_index, base_dir=TRAJ_DIR):
    return os.path.join(base_dir, f"{env_name}_traj_{model_index}.pt")

def get_option_path(env_name, model_index, base_dir=OPTION_DIR):
    return os.path.join(base_dir, f"{env_name}_option_{model_index}.pt")


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
        if kwargs['visitation_bonus'] == 1:
            env = PositionBonus(env, scale=0.001)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

    
def generate_trajectories_combogrid(args, model_idx):
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env = gym.vector.SyncVectorEnv(
    [make_combogrid_env(IDX_TO_COMBO[model_idx], width=args.game_width) for i in range(1)],
    )
    rnn = GruAgent(env,args.hidden_size, feature_extractor=True, option_len=0, greedy=True, critic_layer_size=args.critic_layer_size, actor_layer_size=args.actor_layer_size)
    rnn.load_state_dict(torch.load(get_model_path("combogrid", model_idx)))
    rnn.eval()
    trajectories = []
    counter = 0
    next_rnn_state = rnn.init_hidden()
    next_done = torch.zeros(1).to(device)
    next_obs, _ = env.reset()
    for i in range(args.game_width):
        for j in range(args.game_width):
            traj = Trajectory()
            env = ComboGridEnv(args.game_width, args.game_width, IDX_TO_COMBO[model_idx], multiple_initial_states=False, visitation_bonus=args.visitation_bonus)
            env._matrix_unit = np.zeros((args.game_width, args.game_width))
            env._matrix_unit[i][j] = 1
            env._x, env._y = (i, j)
            next_rnn_state = rnn.init_hidden()
            counter = 0
            next_obs = env.get_observation()
            while not env.is_over() and counter < args.episode_length:
                next_obs = torch.Tensor(next_obs).to(device)
                action, logprob, _, value, next_rnn_state = rnn.get_action_and_value(next_obs, next_rnn_state, next_done)
                traj.add_pair(copy.deepcopy(env), int(action[0]))
                env.apply_action(action.cpu().numpy()[0])
                next_obs = env.get_observation()
                counter += 1
            if len(traj.get_trajectory()) > 0 and env.is_over():
                trajectories.append(traj)
    try:
        os.makedirs(TRAJ_DIR)
    except:
        pass

    with open (get_trajectory_path('combogrid', model_idx), 'wb') as f:
        pickle.dump(trajectories, f)



def generate_trajectories_minigrid(args, env, model_idx):
    if env == 'simple-crossing':
        env = gym.vector.SyncVectorEnv(
        [make_env_simple_crossing(view_size=3, seed=args.seed, max_episode_steps=args.episode_length) for i in range(1)],
        )
    rnn = GruAgent(env, args.hidden_size, feature_extractor=True, greedy=True, critic_layer_size=args.critic_layer_size, actor_layer_size=args.actor_layer_size)
    rnn.load_state_dict(torch.load(get_model_path(env, model_idx), weights_only=True))
    rnn.eval()
    trajectories = []
    next_done = torch.zeros(1).to(device)
    traj = Trajectory()
    env = MiniGridWrap(
        CrossingEnv(obstacle_type=Wall, max_steps=args.episode_length),
        seed=args.seed,
        n_discrete_actions=3,
        view_size=3,
        step_reward=-1, 
        options=None)
    next_rnn_state = rnn.init_hidden()
    next_obs, _ = env.reset(seed=args.seed)
    done = False
    counter = 0
    while not done:
        next_obs = torch.Tensor(next_obs).to(device)
        action, logprob, _, value, next_rnn_state = rnn.get_action_and_value(next_obs, next_rnn_state, next_done)
        traj.add_pair(copy.deepcopy(env), int(action[0]))
        next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
        done = terminated or truncated
    if len(traj.get_trajectory()) > 0 and terminated:
        trajectories.append(traj)

    try:
        os.makedirs(TRAJ_DIR)
    except:
        pass

    with open (get_trajectory_path(env, model_idx), 'wb') as f:
        pickle.dump(trajectories, f)


def main():
    args = tyro.cli(Args)
    generate_trajectories_minigrid(args, 'simple-crossing')

if __name__ == "__main__":
    main()