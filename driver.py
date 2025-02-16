import multiprocessing
from agents.train_ppo_agent import train_model
from agents.train_ppo_agent_positive import train_model_positive
import tyro
from args import Args
from models.model_recurrent import GruAgent
import os
import torch
from envs.combogrid import ComboGridEnv
import copy
import numpy as np
import pickle
from utils.selecting_options_rewire import extract_options
import gymnasium as gym
from envs.combogrid_gym import ComboGridGym
import random

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

def make_env( problem, episode_length=None, width=5):
    def thunk():
        env = ComboGridGym(rows=width, columns=width, problem=problem, random_initial=False, episode_length=episode_length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence
    
def generate_trajectories(args, problem):
        env = gym.vector.SyncVectorEnv(
        [make_env(problem, width=args.game_width) for i in range(1)],
        )
        rnn = GruAgent(env,args.hidden_size, option_len=0, greedy=True)
        rnn.load_state_dict(torch.load(f'training_data/models/{args.seed}/{problem}.pt'))
        rnn.eval()
        trajectories = []
        counter = 0
        next_rnn_state = rnn.init_hidden()
        next_done = torch.zeros(1).to(device)
        next_obs, _ = env.reset()
        for i in range(args.game_width):
            for j in range(args.game_width):
                traj = Trajectory()
                env = ComboGridEnv(args.game_width, args.game_width, problem, multiple_initial_states=False)
                env._matrix_unit = np.zeros((args.game_width, args.game_width))
                env._matrix_unit[i][j] = 1
                env._x, env._y = (i, j)
                next_rnn_state = rnn.init_hidden()
                counter = 0
                next_obs = env.get_observation()
                # print(i, j, '\n')
                while not env.is_over() and counter < args.episode_length:
                    next_obs = torch.Tensor(next_obs).to(device)
                    action, logprob, _, value, next_rnn_state = rnn.get_action_and_value(next_obs, next_rnn_state, next_done)
                    # print(action[0])
                    # next_obs, reward, terminations, truncations, infos = env.step(action.cpu().numpy())
                    traj.add_pair(copy.deepcopy(env), int(action[0]))
                    env.apply_action(action.cpu().numpy()[0])
                    next_obs = env.get_observation()
                    # print(env)
                    counter += 1
                if len(traj.get_trajectory()) > 0 and env.is_over():
                    trajectories.append(traj)
        # if not os.path.exists(f'training_data/trajectories/{args.seed}'):
        try:
            os.mkdir(f'training_data/trajectories/{args.seed}')
        except:
            pass
        # print(trajectories)
        with open (f'training_data/trajectories/{args.seed}/{problem}.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

if __name__ =="__main__":

    problems = ["BL-TR", "TR-BL","TL-BR","BR-TL"]
    args = tyro.cli(Args)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
    #     pool.map(train_model, problems)

    #generate trajectories
    # with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
    #     pool.starmap(generate_trajectories, [(args, p) for p in problems])

    # #select options
    # extract_options(args.seed, args.game_width)

    #TODO train test model with and without options
    options = [None, f"training_data/options{args.seed}/selected_options.pkl"]
    with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
        pool.starmap(train_model_positive, [("test", o, args.seed) for o in options])
        pool.starmap(train_model, [("test", o) for o in options])