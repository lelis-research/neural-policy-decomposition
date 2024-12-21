import multiprocessing
from agent_recurrent import train_model
import tyro
from args import Args
from model_recurrent import GruAgent
import os
import torch
from combo import Game
import copy
import numpy as np
import pickle
from selecting_options_rewire import extract_options

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence
    
def generate_trajectories(args, problem):
        rnn = GruAgent(env,args.hidden_size, option_len=0, greedy=True)
        rnn.load_state_dict(torch.load(f'models/{args.seed}/{problem}.pt'))
        rnn.eval()
        trajectories = []
        counter = 0
        next_rnn_state = rnn.init_hidden()
        next_done = torch.zeros(1).to(device)
        next_obs, _ = env.reset()
        for i in range(5):
            for j in range(5):
                traj = Trajectory()
                env = Game(5, 5, problem, multiple_initial_states=False)
                env._matrix_unit = np.zeros((5, 5))
                env._matrix_unit[i][j] = 1
                env._x, env._y = (i, j)
                next_rnn_state = rnn.init_hidden()
                counter = 0
                next_obs = env.get_observation()
                # print(i, j, '\n')
                while not env.is_over() and counter<30:
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
        os.mkdir(f'trajectories/{args.seed}')
        with open (f'trajectories/{args.seed}/{problem}.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

if __name__ =="__main__":

    problems = ["BL-TR", "TR-BL","TL-BR","BR-TL"]
    args = tyro.cli(Args)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))


    with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
        pool.map(train_model, problems)

    #generate trajectories
    with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
        pool.starmap(generate_trajectories, [(args, p) for p in problems])

    #select options
    extract_options(args.seed)

    #TODO train test model with and without options
    options = [None, f"options/{args.seed}/selected_options.pkl"]
    with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
        pool.starmap(train_model, [("test", o) for o in options])