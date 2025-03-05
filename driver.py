import os
os.environ["OMP_NUM_THREADS"] = "1"
from utils.functions import set_seed
import tyro
from args import Args

#SET SEED
args = tyro.cli(Args)
set_seed(args.seed)

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

if __name__ =="__main__":
    args = tyro.cli(Args)
    if args.env_name == "combogrid":
        model_indices = [0, 1, 2, 3]
    elif args.env_name == "simple-crossing":
        model_indices = [1, 2, 3]
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    
    # with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
    #     pool.starmap(train_model, [(args.env_name, i) for i in model_indices])

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