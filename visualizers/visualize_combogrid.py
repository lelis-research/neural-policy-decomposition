import tyro
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper
from utils.utils import get_logger
from agents.trajectory import Trajectory
from environments.environments_combogrid_gym import ComboGym
from agents.policy_guided_agent import PPOAgent
from pipelines.option_discovery import Args, process_args, regenerate_trajectories

def visualize(env: ComboGym, verbose=True):
    print(env._game)

def run(agent, env: ComboGym, length_cap=None, verbose=False):

        trajectory = Trajectory()
        current_length = 0
        agent.actor.requires_grad = False

        o, _ = env.reset()
        
        done = False

        if verbose: print('Beginning Trajectory')
        while not done:
            o = torch.tensor(o, dtype=torch.float32)
            
            a, _, _, _, logits = agent.get_action_and_value(o)
            trajectory.add_pair(copy.deepcopy(env), a.item(), logits)

            if verbose:
                print("--------------------------------")
                print("processed obs:", o)
                print("logits:", logits)
                print(env, a)
                print()
                visualize(env, verbose)

            next_o, _, terminal, truncated, _ = env.step(a.item())
            
            current_length += 1
            if (length_cap is not None and current_length > length_cap) or \
                terminal or truncated:
                done = True     

            o = next_o   
        
        agent._h = None
        if verbose: print("End Trajectory \n\n")
        print("trajectory length:", trajectory.get_length())
        return trajectory


def visualize_on_four_rooms(args):
    trajectories = {}
    verbose = True
    for problem, model_directory in zip(args.problems, args.model_paths):

        model_path = f'binary/models/{model_directory}/ppo_first_MODEL.pt'

        for other_seeds in args.seeds: 
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=args.game_width, seed=other_seeds)
            elif args.env_id == "MiniGrid-FourRooms-v0":
                env = get_test_tasks_fourrooms(view_size=args.game_width, seed=other_seeds)
            import minigrid.core.constants

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            
            agent.load_state_dict(torch.load(model_path))

            trajectory = run(agent, env, verbose=verbose)
            trajectories[problem] = trajectory


def visualize_trained_agents(args):
    trajectories = {}
    verbose = False
    for seed, problem, model_directory in zip(args.seeds, args.problems, args.model_paths):
        print(problem, model_directory)
        model_path = f'binary/models/{model_directory}/ppo_first_MODEL.pt'
        
        env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)

        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory = run(agent, env, verbose=verbose)
        trajectories[problem] = trajectory


def try_on_other_environments(args):
    verbose = False
    impossibles = set([
        (1,0), (2,0), (0,8), (1,8)
    ])
    other_seeds = (3, 4, 5, 6, 7, 8, 9)
    for seed, problem, model_directory in zip(args.seeds, args.problems, args.model_paths):

        model_path = f'binary/models/{model_directory}/ppo_first_MODEL.pt'
        for other_seed in other_seeds:
            if seed == other_seed:
                continue
            if (seed, other_seed) in impossibles:
                continue
            print(f"Try seed={seed} on other_seed={other_seed}")
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=args.game_width, seed=other_seed)
            elif args.env_id == "MiniGrid-FourRooms-v0":
                raise NotImplementedError("Environment creation not implemented!")
            import minigrid.core.constants

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            
            agent.load_state_dict(torch.load(model_path))

            trajectory = run(agent, env, verbose=verbose)


def visualize_envs(args):
    for seed in args.seeds:
        print(f"seed = {seed}")
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(view_size=args.game_width, seed=seed)
        elif args.env_id == "MiniGrid-FourRooms-v0":
            env = get_test_tasks_fourrooms(view_size=args.game_width, seed=seed)
        visualize(env, verbose=False)


def main(args):
    # visualize_trained_agents(args)
    args.seeds = (0, 1, 2, 3)
    # args.seeds = (41, 51, 8)
    # args.env_id = "MiniGrid-FourRooms-v0"
    args.env_id = "ComboGrid"
    args.problems = (
        "TL-BR", 
                     "TR-BL", 
                     "BR-TL", 
                     "BL-TR"
                     )
    args.model_paths = [
        'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd0_TL-BR',
        'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd1_TR-BL',
        'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd2_BR-TL',
        'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd3_BL-TR'
    ]
    args.hidden_size = 64
    visualize_trained_agents(args)
    # try_on_other_environments(args)
    # visualize_envs(args)
    # visualize_on_four_rooms(args)


if __name__ == '__main__':
    args = process_args()
    main(args)