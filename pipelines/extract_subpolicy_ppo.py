import copy
import itertools
import math
import random
import torch
import time
import tyro
from typing import Union, List
import numpy as np
import gymnasium as gym
from typing import Tuple
from utils import utils
import concurrent.futures
from dataclasses import dataclass
from pipelines.args import Args
from pipelines.losses import LevinLossActorCritic, LogitsLossActorCritic
from agents.policy_guided_agent import PPOAgent
from environemnts.environments_combogrid_gym import ComboGym, make_env
from environemnts.environments_minigrid import make_env_four_rooms
from environemnts.environments_minigrid import get_training_tasks_simplecross, make_env_simple_crossing
from utils.utils import timing_decorator
from utils.utils import timing_decorator, get_ppo_model_file_name
from training.train_ppo_agent import train_ppo
from torch.utils.tensorboard import SummaryWriter

def load_trajectories(problems, args: Args, num_envs=4, seeds=None, verbose=False, logger=None):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    
    for i, (seed, problem) in enumerate(zip(seeds, problems)):
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            model_path = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
            env = get_training_tasks_simplecross(view_size=args.game_width, seed=seed)
        elif args.env_id == "FourRooms":
            model_path = f'binary/four-rooms/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
            raise NotImplementedError("Environment creation not implemented!")
        else:
            model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_MODEL.pt'
            env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size) # TODO: move to cuda
        
        agent.load_state_dict(torch.load(model_path))

        trajectory = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

        if verbose:
            logger.info(f"The trajectory length: {len(trajectory.get_state_sequence())}")

    return trajectories


def save_options(models, args:Args, seeds=None):
    # if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
    #     model_path = f'binary/options/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
    #                     f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
    #     env = get_training_tasks_simplecross(view_size=args.game_width, seed=seed)
    # elif args.env_id == "FourRooms":
    #     model_path = f'binary/four-rooms/PPO-gw{args.game_width}' + \
    #                     f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
    #     raise NotImplementedError("Environment creation not implemented!")
    # else:
    #     model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_MODEL.pt'
    #     env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)
    pass


def evaluate_all_masks_for_ppo_model(masks, selected_models_of_masks, model, problem, trajectories, number_actions, number_iterations, hidden_size):
    """
    Function that evaluates all masks for a given model. It returns the best mask (the one that minimizes the Levin loss)
    for the current set of selected masks. It also returns the Levin loss of the best mask. 
    """
    values = [-1, 0, 1]

    best_mask = None
    best_value = None
    loss = LevinLossActorCritic()

    combinations = itertools.product(values, repeat=hidden_size)

    for value in combinations:
        current_mask = torch.tensor(value, dtype=torch.int8).view(1, -1)
        
        value = loss.compute_loss(masks + [current_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_iterations)

        if best_mask is None or value < best_value:
            best_value = value
            best_mask = copy.deepcopy(current_mask)
            print(best_mask, best_value)
                            
    return best_mask, best_value


@timing_decorator
def evaluate_all_masks_levin_loss():
    """
    This function implements the greedy approach for selecting masks (options) from Alikhasi and Lelis (2024).
    This method evaluates all possible masks of a given model and adds to the pool of options the one that minimizes
    the Levin loss. This process is repeated while we can minimize the Levin loss. 

    This method should only be used with small neural networks, as there are 3^n masks, where n is the number of neurons
    in the hidden layer. 
    """
    hidden_size = 32
    number_iterations = 3
    game_width = 5
    number_actions = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

    params = {
        'hidden_size': hidden_size,
        'number_iterations': number_iterations,
        'game_width': game_width,
        'number_actions': number_actions,
        'problems': problems
    }

    print("Parameters:")
    for key, value in params.items():
        print(f"- {key}: {value}")

    trajectories = load_trajectories(problems, hidden_size, game_width)
    

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic()

    selected_masks = []
    selected_models_of_masks = []
    selected_options_problem = []

    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask = None
        model_best_mask = None
        problem_mask = None

        for problem in problems:
            print('Problem: ', problem)
            model_file = get_ppo_model_file_name(hidden_size=hidden_size, game_width=game_width, problem=problem)
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            agent = PPOAgent(env, hidden_size=hidden_size)
            agent.load_state_dict(torch.load(model_file))

            mask, levin_loss = evaluate_all_masks_for_ppo_model(selected_masks, selected_models_of_masks, agent, problem, trajectories, number_actions, number_iterations, hidden_size)

            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask = mask
                model_best_mask = agent
                problem_mask = problem

                print('Best Loss so far: ', best_loss, problem)

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting automata
        selected_masks.append(best_mask)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, number_actions, number_iterations)

        print("Levin loss of the current set: ", best_loss)

    # remove the last automaton added
    num_options = len(selected_masks)
    selected_masks = selected_masks[0:num_options - 1]
    selected_models_of_masks = selected_models_of_masks[:num_options - 1]

    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, number_iterations)

    # printing selected options
    for i in range(len(selected_masks)):
        print(selected_masks[i])

    print("Testing on each grid cell")
    for problem in problems:
        print("Testing...", problem)
        loss.evaluate_on_each_cell(selected_models_of_masks, selected_masks, problem, game_width)

def hill_climbing_iter(
    i: int, 
    args: Args, 
    values: List, 
    masks: List, 
    selected_models_of_masks: List, 
    model: PPOAgent, 
    problem: str, 
    trajectories: List, 
    number_actions: int, 
    selected_options_n_iterations: List, 
    number_iterations: int, 
    default_loss: float, 
    loss: LevinLossActorCritic
):
    # Initialize the value depending on whether it's the last restart or not
    if i == args.number_restarts:
        value = [-1 for _ in range(args.hidden_size)]
    else:
        value = random.choices(values, k=args.hidden_size)
    
    # Initialize the current mask
    current_mask = torch.tensor(value, dtype=torch.int8).view(1, -1)
    init_mask = current_mask
    applicable = False

    # Compute initial loss
    best_value = loss.compute_loss(
        masks + [current_mask], 
        selected_models_of_masks + [model], 
        problem, 
        trajectories, 
        number_actions, 
        number_steps=selected_options_n_iterations + [number_iterations]
    )

    # Check against default loss
    if best_value < default_loss:
        applicable = True

    n_steps = 0
    while True:
        made_progress = False
        # Iterate through each element of the current mask
        for j in range(len(current_mask[0])):
            modifiable_current_mask = copy.deepcopy(current_mask)
            # Try each possible value for the current position
            for v in values:
                if v == current_mask[0][j]:
                    continue
                
                modifiable_current_mask[0][j] = v
                eval_value = loss.compute_loss(
                    masks + [modifiable_current_mask], 
                    selected_models_of_masks + [model], 
                    problem, 
                    trajectories, 
                    number_actions, 
                    number_steps=selected_options_n_iterations + [number_iterations]
                )

                if eval_value < default_loss:
                    applicable = True

                # Update the best value and mask if improvement is found
                if 'best_mask' not in locals() or eval_value < best_value:
                    best_value = eval_value
                    best_mask = copy.deepcopy(modifiable_current_mask)
                    made_progress = True

            # Update current mask to the best found so far
            current_mask = copy.deepcopy(best_mask)

        # Break the loop if no progress was made in the current iteration
        if not made_progress:
            break

        n_steps += 1

    # Optional logging (uncomment if needed)
    # logger.info(f"#{i}: {n_steps} steps taken. For option length {number_iterations}")

    # Optionally return the best mask and the best value if needed
    return i, best_value, current_mask, init_mask, n_steps, applicable
            

def hill_climbing(
        masks: List, 
        selected_models_of_masks: List, 
        selected_options_n_iterations: List, 
        model: PPOAgent, 
        problem: str, 
        trajectories: dict, 
        loss: LevinLossActorCritic, 
        number_actions: int, 
        number_iterations_ls: List, 
        args: Args, 
        logger):
    """
    Performs Hill Climbing in the mask space for a given agent. Note that when computing the loss of a mask (option), 
    we pass the name of the problem in which the mask is used. That way, we do not evaluate an option on the problem in 
    which the option's model was trained. 

    Larger number of restarts will result in computationally more expensive searches, with the possibility of finding 
    a mask that better optimizes the Levin loss. 
    """
    best_mask = None
    mask_values = [-1, 0, 1]
    best_overall = None
    best_n_iterations = None
    best_value_overall = None
    default_loss = loss.compute_loss(masks, selected_models_of_masks, problem, trajectories, number_actions, number_steps=selected_options_n_iterations)

    
    for number_iterations in number_iterations_ls:
        n_applicable = [0] * (args.number_restarts + 1)

        if args.cpus == 1: 
            for i in range(args.number_restarts + 1):
                _, best_value, best_mask, init_mask, n_steps, applicable = hill_climbing_iter(i, args, mask_values, masks, selected_models_of_masks,
                        model, problem, trajectories, number_actions, selected_options_n_iterations,
                        number_iterations, default_loss, loss)

                logger.info(f'option length {number_iterations}, restart #{i}')
                if applicable:
                    n_applicable[i] = 1
                if i == args.number_restarts:
                    logger.info(f'iteration {i}, Resulting Mask from the original Model: {best_mask}, Loss: {best_value}, using {number_iterations} iterations.')
                if best_overall is None or best_value < best_value_overall:
                    best_overall = copy.deepcopy(best_mask)
                    best_value_overall = best_value
                    best_n_iterations = number_iterations

                    logger.info(f'iteration {i}, Best Mask Overall: {best_overall}, Best Loss: {best_value_overall}, Best number of iterations: {best_n_iterations}')
                    logger.info(f'iteration {i}, {n_steps} steps taken.\n Starting mask: {init_mask}\n Resulted Mask: {best_mask}')
        
        else:
            # Use ProcessPoolExecutor to run the hill climbing iterations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = [
                    executor.submit(
                        hill_climbing_iter, i, args, mask_values, masks, selected_models_of_masks,
                        model, problem, trajectories, number_actions, selected_options_n_iterations,
                        number_iterations, default_loss, loss
                    )
                    for i in range(args.number_restarts + 1)
                ]

                # Process the results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        i, best_value, best_mask, init_mask, n_steps, applicable = future.result()

                        if applicable:
                            n_applicable[i] = 1
                        if i == args.number_restarts:
                            logger.info(f'iteration {i}, Resulting Mask from the original Model: {best_mask}, Loss: {best_value}, using {number_iterations} iterations.')
                        if best_overall is None or best_value < best_value_overall:
                            best_overall = copy.deepcopy(best_mask)
                            best_value_overall = best_value
                            best_n_iterations = number_iterations

                            logger.info(f'iteration {i}, Best Mask Overall: {best_overall}, Best Loss: {best_value_overall}, Best number of iterations: {best_n_iterations}')
                            logger.info(f'iteration {i}, {n_steps} steps taken.\n Starting mask: {init_mask}\n Resulted Mask: {best_mask}')

                    except Exception as exc:
                        logger.error(f'iteration {i} generated an exception: {exc}')

        
        logger.info(f'Out of {args.number_restarts}, {sum(n_applicable)} found applicable options with {number_iterations} iterations.')
    
    return best_overall, best_value_overall, best_n_iterations


@timing_decorator
def hill_climbing_mask_space_training_data():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = tyro.cli(Args)
    args.log_path += "_" + args.exp_name
    
    # Logger configurations
    logger = utils.get_logger('hill_climbing_logger', args.log_level, args.log_path)

    hidden_size = args.hidden_size    
    game_width = args.game_width
    number_restarts = args.number_restarts
    number_actions = 3
    l1_lambda = args.l1_lambda
    
    if isinstance(args.seeds, list) or isinstance(args.seeds, tuple):
        seeds = list(map(int, args.seeds))
    elif isinstance(args.seeds, str):
        start, end = map(int, args.seeds.split(","))
        seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        seeds = seeds * (len(problems)//len(seeds) + 1)
        seeds = seeds[:len(problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        problems = [args.env_id + str(seed) for seed in seeds]

    trajectories = load_trajectories(problems, args, seeds=seeds, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])
    option_length = list(range(2, max_length + 1))

    params = {
        'hidden_size': hidden_size,
        'option_length': option_length,
        'game_width': game_width,
        'number_restarts': number_restarts,
        'number_actions': number_actions,
        'l1_lambda': l1_lambda,
        'problems': problems
    }

    logger.info("Parameters:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

    utils.logger_flush(logger)

    previous_loss = None
    best_loss = None

    loss = LogitsLossActorCritic(logger)
    # loss = LevinLossActorCritic(logger)

    selected_masks = []
    selected_options_n_iterations = []
    selected_models_of_masks = []
    selected_options_problem = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask = None
        best_n_iterations = None
        model_best_mask = None
        problem_mask = None

        for seed, problem in zip(seeds, problems):
            logger.info(f'Problem: {problem} Seeds: {seed}')
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
                model_path = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
            else:
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
                model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_MODEL.pt'
            
            agent = PPOAgent(env, hidden_size=hidden_size)
            agent.load_state_dict(torch.load(model_path))

            mask, levin_loss, n_iterations = hill_climbing(masks=selected_masks, 
                                                           selected_models_of_masks=selected_models_of_masks, 
                                                           selected_options_n_iterations=selected_options_n_iterations, 
                                                           model=agent, 
                                                           problem=problem, 
                                                           trajectories=trajectories, 
                                                           loss=loss,
                                                           number_actions=number_actions, 
                                                           number_iterations_ls=option_length, 
                                                           args=args, 
                                                           logger=logger)

            logger.info(f'Search Summary for {problem},seed={seed}: \nBest Mask:{mask}, levin_loss={levin_loss}, n_iterations={n_iterations}\nPrevious Loss: {best_loss}, Previous selected loss:{previous_loss}, n_selected_masks={len(selected_masks)}')
            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask = mask
                model_best_mask = agent
                best_n_iterations = n_iterations
                problem_mask = problem
            utils.logger_flush(logger)
        logger.debug("\n")

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask)
        selected_options_n_iterations.append(best_n_iterations)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, number_actions, selected_options_n_iterations)

        logger.info(f"Levin loss of the current set: {best_loss}")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_masks)
    selected_masks = selected_masks[0:num_options - 1]
    selected_models_of_masks = selected_models_of_masks[:num_options - 1]
    selected_options_n_iterations = selected_options_n_iterations[:num_options - 1]
    for mask, model, n_iterations in zip(selected_masks, selected_models_of_masks, selected_options_n_iterations):
        model.to_option(mask, n_iterations)

    utils.logger_flush(logger)
    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, selected_options_n_iterations, logger=logger)

    utils.logger_flush(logger)
    # printing selected options
    logger.info("Selected masks:")
    for i in range(len(selected_masks)):
        logger.info(selected_masks[i])

    utils.logger_flush(logger)
    logger.info("Testing on each grid cell")
    for seed, problem in zip(seeds, problems):
        logger.info(f"Testing on each cell..., {problem}")
        loss.evaluate_on_each_cell(selected_models_of_masks, selected_masks, selected_options_problem, problem, args=args, seed=seed, logger=logger)

    utils.logger_flush(logger)
    if isinstance(args.test_seeds, list) or isinstance(args.test_seeds, tuple):
        test_seeds = list(map(int, args.test_seeds))
    elif isinstance(args.test_seeds, str):
        start, end = map(int, args.test_seeds.split(","))
        test_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        test_problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        test_seeds = test_seeds * (len(test_problems)//len(test_seeds) + 1)
        test_seeds = test_seeds[:len(test_problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        test_problems = [args.test_env_id + str(seed) for seed in test_seeds]

    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    for i, (problem, seed) in enumerate(zip(test_problems, test_seeds)):
        logger.info(f"Retraining on {problem}")
        args.learning_rate = lrs[i]
        args.clip_coef = clip_coef[i]
        args.ent_coef = ent_coef[i]
        train_extended_ppo(selected_models_of_masks, problem, seed, args, logger)


@timing_decorator
def hill_climbing_all_segments():
    
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = tyro.cli(Args)
    args.log_path += "_" + args.exp_name
    
    # Logger configurations
    logger = utils.get_logger('hc_all_segments_logger', args.log_level, args.log_path)

    hidden_size = args.hidden_size    
    game_width = args.game_width
    number_restarts = args.number_restarts
    number_actions = 3
    l1_lambda = args.l1_lambda
    
    if isinstance(args.seeds, list) or isinstance(args.seeds, tuple):
        seeds = list(map(int, args.seeds))
    elif isinstance(args.seeds, str):
        start, end = map(int, args.seeds.split(","))
        seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        seeds = seeds * (len(problems)//len(seeds) + 1)
        seeds = seeds[:len(problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        problems = [args.env_id + str(seed) for seed in seeds]

    trajectories = load_trajectories(problems, args, seeds=seeds, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])
    option_length = list(range(2, max_length + 1))

    params = {
        'hidden_size': hidden_size,
        'option_length': option_length,
        'game_width': game_width,
        'number_restarts': number_restarts,
        'number_actions': number_actions,
        'l1_lambda': l1_lambda,
        'problems': problems
    }

    logger.info("Parameters:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

    utils.logger_flush(logger)

    loss = LogitsLossActorCritic(logger)
    # loss = LevinLossActorCritic(logger)

    all_masks_info = []
    all_masks = []
    all_masks_problems = []
    all_masks_n_iterations = []
    all_trajectories = []

    for seed, problem in zip(seeds, problems):
        logger.info(f'Problem: {problem} Seeds: {seed}')
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
            model_path = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                        f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
        else:
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_MODEL.pt'
        
        agent = PPOAgent(env, hidden_size=hidden_size)
        agent.load_state_dict(torch.load(model_path))

        t_length = trajectories[problem].get_length()

        for length in range(2, t_length + 1):
            for s in range(t_length - length):
                logger.info(f"Processing option length {length}, segment {s}..")
                option_length = [length]
                sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
                mask, levin_loss, n_iterations = hill_climbing(masks=[], 
                                                                selected_models_of_masks=[], 
                                                                selected_options_n_iterations=[], 
                                                                model=agent, 
                                                                problem="", 
                                                                trajectories=sub_trajectory, 
                                                                loss=loss,
                                                                number_actions=number_actions, 
                                                                number_iterations_ls=option_length, 
                                                                args=args, 
                                                                logger=logger)
                all_masks_info.append((mask, problem, n_iterations, model_path))
            utils.logger_flush(logger)
        
    logger.debug("\n")

    selected_masks = []
    selected_options_n_iterations = []
    selected_models_of_masks = []
    selected_options_problem = []

    previous_loss = None
    best_loss = None

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask = None
        best_n_iterations = None
        model_best_mask = None
        problem_mask = None

        for mask, problem, n_iterations, model_path in all_masks_info:
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
                model_path = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
            else:
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
                model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_MODEL.pt'
            
            agent = PPOAgent(env, hidden_size=hidden_size)
            agent.load_state_dict(torch.load(model_path))

            levin_loss = loss.compute_loss(masks=selected_masks + [mask], 
                                           models=selected_models_of_masks + [agent], 
                                           problem_str=problem, 
                                           trajectories=trajectories, 
                                           number_actions=number_actions, 
                                           number_steps=selected_options_n_iterations + [n_iterations])


            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask = mask
                model_best_mask = agent
                best_n_iterations = n_iterations
                problem_mask = problem

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask)
        selected_options_n_iterations.append(best_n_iterations)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, number_actions, selected_options_n_iterations)

        logger.info(f"Levin loss of the current selected set: {best_loss} on all trajectories")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_masks)
    selected_masks = selected_masks[0:num_options - 1]
    selected_models_of_masks = selected_models_of_masks[:num_options - 1]
    selected_options_n_iterations = selected_options_n_iterations[:num_options - 1]
    for mask, model, n_iterations in zip(selected_masks, selected_models_of_masks, selected_options_n_iterations):
        model.to_option(mask, n_iterations)

    utils.logger_flush(logger)
    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, selected_options_n_iterations, logger=logger)

    utils.logger_flush(logger)
    # printing selected options
    logger.info("Selected masks:")
    for i in range(len(selected_masks)):
        logger.info(selected_masks[i])

    utils.logger_flush(logger)
    logger.info("Testing on each grid cell")
    for seed, problem in zip(seeds, problems):
        logger.info(f"Testing on each cell..., {problem}")
        loss.evaluate_on_each_cell(selected_models_of_masks, selected_masks, selected_options_problem, problem, args=args, seed=seed, logger=logger)

    utils.logger_flush(logger)
    if isinstance(args.test_seeds, list) or isinstance(args.test_seeds, tuple):
        test_seeds = list(map(int, args.test_seeds))
    elif isinstance(args.test_seeds, str):
        start, end = map(int, args.test_seeds.split(","))
        test_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        test_problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        test_seeds = test_seeds * (len(test_problems)//len(test_seeds) + 1)
        test_seeds = test_seeds[:len(test_problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        test_problems = [args.test_env_id + str(seed) for seed in test_seeds]

    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    for i, (problem, seed) in enumerate(zip(test_problems, test_seeds)):
        logger.info(f"Retraining on {problem}")
        args.learning_rate = lrs[i]
        args.clip_coef = clip_coef[i]
        args.ent_coef = ent_coef[i]
        train_extended_ppo(selected_models_of_masks, problem, seed, args, logger)

def whole_dec_options_training_data_levin_loss():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = tyro.cli(Args)
    args.log_path += "_" + args.exp_name
    
    # Logger configurations
    logger = utils.get_logger('whole_dec_option_logger', args.log_level, args.log_path)

    hidden_size = args.hidden_size
    # option_length = args.option_length
    option_length = list(range(2,5))
    game_width = args.game_width
    number_restarts = args.number_restarts
    number_actions = 3
    l1_lambda = args.l1_lambda
    
    if isinstance(args.seeds, list) or isinstance(args.seeds, tuple):
        seeds = list(map(int, args.seeds))
    elif isinstance(args.seeds, str):
        start, end = map(int, args.seeds.split(","))
        seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
        
    if args.env_id == "ComboGrid":
        problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        seeds = seeds * (len(problems)//len(seeds) + 1)
        seeds = seeds[:len(problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        problems = [args.env_id + str(seed) for seed in seeds]

    params = {
        'hidden_size': hidden_size,
        'option_length': option_length,
        'game_width': game_width,
        'number_restarts': number_restarts,
        'number_actions': number_actions,
        'l1_lambda': l1_lambda,
        'problems': problems
    }

    logger.info("Parameters:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

    trajectories = load_trajectories(problems, args, seeds=seeds, verbose=True, logger=logger)
    utils.logger_flush(logger)

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic()

    selected_masks = []
    selected_options_n_iterations = []
    selected_models_of_masks = []
    selected_options_problem = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask = None
        best_n_iterations = None
        model_best_mask = None
        problem_mask = None

        for seed, problem in zip(seeds, problems):
            logger.info(f'Problem: {problem} Seeds: {seed}')
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
                model_path = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
            else:
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
                model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_MODEL.pt'
            
            agent = PPOAgent(env, hidden_size=hidden_size)
            agent.load_state_dict(torch.load(model_path))

            for i in range(2, 15):
                mask = torch.tensor([-1] * hidden_size).view(1,-1)
                levin_loss = loss.compute_loss(selected_masks + [mask], selected_models_of_masks + [agent], problem, trajectories, number_actions, selected_options_n_iterations + [i])
            
                if best_loss is None or levin_loss < best_loss:
                    best_loss = levin_loss
                    best_mask = mask
                    model_best_mask = agent
                    best_n_iterations = i
                    problem_mask = problem
            utils.logger_flush(logger)
        logger.debug("\n")

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask)
        selected_options_n_iterations.append(best_n_iterations)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, number_actions, selected_options_n_iterations)

        logger.info(f"Levin loss of the current set: {best_loss}")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_masks)
    selected_masks = selected_masks[0:num_options - 1]
    selected_models_of_masks = selected_models_of_masks[:num_options - 1]
    selected_options_n_iterations = selected_options_n_iterations[:num_options - 1]
    for mask, model, n_iterations in zip(selected_masks, selected_models_of_masks, selected_options_n_iterations):
        model.to_option(mask, n_iterations)

    utils.logger_flush(logger)
    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, selected_options_n_iterations, logger=logger)

    utils.logger_flush(logger)
    # printing selected options
    logger.info("Selected masks:")
    for i in range(len(selected_masks)):
        logger.info(selected_masks[i])

    utils.logger_flush(logger)
    logger.info("Testing on each grid cell")
    for seed, problem in zip(seeds, problems):
        logger.info(f"Testing on each cell..., {problem}")
        loss.evaluate_on_each_cell(selected_models_of_masks, selected_masks, selected_options_problem, problem, args=args, seed=seed, logger=logger)

    utils.logger_flush(logger)
    if isinstance(args.test_seeds, list) or isinstance(args.test_seeds, tuple):
        test_seeds = list(map(int, args.test_seeds))
    elif isinstance(args.test_seeds, str):
        start, end = map(int, args.test_seeds.split(","))
        test_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        test_problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        test_seeds = test_seeds * (len(test_problems)//len(test_seeds) + 1)
        test_seeds = test_seeds[:len(test_problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        test_problems = [args.test_env_id + str(seed) for seed in test_seeds]

    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    for i, (problem, seed) in enumerate(zip(test_problems, test_seeds)):
        logger.info(f"Retraining on {problem}")
        args.learning_rate = lrs[i]
        args.clip_coef = clip_coef[i]
        args.ent_coef = ent_coef[i]
        train_extended_ppo(selected_models_of_masks, problem, seed, args, logger)


def train_extended_ppo(options, problem, seed, args: Args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    game_width = args.game_width
    hidden_size = args.hidden_size
    l1_lambda = args.l1_lambda
    number_restarts = args.number_restarts
    number_actions = 3

    if args.test_env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        model_path = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                        f'-h{args.hidden_size}-sd{seed}_retrained_MODEL.pt'
        raise NotImplementedError
    elif args.test_env_id == "MiniGrid-FourRooms-v0":
        model_path = f'binary/four-rooms/PPO-gw{args.game_width}' + \
                        f'-h{args.hidden_size}-sd{seed}_retrained_MODEL.pt'
        envs = gym.vector.SyncVectorEnv(
        [make_env_four_rooms(view_size=game_width, seed=seed + i, options=options) for i in range(args.num_envs)],
    )
    else:
        model_path = f'binary/PPO-{problem}-gw{args.game_width}-h{args.hidden_size}-l1l{args.l1_lambda}_retrained_MODEL.pt'
        raise NotImplementedError
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    exp_name = f"PPO_{problem}-{hidden_size}_{game_width}_{l1_lambda}_{number_restarts}_{number_actions}"
    run_name = f"{exp_name}__{seed}__{int(time.time())}_{args.exp_name}_retrained"
    writer = SummaryWriter(f"outputs/tensorboard/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger.info(f"Constructing tensorboard summary writer on outputs/tensorboard/runs/{run_name}")

    problem_setting = {"problem": problem}
    problem_setting.update({"hidden_size": hidden_size, "game_width": game_width, "l1_lambda":l1_lambda, "number_restarts":number_restarts, "number_actions":number_actions})
    problem_setting.update({f"option{i}":option.mask.tolist() for i, option in enumerate(options)})
    writer.add_text(
        "problem_setting",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in problem_setting.items()])),)

    train_ppo(envs=envs, 
              args=args, 
              model_file_name=model_path, 
              device=device,
              seed=seed,
              writer=writer, 
              logger=logger)


def main():
    # evaluate_all_masks_levin_loss()
    # hill_climbing_mask_space_training_data()
    # whole_dec_options_training_data_levin_loss()
    hill_climbing_all_segments()

if __name__ == "__main__":
    main()