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
from dataclasses import dataclass
from args import Args
from losses import LevinLossActorCritic
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


def hill_climbing(masks, selected_models_of_masks, selected_options_n_iterations, model, problem, trajectories, number_actions, number_iterations_ls, number_restarts, hidden_size, logger):
    """
    Performs Hill Climbing in the mask space for a given agent. Note that when computing the loss of a mask (option), 
    we pass the name of the problem in which the mask is used. That way, we do not evaluate an option on the problem in 
    which the option's model was trained. 

    Larger number of restarts will result in computationally more expensive searches, with the possibility of finding 
    a mask that better optimizes the Levin loss. 
    """
    best_mask = None
    values = [-1, 0, 1]
    best_overall = None
    best_n_iterations = None
    best_value_overall = None
    loss = LevinLossActorCritic()

    for number_iterations in number_iterations_ls:
        for i in range(number_restarts + 1):
            # if i == number_restarts:
            #     value = list(range(hidden_size))
            #     logger.info("Starting the search with the original model")        
            value = random.choices(values, k=hidden_size)
            current_mask = torch.tensor(value, dtype=torch.int8).view(1, -1)
            init_mask = current_mask

            best_value = loss.compute_loss(masks + [current_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_steps=selected_options_n_iterations + [number_iterations])

            n_steps = 0
            while True:
                made_progress = False
                for j in range(len(current_mask)):
                    modifiable_current_mask = copy.deepcopy(current_mask)
                    for v in values:
                        if v == current_mask[0][j]:
                            continue
                        modifiable_current_mask[0][j] = v
                        eval_value = loss.compute_loss(masks + [modifiable_current_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_steps=selected_options_n_iterations + [number_iterations])

                        if best_mask is None or eval_value < best_value:
                            best_value = eval_value
                            best_mask = copy.deepcopy(modifiable_current_mask)

                            made_progress = True

                    current_mask = copy.deepcopy(best_mask)

                if not made_progress:
                    break
                n_steps += 1
            # logger.info(f"#{i}: {n_steps} steps taken. For option length {number_iterations}")

            if i == number_restarts:
                logger.info(f'Resulting Mask from the original Model: {best_mask}, Loss: {best_value}, using {number_iterations} iterations.')
            if best_overall is None or best_value < best_value_overall:
                best_overall = copy.deepcopy(best_mask)
                best_value_overall = best_value
                best_n_iterations = number_iterations

                logger.info(f'Best Mask Overall: {best_overall}, Best Loss: {best_value_overall}, Best number of iterations: {best_n_iterations}')
                logger.info(f'{n_steps} steps taken.\n Starting mask: {init_mask}\n Resulted Mask: {best_mask}')
    return best_overall, best_value_overall, best_n_iterations


@timing_decorator
def hill_climbing_mask_space_training_data_levin_loss():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = tyro.cli(Args)
    
    # Logger configurations
    logger = utils.get_logger('hill_climbing_logger', args.log_level, args.log_path)

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

            mask, levin_loss, n_iterations = hill_climbing(selected_masks, selected_models_of_masks, selected_options_n_iterations, agent, problem, trajectories, number_actions, option_length, number_restarts, hidden_size, logger=logger)

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


def whole_dec_options_training_data_levin_loss():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = tyro.cli(Args)
    
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
    run_name = f"{exp_name}__{seed}__{int(time.time())}_retrained"
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
    # hill_climbing_mask_space_training_data_levin_loss()
    whole_dec_options_training_data_levin_loss()

if __name__ == "__main__":
    main()