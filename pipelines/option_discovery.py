import copy
import os
import random
import torch
import tyro
import pickle
import glob
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import List
from utils import utils
from dataclasses import dataclass
import itertools
import logging
from typing import Union, List, Tuple
import concurrent.futures
from pipelines.losses import LevinLossActorCritic, LogitsLossActorCritic
from agents.policy_guided_agent import PPOAgent
from environments.environments_combogrid_gym import ComboGym
from environments.environments_combogrid import SEEDS, PROBLEM_NAMES as COMBO_PROBLEM_NAMES
from environments.environments_minigrid import get_training_tasks_simplecross
from utils.utils import timing_decorator
from utils.utils import timing_decorator


@dataclass
class Args:
    exp_name: str = "extract_learnOption"
    # exp_name: str = "debug"
    # exp_name: str = "extract_decOptionWhole_sparseInit"
    # exp_name: str = "extract_learnOptions_randomInit_discreteMasks"
    # exp_name: str = "extract_learnOptions_randomInit_pitisFunction"
    """the name of this experiment"""
    env_seeds: Union[List, str, Tuple] = (0,1,2,3)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    # model_paths: List[str] = (
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    # model_paths: List[str] = (
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    # model_paths: List[str] = (
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2',
    #     )
    model_paths: List[str] = (
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
    )

    # These attributes will be filled in the runtime
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    problems: List[str] = ()
    """the name of the problems the agents were trained on; To be filled in runtime"""

    # Algorithm specific arguments
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameters
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""

    # hill climbing arguments
    number_restarts: int = 400
    """number of hill climbing restarts for finding one option"""

    # mask learning
    mask_learning_rate: float = 0.001
    """"""
    mask_learning_steps: int = 2_000
    """"""
    max_grad_norm: float = 1.0
    """"""
    input_update_frequency: int = 1
    """"""
    mask_type: str = "input"
    """It's one of these: [internal, input, both]"""
    mask_transform_type: str = "softmax"
    """It's either `softmax` or `quantize`"""
    selection_type: str = "local_search"

    # Script arguments
    seed: int = 0
    """The seed used for reproducibilty of the script"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "BASELINE0_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""


def get_single_environment(args: Args, seed):
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        env = get_training_tasks_simplecross(view_size=args.game_width, seed=seed)
    elif args.env_id == "ComboGrid":
        problem = COMBO_PROBLEM_NAMES[seed]
        env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)
    else:
        raise NotImplementedError
    return env


def process_args() -> Args:
    args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_l1{args.l1_lambda}' + \
        f'_r{args.number_restarts}_envsd{",".join(map(str, args.env_seeds))}'
        if 'mask_type' in vars(args):
            args.exp_id += f'_mskType{args.mask_type}'
        if 'mask_transform_type' in vars(args):
            args.exp_id += f'_mskTransform{args.mask_transform_type}'
        if 'selection_type' in vars(args):
            args.exp_id += f'_selectType{args.selection_type}'

    # updating log path
    args.log_path = os.path.join(args.log_path, args.exp_id, f"seed={str(args.seed)}")
    
    # Processing seeds from commands
    if isinstance(args.env_seeds, list) or isinstance(args.env_seeds, tuple):
        args.env_seeds = list(map(int, args.env_seeds))
    elif isinstance(args.env_seeds, str):
        start, end = map(int, args.env_seeds.split(","))
        args.env_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        args.problems = [COMBO_PROBLEM_NAMES[seed] for seed in args.env_seeds]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        args.problems = [args.env_id + f"_{seed}" for seed in args.env_seeds]
        
    return args


def regenerate_trajectories(args: Args, verbose=False, logger=None):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        env = get_single_environment(args, seed=seed)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

        if verbose:
            logger.info(f"The trajectory length: {len(trajectory.get_state_sequence())}")

    return trajectories


def save_options(options: List[PPOAgent], trajectories: dict, args: Args, logger):
    """
    Save the options (masks, models, and number of iterations) to the specified directory.

    Parameters:
        options (List[PPOAgent]): The models corresponding to the masks.
        trajectories (Dict[str, Trajectory]): The trajectories corresponding to the these options
        save_dir (str): The directory where the options will be saved.
    """
    save_dir = f"binary/options/{args.exp_id}/seed={args.seed}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Removing every file or directory in the path 
        for file_path in glob.glob(os.path.join(save_dir, "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)

    trajectories_path = os.path.join(save_dir, f'trajectories.pickle')
    with open(trajectories_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    logger.info(f"Trajectories saved to {trajectories_path}")
    
    # Save each model with its mask and iteration count
    for i, model in enumerate(options):
        
        model_path = os.path.join(save_dir, f'ppo_model_option_{i}.pth')
        torch.save({
            'id': i,
            'model_state_dict': model.state_dict(),
            'mask': model.mask,
            'n_iterations': model.option_size,
            'problem': model.problem_id,
            'environment_args': vars(args),
            'extra_info': model.extra_info
        }, model_path)
    
    logger.info(f"Options saved to {save_dir}")


def load_options(args, logger):
    """
    Load the saved options (masks, models, and number of iterations) from the specified directory.

    Parameters:
        save_dir (str): The directory where the options, and trajectories are saved.

    Returns:
        options (List[PPOAgent]): Loaded models.
        loaded_trajectories (List[Trajectory]): Loaded trajectories.
    """

    # Load the models and iterations
    save_dir = f"binary/options/{args.exp_id}/seed={args.seed}"

    logger.info(f"Option directory: {save_dir}")

    model_files = sorted([f for f in os.listdir(save_dir) if f.startswith('ppo_model_option_') and f.endswith('.pth')])
    
    logger.info(f"Found options: {model_files}")

    n = len(model_files)
    options = [None] * n

    print(model_files)

    for model_file in model_files:
        model_path = os.path.join(save_dir, model_file)
        checkpoint = torch.load(model_path)
        
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            if 'environment_args' in checkpoint:
                seed = int(checkpoint['environment_args']['seed'])
                game_width = int(checkpoint['environment_args']['game_width'])
            else:
                seed = int(checkpoint['problem'][-1])
                game_width = args.game_width
            envs = get_training_tasks_simplecross(view_size=game_width, seed=seed)
        elif args.env_id == "MiniGrid-FourRooms-v0":
            raise NotImplementedError("Environment creation not implemented!")
        elif args.env_id == "ComboGrid":
            game_width = int(checkpoint['environment_args']['game_width'])
            problem = checkpoint['problem']
            envs = ComboGym(rows=game_width, columns=game_width, problem=problem)
        else:
            raise NotImplementedError

        model = PPOAgent(envs=envs, hidden_size=args.hidden_size)  # Create a new PPOAgent instance with default parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to_option(checkpoint['mask'], checkpoint['n_iterations'], checkpoint['problem'])
        model.extra_info = checkpoint['extra_info'] if 'extra_info' in checkpoint else {}
        model.environment_args = checkpoint['environment_args'] if 'environment_args' in checkpoint else {}

        i = checkpoint['id']
        options[i] = model
        
    trajectories_path = os.path.join(save_dir, f'trajectories.pickle')

    with open(trajectories_path, 'rb') as f:
        loaded_trajectory = pickle.load(f)

    return options, loaded_trajectory


def hill_climbing_iter(
    i: int, 
    agent: PPOAgent,
    option_size: int, 
    problem_str: str,
    number_actions: int, 
    mask_values: List, 
    trajectories: dict, 
    selected_masks: List, 
    selected_mask_models: List, 
    selected_option_sizes: List, 
    initial_loss: float, 
    args: Args, 
    loss: LevinLossActorCritic
):
    # Initialize the value depending on whether it's the last restart or not
    if i == args.number_restarts:
        mask_seq = [-1 for _ in range(args.hidden_size)]
    else:
        mask_seq = random.choices(mask_values, k=args.hidden_size)
    
    # Initialize the current mask
    current_mask = torch.tensor(mask_seq, dtype=torch.int8).view(1, -1)
    init_mask = current_mask
    applicable = False

    # Compute initial loss
    best_value = loss.compute_loss(
        selected_masks + [current_mask], 
        selected_mask_models + [agent], 
        problem_str, 
        trajectories, 
        number_actions, 
        number_steps=selected_option_sizes + [option_size]
    )

    # Check against default loss
    if best_value < initial_loss:
        applicable = True

    n_steps = 0
    while True:
        made_progress = False
        # Iterate through each element of the current mask
        for j in range(len(current_mask[0])):
            modifiable_current_mask = copy.deepcopy(current_mask)
            # Try each possible value for the current position
            for v in mask_values:
                if v == current_mask[0][j]:
                    continue
                
                modifiable_current_mask[0][j] = v
                eval_value = loss.compute_loss(
                    selected_masks + [modifiable_current_mask], 
                    selected_mask_models + [agent], 
                    problem_str, 
                    trajectories, 
                    number_actions, 
                    number_steps=selected_option_sizes + [option_size]
                )

                if eval_value < initial_loss:
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

    # Optionally return the best mask and the best value if needed
    return i, best_value, current_mask, init_mask, n_steps, applicable
            

def hill_climbing(
        agent: PPOAgent, 
        problem_str: str,
        number_actions: int, 
        trajectories: dict, 
        selected_masks: List, 
        selected_masks_models: List, 
        selected_option_sizes: List, 
        possible_option_sizes: List, 
        loss: LevinLossActorCritic, 
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
    best_option_sizes = None
    best_value_overall = None
    
    def _update_best(i, best_value, best_mask, n_steps, init_mask, applicable, n_applicable, option_size):
        nonlocal best_overall, best_value_overall, best_option_sizes
        if applicable:
            n_applicable[i] = 1
        if i == args.number_restarts:
            logger.info(f'restart #{i}, Resulting Mask from the original Model: {best_mask}, Loss: {best_value}, using {option_size} iterations.')
        if best_overall is None or best_value < best_value_overall:
            best_overall = copy.deepcopy(best_mask)
            best_value_overall = best_value
            best_option_sizes = option_size

            logger.info(f'restart #{i}, Best Mask Overall: {best_overall}, Best Loss: {best_value_overall}, Best number of iterations: {best_option_sizes}')
            logger.info(f'restart #{i}, {n_steps} steps taken.\n Starting mask: {init_mask}\n Resulted Mask: {best_mask}')

    default_loss = loss.compute_loss(selected_masks, selected_masks_models, problem_str, trajectories, number_actions, number_steps=selected_option_sizes)
    
    for option_size in possible_option_sizes:
        logger.info(f'Selecting option #{len(selected_masks_models)} - option size {option_size}')
        n_applicable = [0] * (args.number_restarts + 1)

        if args.cpus == 1: 
            for i in range(args.number_restarts + 1):
                _, best_value, best_mask, init_mask, n_steps, applicable = hill_climbing_iter(i=i, 
                                                    agent=agent,
                                                    problem_str=problem_str,
                                                    option_size=option_size,
                                                    number_actions=number_actions,
                                                    mask_values=mask_values,
                                                    trajectories=trajectories,
                                                    selected_masks=selected_masks,
                                                    selected_mask_models=selected_masks_models,
                                                    selected_option_sizes=selected_option_sizes,
                                                    initial_loss=default_loss,
                                                    args=args,
                                                    loss=loss)
                _update_best(i=i, 
                             best_value=best_value, 
                             best_mask=best_mask,
                             n_steps=n_steps,
                             init_mask=init_mask,
                             applicable=applicable,
                             n_applicable=n_applicable,
                             option_size=option_size)
                
                if i % 100 == 0:
                    logger.info(f'Progress: {i}/{args.number_restarts}')
        else:
            # Use ProcessPoolExecutor to run the hill climbing iterations in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = [
                    executor.submit(
                        hill_climbing_iter, i, agent, option_size, problem_str, number_actions, 
                        mask_values, trajectories, selected_masks, selected_masks_models, 
                        selected_option_sizes, default_loss, args, loss
                    )
                    for i in range(args.number_restarts + 1)
                ]

                # Process the results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        i, best_value, best_mask, init_mask, n_steps, applicable = future.result()
                        _update_best(i=i, 
                                    best_value=best_value, 
                                    best_mask=best_mask,
                                    n_steps=n_steps, 
                                    init_mask=init_mask, 
                                    applicable=applicable,
                                    n_applicable=n_applicable, 
                                    option_size=option_size)
                        if i % 100 == 0:
                            logger.info(f'Progress: {i}/{args.number_restarts}')
                    except Exception as exc:
                        logger.error(f'restart #{i} generated an exception: {exc}')

        logger.info(f'Out of {args.number_restarts}, {sum(n_applicable)} options where applicable with size={option_size} .')
    
    return best_overall, best_value_overall, best_option_sizes


@timing_decorator
def hill_climbing_mask_space_training_data():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = process_args()
    
    # Logger configurations
    logger = utils.get_logger('hill_climbing_logger', args.log_level, args.log_path)

    game_width = args.game_width
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])
    option_length = list(range(2, max_length + 1))
    # option_length = list(range(2, 5))
    args.exp_id += f'_olen{",".join(map(str, option_length))}'

    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)

    previous_loss = None
    best_loss = None

    # loss = LogitsLossActorCritic(logger)
    loss = LevinLossActorCritic(logger)

    selected_masks = []
    selected_mask_models = []
    selected_option_sizes = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask_model = None

        for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
            model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
            logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
            env = get_single_environment(args, seed=seed)

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            mask, levin_loss, option_size = hill_climbing(agent=agent, 
                                                problem_str=problem, 
                                                number_actions=number_actions, 
                                                trajectories=trajectories, 
                                                selected_masks=selected_masks, 
                                                selected_masks_models=selected_mask_models, 
                                                selected_option_sizes=selected_option_sizes, 
                                                possible_option_sizes=option_length, 
                                                loss=loss, 
                                                args=args, 
                                                logger=logger)

            logger.info(f'Search Summary for {problem}, seed={seed}: \nBest Mask:{mask}, levin_loss={levin_loss}, n_iterations={option_size}\nPrevious Loss: {best_loss}, Previous selected loss:{previous_loss}, n_selected_masks={len(selected_masks)}')
            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask_model = agent
                agent.to_option(mask, option_size, problem)
                if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                    agent.environment_args = {
                        "seed": seed,
                        "game_width": game_width
                    }
                else:
                    agent.environment_args = {
                        "game_width": game_width
                    }
            utils.logger_flush(logger)
        logger.debug("\n")

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask_model.mask)
        selected_mask_models.append(best_mask_model)
        selected_option_sizes.append(best_mask_model.option_size)
        best_loss = loss.compute_loss(selected_masks, selected_mask_models, "", trajectories, number_actions, selected_option_sizes)

        logger.info(f"Levin loss of the current set: {best_loss}")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_mask_models)
    selected_mask_models = selected_mask_models[:num_options - 1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_mask_models)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_mask_models[i].mask}\n" +
                    f"size={selected_mask_models[i].option_size}\n" +
                    f"problem={selected_mask_models[i].problem_id}")

    save_options(options=selected_mask_models, 
                 trajectories=trajectories,
                 args=args, 
                 logger=logger)
    
    utils.logger_flush(logger)


@timing_decorator
def hill_climbing_all_segments(args: Args, logger: logging.Logger):
    
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """

    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)

    logits_loss = LogitsLossActorCritic(logger)
    levin_loss = LevinLossActorCritic(logger)

    all_masks_info = []

    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
        env = get_single_environment(args, seed=seed)
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        agent.load_state_dict(torch.load(model_path))

        t_length = trajectories[problem].get_length()

        for length in range(2, t_length + 1):
            for s in range(t_length - length):
                logger.info(f"Processing option length={length}, segment={s}..")
                option_length = [length]
                sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
                mask, loss_value, option_size = hill_climbing(agent=agent, 
                                                problem_str="", 
                                                number_actions=number_actions, 
                                                trajectories=sub_trajectory, 
                                                selected_masks=[], 
                                                selected_masks_models=[], 
                                                selected_option_sizes=[], 
                                                possible_option_sizes=option_length, 
                                                loss=logits_loss, 
                                                args=args, 
                                                logger=logger)
                all_masks_info.append((mask, problem, option_size, model_path, s))
            utils.logger_flush(logger)
        
    logger.debug("\n")

    selected_masks = []
    selected_option_sizes = []
    selected_mask_models = []
    selected_options_problem = []

    previous_loss = None
    best_loss = None

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask_model = None

        for mask, problem, option_size, model_path, segment in all_masks_info:
            logger.info(f'Extracting from the agent trained on problem={problem}, seed={seed}, segment=({segment},{segment+option_size})')
            env = get_single_environment(args, seed=seed)
            
            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            loss_value = levin_loss.compute_loss(masks=selected_masks + [mask], 
                                           agents=selected_mask_models + [agent], 
                                           problem_str=problem, 
                                           trajectories=trajectories, 
                                           number_actions=number_actions, 
                                           number_steps=selected_option_sizes + [option_size])


            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_mask_model = agent
                agent.to_option(mask, option_size, problem)

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask_model.mask)
        selected_option_sizes.append(best_mask_model.option_size)
        selected_mask_models.append(best_mask_model)
        selected_options_problem.append(best_mask_model.problem_id)
        best_loss = levin_loss.compute_loss(selected_masks, selected_mask_models, "", trajectories, number_actions, selected_option_sizes)

        logger.info(f"Added option #{len(selected_mask_models)}; Levin loss of the current selected set: {best_loss} on all trajectories")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_mask_models)
    selected_mask_models = selected_mask_models[:num_options - 1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_mask_models)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_mask_models[i].mask}\n" +
                    f"size={selected_mask_models[i].option_size}\n" +
                    f"problem={selected_mask_models[i].problem_id}")

    save_options(options=selected_mask_models, 
                 trajectories=trajectories,
                 args=args,  
                 logger=logger)
    
    utils.logger_flush(logger)

    levin_loss.print_output_subpolicy_trajectory(selected_mask_models, trajectories, logger=logger)
    utils.logger_flush(logger)


@timing_decorator
def whole_dec_options_training_data_levin_loss(args: Args, logger: logging.Logger):
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic(logger)

    selected_masks = []
    selected_option_sizes = []
    selected_mask_models = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask_model = None

        for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
            model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
            logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
            env = get_single_environment(args, seed=seed)

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            for i in range(2, max_length + 1):
                mask = torch.tensor([-1] * args.hidden_size).view(1,-1)
                levin_loss = loss.compute_loss(selected_masks + [mask], selected_mask_models + [agent], problem, trajectories, number_actions, selected_option_sizes + [i])
            
                if best_loss is None or levin_loss < best_loss:
                    best_loss = levin_loss
                    best_mask_model = agent
                    agent.to_option(mask, i, problem)

        logger.info(f'Summary of option #{len(selected_mask_models)}: \nBest Mask:{best_mask_model.mask}, best_loss={best_loss}, option_size={best_mask_model.option_size}, option problem={best_mask_model.problem_id}\nPrevious selected loss:{previous_loss}')
        utils.logger_flush(logger)
        logger.debug("\n")

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask_model.mask)
        selected_mask_models.append(best_mask_model)
        selected_option_sizes.append(best_mask_model.option_size)
        best_loss = loss.compute_loss(selected_masks, selected_mask_models, "", trajectories, number_actions, selected_option_sizes)

        logger.info(f"Levin loss of the current set: {best_loss}")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_mask_models)
    selected_mask_models = selected_mask_models[:num_options - 1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_mask_models)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_mask_models[i].mask}\n" +
                    f"size={selected_mask_models[i].option_size}\n" +
                    f"problem={selected_mask_models[i].problem_id}")

    save_options(options=selected_mask_models, 
                 trajectories=trajectories,
                 args=args, 
                 logger=logger)

    utils.logger_flush(logger)

    loss.print_output_subpolicy_trajectory(selected_mask_models, trajectories, logger=logger)
    utils.logger_flush(logger)


class STESoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Selects the highest probability (argmax) and returns a one-hot encoding.
        """
        max_indices = torch.argmax(x, dim=0, keepdim=True)
        
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(0, max_indices, 1.0)
        
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Allows gradient to flow through as if softmax was used.
        """
        return grad_output


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Define the quantization levels
        quantization_levels = torch.tensor([-1, 0, 1])
        distances = torch.abs(x.unsqueeze(1) - quantization_levels)
        quantized_indices = torch.argmin(distances, dim=1)
        return quantization_levels[quantized_indices]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LearnOptions:
    def __init__(self, args: Args, logger: logging.Logger, mask_type: str = "internal", mask_transform_type: str = "softmax", selection_type="local_search"):
        """
        `mask_type` can be either "internal", "input", or "both"
        `mask_transform_type` can be either "softmax" or "quantize"
        `selection_type` can be either "greedy" or "local_search"
        """
        
        self.args = args
        self.logger = logger
        self.mask_transform_type = mask_transform_type
        self._mask_transform_func = self._get_transform_func()
        self.mask_type = mask_type
        self.levin_loss = LevinLossActorCritic(self.logger, self.mask_type, self.mask_transform_type)
        self.number_actions = 3

        self.selection_type = selection_type

    def _get_transform_func(self):
        if self.mask_transform_type == "softmax":
            return LearnOptions._softmax_transform
        elif self.mask_transform_type == "quantize":
            return LearnOptions._tanh_transform
        else:
            raise ValueError(f"Invalid mask transform type: {self.mask_transform_type}")

    @staticmethod
    def _tanh_transform(mask):
        mask_transformed = 1.5 * torch.tanh(mask) + 0.5 * torch.tanh(-3 * mask)
        mask_transformed = STEQuantize.apply(mask_transformed)
        return mask_transformed

    @staticmethod
    def _softmax_transform(mask):
            return STESoftmax.apply(torch.softmax(mask, dim=0))

    @timing_decorator
    def discover(self):
        """
        This function performs hill climbing in the space of masks of a ReLU neural network
        to minimize the Levin loss of a given data set. It uses gumbel_softmax to extract 
        """

        trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)

        option_candidates = []

        for target_seed, target_problem in zip(self.args.env_seeds, self.args.problems):
            self.logger.info(f'Extracting options using trajectory segments from {target_problem}, env_seed={target_seed}')
            mimicing_agents = {}

            t_length = trajectories[target_problem].get_length()

            for primary_seed, primary_problem, primary_model_directory in zip(self.args.env_seeds, self.args.problems, self.args.model_paths):
                if primary_problem == target_problem:
                    continue
                model_path = f'binary/models/{primary_model_directory}/seed={self.args.seed}/ppo_first_MODEL.pt'
                primary_env = get_single_environment(self.args, seed=primary_seed)
                parimary_agent = PPOAgent(primary_env, hidden_size=self.args.hidden_size)
                parimary_agent.load_state_dict(torch.load(model_path))
                mimicing_agents[primary_problem] = (primary_seed, model_path, parimary_agent)

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = []
                for length in range(2, t_length + 1):
                    # if length != 3:
                    #     continue
                    for s in range(0, t_length - length + 1, 3):
                        # actions = trajectories[target_problem].slice(s, n=length).get_action_sequence()
                        # if actions not in [[0,0,1], [2,1,0],[1,0,2], [0,1,2]]:
                        #     continue
                        for primary_problem, (primary_seed, primary_model_path, parimary_agent) in mimicing_agents.items():
                            future = executor.submit(
                                self._train_mask_iter, trajectories, target_problem, s, length, parimary_agent)
                            future.s = s,
                            future.length = length
                            future.primary_problem = primary_problem
                            future.primary_env_seed = primary_seed
                            future.primary_model_path = primary_model_path
                            futures.append(future)

                # Process the results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        mask, init_loss, final_loss = future.result()
                        option_candidates.append((mask, 
                                            future.primary_problem, 
                                            target_problem, 
                                            future.primary_env_seed, 
                                            target_seed, 
                                            future.length, 
                                            future.primary_model_path, 
                                            future.s))
                        self.logger.info(f'Progress: segment:{future.s} of length {future.length}, primary_problem={future.primary_problem} done. init_loss={init_loss}, final_loss={final_loss}')
                    except Exception as exc:
                        self.logger.error(f'Segment:{future.s} of length {future.length} with primary_problem={future.primary_problem} generated an exception: {exc}')
            utils.logger_flush(self.logger)
        self.logger.debug("\n")

        if self.selection_type == "greedy":
            selected_options = self.select_greedy(option_candidates, trajectories)
        elif self.selection_type == "local_search":
            selected_options = self.select_by_local_search(option_candidates, trajectories)
        else:
            raise ValueError(f"Invalid selection type: {self.selection_type}")

        # printing selected options
        self.logger.info("Selected options:")
        for i in range(len(selected_options)):
            self.logger.info(f"Option #{i}:\n" + 
                        f"mask={selected_options[i].mask}\n" +
                        f"size={selected_options[i].option_size}\n" +
                        f"problem={selected_options[i].problem_id}")

        save_options(options=selected_options, 
                    trajectories=trajectories,
                    args=self.args, 
                    logger=self.logger)

        utils.logger_flush(self.logger)

        self.levin_loss.print_output_subpolicy_trajectory(selected_options, trajectories, logger=self.logger)
        utils.logger_flush(self.logger)

    def select_greedy(self, option_candidates, trajectories):
        selected_masks = []
        selected_option_sizes = []
        selected_options = []
        selected_options_problem = []

        previous_loss = None
        best_loss = None

        # the greedy loop of selecting options (masks)
        while previous_loss is None or best_loss < previous_loss:
            previous_loss = best_loss

            best_loss = None
            best_mask_model = None

            for mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
                self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
                env = get_single_environment(self.args, seed=primary_env_seed)
                agent = PPOAgent(env, hidden_size=self.args.hidden_size)
                agent.load_state_dict(torch.load(model_path))

                loss_value = self.levin_loss.compute_loss(masks=selected_masks + [mask], 
                                            agents=selected_options + [agent], 
                                            problem_str="", 
                                            trajectories=trajectories, 
                                            number_actions=self.number_actions, 
                                            number_steps=selected_option_sizes + [option_size])

                if best_loss is None or loss_value < best_loss:
                    best_loss = loss_value
                    best_mask_model = agent
                    agent.to_option(mask, option_size, target_problem)
                    agent.extra_info['primary_problem'] = primary_problem
                    agent.extra_info['primary_env_seed'] = primary_env_seed
                    agent.extra_info['target_problem'] = target_problem
                    agent.extra_info['target_env_seed'] = target_env_seed
                    agent.extra_info['segment'] = segment

            # we recompute the Levin loss after the automaton is selected so that we can use 
            # the loss on all trajectories as the stopping condition for selecting masks
            selected_masks.append(best_mask_model.mask)
            selected_option_sizes.append(best_mask_model.option_size)
            selected_options.append(best_mask_model)
            selected_options_problem.append(best_mask_model.problem_id)
            best_loss = self.levin_loss.compute_loss(selected_masks, selected_options, "", trajectories, self.number_actions, selected_option_sizes)

            if previous_loss is None or best_loss < previous_loss:
                self.logger.info(f"Added option #{len(selected_options)}; Levin loss of the current selected set: {best_loss} on all trajectories")
            utils.logger_flush(self.logger)

        # remove the last automaton added
        num_options = len(selected_options)
        selected_options = selected_options[:num_options - 1]
        return selected_options

    def _search_options_subset(self, max_num_options, all_options, chained_trajectory, joint_problem_name_list, max_steps):
        max_num_options = min(max_num_options, len(all_options))
        subset_length = random.choices(range(max_num_options + 1), weights=[1/(i+2) for i in range(max_num_options + 1)], k=1)[0]
        selected_options = set(random.sample(all_options, k=subset_length))
        best_cost = self.levin_loss.compute_loss_cached(list(selected_options), chained_trajectory, joint_problem_name_list, "", self.number_actions)
        previous_cost = float('Inf')
        steps = 0
        while (best_cost < previous_cost or steps == 0) and steps < max_steps:
            previous_cost = best_cost
            neighbours = []
            for option in all_options:
                if option not in selected_options:
                    if len(selected_options) < max_num_options:
                        neighbour = selected_options | {option}
                        neighbours.append(neighbour)
                    for option2 in selected_options:
                        neighbour2 = selected_options - {option2} | {option}
                        neighbours.append(neighbour2)
                else:
                    neighbour = selected_options - {option}
                    neighbours.append(neighbour)
            for neighbour in neighbours:
                cost = self.levin_loss.compute_loss_cached(list(neighbour), chained_trajectory, joint_problem_name_list, "", self.number_actions)
                if cost < best_cost:
                    selected_options = neighbour
                    best_cost = cost
            steps += 1
        
        return best_cost, selected_options

    def select_by_local_search(self, option_candidates, trajectories):
        all_options = []

        for mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
            self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
            env = get_single_environment(self.args, seed=primary_env_seed)
            agent = PPOAgent(env, hidden_size=self.args.hidden_size)
            agent.load_state_dict(torch.load(model_path))
            agent.to_option(mask, option_size, target_problem)
            agent.extra_info['primary_problem'] = primary_problem
            agent.extra_info['primary_env_seed'] = primary_env_seed
            agent.extra_info['target_problem'] = target_problem
            agent.extra_info['target_env_seed'] = target_env_seed
            agent.extra_info['segment'] = segment
            all_options.append(agent)

        print("Number of option_candidates", len(all_options))
        # return 

        chained_trajectory = None
        joint_problem_name_list = []
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
            else:
                chained_trajectory.concat(trajectory)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list

        restarts = 1000
        max_steps = 1000
        max_num_options = 10
        best_selected_options = []
        best_levin_loss_total = float('Inf')
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
            # Submit tasks to the executor with all required arguments
            futures = set()
            for i in range(restarts):
                if len(futures) >= self.args.cpus * 2:
                        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        for f in done:
                            try:
                                best_cost, selected_options = future.result()
                                if best_cost < best_levin_loss_total:
                                    best_levin_loss_total = best_cost
                                    best_selected_options = selected_options
                                completed += 1
                                self.logger.info(f"Restart {completed} of {restarts}")
                            except Exception as exc:
                                self.logger.error(f'Exception: {exc}')
                future = executor.submit(
                    self._search_options_subset, max_num_options, all_options, chained_trajectory, joint_problem_name_list, max_steps)
                futures.add(future)
            
            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    best_cost, selected_options = future.result()
                    if best_cost < best_levin_loss_total:
                        best_levin_loss_total = best_cost
                        best_selected_options = selected_options
                    completed += 1
                    self.logger.info(f"Restart {completed} of {restarts}")
                except Exception as exc:
                    self.logger.error(f'Exception: {exc}')

        self.levin_loss.remove_cache()
        return list(best_selected_options)


    def _train_mask_iter(self, trajectories, problem, s, length, agent: PPOAgent):
        sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
        # learn option with this length
        if self.mask_type != "both":
            if self.mask_type == "internal" and self.mask_transform_type == "quantize":
                mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_mask
            elif self.mask_type == "internal" and self.mask_transform_type == "softmax":
                mask = torch.nn.Parameter(torch.randn(3, self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_mask_softmax
            elif self.mask_type == "input" and self.mask_transform_type == "quantize":
                mask = torch.nn.Parameter(torch.randn(agent.observation_space_size), requires_grad=True)
                rollout_func = agent.run_with_input_mask
            elif self.mask_type == "input" and self.mask_transform_type == "softmax":
                mask = torch.nn.Parameter(torch.randn(3, agent.observation_space_size), requires_grad=True)
                rollout_func = agent.run_with_input_mask_softmax
            return self._train_mask(mask, agent, sub_trajectory, rollout_func)
        else:
            if self.mask_transform_type == "quantize":
                input_mask = torch.nn.Parameter(torch.randn(agent.observation_space_size), requires_grad=True)
                internal_mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_both_masks
            elif self.mask_transform_type == "softmax":
                input_mask = torch.nn.Parameter(torch.randn(3, agent.observation_space_size), requires_grad=True)
                internal_mask = torch.nn.Parameter(torch.randn(3, self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_both_masks_softmax
            return self._train_mask_both(input_mask, internal_mask, agent, sub_trajectory, rollout_func)

    def _train_mask(self, mask, agent: PPOAgent, trajectories: dict, rollout_func):
        # Initialize the masks as trainable parameters with random values
        # mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
        optimizer = torch.optim.Adam([mask], lr=self.args.mask_learning_rate)

        init_loss = None
        best_loss = None
        best_mask = None

        steps = 0
        # with tqdm(total=self.args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
        while steps < self.args.mask_learning_steps:
            # Iterate over trajectories
            for _, trajectory in trajectories.items():
                if steps >= self.args.mask_learning_steps:
                    break

                # Forward pass with mask
                envs = trajectory.get_state_sequence()
                actions = trajectory.get_action_sequence()
                agent.eval()

                # Apply transformations to the mask
                mask_transformed = self._mask_transform_func(mask)

                # Generate a rollout with the mask
                new_trajectory = rollout_func(envs, mask_transformed, trajectory.get_length())

                # Calculate the mask loss            
                # input_probs = torch.nn.functional.log_softmax(torch.stack(new_trajectory.logits), dim=0)
                # target_probs = torch.nn.functional.softmax(torch.stack(trajectory.logits) , dim=0)
                # loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                # mask_loss = loss_fn(input_probs, target_probs)
                loss_fn = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions))
                if not init_loss:
                    init_loss = mask_loss.item()
                    best_loss = init_loss
                    best_mask = mask_transformed

                # Backward pass and optimization
                if mask_loss.item() < best_loss:
                    best_loss = mask_loss.item()
                    best_mask = mask_transformed
                optimizer.zero_grad()
                mask_loss.backward()
                
                # torch.nn.utils.clip_grad_norm_([mask], max_norm=self.args.max_grad_norm)
                optimizer.step()

                # Update progress
                steps += 1

        return best_mask.detach().data, init_loss, best_loss

    def _train_mask_both(self, input_mask, internal_mask, agent: PPOAgent, trajectories: dict, rollout_func):
        # Initialize the masks as trainable parameters with random values
        input_optimizer = torch.optim.Adam([input_mask], lr=self.args.mask_learning_rate)
        internal_optimizer = torch.optim.Adam([internal_mask], lr=self.args.mask_learning_rate)

        init_loss = None
        best_loss = None
        best_input_mask = None
        best_internal_mask = None

        steps = 0
        # with tqdm(total=self.args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
        while steps < self.args.mask_learning_steps:
            # Iterate over trajectories
            for _, trajectory in trajectories.items():
                if steps >= self.args.mask_learning_steps:
                    break

                # Forward pass with mask
                envs = trajectory.get_state_sequence()
                actions = trajectory.get_action_sequence()
                agent.eval()

                # Apply transformations to the mask
                input_mask_discretized = self._mask_transform_func(input_mask)

                internal_mask_discretized = self._mask_transform_func(internal_mask)

                new_trajectory = rollout_func(envs, input_mask_discretized, internal_mask_discretized, trajectory.get_length())

                loss_fn = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions)) 

                if not init_loss:
                    init_loss = mask_loss.item()
                    best_loss = init_loss
                    best_input_mask = input_mask_discretized
                    best_internal_mask = internal_mask_discretized

                # Backward pass and optimization
                if mask_loss.item() < best_loss:
                    best_loss = mask_loss.item()
                    best_input_mask = input_mask_discretized
                    best_internal_mask = internal_mask_discretized

                internal_optimizer.zero_grad()
                mask_loss.backward(retain_graph=True)  
                internal_optimizer.step()

                if steps % self.args.input_update_frequency == 0:  
                    input_optimizer.zero_grad()
                    mask_loss.backward()  
                    input_optimizer.step()
                
                # torch.nn.utils.clip_grad_norm_([mask], max_norm=self.args.max_grad_norm)

                # Update progress
                steps += 1

        return (best_input_mask.detach().data, best_internal_mask.detach().data), init_loss, mask_loss.item()


def evaluate_all_masks_for_model(masks, agents, num_steps, problem, trajectories, loss_evaluator, args, number_actions):
    """
    Function that evaluates all masks for a given model. It returns the best mask (the one that minimizes the Levin loss)
    for the current set of selected masks. It also returns the Levin loss of the best mask. 
    """
    values = [-1, 0, 1]

    best_mask = None
    best_value = None

    combinations = itertools.product(values, repeat=args.hidden_size)

    for value in combinations:
        current_mask = torch.tensor(value, dtype=torch.int8).view(1, -1)
        
        loss = loss_evaluator.compute_loss(masks + [current_mask], agents, problem, trajectories, number_actions, num_steps)

        if best_mask is None or loss < best_value:
            best_value = loss
            best_mask = copy.deepcopy(current_mask)
            print(f"Best Mask so far: {best_mask.numpy()}, Best Levin Loss: {best_value}")
                            
    return best_mask, best_value


def evaluate_all_masks_parallel(args_tuple):
    """Wrapper function to call evaluate_all_masks_for_model with unpacked arguments."""
    return evaluate_all_masks_for_model(*args_tuple)


@timing_decorator
def evaluate_all_masks_levin_loss(args: Args, logger: logging.Logger):
    """
    This function implements the greedy approach for selecting masks (options) from Alikhasi and Lelis (2024).
    This method evaluates all possible masks of a given model and adds to the pool of options the one that minimizes
    the Levin loss. This process is repeated while we can minimize the Levin loss. 

    This method should only be used with small neural networks, as there are 3^n masks, where n is the number of neurons
    in the hidden layer. 
    """
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])

    previous_loss = None
    best_loss = None

    loss_evaluator = LevinLossActorCritic(logger)

    selected_masks = []
    selected_options_problem = []
    selected_options = []
    selected_options_lengths = []

    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_option = None

        for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
            logger.info(f'Evaluating Problem: {problem}')
            model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
            env = get_single_environment(args, seed=seed)
            agent = PPOAgent(envs=env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))


            args_list = [
                (
                    selected_masks,
                    selected_options + [agent],
                    selected_options_lengths + [num_step],
                    problem,
                    trajectories,
                    loss_evaluator,
                    args,
                    number_actions
                )
                for num_step in range(2, max_length + 1)
            ]


            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Submit tasks
                futures = {executor.submit(evaluate_all_masks_parallel, arg): arg for arg in args_list}

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    mask, levin_loss = future.result()
                    num_step = futures[future][2][-1]  # Extract num_step from arguments

                    if best_loss is None or levin_loss < best_loss:
                        best_loss = levin_loss
                        best_option = agent
                        best_option.to_option(mask, num_step, problem)
                        logger.info(f'Best Loss so far: {mask}, {best_loss}, {problem}, {num_step} steps')

            # for num_step in range(2, max_length + 1):
            #     mask, levin_loss = evaluate_all_masks_for_model(masks=selected_masks, 
            #                                                     agents=selected_options + [agent], 
            #                                                     num_steps=selected_options_lengths + [num_step],
            #                                                     problem=problem, 
            #                                                     trajectories=trajectories,
            #                                                     loss_evaluator=loss_evaluator, 
            #                                                     args=args, 
            #                                                     number_actions=number_actions
            #                                                     )

            #     if best_loss is None or levin_loss < best_loss:
            #         best_loss = levin_loss
            #         best_option = agent
            #         best_option.to_option(mask, num_step, problem)
            #         logger.info(f'Best Loss so far: {mask}, {best_loss}, {problem}, {num_step} steps')

        selected_masks.append(best_option.mask)
        selected_options.append(best_option)
        selected_options_problem.append(best_option.problem_id)
        selected_options_lengths.append(best_option.option_size)
        best_loss = loss_evaluator.compute_loss(selected_masks, selected_options, "", trajectories, number_actions, selected_options_lengths)

        logger.info(f"Levin loss of the current set: {best_loss}")

    # remove the last automaton added
    selected_options = selected_options[:-1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_options)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_options[i].mask}\n" +
                    f"size={selected_options[i].option_size}\n" +
                    f"problem={selected_options[i].problem_id}")

    save_options(options=selected_options, 
                 trajectories=trajectories,
                 args=args, 
                 logger=logger)

    utils.logger_flush(logger)

    loss_evaluator.print_output_subpolicy_trajectory(selected_options, trajectories, logger=logger)
    utils.logger_flush(logger)


def main():
    args = process_args()

    # Logger configurations
    logger, args.log_path = utils.get_logger(args.exp_name, args.log_level, args.log_path)

    run_name = f'{args.exp_id}_sd{args.seed}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.exp_id,
            job_type="eval",
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)

    logger.info(f'mask_type="{args.mask_type}", mask_transform_type="{args.mask_transform_type}"')

    module_extractor = LearnOptions(args, logger, 
                                    mask_type=args.mask_type, 
                                    mask_transform_type=args.mask_transform_type, 
                                    selection_type=args.selection_type)
    module_extractor.discover()

    # evaluate_all_masks_levin_loss(args, logger)
    # hill_climbing_mask_space_training_data()
    # whole_dec_options_training_data_levin_loss()
    # hill_climbing_all_segments()
    # learn_options(args, logger)

    logger.info(f"Run id: {run_name}")
    logger.info(f"logs saved at {args.log_path}")


if __name__ == "__main__":
    main()