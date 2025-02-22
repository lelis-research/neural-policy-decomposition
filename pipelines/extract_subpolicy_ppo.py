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
from typing import Union, List
import concurrent.futures
from pipelines.losses import LevinLossActorCritic, LogitsLossActorCritic
from agents.policy_guided_agent import PPOAgent
from environments.environments_combogrid_gym import ComboGym
from environments.environments_combogrid import SEEDS
from environments.environments_minigrid import get_training_tasks_simplecross
from utils.utils import timing_decorator
from utils.utils import timing_decorator


@dataclass
class Args:
    exp_id: str = ""
    """the id to be set for the experiment"""
    # exp_name: str = "extract_decOptionWhole_randomInit"
    exp_name: str = "extract_decOptionWhole_sparseInit"
    # exp_name: str = "extract_learnOptions_randomInit_discreteMasks"
    # exp_name: str = "extract_learnOptions_randomInit_pitisFunction"
    """the name of this experiment"""
    problems: List[str] = ("TL-BR", "TR-BL", "BR-TL", "BL-TR")
    """"""
    env_seeds: Union[List[int], str] = (0,1,2)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    # model_paths: List[str] = (
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    model_paths: List[str] = (
        'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
        'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
        'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2',
        )
    # model_paths: List[str] = (
    #     'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd0_TL-BR',
    #     'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd1_TR-BL',
    #     'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd2_BR-TL',
    #     'train_ppo_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_sd3_BL-TR',
    # )
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    # env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameter arguments
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""
    option_length: int = 3
    """number of while loops for applying the option"""
    number_restarts: int = 400
    """number of hill climbing restarts for finding one option"""

    # mask learning
    mask_learning_rate: float = 0.001
    """"""
    mask_learning_steps: int = 1_000
    """"""
    max_grad_norm: float = 1.0

    # Script arguments
    seed: int = 0
    """The seed used for reproducibilty of the script"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "LMNOP"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""


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
        args.env_seeds = [SEEDS[problem] for problem in args.problems]
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
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(view_size=args.game_width, seed=seed)
        elif args.env_id == "MiniGrid-FourRooms-v0":
            raise NotImplementedError("Environment creation not implemented!")
        else:
            env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

        # if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        #     assert trajectories[problem].get_length() < 16, f"model not optimized! {(seed, problem, model_directory)}"

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
            'environment_args': model.environment_args
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
        if 'environment_args' in checkpoint:
            model.environment_args = checkpoint['environment_args']
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
            # # Use ProcessPoolExecutor to run the hill climbing iterations in parallel
            # with concurrent.futures.ThreadPoolExecutor(max_workers=args.cpus) as executor:
            #     # Submit tasks to the executor with all required arguments
            #     futures = [
            #         executor.submit(
            #             hill_climbing_iter, i, agent, option_size, problem_str, number_actions, 
            #             mask_values, trajectories, selected_masks, selected_masks_models, 
            #             selected_option_sizes, default_loss, args, loss
            #         )
            #         for i in range(args.number_restarts + 1)
            #     ]

            #     # Process the results as they complete
            #     for future in concurrent.futures.as_completed(futures):
            #         try:
            #             i, best_value, best_mask, init_mask, n_steps, applicable = future.result()
            #             _update_best(i=i, 
            #                          best_value=best_value, 
            #                          best_mask=best_mask,
            #                          n_steps=n_steps, 
            #                          init_mask=init_mask, 
            #                          applicable=applicable,
            #                          n_applicable=n_applicable, 
            #                          option_size=option_size)
            #             if i % 100 == 0:
            #                 logger.info(f'Progress: {i}/{args.number_restarts}')
            #         except Exception as exc:
            #             logger.error(f'restart #{i} generated an exception: {exc}')

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
    logger = utils.get_logger('hill_climbing_logger', args.log_level, args.log_path, suffix="option_extraction")

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
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
            elif args.env_id == "ComboGrid":
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            else:
                raise NotImplementedError

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
def hill_climbing_all_segments():
    
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = process_args()
    
    # Logger configurations
    logger = utils.get_logger('hc_all_segments_logger', args.log_level, args.log_path, suffix="option_extraction")

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

    logits_loss = LogitsLossActorCritic(logger)
    levin_loss = LevinLossActorCritic(logger)

    all_masks_info = []

    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
        elif args.env_id == "ComboGrid":
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
        
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
            model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
            logger.info(f'Extracting from the agent trained on problem={problem}, seed={seed}, segment=({segment},{segment+option_size})')
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
            elif args.env_id == "ComboGrid":
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            else:
                raise NotImplementedError
            
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
                if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                    agent.environment_args = {
                        "seed": seed,
                        "game_width": game_width
                    }
                else:
                    agent.environment_args = {
                        "game_width": game_width
                    }

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
def whole_dec_options_training_data_levin_loss():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = process_args()
    
    # Logger configurations
    logger = utils.get_logger('whole_dec_options', args.log_level, args.log_path, suffix="option_extraction")
    logger.info("Extract Dec-Option Whole ... ")
    

    game_width = args.game_width
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])
    option_length = list(range(2, max_length + 1))
    # args.exp_id += f'_olen{",".join(map(str, option_length))}'
    
    run_name = f'{args.exp_id}_sd{args.seed}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.exp_id,
            job_type="eval",
            entity=args.wandb_entity,
            sync_tensorboard=True,
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
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
            elif args.env_id == "ComboGrid":
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            else:
                raise NotImplementedError

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            for i in range(2, max_length + 1):
                mask = torch.tensor([-1] * args.hidden_size).view(1,-1)
                levin_loss = loss.compute_loss(selected_masks + [mask], selected_mask_models + [agent], problem, trajectories, number_actions, selected_option_sizes + [i])
            
                if best_loss is None or levin_loss < best_loss:
                    best_loss = levin_loss
                    best_mask_model = agent
                    agent.to_option(mask, i, problem)
                    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                        agent.environment_args = {
                            "seed": seed,
                            "game_width": game_width
                        }
                    else:
                        agent.environment_args = {
                            "game_width": game_width
                        }

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

    # logger.info("Testing on each grid cell")
    # for seed, problem in zip(args.env_seeds, args.problems):
    #     logger.info(f"Testing on each cell..., {problem}")
    #     loss.evaluate_on_each_cell(selected_mask_models, problem, args=args, seed=seed, logger=logger)

    # utils.logger_flush(logger)


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


def learn_mask(agent: PPOAgent, trajectories: dict, args: Args, discrete_masks=True, logger=None):
    # Initialize the masks as trainable parameters with random values
    mask = torch.nn.Parameter(torch.randn(args.hidden_size), requires_grad=True)
    optimizer = torch.optim.Adam([mask], lr=args.mask_learning_rate)

    initial_mask = torch.t_copy(mask)

    init_loss = None

    steps = 0
    # with tqdm(total=args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
    while steps < args.mask_learning_steps:
        # Iterate over trajectories
        for _, trajectory in trajectories.items():
            if steps >= args.mask_learning_steps:
                break

            # Forward pass with mask
            envs = trajectory.get_state_sequence()
            agent.eval()

            # Apply transformations with requires_grad

            if discrete_masks:
                mask_transformed = STEQuantize.apply(mask)
            else:
                mask_transformed = 1.5 * torch.tanh(mask) + 0.5 * torch.tanh(-3 * mask)
                mask_transformed = mask_transformed.clamp(-0.5, 0.5) * 2

            new_trajectory = agent.run_with_mask(envs, mask_transformed, trajectory.get_length())

            # Compute the mask loss
            # cross_entropy_loss = torch.nn.CrossEntropyLoss()
            
            input_probs = torch.nn.functional.log_softmax(torch.stack(new_trajectory.logits), dim=0)
            target_probs = torch.nn.functional.softmax(torch.stack(trajectory.logits) , dim=0)
            loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            mask_loss = loss_fn(input_probs, target_probs)
            if not init_loss:
                init_loss = mask_loss.item()

            # logger.info(f"loss={mask_loss.item()}, {torch.stack(new_trajectory.logits).shape}, {torch.stack(trajectory.logits).shape}")

            # Backward pass and optimization
            optimizer.zero_grad()
            mask_loss.backward()
            
            torch.nn.utils.clip_grad_norm_([mask], max_norm=args.max_grad_norm)
            optimizer.step()

            # Update progress
            steps += 1
                # pbar.update(1)  # Update the outer progress bar

    mask_transformed = 1.5 * torch.tanh(mask) + 0.5 * torch.tanh(-3 * mask)
    mask_transformed = mask_transformed.clamp(-0.5, 0.5) * 2
    return mask_transformed.detach().data, init_loss, mask_loss.item()


def learn_mask_iter(trajectories, problem, s, length, agent, args, model_path, logger):
    sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
    # learn option with this length using gumbel softmax
    mask, init_loss, final_loss = learn_mask(agent, sub_trajectory, args, logger=logger)
    return (mask, problem, length, model_path, s), init_loss, final_loss


@timing_decorator
def learn_options():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. It uses gumbel_softmax to extract 
    """
    args = process_args()
    
    # Logger configurations
    logger = utils.get_logger('learn_options', args.log_level, args.log_path, suffix="learn_option")
    logger.info("Learning Options ... ")

    game_width = args.game_width
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])
    # option_length = list(range(2, max_length + 1))
    # args.exp_id += f'_olen{",".join(map(str, option_length))}'

    run_name = f'{args.exp_id}_sd{args.seed}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.exp_id,
            job_type="eval",
            entity=args.wandb_entity,
            sync_tensorboard=True,
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


    levin_loss = LevinLossActorCritic(logger)

    all_masks_info = []

    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        logger.info(f'Extracting from the agent trained on {problem}, env_seed={seed}')
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
        elif args.env_id == "ComboGrid":
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
        else:
            raise NotImplementedError
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        agent.load_state_dict(torch.load(model_path))

        t_length = trajectories[problem].get_length()

        
        # for length in range(2, t_length + 1):
        #     for s in range(t_length - length):
        #         sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
        #         # learn option with this length using gumbel softmax
        #         mask, init_loss, final_loss = learn_mask(agent, sub_trajectory, args, logger=logger)
        #         all_masks_info.append((mask, problem, length, model_path, s))
        #         logger.info(f'Progress: segment:{s} of length{length} done. init_loss={init_loss}, final_loss={final_loss}')

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.cpus) as executor:
            # Submit tasks to the executor with all required arguments
            futures = []
            for length in range(2, t_length + 1):
                for s in range(t_length - length):
                    future = executor.submit(
                        learn_mask_iter, trajectories, problem, s, length, agent, 
                        args, model_path, logger
                    )
                    future.s = s,
                    future.length = length
                    futures.append(future)

            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    (mask, problem, length, model_path, s), init_loss, final_loss = future.result()
                    all_masks_info.append((mask, problem, length, model_path, s))
                    logger.info(f'Progress: segment:{s} of length{length} done. init_loss={init_loss}, final_loss={final_loss}')
                except Exception as exc:
                    logger.error(f'Segment:{future.s} of length{future.length}generated an exception: {exc}')
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
            model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
            logger.info(f'Evaluating the option from problem={problem}, env_seed={seed}, segment=({segment},{segment+option_size})')
            
            if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                env = get_training_tasks_simplecross(view_size=game_width, seed=seed)
            elif args.env_id == "ComboGrid":
                env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            else:
                raise NotImplementedError

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
                if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                    agent.environment_args = {
                        "seed": seed,
                        "game_width": game_width
                    }
                else:
                    agent.environment_args = {
                        "game_width": game_width
                    } 

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


def main():
    # hill_climbing_mask_space_training_data()
    # whole_dec_options_training_data_levin_loss()
    # hill_climbing_all_segments()
    learn_options()

    
if __name__ == "__main__":
    main()