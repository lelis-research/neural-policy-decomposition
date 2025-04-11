import os
import random
import time
import torch
import tyro
import numpy as np
import gymnasium as gym
from utils import utils
from typing import Union, List, Tuple
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from environments.environments_combogrid_gym import make_env as make_env_combogrid
from environments.environments_combogrid import PROBLEM_NAMES as COMBOGRID_PROBLEMS
from environments.environments_minigrid import make_env_simple_crossing, make_env_four_rooms
from training.train_ppo_agent import train_ppo


@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    exp_name: str = "train_ppoAgent"
    """the name of this experiment"""
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0, MiniGrid-FourRooms-v0]
    """
    # env_seeds: Union[List[int], str] = (0,1,2) # SimpleCrossing
    env_seeds: Union[List, str, Tuple] = (0,1,2,3) # ComboGrid
    # env_seeds: Union[List[int], str] = (41,51,8) # FourRooms
    # env_seeds: Union[List[int], str] = (8,) # FourRooms
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'.
    This determines the exact environments that will be separately used for training.
    """
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track_tensorboard: bool = False
    """if toggled, this experiment will be tracked with Tensorboard SummaryWriter"""
    wandb_project_name: str = "BASELINE0_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    cpus: int = 0
    """"Not used in this experiment"""
    
    # hyperparameter arguments
    game_width: int = 5
    """the length of the combo/mini-grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""
    number_actions: int = 3

    # Specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps for testinging"""
    learning_rate: Union[Tuple[float, ...], float] = (2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4) # ComboGrid
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 5e-05) # Vanilla RL FourRooms
    # learning_rate: Union[List[float], float] = (5e-05,) # Vanilla RL FourRooms
    # learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001) # SimpleCrossing
    """the learning rate of the optimize for testinging"""
    num_envs: int = 4
    """the number of parallel game environments for testinging"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout for testinging"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks for testinging"""
    gamma: float = 0.99
    """the discount factor gamma for testinging"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation for testinging"""
    num_minibatches: int = 4
    """the number of mini-batches for testinging"""
    update_epochs: int = 4
    """the K epochs to update the policy for testinging"""
    norm_adv: bool = True
    """Toggles advantages normalization for testinging"""
    clip_coef: Union[Tuple[float, ...], float] = (0.2, 0.2, 0.2, 0.2) # ComboGrid
    # clip_coef: Union[List[float], float] = (0.15, 0.1, 0.2) # Vanilla RL FourRooms
    # clip_coef: Union[List[float], float] = (0.2,) # Vanilla RL FourRooms
    # clip_coef: Union[List[float], float] = (0.25, 0.2, 0.2) # SimpleCrossing
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef:Union[Tuple[float, ...], float] = (0.01, 0.01, 0.01, .01) # ComboGrid
    # ent_coef: Union[List[float], float] = (0.05, 0.2, 0.0) # Vanilla RL FourRooms
    # ent_coef: Union[List[float], float] = (0.1, 0.1, 0.1) # SimpleCrossing
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    env_seed: int = -1
    """the seed of the environment (set in runtime)"""
    seed: int = 2
    """experiment randomness seed (set in runtime)"""
    problem: str = ""
    """"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""


@utils.timing_decorator
def main(args: Args):
    
    run_name = f"{args.exp_id}_sd{args.seed}_training_t{int(time.time())}" 
    run_index = f"train_ppo_t{int(time.time())}" 
    
    log_path = os.path.join(args.log_path, args.exp_id, f"seed={args.seed}", "train_ppo")

    logger, _ = utils.get_logger('ppo_trainer_logger_' + str(args.env_seed) + "_" + args.exp_name, args.log_level, log_path)

    logger.info(f"\n\nExperiment: {args.exp_id}\n\n")

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
    
    writer = None
    if args.track_tensorboard:
        # Setting up tensorboard summary writer
        writer_path = f"{args.exp_id}/seed={args.seed}/{run_index}"
        writer = SummaryWriter(f"outputs/tensorboard/runs/{writer_path}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger.info(f"Constructing tensorboard summary writer on outputs/tensorboard/runs/{run_name}")
        

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Parameter logging
    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)

    # Environment creation
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        envs = gym.vector.SyncVectorEnv( 
            [make_env_simple_crossing(view_size=args.game_width, seed=args.env_seed) for _ in range(args.num_envs)])
    elif "ComboGrid" in args.env_id:
        problem = args.problem
        envs = gym.vector.SyncVectorEnv(
            [make_env_combogrid(rows=args.game_width, columns=args.game_width, problem=problem) for _ in range(args.num_envs)],
        )    
    elif args.env_id == "MiniGrid-FourRooms-v0":
        envs = gym.vector.SyncVectorEnv( 
            [make_env_four_rooms(view_size=args.game_width, seed=args.env_seed) for _ in range(args.num_envs)])
    else:
        raise NotImplementedError
    
    model_path = f'binary/models/{args.exp_id}/seed={args.seed}/ppo_first_MODEL.pt'

    train_ppo(envs=envs, 
              seed=args.env_seed, 
              args=args, 
              model_file_name=model_path, 
              device=device, 
              logger=logger, 
              writer=writer, 
              sparse_init=False)
    if args.track:
        wandb.finish()
    # wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_l1{args.l1_lambda}'
    
    # Processing seeds from arguments
    if isinstance(args.env_seeds, list) or isinstance(args.env_seeds, tuple):
        args.env_seeds = list(map(int, args.env_seeds))
    elif isinstance(args.env_seeds, str):
        start, end = map(int, args.env_seeds.split(","))
        args.env_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    # Parameter specification for each problem
    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    exp_id = args.exp_id
    for i in range(1, len(args.env_seeds)):
        args.env_seed = args.env_seeds[i]
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        args.ent_coef = ent_coef[i]
        args.clip_coef = clip_coef[i]
        args.learning_rate = lrs[i]
        args.exp_id = f'{exp_id}_lr{args.learning_rate}_clip{args.clip_coef}_ent{args.ent_coef}_envsd{args.env_seed}'
        if args.env_id == "ComboGrid":
            args.problem = COMBOGRID_PROBLEMS[args.env_seed]
            args.exp_id += f'_{args.problem}'
        main(args)
    