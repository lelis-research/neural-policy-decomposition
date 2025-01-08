import os
import random
import time
import torch
import tyro
import numpy as np
import gymnasium as gym
from utils import utils
from typing import Union, List
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from environemnts.environments_combogrid_gym import make_env
from environemnts.environments_combogrid import SEEDS as COMBOGRID_SEEDS
from environemnts.environments_minigrid import make_env_simple_crossing, make_env_four_rooms
from training.train_ppo_agent import train_ppo


@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    exp_name: str = "train_ppo_agent"
    """the name of this experiment"""
    env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    env_seeds: Union[List[int], str] = (0,1,2)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
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

    # combogrid parameters
    combogrid_problems: List[str] = ("TL-BR", "TR-BL", "BR-TL", "BL-TR")
    
    # Specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps for testinging"""
    # learning_rate: Union[List[float], float] = (2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4) # ComboGrid
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 5e-05) # Vanilla RL MiniGrid
    learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001)
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
    # clip_coef: Union[List[float], float] = (0.2, 0.2, 0.2, 0.2) # ComboGrid
    # clip_coef: Union[List[float], float] = (0.15, 0.1, 0.2) # Vanilla RL
    clip_coef: Union[List[float], float] = (0.25, 0.2, 0.2) 
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # ent_coef: Union[List[float], float] = (0.01, 0.01, 0.01, .01) # ComboGrid
    # ent_coef: Union[List[float], float] = (0.05, 0.2, 0.0) # Vanilla RL
    ent_coef: Union[List[float], float] = (0.1, 0.1, 0.1)
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
    seed: int = -1
    """the seed (set in runtime)"""
    problem: str = ""
    """"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""


@utils.timing_decorator
def main(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.exp_id}_training_t{int(time.time())}" 
    run_index = f"train_ppo_t{int(time.time())}" 
    
    log_path = os.path.join(args.log_path, args.exp_id, "train_ppo")
    suffix = f"/training_ppo"

    logger = utils.get_logger('ppo_trainer_logger_' + str(args.seed) + "_" + args.exp_name, args.log_level, log_path, suffix=suffix)

    logger.info(f"\n\nExperiment: {args.exp_id}\n\n")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Setting up tensorboard summary writer
    writer = SummaryWriter(f"outputs/tensorboard/runs/{args.exp_id}/{run_index}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger.info(f"Constructing tensorboard summary writer on outputs/tensorboard/runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    game_width = args.game_width
    hidden_size = args.hidden_size
    problem = args.env_id
    l1_lambda = args.l1_lambda
    
    # Parameter logging
    params = {
        'seed': seed,
        'hidden_size': hidden_size,
        'game_width': game_width,
        'l1_lambda': l1_lambda,
        'problem': problem
    }

    buffer = "\nParameters:"
    for key, value in params.items():
        buffer += f"\n- {key}: {value}"
    logger.info(buffer)

    # Environment creation
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        envs = gym.vector.SyncVectorEnv( 
            [make_env_simple_crossing(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    elif "ComboGrid" in args.env_id:
        problem = args.problem
        envs = gym.vector.SyncVectorEnv(
            [make_env(rows=game_width, columns=game_width, problem=problem) for _ in range(args.num_envs)],
        )    
    elif args.env_id == "MiniGrid-FourRooms-v0":
        envs = gym.vector.SyncVectorEnv( 
            [make_env_four_rooms(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    else:
        raise NotImplementedError
    
    model_path = f'binary/models/{args.exp_id}/ppo_first_MODEL.pt'

    train_ppo(envs=envs, 
              seed=seed, 
              args=args, 
              model_file_name=model_path, 
              device=device, 
              logger=logger, 
              writer=writer)


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
    
    if args.env_id == "ComboGrid":
        args.env_seeds = [COMBOGRID_SEEDS[problem] for problem in args.combogrid_problems]

    # Parameter specification for each problem
    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    exp_id = args.exp_id
    for i in range(len(args.env_seeds)):
        args.seed = args.env_seeds[i]
        args.ent_coef = ent_coef[i]
        args.clip_coef = clip_coef[i]
        args.learning_rate = lrs[i]
        args.exp_id = f'{exp_id}_lr{args.learning_rate}_clip{args.clip_coef}_ent{args.ent_coef}_sd{args.seed}'
        if args.env_id == "ComboGrid":
            args.problem = args.combogrid_problems[i]
            args.exp_id += f'_{args.problem}'
        main(args)
    