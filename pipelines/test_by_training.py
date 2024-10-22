import os
import torch
import time
import tyro
import gymnasium as gym
from utils import utils
from logging import Logger
from typing import Union, List
from agents.policy_guided_agent import PPOAgent
from torch.utils.tensorboard import SummaryWriter
from extract_subpolicy_ppo import load_options
from dataclasses import dataclass
from training.train_ppo_agent import train_ppo
from environemnts.environments_minigrid import make_env_four_rooms


@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment"""
    env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    seeds: Union[List[int], str] = (0,1,2)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameter arguments
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""
    number_actions: int = 3
    
    # Testing specific arguments
    test_exp_id: str = ""
    """The ID of the new experiment"""
    test_exp_name: str = "Option Extraction"
    """the name of this experiment"""
    test_env_id: str = "MiniGrid-FourRooms-v0"
    """the id of the environment for testing
    choices from [ComboGrid, MiniGrid-FourRooms-v0]"""
    test_problems: List[str] = []
    """"""
    test_seeds: Union[List[int], str] = (1,2,3)
    """the seeds of the environment for testing"""
    total_timesteps: int = 1_500_000
    """total timesteps for testing"""
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 5e-05) # Vanilla RL
    learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001)
    """the learning rate of the optimize for testing"""
    num_envs: int = 4
    """the number of parallel game environments for testing"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout for testing"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks for testing"""
    gamma: float = 0.99
    """the discount factor gamma for testing"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation for testing"""
    num_minibatches: int = 4
    """the number of mini-batches for testing"""
    update_epochs: int = 4
    """the K epochs to update the policy for testing"""
    norm_adv: bool = True
    """Toggles advantages normalization for testing"""
    # clip_coef: Union[List[float], float] = (0.15, 0.1, 0.2) # Vanilla RL
    clip_coef: Union[List[float], float] = (0.25, 0.2, 0.2)
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
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
    
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""


def train_ppo_with_options(options: List[PPOAgent], test_exp_id: str, seed: int, args: Args, logger: Logger):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    if args.test_env_id == "MiniGrid-FourRooms-v0":
        envs = gym.vector.SyncVectorEnv(
        [make_env_four_rooms(view_size=args.game_width, seed=seed, options=options) 
         for _ in range(args.num_envs)],
    )
    else:
        raise NotImplementedError
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    model_path = f'binary/models/{test_exp_id}/extended_MODEL.pt'
    os.makedirs(model_path, exist_ok=True)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{test_exp_id}_trained_with_options_t{int(time.time())}"
    writer = SummaryWriter(f"outputs/tensorboard/runs/{run_name}")
    hyperparameters = dict(vars(args))
    hyperparameters.update({"test_exp_id": test_exp_id})
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyperparameters.items()])),
    )
    options_info = {f"option{i}":(option.mask.tolist(), option.option_size, option.problem_id) for i, option in enumerate(options)}
    writer.add_text(
        "options_setting",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in options_info.items()])),)
    logger.info(f"Reporting tensorboard summary writer on outputs/tensorboard/runs/{run_name}")

    train_ppo(envs=envs, 
              seed=seed, 
              args=args, 
              model_file_name=model_path, 
              device=device, 
              logger=logger, 
              writer=writer)


def main(args: Args):

    logger = utils.get_logger("testing_by_training_logger", args.log_level, args.log_path)

    options, _ = load_options(args.exp_id)

    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    for i, (problem, seed) in enumerate(zip(args.test_problems, args.test_seeds)):
        logger.info(f"Testing by training on {problem}")
        args.learning_rate = lrs[i]
        args.clip_coef = clip_coef[i]
        args.ent_coef = ent_coef[i]
        test_exp_id = f'{args.test_exp_id}_lr{args.learning_rate}_clip{args.clip_coef}_ent{args.ent_coef}_sd{seed}'
        train_ppo_with_options(options, test_exp_id, seed, args, logger)
        utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.test_exp_id == "":
        args.test_exp_id = f'{args.test_exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_l1{args.l1_lambda}'
    args.log_path = os.path.join(args.log_path, args.exp_id)
    args.log_path += f"/test_by_training"
    
    if isinstance(args.test_seeds, list) or isinstance(args.test_seeds, tuple):
        args.test_seeds = list(map(int, args.test_seeds))
    elif isinstance(args.test_seeds, str):
        start, end = map(int, args.test_seeds.split(","))
        args.test_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        args.test_problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        args.test_seeds = args.test_seeds * (len(args.test_problems)//len(args.test_seeds) + 1)
        args.test_seeds = args.test_seeds[:len(args.test_problems)]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        args.test_problems = [args.test_env_id + str(seed) for seed in args.test_seeds]

    main(args)