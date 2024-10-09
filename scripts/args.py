import os
from typing import List, Union, Tuple
from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = ""
    """the name of this experiment"""
    seeds: List[int] = (0,1,2)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MiniGrid-FourRooms-v0"
    """the id of the environment; choices fro [MiniGrid-SimpleCrossingS9N1-v0, ComboGrid, MiniGrid-FourRooms-v0]"""
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    # learning_rate: Union[float, List] = (2.5e-4, 2.5e-4, 2.5e-4) # ComboGrid
    learning_rate: Union[float, List] = (0.0005, 0.0005, 5e-05) # MiniGrid-FourRooms-v0
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    # clip_coef: Union[float, List] = (0.2, 0.2, 0.2) # ComboGrid
    clip_coef: Union[float, List] = (0.15, 0.1, 0.2) # MiniGrid-FourRooms-v0
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # ent_coef: Union[float, List] = (0.01, 0.01, 0.01) # ComboGrid
    ent_coef: Union[float, List] = (0.05, 0.2, 0.0) # MiniGrid-FourRooms-v0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # hyperparameter arguments
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Logging
    log_path: str = "logfile"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""