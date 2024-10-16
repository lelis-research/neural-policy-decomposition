from dataclasses import dataclass
from typing import Union, List

@dataclass
class Args:
    exp_name: str = "Option Extraction"
    """the name of this experiment"""
    seeds: Union[List[int], str] = (0,1,2)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
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
    number_restarts: int = 1000
    """number of hill climbing restarts for finding one option"""
    
    # Retraining specific arguments
    test_env_id: str = "MiniGrid-FourRooms-v0"
    """the id of the environment for retraining
    choices from [ComboGrid, MiniGrid-FourRooms-v0]"""
    test_seeds: Union[List[int], str] = (1,2,3)
    """the seeds of the environment for retraining"""
    total_timesteps: int = 1_500_000
    """total timesteps for retraining"""
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 5e-05) # Vanilla RL
    learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001)
    """the learning rate of the optimize for retraining"""
    num_envs: int = 4
    """the number of parallel game environments for retraining"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout for retraining"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks for retraining"""
    gamma: float = 0.99
    """the discount factor gamma for retraining"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation for retraining"""
    num_minibatches: int = 4
    """the number of mini-batches for retraining"""
    update_epochs: int = 4
    """the K epochs to update the policy for retraining"""
    norm_adv: bool = True
    """Toggles advantages normalization for retraining"""
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

    log_path: str = "logfile"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""