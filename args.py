import os
from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PPOwithRandomInitial3x3"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    # env_id: str = "BreakoutNoFrameskip-v4"
    env_id: str = "ComboGrid-v0"
    """the id of the environment"""
    total_timesteps: int = 7000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    value_learning_rate: float = 1.5e-4
    """the learning rate of the optimizer for value network"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 60
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 6
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.07
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rnn_type: str = "gru"
    """RNN model type"""
    weight_decay: float = 0
    "weight decay for l2 regularization"
    l1_lambda: float = 0
    "l1 lambda regularization"
    hidden_size: int = 64
    "size of RNN hidden states"
    problem: str = "test"
    "Problem"
    fine_tune: bool = False
    "toggles fine tuning mode"
    episode_length: int = 20
    "maximum episode length"
    game_width: int = 3
    "the width of the grid"
    visitation_bonus: bool = True
    "toggles using visitation bonus in calculating reward"
    use_options: int = 0
    "set to 0 for not using options, and 1 for using options when training"
    quantized: int = 1
    "set to 0 for models without quantized hidden states, and 1 models with quantized hidden states"

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
