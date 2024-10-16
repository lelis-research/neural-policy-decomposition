import os
import random
import time
import torch
import tyro
import numpy as np
import gymnasium as gym
from scripts.args import Args
from utils import utils
from torch.utils.tensorboard import SummaryWriter
from environemnts.environments_combogrid_gym import make_env
from environemnts.environments_minigrid import make_env_simple_crossing, make_env_four_rooms
from training.train_ppo_agent import train_ppo

@utils.timing_decorator
def main(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.exp_name += f"{args.learning_rate}"
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_" + args.exp_name 
    args.log_path += "_" + args.exp_name

    logger = utils.get_logger('ppo_trainer_logger_' + str(args.seed) + "_" + args.exp_name, args.log_level, args.log_path)

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
    writer = SummaryWriter(f"outputs/tensorboard/runs/{run_name}")
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

    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        model_file_name = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                        f'-h{args.hidden_size}-lr{args.learning_rate}-sd{seed}_MODEL.pt'
        envs = gym.vector.SyncVectorEnv( 
            [make_env_simple_crossing(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    elif "ComboGrid" in args.env_id:
        problem = args.env_id[len("ComboGrid_"):]
        model_file_name = f'binary/PPO-{problem}-gw{game_width}-h{hidden_size}-l1l{l1_lambda}_MODEL.pt'
        envs = gym.vector.SyncVectorEnv(
            [make_env(rows=game_width, columns=game_width, problem=problem) for _ in range(args.num_envs)],
        )    
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        logger.info("envs.action_space.n", envs.action_space[0].n)
    elif args.env_id == "MiniGrid-FourRooms-v0":
        model_file_name = f'binary/four-rooms/PPO-gw{args.game_width}' + \
                        f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
        envs = gym.vector.SyncVectorEnv( 
            [make_env_four_rooms(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    else:
        raise NotImplementedError
    
        
    train_ppo(envs, args, model_file_name, device, writer, logger=logger, seed=seed)


if __name__ == "__main__":
    args = tyro.cli(Args)
    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    for i in range(len(args.seeds)):
        args.seed = args.seeds[i]
        args.ent_coef = ent_coef[i]
        args.clip_coef = clip_coef[i]
        args.learning_rate = lrs[i]
        main(args)
    