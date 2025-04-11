from pipelines.option_discovery import regenerate_trajectories, process_args, get_single_environment, load_options
from utils import utils
from agents.policy_guided_agent import PPOAgent
import random
import numpy as np
import torch

args = process_args()

args.model_paths = (
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR'
        )
# args.model_paths = (
#         # 'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
#         'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
#         'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2',
#         )

lengths = {}
logger, _ = utils.get_logger('', args.log_level, args.log_path)
args.env_seeds = (1,2)
# args.env_id = "MiniGrid-SimpleCrossingS9N1-v0"
args.env_id = "ComboGrid"
args.exp_id = "extract_learnOption_both_softmax_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3"
args.game_width = 5
args.hidden_size = 64

for i in range(1):
    args.seed = i
    env = get_single_environment(args, seed=i)
    options, _ = load_options(args, logger)
    for option in options:
        print(option.mask, option.problem_id, option.option_size, option.extra_info)


for p, ls in lengths.items():
    print(f"problem: {p}")
    for i, l in enumerate(ls):
        print(f"seed={i}, l={l}")