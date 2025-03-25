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

lengths = {}
logger, _ = utils.get_logger('hill_climbing_logger', args.log_level, args.log_path)

for i in range(10):
    trajectories = regenerate_trajectories(args, verbose=False, logger=None)
    for problem, t in trajectories.items():
        if problem not in lengths:
            lengths[problem] = []
        lengths[problem].append(t.get_length())


for p, ls in lengths.items():
    print(f"problem: {p}")
    for i, l in enumerate(ls):
        print(f"seed={i}, l={l}")