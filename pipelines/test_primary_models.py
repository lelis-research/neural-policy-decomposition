from pipelines.extract_subpolicy_ppo import regenerate_trajectories, process_args
from utils import utils

args = process_args()

args.model_paths = (
        'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
        'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
        'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2',
        )

lengths = {}

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