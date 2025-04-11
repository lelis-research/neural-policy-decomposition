import os
import tyro
from utils import utils
from typing import Union, List, Tuple
from pipelines.losses import LevinLossActorCritic
from pipelines.option_discovery import load_options
from dataclasses import dataclass
from environments.environments_combogrid import PROBLEM_NAMES as COMMBOGRID_NAMES


@dataclass
class Args:
    exp_id: str = "extract_learnOption_maxlength3_aligned_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinput_mskTransformsoftmax"
    """The ID of the finished experiment"""
    # env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    problems: List[str] = tuple()
    """"""
    env_seeds: Union[Tuple[int, ...], str] = (0,1,2,3)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""

    # model_paths: List[str] = (
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
    # )
    
    # script arguments
    seed: int = 0
    """run seed"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""


def main(args: Args):
    log_path = os.path.join(args.log_path, args.exp_id)
    logger, args.log_path = utils.get_logger("whole_grid_testing_logger", args.log_level, log_path)
    
    options, trajectories = load_options(args, logger)
    assert all(option.option_size == 3 for option in options)
    mask_type = options[0].mask_type
    assert all(option.mask_type == mask_type for option in options)
    mask_transform_type = options[0].mask_transform_type
    assert all(option.mask_transform_type == mask_transform_type for option in options)

    loss = LevinLossActorCritic(logger, mask_type, mask_transform_type)

    logger.info("Testing on each grid cell")
    for seed, problem in zip(args.env_seeds, args.problems):
        logger.info(f"Testing on each cell..., {problem}")
        loss.evaluate_on_each_cell(options=options, 
                                   trajectories=trajectories,
                                   problem_test=problem, 
                                   args=args, 
                                   seed=seed, 
                                   logger=logger)

    utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.env_id == "ComboGrid":
        args.problems = [COMMBOGRID_NAMES[i] for i in args.env_seeds]
    main(args)