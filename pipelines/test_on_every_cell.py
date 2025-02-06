import os
import tyro
from utils import utils
from typing import Union, List
from pipelines.losses import LogitsLossActorCritic
from pipelines.extract_subpolicy_ppo import load_options
from dataclasses import dataclass


@dataclass
class Args:
    exp_id: str = "extract_learn_options__ComboGrid_gw5_h64_l10_r400_sd0,1,2,3_olen2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"
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
    problems: List[str] = ("TL-BR", "TR-BL", "BR-TL", "BL-TR")
    """"""
    seeds: Union[List[int], str] = (0,1,2,3)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    
    # script arguments
    seed: int = 0
    """run seed"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""


def main(args):
    log_path = os.path.join(args.log_path, args.exp_id)
    logger = utils.get_logger("whole_grid_testing_logger", args.log_level, log_path)
    
    loss = LogitsLossActorCritic(logger)

    options, _ = load_options(args, logger)

    logger.info("Testing on each grid cell")
    for seed, problem in zip(args.seeds, args.problems):
        logger.info(f"Testing on each cell..., {problem}")
        loss.evaluate_on_each_cell(options=options, 
                                   problem_test=problem, 
                                   args=args, 
                                   seed=seed, 
                                   logger=logger)

    utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)