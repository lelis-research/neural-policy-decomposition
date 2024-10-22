import os
import tyro
from utils import utils
from typing import Union, List
from pipelines.losses import LogitsLossActorCritic
from extract_subpolicy_ppo import load_options
from dataclasses import dataclass


@dataclass
class Args:
    
    exp_id: str = "exp_01"
    """The ID of the finished experiment"""
    env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    game_width: int = 5
    """the length of the combo/mini grid square"""
    seeds: Union[List[int], str] = (0,1,2)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""


def main(args):
    log_path = os.path.join(args.log_path, args.exp_id)
    log_path += f"/whole_grid_testing"
    logger = utils.get_logger("whole_grid_testing_logger", args.log_level, log_path)
    
    loss = LogitsLossActorCritic(logger)

    options, _ = load_options(args.exp_id)

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