import os
import tyro
from utils import utils
from losses import LogitsLossActorCritic
from extract_subpolicy_ppo import load_options
from dataclasses import dataclass


@dataclass
class Args:
    exp_id: str = "exp_01"
    """The ID of the finished experiment"""

    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""


def main(args):
    log_path = os.path.join(args.log_path, args.exp_id)
    log_path += f"/occurances"
    logger = utils.get_logger("print_option_occurances_logger", args.log_level, log_path)
    loss = LogitsLossActorCritic(logger)

    options, trajectories = load_options(args.exp_id, args)

    loss.print_output_subpolicy_trajectory(options=options, 
                                            trajectories=trajectories, 
                                            logger=logger)

    utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)