import os
import tyro
from utils import utils
from pipelines.losses import LogitsLossActorCritic, LevinLossActorCritic
from pipelines.option_discovery import load_options
from dataclasses import dataclass


@dataclass
class Args:
    # exp_id: str = "extract_learnOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3"
    exp_id: str = "extract_learnOption_maxlength3_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3"
    """The ID of the finished experiment"""
    seed: int = 0
    """run seed"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""

    env_id = "ComboGrid"
    hidden_size = 64
    game_width = 5


def main(args):
    log_path = os.path.join(args.log_path, args.exp_id)
    log_path += f"/occurances"
    logger, args.log_path = utils.get_logger("print_option_occurances_logger", args.log_level, log_path)
    loss = LevinLossActorCritic(logger, "input", "softmax")

    options, trajectories = load_options(args, logger)
    options = options[:1]
    option = options[0]
    print(option.extra_info)
    best_loss = loss.compute_loss([option.mask], [option], "", trajectories, 3, [option.option_size])
    print("Best loss:", best_loss)

    loss.print_output_subpolicy_trajectory(options=options, 
                                            trajectories=trajectories, 
                                            logger=logger)

    utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)