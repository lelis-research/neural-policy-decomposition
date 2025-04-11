import copy
import torch
import math
import numpy as np
from typing import List
import torch.nn.functional as F
# from pipelines.test_on_every_cell import Args as EachCellTestArgs
from agents.policy_guided_agent import PPOAgent
from environments.environments_combogrid_gym import ComboGym
from environments.environments_combogrid import DIRECTIONS, PROBLEM_NAMES as COMBO_PROBLEM_NAMES
from environments.environments_minigrid import get_training_tasks_simplecross


def regenerate_trajectories(args, verbose=False, logger=None):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    def get_single_environment(args, seed):
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(view_size=args.game_width, seed=seed)
        elif args.env_id == "ComboGrid":
            problem = COMBO_PROBLEM_NAMES[seed]
            env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)
        else:
            raise NotImplementedError
        return env
    
    trajectories = {}
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        env = get_single_environment(args, seed=seed)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

        if verbose:
            logger.info(f"The trajectory length: {len(trajectory.get_state_sequence())}")

    return trajectories


class LevinLossActorCritic:
    def __init__(self, logger, mask_type="internal", mask_transform_type="quantize"):
        """
        `mask_type` is a string that can be either "internal", "input", or "both".
        `mask_transform_type` is a string that can be either "quantize" or "softmax".
        """
        self.logger = logger
        self.mask_type = mask_type
        self.mask_transform_type = mask_transform_type
        assert self.mask_type in ["internal", "input", "both"]
        assert self.mask_transform_type in ["quantize", "softmax"]
        self.cache = {}

    def remove_cache(self):
        """
        This function removes the cache of the Levin loss. 
        """
        self.logger.info(f"Cache size: {len(self.cache)}")
        self.cache = {}
        self.logger.info("Cache removed.")

    def is_applicable(self, trajectory, actions, start_index):
        """
        This function checks whether an MLP is applicable in a given state. 

        An actor-critic agent is applicable if the sequence of actions it produces matches
        the sequence of actions in the trajectory. Note that we do not consider an
        actor-critic agent if it has less than 2 actions, as it would be equivalent to a 
        primitive action. 
        """
        if len(actions) <= 1 or len(actions) + start_index > len(trajectory):
            return False
        
        for i in range(len(actions)):
            if actions[i] != trajectory[i + start_index][1]:
                return False
        return True

    def _run(self, env: ComboGym, mask: list, agent: PPOAgent, numbers_steps: int):
        """
        This function executes an option, which is given by a mask, an agent, and a number of steps. 

        It runs the masked model of the agent for the specified number of steps and it returns the actions taken for those steps. 
        """
        if self.mask_type == "internal":
            if self.mask_transform_type == "quantize":
                trajectory = agent.run_with_mask(env, mask, numbers_steps)
            elif self.mask_transform_type == "softmax":
                trajectory = agent.run_with_mask_softmax(env, mask, numbers_steps)
        elif self.mask_type == "input":
            if self.mask_transform_type == "quantize":
                trajectory = agent.run_with_input_mask(env, mask, numbers_steps)
            elif self.mask_transform_type == "softmax": 
                trajectory = agent.run_with_input_mask_softmax(env, mask, numbers_steps)
        elif self.mask_type == "both":
            if self.mask_transform_type == "quantize":
                trajectory = agent.run_with_both_masks(env, mask[0], mask[1], numbers_steps)
            elif self.mask_transform_type == "softmax": 
                trajectory = agent.run_with_both_masks_softmax(env, mask[0], mask[1], numbers_steps)

        actions = []
        for _, action in trajectory.get_trajectory():
            actions.append(action)

        return actions

    def loss(self, masks, models, trajectory, number_actions, joint_problem_name_list, problem_str, number_steps):
        """
        This function implements the dynamic programming method from Alikhasi & Lelis (2024). 

        Note that the if-statement with the following code is in a different place. I believe there is
        a bug in the pseudocode of Alikhasi & Lelis (2024).

        M[j] = min(M[j - 1] + 1, M[j])
        """
        t = trajectory.get_trajectory()
        M = np.arange(len(t) + 1)

        for j in range(len(t) + 1):
            if j > 0:
                M[j] = min(M[j - 1] + 1, M[j])
            if j < len(t):
                # the mask being considered for selection cannot be evaluated on the trajectory
                # generated by the MLP trained to solve the problem.
                if joint_problem_name_list[j] == problem_str:
                    continue
                for i in range(len(masks)):
                    # if j + number_steps[i] >= len(t):
                    #     continue
                    if any([joint_problem_name_list[min(j+k, len(t) - 1)] == problem_str for k in range(1, number_steps[i])]):
                        continue
                    actions = self._run(copy.deepcopy(t[j][0]), masks[i], models[i], number_steps[i])

                    if self.is_applicable(t, actions, j):
                        M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)
        uniform_probability = (1/(len(masks) + number_actions)) 
        depth = len(t) + 1
        number_decisions = M[len(t)]

        # use the Levin loss in log space to avoid numerical issues
        log_depth = math.log(depth)
        log_uniform_probability = math.log(uniform_probability)
        return log_depth - number_decisions * log_uniform_probability

    def compute_loss(self, masks, agents, problem_str, trajectories, number_actions, number_steps):
        """
        This function computes the Levin loss of a set of masks (programs). Each mask in the set is 
        what we select as a set of options, according to Alikhasi & Lelis (2024). 

        The loss is computed for a set of trajectories, one for each training task. Instead of taking
        the average loss across all trajectories, in this function we stich all trajectories together
        forming one long trajectory. The function is implemented this way for debugging purposes. 
        Since a mask k extracted from MLP b cannot be evaluated in the trajectory
        b generated, this "leave one out" was more difficult to debug. Stiching all trajectories
        into a single one makes it easier (see chained_trajectory below). 

        We still do not evaluate a mask on the data it was used to generate it. This is achieved
        with the vector joint_problem_name_list below, which is passed to the loss function. 
        """
        chained_trajectory = None
        joint_problem_name_list = []
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
            else:
                chained_trajectory.concat(trajectory)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list
        return self.loss(masks, agents, chained_trajectory, number_actions, joint_problem_name_list, problem_str, number_steps)

    def compute_loss_cached(self, options, trajectory, joint_problem_name_list, problem_str, number_actions):
        t = trajectory.get_trajectory()
        M = np.arange(len(t) + 1)

        for j in range(len(t) + 1):
            if j > 0:
                M[j] = min(M[j - 1] + 1, M[j])
            if j < len(t):
                # the mask being considered for selection cannot be evaluated on the trajectory
                # generated by the MLP trained to solve the problem.
                if joint_problem_name_list[j] == problem_str:
                    continue
                for i in range(len(options)):
                    option = options[i]
                    # if j + number_steps[i] >= len(t):
                    #     continue
                    if any([joint_problem_name_list[min(j+k, len(t) - 1)] == problem_str for k in range(1, option.option_size)]):
                        continue
                    if (option, j) in self.cache:
                        if self.cache[(option, j)][0] == True:
                            actions = self.cache[(option, j)][1]
                            M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)
                    else:
                        actions = self._run(copy.deepcopy(t[j][0]), option.mask, option, option.option_size)
                        self.cache[(option, j)] = (False, actions)
                        if self.is_applicable(t, actions, j):
                            M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)
                            self.cache[(option, j)] = (True, actions)
        uniform_probability = (1/(len(options) + number_actions)) 
        depth = len(t) + 1
        number_decisions = M[len(t)]

        # use the Levin loss in log space to avoid numerical issues
        log_depth = math.log(depth)
        log_uniform_probability = math.log(uniform_probability)
        return log_depth - number_decisions * log_uniform_probability

    def print_output_subpolicy_trajectory(self, options: List[PPOAgent], trajectories, logger):
        """
        This function prints the "behavior" of the options encoded in a set of masks. It will show
        when each option is applicable in different states of the different trajectories. Here is 
        a typical output of this function.

        BL-TR
        Mask:  o0
        001001102102001102001102
        -----000----------------
        --------------000-------
        --------------------000-

        Mask:  o3
        001001102102001102001102
        ------333---------------
        ---------------333------
        ----------------333-----
        ---------------------333
        ----------------------33

        Number of Decisions:  18

        It shows how different masks are used in a given sequence. In the example above, option o0
        is used in the sequence 110, while option o3 is used in some of the occurrences of 102. 
        """
        for problem, trajectory in trajectories.items():  
            logger.info(f"Option Occurrences in {problem}")

            mask_usage = {}
            t = trajectory.get_trajectory()
            M = np.arange(len(t) + 1)

            for j in range(len(t) + 1):
                if j > 0:
                    if M[j - 1] + 1 < M[j]:
                        M[j] = M[j - 1] + 1

                if j < len(t):
                    for i in range(len(options)):

                        if options[i].problem_id == problem:
                            continue

                        actions = self._run(copy.deepcopy(t[j][0]), options[i].mask, options[i], options[i].option_size)

                        if self.is_applicable(t, actions, j):
                            M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)

                            if isinstance(options[i].mask, torch.Tensor):
                                mask_name = 'o' + str(i) + "-" + str(options[i].mask.detach().cpu().numpy())
                            else:
                                mask_name = 'o' + str(i) + "-" + str(options[i].mask[0])
                                mask_name += 'o' + str(i) + "-" + str(options[i].mask[1])

                            if mask_name not in mask_usage:
                                mask_usage[mask_name] = []

                            usage = ['-' for _ in range(len(t))]
                            for k in range(j, j+len(actions)):
                                usage[k] = str(i)
                            mask_usage[mask_name].append(usage)
            
            for mask, matrix in mask_usage.items():
                logger.info(f'Mask: {mask}')
                buffer = "\n"
                for _, action in t:
                    buffer += str(action)
                buffer += "\n"
                for use in matrix:
                    for v in use:
                        buffer += str(v)
                    buffer += "\n"
                logger.info(buffer)
            logger.info(f'Number of Decisions:  {M[len(t)]}')

    def evaluate_on_each_cell(self, options: List[PPOAgent], trajectories: dict, problem_test, args, seed: int, logger=None):
        """
        This test is to see for each cell, options will give which sequence of actions
        """
        def _display_options(action_seq, game_width, indent=24):
            buffer = "\n"
            for i in range(game_width):
                for j in range(game_width):
                    if env.is_over(loc=(i,j)):
                        buffer += ("Goal" + " " * 6)
                        continue
                    buffer += (",".join(map(str, action_seq[(i,j)])) + " " * indent)[:indent]
                buffer += "\n"
            logger.info(buffer)

        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            env = get_training_tasks_simplecross(args.game_width, seed=seed)
            directions = ["R", "D", "L", "U"]
            game_width = 7
        elif args.env_id == "ComboGrid":
            env = ComboGym(args.game_width, args.game_width, problem_test)
            directions = ["NA"]
            game_width = args.game_width
        else:
            raise NotImplementedError
        
        for problem, t in trajectories.items():
            actions = t.get_action_sequence()
            t_str = [DIRECTIONS[tuple(actions[i:i+3])] for i in range(0, t.get_length(), 3)]
            print(f"Problem {problem}, model trajectory: {t_str}, {t.get_logits_sequence()}")
        
        for idx, agent in enumerate(options):
            # Evaluating the performance of options
            logger.info(f"\n {idx} Option: {agent.mask.cpu().numpy()} {agent.problem_id}, {agent.extra_info}")
            for direction in directions:
                logger.info(f"Direction: {direction}")
                action_seq = {}
                for i in range(game_width):
                    for j in range(game_width):    
                        env.reset(init_loc=(i,j), init_dir=direction)  
                        if env.is_over(loc=(i,j)):
                            continue
                                              
                        actions = self._run(env, agent.mask, agent, agent.option_size)
                        action_seq[(i,j)] = actions

                logger.info("Option Outputs:")
                _display_options(action_seq, game_width)

                # Evaluating the performance of original agents
                action_seq = {}
                for i in range(game_width):
                    for j in range(game_width):    
                        if env.is_over(loc=(i,j)):
                            continue
                        env.reset(init_loc=(i,j), init_dir=direction)
                        trajectory = agent.run(env, length_cap=agent.option_size - 1)
                        actions = trajectory.get_action_sequence()
                        action_seq[(i,j)] = actions

                logger.info("Original Agent's Outputs:")
                _display_options(action_seq, game_width)
        logger.info("#### ### ###\n")


class LogitsLossActorCritic(LevinLossActorCritic): 
    def __init__(self, logger):
        self.logger = logger

    def _run_for_logits(self, env, mask, agent, numbers_steps):
        """
        This function executes an option, which is given by a mask, an agent, and a number of steps. 

        It runs the masked model of the agent for the specified number of steps and it returns the actions taken for those steps. 
        """
        trajectory = agent.run_with_mask(env, mask, numbers_steps)

        actions = []
        logits_ls = []
        for ( _, action), logits in zip(trajectory.get_trajectory(), trajectory.get_logits_sequence()):
            actions.append(action)
            logits_ls.append(logits)

        return actions, logits_ls

    def loss(self, masks, models, trajectory, number_actions, joint_problem_name_list, problem_str, number_steps, cross_entropy_clip_coef=50):
        """
        This function implements the dynamic programming method from Alikhasi & Lelis (2024). 

        Note that the if-statement with the following code is in a different place. I believe there is
        a bug in the pseudocode of Alikhasi & Lelis (2024).

        M[j] = min(M[j - 1] + 1, M[j])
        """
        t = trajectory.get_trajectory()
        M = np.arange(len(t) + 1)

        mae_loss = torch.nn.L1Loss()
        losses_dist = {i:[] for i in number_steps}
        steps_dist = {i:[] for i in number_steps}
                            
        for j in range(len(t) + 1):
            if j > 0:
                M[j] = min(M[j - 1] + 1, M[j])
            if j < len(t):
                for i in range(len(masks)):
                    # the mask being considered for selection cannot be evaluated on the trajectory
                    # generated by the MLP trained to solve the problem.
                    if joint_problem_name_list[j] == problem_str or j + number_steps[i] >= len(t):
                        continue
                    actions, logits_ls = self._run_for_logits(copy.deepcopy(t[j][0]), masks[i], models[i], number_steps[i])

                    if self.is_applicable(t, actions, j):
                        loss = 0
                        n_actions = len(actions)
                        for logits, target_logits in zip(logits_ls, trajectory.get_logits_sequence()[j:j+n_actions]):
                            T = torch.e**2
                            probs = F.softmax(logits/T, dim=-1)
                            target_probs = F.softmax(target_logits/T, dim=-1)
                            l = mae_loss(probs, target_probs)

                            # l = torch.sum(-F.softmax(target_logits, dim=-1) * F.log_softmax(logits, dim=-1))
                            # self.logger.info(f"cross entropy sample loss: {l}")
                            losses_dist[number_steps[i]].append(l.item())

                            loss += l / n_actions
                        
                        # self.logger.info(f"average loss: {loss}")
                        # n_steps = int(1 + loss/cross_entropy_clip_coef * (n_actions - 1))
                        n_steps = int(1 + loss * (n_actions - 1))
                        steps_dist[number_steps[i]].append(n_steps)
                        # self.logger.info(f"n_steps: {n_steps}")
                        M[j + n_actions] = min(M[j + n_actions], M[j] + n_steps)
                        # utils.logger_flush(self.logger)
                    
        uniform_probability = (1/(len(masks) + number_actions)) 
        depth = len(t) + 1
        number_decisions = M[len(t)]

        # Monitoring some probability properties of the loss function
        # for i in number_steps:
        #     self.logger.info(f"\n Max steps {i}: Logits Loss Distribution: {Series(losses_dist[i], dtype=np.float64).describe()}")
        #     self.logger.info(f"\n Max steps {i}: Steps Distribution: {Series(steps_dist[i], dtype=np.float64).describe()}")

        # use the Levin loss in log space to avoid numerical issues
        log_depth = math.log(depth)
        log_uniform_probability = math.log(uniform_probability)
        return log_depth - number_decisions * log_uniform_probability