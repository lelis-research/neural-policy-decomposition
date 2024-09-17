import copy
import itertools
import math
import random
import torch
import time
import numpy as np
import gymnasium as gym
from typing import List
from environments_combogrid import Game
from agents.policy_guided_agent import PPOAgent
from environments_combogrid_gym import ComboGym, make_env
from utils.utils import timing_decorator
from torch.distributions.categorical import Categorical
from utils.utils import timing_decorator, get_ppo_model_file_name
from ppo import Args, train_ppo
from torch.utils.tensorboard import SummaryWriter


class LevinLossActorCritic:
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

    def _run(self, env, mask, agent, numbers_steps):
        """
        This function executes an option, which is given by a mask, an agent, and a number of steps. 

        It runs the masked model of the agent for the specified number of steps and it returns the actions taken for those steps. 
        """
        trajectory = agent.run_with_mask(env, mask, numbers_steps)

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
                for i in range(len(masks)):
                    # the mask being considered for selection cannot be evaluated on the trajectory
                    # generated by the MLP trained to solve the problem.
                    if joint_problem_name_list[j] == problem_str:
                        continue
                    actions = self._run(copy.deepcopy(t[j][0]), masks[i], models[i], number_steps)

                    if self.is_applicable(t, actions, j):
                        M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)
        uniform_probability = (1/(len(masks) + number_actions)) 
        depth = len(t) + 1
        number_decisions = M[len(t)]

        # use the Levin loss in log space to avoid numerical issues
        log_depth = math.log(depth)
        log_uniform_probability = math.log(uniform_probability)
        return log_depth - number_decisions * log_uniform_probability

    def compute_loss(self, masks, models, problem_str, trajectories, number_actions, number_steps):
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
                chained_trajectory._sequence = chained_trajectory._sequence + copy.deepcopy(trajectory._sequence)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list
        return self.loss(masks, models, chained_trajectory, number_actions, joint_problem_name_list, problem_str, number_steps)

    def print_output_subpolicy_trajectory(self, models, masks, masks_problems, trajectories, number_steps):
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
            print(problem)

            mask_usage = {}
            t = trajectory.get_trajectory()
            M = np.arange(len(t) + 1)

            for j in range(len(t) + 1):
                if j > 0:
                    if M[j - 1] + 1 < M[j]:
                        M[j] = M[j - 1] + 1

                if j < len(t):
                    for i in range(len(masks)):

                        if masks_problems[i] == problem:
                            continue

                        actions = self._run(copy.deepcopy(t[j][0]), masks[i], models[i], number_steps)

                        if self.is_applicable(t, actions, j):
                            M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)

                            mask_name = 'o' + str(i) + "-" + str(masks[i].cpu().numpy())
                            if mask_name not in mask_usage:
                                mask_usage[mask_name] = []

                            usage = ['-' for _ in range(len(t))]
                            for k in range(j, j+len(actions)):
                                usage[k] = str(i)
                            mask_usage[mask_name].append(usage)

            for mask, matrix in mask_usage.items():
                print('Mask: ', mask)
                for _, action in t:
                    print(action, end="")
                print()
                for use in matrix:
                    for v in use:
                        print(v, end='')
                    print()
                print()
            print('Number of Decisions: ',  M[len(t)])

    def evaluate_on_each_cell(self, test_agents: List[PPOAgent], masks, trained_problems, problem_test, game_width, label=""):
        """
        This test is to see for each cell, options will give which sequence of actions
        """
        env = ComboGym(game_width, game_width, problem_test)
        for agent, idx, mask, trained_problem in zip(test_agents, range(len(test_agents)), masks, trained_problems):
            # Evaluating the performance of options
            print("\n",label, idx, "Option:", mask.cpu().numpy(), trained_problem)
            options = {}
            for i in range(game_width):
                for j in range(game_width):    
                    if env.is_over(loc=(i,j)):
                        continue
                    env.reset(init_loc=(i,j))
                    trajectory = agent.run_with_mask(env, mask, max_size_sequence=3)
                    actions = trajectory.get_action_sequence()
                    options[(i,j)] = actions
            state = trajectory.get_state_sequence()[0]

            print("Option Outputs:")
            for i in range(game_width):
                for j in range(game_width):
                    if env.is_over(loc=(i,j)):
                        continue
                    print(options[(i,j)], end=" ")
                print()
            print(state.represent_options(options))

            # Evaluating the performance of original agents
            options = {}
            for i in range(game_width):
                for j in range(game_width):    
                    if env.is_over(loc=(i,j)):
                        continue
                    env.reset(init_loc=(i,j))
                    trajectory = agent.run(env, length_cap=2)
                    actions = trajectory.get_action_sequence()
                    options[(i,j)] = actions
            state = trajectory.get_state_sequence()[0]

            print("Original Agent's Outputs:")
            for i in range(game_width):
                for j in range(game_width):
                    if env.is_over(loc=(i,j)):
                        continue
                    print(options[(i,j)], end=" ")
                print()
            print(state.represent_options(options))
        print("#### ### ###\n")


def load_trajectories(problems, hidden_size, game_width, l1_lambda=None, num_envs=4, verbose=False):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    for problem in problems:
        model_file = get_ppo_model_file_name(hidden_size=hidden_size, game_width=game_width, problem=problem, l1_lambda=l1_lambda)
        if verbose:
            print(f"Loading Trajectories from {model_file} ...")
        env = ComboGym(rows=game_width, columns=game_width, problem=problem)
        agent = PPOAgent(env, hidden_size=hidden_size) # TODO: move to cuda
        
        agent.load_state_dict(torch.load(model_file))

        trajectory = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

    return trajectories


def evaluate_all_masks_for_ppo_model(masks, selected_models_of_masks, model, problem, trajectories, number_actions, number_iterations, hidden_size):
    """
    Function that evaluates all masks for a given model. It returns the best mask (the one that minimizes the Levin loss)
    for the current set of selected masks. It also returns the Levin loss of the best mask. 
    """
    values = [-1, 0, 1]

    best_mask = None
    best_value = None
    loss = LevinLossActorCritic()

    combinations = itertools.product(values, repeat=hidden_size)

    for value in combinations:
        current_mask = torch.tensor(value, dtype=torch.int8).view(1, -1)
        
        value = loss.compute_loss(masks + [current_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_iterations)

        if best_mask is None or value < best_value:
            best_value = value
            best_mask = copy.deepcopy(current_mask)
            print(best_mask, best_value)
                            
    return best_mask, best_value


@timing_decorator
def evaluate_all_masks_levin_loss():
    """
    This function implements the greedy approach for selecting masks (options) from Alikhasi and Lelis (2024).
    This method evaluates all possible masks of a given model and adds to the pool of options the one that minimizes
    the Levin loss. This process is repeated while we can minimize the Levin loss. 

    This method should only be used with small neural networks, as there are 3^n masks, where n is the number of neurons
    in the hidden layer. 
    """
    hidden_size = 32
    number_iterations = 3
    game_width = 5
    number_actions = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

    params = {
        'hidden_size': hidden_size,
        'number_iterations': number_iterations,
        'game_width': game_width,
        'number_actions': number_actions,
        'problems': problems
    }

    print("Parameters:")
    for key, value in params.items():
        print(f"- {key}: {value}")

    trajectories = load_trajectories(problems, hidden_size, game_width)
    

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic()

    selected_masks = []
    selected_models_of_masks = []
    selected_options_problem = []

    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask = None
        model_best_mask = None
        problem_mask = None

        for problem in problems:
            print('Problem: ', problem)
            model_file = get_ppo_model_file_name(hidden_size=hidden_size, game_width=game_width, problem=problem)
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            agent = PPOAgent(env, hidden_size=hidden_size)
            agent.load_state_dict(torch.load(model_file))

            mask, levin_loss = evaluate_all_masks_for_ppo_model(selected_masks, selected_models_of_masks, agent, problem, trajectories, number_actions, number_iterations, hidden_size)

            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask = mask
                model_best_mask = agent
                problem_mask = problem

                print('Best Loss so far: ', best_loss, problem)

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting automata
        selected_masks.append(best_mask)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, number_actions, number_iterations)

        print("Levin loss of the current set: ", best_loss)

    # remove the last automaton added
    num_options = len(selected_masks)
    selected_masks = selected_masks[0:num_options - 1]
    selected_models_of_masks = selected_models_of_masks[:num_options - 1]

    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, number_iterations)

    # printing selected options
    for i in range(len(selected_masks)):
        print(selected_masks[i])

    print("Testing on each grid cell")
    for problem in problems:
        print("Testing...", problem)
        loss.evaluate_on_each_cell(selected_models_of_masks, selected_masks, problem, game_width)


def hill_climbing(masks, selected_models_of_masks, model, problem, trajectories, number_actions, number_iterations, number_restarts, hidden_size):
    """
    Performs Hill Climbing in the mask space for a given agent. Note that when computing the loss of a mask (option), 
    we pass the name of the problem in which the mask is used. That way, we do not evaluate an option on the problem in 
    which the option's model was trained. 

    Larger number of restarts will result in computationally more expensive searches, with the possibility of finding 
    a mask that better optimizes the Levin loss. 
    """
    best_mask = None
    values = [-1, 0, 1]
    best_overall = None
    best_value_overall = None
    loss = LevinLossActorCritic()

    for i in range(number_restarts):        
        value = random.choices(values, k=hidden_size)
        current_mask = torch.tensor(value, dtype=torch.int8).view(1, -1)

        best_value = loss.compute_loss(masks + [current_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_iterations)

        while True:
            made_progress = False
            for i in range(len(current_mask)):
                modifiable_current_mask = copy.deepcopy(current_mask)
                for v in values:
                    if v == current_mask[0][i]:
                        continue
                    modifiable_current_mask[0][i] = v
                    eval_value = loss.compute_loss(masks + [modifiable_current_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_iterations)

                    if best_mask is None or eval_value < best_value:
                        best_value = eval_value
                        best_mask = copy.deepcopy(modifiable_current_mask)

                        made_progress = True
                    
                current_mask = copy.deepcopy(best_mask)

            if not made_progress:
                break
        
        if best_overall is None or best_value < best_value_overall:
            best_overall = copy.deepcopy(best_mask)
            best_value_overall = best_value

            print('Best Mask Overall: ', best_overall, best_value_overall)
    return best_overall, best_value_overall


@timing_decorator
def hill_climbing_mask_space_training_data_levin_loss():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    hidden_size = 64
    number_iterations = 3
    game_width = 5
    number_restarts = 400
    number_actions = 3
    l1_lambda = 0
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

    params = {
        'hidden_size': hidden_size,
        'number_iterations': number_iterations,
        'game_width': game_width,
        'number_restarts': number_restarts,
        'number_actions': number_actions,
        'l1_lambda': l1_lambda,
        'problems': problems
    }

    print("Parameters:")
    for key, value in params.items():
        print(f"- {key}: {value}")

    trajectories = load_trajectories(problems, hidden_size, game_width, l1_lambda=l1_lambda)

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic()

    selected_masks = []
    selected_models_of_masks = []
    selected_options_problem = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask = None
        model_best_mask = None
        problem_mask = None

        for problem in problems:
            print('Problem: ', problem)
            model_file = get_ppo_model_file_name(hidden_size=hidden_size, game_width=game_width, problem=problem, l1_lambda=l1_lambda)
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            agent = PPOAgent(env, hidden_size=hidden_size)
            agent.load_state_dict(torch.load(model_file))

            mask, levin_loss = hill_climbing(selected_masks, selected_models_of_masks, agent, problem, trajectories, number_actions, number_iterations, number_restarts, hidden_size)

            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask = mask
                model_best_mask = agent
                problem_mask = problem
        print()

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, number_actions, number_iterations)

        print("Levin loss of the current set: ", best_loss)

    # remove the last automaton added
    num_options = len(selected_masks)
    selected_masks = selected_masks[0:num_options - 1]
    selected_models_of_masks = selected_models_of_masks[:num_options - 1]
    for model, mask in zip(selected_models_of_masks, selected_masks):
        model.set_mask(mask)

    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, number_iterations)

    # printing selected options
    for i in range(len(selected_masks)):
        print(selected_masks[i])

    print("Testing on each grid cell")
    for problem in problems:
        print("Testing...", problem)
        loss.evaluate_on_each_cell(selected_models_of_masks, selected_masks, selected_options_problem, problem, game_width)

    for problem in problems:
        print("Retraining on", problem)
        train_extended_ppo(selected_models_of_masks, problem, params)


def train_extended_ppo(options, problem, params):
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    game_width = params['game_width']
    hidden_size = params['hidden_size']
    l1_lambda = params['l1_lambda']
    number_restarts = params['number_restarts']
    number_actions = params['number_actions']

    model_file_name = get_ppo_model_file_name(tag="extended",
                                              problem=problem,
                                              game_width=game_width,
                                              hidden_size=hidden_size,
                                              l1_lambda=l1_lambda)
    
    envs = envs = gym.vector.SyncVectorEnv(
        [make_env(rows=game_width, columns=game_width, problem=problem, options=options) for _ in range(args.num_envs)],
    )    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    exp_name = f"PPO_{problem}-{hidden_size}_{game_width}_{l1_lambda}_{number_restarts}_{number_actions}"
    run_name = f"{exp_name}__{args.seed}__{int(time.time())}_retrained"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    problem_setting = {"problem": problem}
    problem_setting.update(params)
    problem_setting.update({f"option{i}":option.mask.tolist() for i, option in enumerate(options)})
    writer.add_text(
        "problem_setting",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in problem_setting.items()])),)

    train_ppo(envs=envs, 
              args=args, 
              hidden_size=hidden_size, 
              l1_lambda=l1_lambda, 
              model_file_name=model_file_name, 
              device=device,
              writer=writer)


def main():
    # evaluate_all_masks_levin_loss()
    hill_climbing_mask_space_training_data_levin_loss()

if __name__ == "__main__":
    main()