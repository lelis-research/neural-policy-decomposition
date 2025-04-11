import itertools
import torch
import copy
from pipelines.option_discovery import regenerate_trajectories
from agents.policy_guided_agent import PPOAgent
from environments.environments_combogrid import ComboGym
from losses import LevinLossActorCritic
from utils.utils import timing_decorator, get_ppo_model_file_name


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

    trajectories = regenerate_trajectories(problems, hidden_size, game_width)
    

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

