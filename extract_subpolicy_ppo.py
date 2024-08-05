import copy
import itertools
import math
import random
import torch
import numpy as np
import gymnasium as gym
from agent import PolicyGuidedAgent
from combo import Game
from model import CustomRelu
import torch.nn as nn
import torch.optim as optim
from combo_gym import ComboGym
from utils import timing_decorator
from torch.distributions.categorical import Categorical
from agent import Trajectory

def make_env(*args, **kwargs):
    def thunk():
        env = ComboGym(*args, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(envs.get_observation_space(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(envs.get_observation_space(), 6)),
            nn.Tanh(),
            layer_init(nn.Linear(6, envs.get_action_space()), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def masked_neuron_operation(self, logits, mask):
        """
        Apply a mask to neuron outputs in a layer.

        Parameters:
            x (torch.Tensor): The pre-activation outputs (linear outputs) from neurons.
            mask (torch.Tensor): The mask controlling the operation, where:
                                1 = pass the linear input
                                0 = pass zero,
                                -1 = compute ReLU as usual (part of the program).

        Returns:
            torch.Tensor: The post-masked outputs of the neurons.
        """
        relu_out = torch.relu(logits)
        output = torch.zeros_like(logits)
        output[mask == -1] = relu_out[mask == -1]
        output[mask == 1] = logits[mask == 1]

        return output

    def masked_forward(self, x, mask):
        hidden_logits = self.actor[0](x)
        hidden = self.masked_neuron_operation(hidden_logits, mask)
        hidden_tanh = self.actor[1](hidden)
        output_logits = self.actor[2](hidden_tanh)

        probs = Categorical(logits=output_logits).probs
        
        return probs

    def run(self, env, length_cap=None, verbose=False):

        trajectory = Trajectory()
        current_length = 0
        self.actor.requires_grad = False

        if verbose: print('Beginning Trajectory')
        while not env.is_over():
            o = torch.Tensor(env.get_obs())
            a, _, _, _ = self.get_action_and_value(o)
            trajectory.add_pair(copy.deepcopy(env), a)

            if verbose:
                print(env, a)
                print()

            env.step(a.cpu().numpy().item())

            current_length += 1
            if length_cap is not None and current_length > length_cap:
                break        
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory

    def run_with_relu_state(self, env, model):
        trajectory = Trajectory()
        current_length = 0

        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            prob_actions, hidden_logits = model.forward_and_return_hidden_logits(x_tensor)
            a = torch.argmax(prob_actions).item()
            
            trajectory.add_pair(copy.deepcopy(env), a)
            print(env.get_observation(), a, (hidden_logits >= 0).float().numpy().tolist())
            env.apply_action(a)

            current_length += 1  

        return trajectory
    
    def run_with_mask(self, env, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        while not env.is_over():
            x_tensor = torch.tensor(env.get_obs(), dtype=torch.float32).view(1, -1)
            # mask_tensor = torch.tensor(mask, dtype=torch.int8).view(1, -1)
            prob_actions = self.masked_forward(x_tensor, mask)
            a = torch.argmax(prob_actions).item()
            
            trajectory.add_pair(copy.deepcopy(env), a)
            env.step(a)

            length += 1

            if length >= max_size_sequence:
                return trajectory


        return trajectory

class LevinLossActorCritic:
    def is_applicable(self, trajectory, actions, j):
        """
        This function checks whether an MLP is applicable in a given state. 

        An actor-critic agent is applicable if the sequence of actions it produces matches
        the sequence of actions in the trajectory. Note that we do not consider an
        actor-critic agent if it has less than 2 actions, as it would be equivalent to a 
        primitive action. 
        """
        if len(actions) <= 1 or len(actions) + j > len(trajectory):
            return False
        
        for i in range(len(actions)):
            if actions[i] != trajectory[i + j][1]:
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

                            mask_name = 'o' + str(i)
                            if mask_name not in mask_usage:
                                mask_usage[mask_name] = []

                            usage = ['-' for _ in range(len(t))]
                            for k in range(j, j+len(actions)):
                                usage[k] = str(i)
                            mask_usage[mask_name].append(usage)

            for mask, matrix in mask_usage.items():
                print('Mask: ', mask)
                for _, action in t:
                    print(action.item(), end="")
                print()
                for use in matrix:
                    for v in use:
                        print(v, end='')
                    print()
                print()
            print('Number of Decisions: ',  M[len(t)])

def load_trajectories(problems, hidden_size, game_width, num_envs=4):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    for problem in problems:
        env = ComboGym(rows=game_width, columns=game_width, problem=problem)
        agent = Agent(env) # TODO: move to cuda
        
        agent.load_state_dict(torch.load(f'binary/PPO-{problem}-game-width{game_width}-hidden{hidden_size}_MODEL.pt'))

        trajectory = agent.run(env, verbose=True)
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

def evaluate_all_masks_levin_loss():
    """
    This function implements the greedy approach for selecting masks (options) from Alikhasi and Lelis (2024).
    This method evaluates all possible masks of a given model and adds to the pool of options the one that minimizes
    the Levin loss. This process is repeated while we can minimize the Levin loss. 

    This method should only be used with small neural networks, as there are 3^n masks, where n is the number of neurons
    in the hidden layer. 
    """
    hidden_size = 6
    number_iterations = 3
    game_width = 5
    number_actions = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

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
            env = ComboGym(rows=game_width, columns=game_width, problem=problem)
            agent = Agent(env)
            agent.load_state_dict(torch.load(f'binary/PPO-{problem}-game-width{game_width}-hidden{hidden_size}_MODEL.pt'))

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
    selected_masks = selected_masks[0:len(selected_masks) - 1]

    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, number_iterations)

    # printing selected options
    for i in range(len(selected_masks)):
        print(selected_masks[i])

def hill_climbing(masks, selected_models_of_masks, model, problem, trajectories, number_actions, number_iterations, number_restarts, hidden_size):
    """
    Performs Hill Climbing in the mask space for a given model. Note that when computing the loss of a mask (option), 
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

def hill_climbing_mask_space_training_data_levin_loss():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    hidden_size = 32
    number_iterations = 3
    game_width = 5
    number_restarts = 100
    number_actions = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

    trajectories = load_trajectories(problems, hidden_size, game_width)

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic()

    selected_options = []
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
            rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size, 3)
            rnn.load_state_dict(torch.load('binary/game-width' + str(game_width) + '-' + problem + '-relu-' + str(hidden_size) + '-model.pth'))

            mask, levin_loss = hill_climbing(selected_options, selected_models_of_masks, rnn, problem, trajectories, number_actions, number_iterations, number_restarts, hidden_size)

            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask = mask
                model_best_mask = rnn
                problem_mask = problem
        print()

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_options.append(best_mask)
        selected_models_of_masks.append(model_best_mask)
        selected_options_problem.append(problem_mask)
        best_loss = loss.compute_loss(selected_options, selected_models_of_masks, "", trajectories, number_actions, number_iterations)

        print("Levin loss of the current set: ", best_loss)

    # remove the last automaton added
    selected_options = selected_options[0:len(selected_options) - 1]

    loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_options, selected_options_problem, trajectories, number_iterations)

    # printing selected options
    for i in range(len(selected_options)):
        print(selected_options[i])

def main():
    evaluate_all_masks_levin_loss()
    # hill_climbing_mask_space_training_data_levin_loss()

if __name__ == "__main__":
    main()