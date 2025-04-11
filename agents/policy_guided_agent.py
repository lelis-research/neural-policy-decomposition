import copy
import random
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from typing import Union
import matplotlib.pyplot as plt
from environments.environments_combogrid_gym import ComboGym
from environments.environments_minigrid import MiniGridWrap
from gymnasium.vector import SyncVectorEnv
from torch.distributions.categorical import Categorical
from models.models_mlp import CustomRNN 
from agents.trajectory import Trajectory

class PolicyGuidedAgent:
    def __init__(self):
        self._h = None
        self._epsilon = 0.3
        self._is_recurrent = False

    def choose_action(self, env, model, greedy=False, verbose=False):
        if random.random() < self._epsilon:
            actions = env.get_actions()
            a = actions[random.randint(0, len(actions) - 1)]
        else:
            if self._is_recurrent and self._h == None:
                self._h = model.init_hidden()
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            if self._is_recurrent:
                prob_actions, self._h = model(x_tensor, self._h)
            else:
                prob_actions = model(x_tensor)
            if greedy:
                a = torch.argmax(prob_actions).item()
            else:
                a = torch.multinomial(prob_actions, 1).item()
        return a
        
    def run(self, env, model, greedy=False, length_cap=None, verbose=False):
        if greedy:
            self._epsilon = 0.0

        if isinstance(model, CustomRNN):
            self._is_recurrent = True

        trajectory = Trajectory()
        current_length = 0

        if verbose: print('Beginning Trajectory')
        while not env.is_over():
            a = self.choose_action(env, model, greedy, verbose)
            trajectory.add_pair(copy.deepcopy(env), a)

            if verbose:
                print(env, a)
                print()

            env.apply_action(a)

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
    
    def run_with_mask(self, env, model, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            # mask_tensor = torch.tensor(mask, dtype=torch.int8).view(1, -1)
            prob_actions = model.masked_forward(x_tensor, mask)
            a = torch.argmax(prob_actions).item()
            
            trajectory.add_pair(copy.deepcopy(env), a)
            env.apply_action(a)

            length += 1

            if length >= max_size_sequence:
                return trajectory


        return trajectory


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_sparse_init(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)
    else:
        raise NotImplementedError
    return m


def sparse_init(tensor, sparsity, type='uniform'):

    if tensor.ndimension() == 2:
        fan_out, fan_in = tensor.shape

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for col_idx in range(fan_out):
                row_indices = torch.randperm(fan_in)
                zero_indices = row_indices[:num_zeros]
                tensor[col_idx, zero_indices] = 0
        return tensor

    elif tensor.ndimension() == 4:
        channels_out, channels_in, h, w = tensor.shape
        fan_in, fan_out = channels_in*h*w, channels_out*h*w

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for out_channel_idx in range(channels_out):
                indices = torch.randperm(fan_in)
                zero_indices = indices[:num_zeros]
                tensor[out_channel_idx].reshape(channels_in*h*w)[zero_indices] = 0
        return tensor
    else:
        raise ValueError("Only tensors with 2 or 4 dimensions are supported")


class PPOAgent(nn.Module):
    def __init__(self, envs, hidden_size=6, sparse_init=False, discrete_masks=True):
        super().__init__()
        if isinstance(envs, ComboGym):
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs, MiniGridWrap):
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs, SyncVectorEnv):
            observation_space_size = envs.observation_space.shape[1]
            action_space_size = envs.action_space[0].n.item()
        else:
            raise NotImplementedError
        
        self.hidden_size = hidden_size
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size

        if sparse_init:
            print(f"Sparse initialization: {sparse_init}")
            layer_init_func = layer_sparse_init
        else:
            layer_init_func = layer_init

        self.critic = nn.Sequential(
                layer_init_func(nn.Linear(observation_space_size, 64)),
                nn.Tanh(),
                layer_init_func(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init_func(nn.Linear(64, 1)),
            )
        self.actor = nn.Sequential(
            layer_init_func(nn.Linear(observation_space_size, hidden_size)),
            nn.ReLU(),
            layer_init_func(nn.Linear(hidden_size, action_space_size)),
        )

        # Option attributes
        self.mask = None
        self.mask_transform_type = None
        self.mask_type = None # internal/input/both
        self.option_size = None
        self.problem_id = None
        self.environment_args = None
        self.discrete_masks = discrete_masks
        self.extra_info = {}
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = probs.probs.max(dim=0)[1]
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), logits
    
    def to_option(self, mask, option_size, problem):
        self.mask = mask
        if isinstance(mask, tuple):
            self.mask_type = "both"
            print(mask[0].shape)
            mask_dim = mask[0].shape[0]
        elif mask.shape[1] == self.observation_space_size:
            self.mask_type = "input"
            mask_dim = mask.shape[0]
        elif mask.shape[1] == self.hidden_size:
            self.mask_type = "internal"
            mask_dim = mask.shape[0]
        else:
            raise Exception(f"The mask {mask} is of invalid type")

        self.mask_transform_type = "softmax" if mask_dim == 3 else "quantize"
        self.option_size = option_size
        self.problem_id = problem
    
    def _masked_input_softmax(self, input, mask):
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        
        return (mask[0] * 0) + (mask[1] * 1) + (mask[2] * input)
    
    def _masked_neuron_operation_softmax(self, logits, mask):
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        relu_out = torch.relu(logits)
        
        return (mask[0] * 0) + (mask[1] * logits) + (mask[2] * relu_out)

    def _masked_input(self, input, mask):
        """
        Apply a mask to the input layer.

        Parameters:
            x (torch.Tensor): Input of the model; we assume binary values. 
            mask (torch.Tensor): The mask controlling the operation, where:
                                1 = pass one
                                0 = pass zero
                                -1 = pass the value of the input

        Returns:
            torch.Tensor: The post-masked inputs of the model.
        """
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        
        return mask * (mask - 1) / 2 * input + mask * (mask + 1) * 1 / 2

    def _masked_neuron_operation(self, logits, mask):
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
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        relu_out = torch.relu(logits)
        
        if self.discrete_masks:            
            output = mask * (mask - 1) / 2 * relu_out + mask * (mask + 1) * logits / 2
        else:
            alpha = 10
            mask_neg_one = torch.sigmoid(-alpha * (mask + 1))  # close to 1 when mask <= -1
            mask_pos_one = torch.sigmoid(alpha * (mask - 1))   # close to 1 when mask >= 1
            mask_between = 1 - mask_neg_one - mask_pos_one     # smooth transition for values in between

            # Create output based on differentiable masks
            output = mask_neg_one * relu_out + mask_pos_one * logits + mask_between * 0
        return output

    def _masked_input_forward_softmax(self, x, mask):
        x = self._masked_input_softmax(x, mask)
        hidden_logits = self.actor[0](x)
        hidden_tanh = self.actor[1](hidden_logits)
        output_logits = self.actor[2](hidden_tanh)

        probs = Categorical(logits=output_logits).probs
        
        return probs, output_logits

    def _masked_input_forward(self, x, mask=None):
        if mask is None:
            mask = self.mask
        x = self._masked_input(x, mask)
        hidden_logits = self.actor[0](x)
        hidden_tanh = self.actor[1](hidden_logits)
        output_logits = self.actor[2](hidden_tanh)

        probs = Categorical(logits=output_logits).probs
        
        return probs, output_logits

    def _both_masked_forward_softmax(self, x, input_mask, internal_mask):
        x = self._masked_input_softmax(x, input_mask)
        hidden_logits = self.actor[0](x)
        hidden = self._masked_neuron_operation_softmax(hidden_logits, internal_mask)
        output_logits = self.actor[2](hidden)

        probs = Categorical(logits=output_logits).probs
        
        return probs, output_logits

    def _both_masked_forward(self, x, input_mask, internal_mask):
        x = self._masked_input(x, input_mask)
        hidden_logits = self.actor[0](x)
        hidden = self._masked_neuron_operation(hidden_logits, internal_mask)
        output_logits = self.actor[2](hidden)

        probs = Categorical(logits=output_logits).probs
        
        return probs, output_logits

    def _masked_forward_softmax(self, x, mask=None):
        if mask is None:
            mask = self.mask
        hidden_logits = self.actor[0](x)
        hidden = self._masked_neuron_operation_softmax(hidden_logits, mask)
        output_logits = self.actor[2](hidden)

        probs = Categorical(logits=output_logits).probs
        
        return probs, output_logits

    def _masked_forward(self, x, mask=None):
        if mask is None:
            mask = self.mask
        hidden_logits = self.actor[0](x)
        hidden = self._masked_neuron_operation(hidden_logits, mask)
        output_logits = self.actor[2](hidden)

        probs = Categorical(logits=output_logits).probs
        
        return probs, output_logits

    def run(self, env: Union[ComboGym, MiniGridWrap], length_cap=None, detach_tensors=True, verbose=False, deterministic=True):

        trajectory = Trajectory()
        current_length = 0
        self.actor.requires_grad = False

        o = env.get_observation()
        
        done = False

        if verbose: print('Beginning Trajectory')
        while not done:
            o = torch.tensor(o, dtype=torch.float32)
            a, _, _, _, logits = self.get_action_and_value(o, deterministic=deterministic)
            trajectory.add_pair(copy.deepcopy(env), a.item(), logits, detach=detach_tensors)

            if verbose:
                print(env, a)
                print()

            next_o, _, terminal, truncated, _ = env.step(a.item())
            
            current_length += 1
            if (length_cap is not None and current_length > length_cap) or \
                terminal or truncated:
                done = True     

            o = next_o   
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory
    
    def _get_action_with_both_masks_softmax(self, x_tensor, input_mask, internal_mask):
        prob_actions, logits = self._both_masked_forward_softmax(x_tensor, input_mask, internal_mask)
        a = torch.argmax(prob_actions).item()
        return a, logits

    def _get_action_with_both_masks(self, x_tensor, input_mask, internal_mask):
        prob_actions, logits = self._both_masked_forward(x_tensor, input_mask, internal_mask)
        a = torch.argmax(prob_actions).item()
        return a, logits

    def _get_action_with_input_mask_softmax(self, x_tensor, mask=None):
        prob_actions, logits = self._masked_input_forward_softmax(x_tensor, mask)
        a = torch.argmax(prob_actions).item()
        return a, logits

    def _get_action_with_input_mask(self, x_tensor, mask=None):
        prob_actions, logits = self._masked_input_forward(x_tensor, mask)
        a = torch.argmax(prob_actions).item()
        return a, logits

    def _get_action_with_mask(self, x_tensor, mask=None):
        prob_actions, logits = self._masked_forward(x_tensor, mask)
        a = torch.argmax(prob_actions).item()
        return a, logits
    
    def _get_action_with_mask_softmax(self, x_tensor, mask=None):
        prob_actions, logits = self._masked_forward_softmax(x_tensor, mask)
        a = torch.argmax(prob_actions).item()
        return a, logits

    def get_action_with_mask(self, x_tensor, mask=None):
        if mask == None:
            mask = self.mask
        if self.mask_type == "internal":
            if self.mask_transform_type == "softmax":
                return self._get_action_with_mask_softmax(x_tensor, mask)
            else:
                return self._get_action_with_mask(x_tensor, mask)
        elif self.mask_type == "input":
            if self.mask_transform_type == "softmax":
                return self._get_action_with_input_mask_softmax(x_tensor, mask)
            else:
                return self._get_action_with_input_mask(x_tensor, mask)
        if self.mask_type == "both":
            if self.mask_transform_type == "softmax":
                return self._get_action_with_both_masks_softmax(x_tensor, input_mask=mask[0], internal_mask=mask[1])
            else:
                return self._get_action_with_both_masks(x_tensor, input_mask=mask[0], internal_mask=mask[1])

    def run_with_mask_softmax(self, envs, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_mask_softmax(x_tensor, mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_mask_softmax(x_tensor, mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory
    
    def run_with_mask(self, envs, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_mask(x_tensor, mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_mask(x_tensor, mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory
    
    def run_with_both_masks_softmax(self, envs, input_mask, internal_mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_both_masks_softmax(x_tensor, input_mask, internal_mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_both_masks_softmax(x_tensor, input_mask, internal_mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory

    def run_with_both_masks(self, envs, input_mask, internal_mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_both_masks(x_tensor, input_mask, internal_mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_both_masks(x_tensor, input_mask, internal_mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory

    def run_with_input_mask_softmax(self, envs, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_input_mask_softmax(x_tensor, mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_input_mask_softmax(x_tensor, mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory

    def run_with_input_mask(self, envs, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_input_mask(x_tensor, mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_input_mask(x_tensor, mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory
