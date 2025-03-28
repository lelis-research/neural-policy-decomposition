# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)  # Quantize to -1 or 1
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
#TO DO: UPDATE LSTM STRUCTURE TO BE ABLE TO ENALBE/DISABLE FEATURE EXTRACTOR AND INPUT_TO_ACTOR
class LstmAgent(nn.Module):
    def __init__(self, envs, h_size=64):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 512)),
        )
        self.lstm = nn.LSTM(512, h_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # self.actor = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], envs.single_action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], 1), std=1)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n)),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            # print('d: ', d)
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        concatenated = torch.cat((hidden, x), dim=1)
        return self.critic(concatenated)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        concatenated = torch.cat((hidden, x), dim=1)
        logits = self.actor(concatenated)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(concatenated), lstm_state

class GruAgent(nn.Module):
    def __init__(self, envs, h_size=64, feature_extractor=False, greedy=False, quantized=0, actor_layer_size=64, critic_layer_size=64):
        super().__init__()
        self.input_to_actor = True
        self.hidden_size = h_size
        self.greedy = greedy
        self.quantized = quantized
        if feature_extractor:
            self.network = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 512)),
            )
            self.gru = nn.GRU(512, h_size, 1)
        else:
            self.network = IdentityLayer()
            self.gru = nn.GRU(envs.single_observation_space.shape[0], h_size, 1)

        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # self.actor = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], envs.single_action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], 1), std=1)
        if self.input_to_actor:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, envs.single_action_space.n),std=np.sqrt(2)*0.01),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, 1)),
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(h_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, envs.single_action_space.n), std=np.sqrt(2)*0.01),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(h_size , critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, 1)),
            )

    def get_states(self, x, gru_state, done):
        hidden = self.network(x)
        # GRU logic
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, gru_state = self.gru(h.unsqueeze(0), (1.0 - d).view(1, -1, 1) * gru_state)
            if self.quantized == 1:
                gru_state = STEQuantize.apply(gru_state)
                new_hidden += [STEQuantize.apply(h)]
            else:
                new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_value(self, x, gru_state, done):
        if self.input_to_actor:
            hidden, _ = self.get_states(x, gru_state, done)
            concatenated = torch.cat((hidden, x), dim=1)
        else:
            hidden, _ = self.get_states(x, gru_state, done)
            concatenated = hidden
        return self.critic(concatenated)

    def get_action_and_value(self, x, gru_state, done, action=None):
        if len(x.shape) == 1:  # If no batch dimension
            x = x[None, ...]
        if self.input_to_actor:
            hidden, gru_state = self.get_states(x, gru_state, done)
            concatenated = torch.cat((hidden, x), dim=1)
        else: 
            hidden, gru_state = self.get_states(x, gru_state, done)
            concatenated = hidden
        logits = self.actor(concatenated)
        probs = Categorical(logits=logits)
        if action is None:
            if self.greedy:
                action = torch.tensor([torch.argmax(logits[i]).item() for i in range(len(logits))])
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(concatenated), gru_state

    #TODO: Used in other codes, will edit it to work for synced envs too
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)