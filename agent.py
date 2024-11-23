import copy
import random
import torch
import numpy as np

from combo import Game
from model import CustomRNN, CustomRelu, GumbelSoftmaxRNN, QuantizedRNN 

class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence

class RandomAgent:
    def run(self, env):
        trajectory = Trajectory()

        while not env.is_over():
            actions = env.get_actions()
            a = actions[random.randint(0, len(actions) - 1)]

            trajectory.add_pair(copy.deepcopy(env.get_observation()), a)
            env.apply_action(a)
        
        return trajectory
    
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
                if verbose:
                    print('Hidden: ', self._h)
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

        if isinstance(model, CustomRNN) or isinstance(model, QuantizedRNN) or isinstance(model, GumbelSoftmaxRNN):
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
                self._h = None
                if verbose: print("End Trajectory \n\n")
                return trajectory, False
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory, True

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

def main():
    hidden_size = 4
    game_width = 3

    # problem = "TL-BR"
    # problem = "TR-BL"
    # problem = "BR-TL"
    problem = "BL-TR"

    while True:
        # rnn = CustomRNN(21, hidden_size, 3)
        rnn = QuantizedRNN(21, hidden_size, 3)
        # rnn = GumbelSoftmaxRNN(21, hidden_size, 3)
        # rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size, 3)

        policy_agent = PolicyGuidedAgent()

        shortest_trajectory_length = np.inf
        best_trajectory = None
        best_model = None
        best_loss = None

        for _ in range(150):
            for _ in range(500):
                env = Game(game_width, game_width, problem)
                trajectory, _ = policy_agent.run(env, rnn, length_cap=shortest_trajectory_length, verbose=False)

                if len(trajectory.get_trajectory()) < shortest_trajectory_length:
                    shortest_trajectory_length = len(trajectory.get_trajectory())
                    best_trajectory = trajectory

            print('Trajectory length: ', len(best_trajectory.get_trajectory()))
            for _ in range(10):
                loss = rnn.train(best_trajectory, l1_coef=0)

                if best_model is None or loss < best_loss:
                    best_loss = loss
                    best_model = copy.deepcopy(rnn)
                print(loss)
            print()

        policy_agent._epsilon = 0.0
        env = Game(game_width, game_width, problem)
        trajectory, _ = policy_agent.run(env, best_model, greedy=True, length_cap=15, verbose=True)

        print('Trajectory length = ', len(trajectory.get_trajectory()))
        if len(trajectory.get_trajectory()) == 12:
            break
    best_model.print_weights()

    # env = Game(game_width, game_width, problem)
    # policy_agent.run_with_relu_state(env, rnn)

    # torch.save(rnn.state_dict(), 'binary/quantized-game-width' + str(game_width) + '-' + problem + '-rnn-noreg-' + str(hidden_size) + '-model.pth')
    torch.save(rnn.state_dict(), 'binary/delayed-quantized-game-width' + str(game_width) + '-' + problem + '-rnn-noreg-' + str(hidden_size) + '-model.pth')

if __name__ == "__main__":
    main()
# for s, a in trajectory.get_trajectory():
#     print(s, a)
# print(trajectory)

# rnn = CustomRNN(18, 256, 3)
# for _ in range(10):
#     loss = rnn.train(trajectory)
#     print(loss)