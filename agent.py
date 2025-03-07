import copy
import random
import torch
import numpy as np
from tqdm import tqdm

from combo import Game
from model import CustomRNN, CustomRelu 

class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence
    
    def get_action_sequence(self):
        return [pair[1] for pair in self._sequence]
    
    def get_state_sequence(self):
        return [pair[0] for pair in self._sequence]

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

def main():
    hidden_size = 64
    game_width = 5
    num_models_per_task = 2

    # problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR", "ML-BR", "ML-TR", "MR-BL", "MR-TL"]
    problems = ["MR-TL"]

    print(f"Parameters: problems:{problems}, hidden_size:{hidden_size}, game_width:{game_width}, num_models_per_task:{num_models_per_task}")

    rnns = {problem:[CustomRelu(game_width**2 * 2 + 3**2, hidden_size, 3) \
                     for _ in range(num_models_per_task)] \
                        for problem in problems}

    policy_agent = PolicyGuidedAgent()

    shortest_trajectory_length = np.inf
    best_trajectory = None

    for problem in tqdm(problems):
        env = Game(game_width, game_width, problem)
        for model_num, rnn in enumerate(rnns[problem]):
            for _ in range(150):
                for _ in range(500):
                    env.reset()
                    trajectory = policy_agent.run(env, rnn, length_cap=shortest_trajectory_length, verbose=False)

                    if len(trajectory.get_trajectory()) < shortest_trajectory_length:
                        shortest_trajectory_length = len(trajectory.get_trajectory())
                        best_trajectory = trajectory

                print('Trajectory length: ', len(best_trajectory.get_trajectory()))
                for _ in range(10):
                    loss = rnn.train(best_trajectory)
                    print(f"loss: {loss.item()}")
                print()

            policy_agent._epsilon = 0.0
            env.reset()
            policy_agent.run(env, rnn, greedy=True, length_cap=None, verbose=True)
            rnn.print_weights()

            env.reset()
            policy_agent.run_with_relu_state(env, rnn)

            torch.save(rnn.state_dict(), f'binary/game-width{game_width}-{problem}-relu-{hidden_size}-model-{model_num}.pth')

if __name__ == "__main__":
    main()