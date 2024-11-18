import copy
import random
import torch
import numpy as np

from agent import PolicyGuidedAgent
from combo import Game
from model import CustomRNN, CustomRelu, GumbelSoftmaxRNN, QuantizedRNN 

class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence

def main():
    hidden_size = 6
    game_width = 3

    # problem = "TL-BR"
    # problem = "TR-BL"
    # problem = "BR-TL"
    problem = "BL-TR"

    # rnn = CustomRNN(21, hidden_size, 3)
    rnn = QuantizedRNN(21, hidden_size, 3)
    # rnn = GumbelSoftmaxRNN(21, hidden_size, 3)
    # rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size, 3)

    policy_agent = PolicyGuidedAgent()

    shortest_trajectory_length = np.inf
    best_trajectory = None
    best_model = None
    best_loss = None

    initial_state_trajectory_map = {}

    for _ in range(200):
        for _ in range(500):
            env = Game(game_width, game_width, problem, multiple_initial_states=True)
            shortest_trajectory_length = np.inf
            if env in initial_state_trajectory_map:
                shortest_trajectory_length = len(initial_state_trajectory_map[env].get_trajectory())
            trajectory = policy_agent.run(copy.deepcopy(env), rnn, length_cap=shortest_trajectory_length, verbose=False)

            if env not in initial_state_trajectory_map:
                initial_state_trajectory_map[env] = trajectory
            elif len(trajectory.get_trajectory()) < len(initial_state_trajectory_map[env].get_trajectory()):
                initial_state_trajectory_map[env] = trajectory

        loss = 0
        for state, trajectory in initial_state_trajectory_map.items():
            print('Trajectory length: ', len(trajectory.get_trajectory()), state._x, state._y)
            loss += rnn.train(trajectory, l1_coef=0)
        print(loss)

    policy_agent._epsilon = 0.0
    env = Game(game_width, game_width, problem)

    for i in range(env._rows):
        for j in range(env._columns):
            env.set_initial_state(i, j)
            trajectory = policy_agent.run(copy.deepcopy(env), rnn, greedy=True, length_cap=15, verbose=True)
            print()

    # env = Game(game_width, game_width, problem)
    # policy_agent.run_with_relu_state(env, rnn)

    torch.save(rnn.state_dict(), 'binary/multi-init-quantized-game-width' + str(game_width) + '-' + problem + '-rnn-noreg-' + str(hidden_size) + '-model.pth')
    # torch.save(rnn.state_dict(), 'binary/delayed-quantized-game-width' + str(game_width) + '-' + problem + '-rnn-noreg-' + str(hidden_size) + '-model.pth')

if __name__ == "__main__":
    main()