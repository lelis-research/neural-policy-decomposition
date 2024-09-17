import torch
import numpy as np
from tqdm import tqdm

from models.models_mlp import CustomRNN, CustomRelu 
from agents.policy_guided_agent import PolicyGuidedAgent
from environemnts.environments_combogrid import Game
from utils.utils import timing_decorator


@timing_decorator
def main():
    hidden_size = 64
    game_width = 5
    num_models_per_task = 2

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