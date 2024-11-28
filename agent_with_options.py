import copy
import random
import torch
import numpy as np
import pickle
from bisect import insort
import tyro
from args import Args
import wandb

from combo import Game
from model import CustomRNN, CustomRelu, QuantizedRNN

class OrderedLimitedList:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def add(self, item):
        # Check if the item already exists based on its trajectory length
        if any(existing_item.get_trajectory() == item.get_trajectory() for existing_item in self.items):
            return

        insort(self.items, item, key=lambda x: len(x.get_trajectory()))
        
        if len(self.items) > self.capacity:
            self.items.pop()
        
    def get_longest_length(self):
        if len(self.items) == 0:
            return np.inf
        return len(self.items[-1].get_trajectory())
    
    def get_shortest_length(self):
        return len(self.items[0].get_trajectory())
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.items, f)
    
    def __repr__(self):
        return repr(self.items)

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
    def __init__(self, options=None):
        self._h = None
        self._epsilon = 0.23
        self._is_recurrent = False
        self._options = options

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
            if self._options is not None:
                if a < len(env.get_actions()):
                    return a, False
                else:
                    is_applicable, actions = self._options[a - len(env.get_actions())].transition(env, apply_actions=True)
                    return a, is_applicable
        return a, False
        
    def run(self, env, model, greedy=False, length_cap=None, verbose=False):
        if greedy:
            self._epsilon = 0.0

        if isinstance(model, CustomRNN) or isinstance(model, QuantizedRNN):
            self._is_recurrent = True

        trajectory = Trajectory()
        current_length = 0
        hidden_states = []
        hidden_states.append(model.init_hidden())
        if verbose: print('Beginning Trajectory')
        while not env.is_over() and current_length < length_cap:
            a, option_applied = self.choose_action(env, model, greedy, verbose)
            if not option_applied and isinstance(a, int):
                trajectory.add_pair(copy.deepcopy(env), a)
                env.apply_action(a)
                hidden_states.append(self._h)
            elif option_applied:
                trajectory.add_pair(copy.deepcopy(env), a)

            if verbose:
                print(env, a)
                print()

            current_length += 1
            if length_cap is not None and current_length > length_cap:
                break        
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory, hidden_states

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
    hidden_size = 32
    game_width = 5
    # args = tyro.cli(Args)
    # problem = args.problem
    # problem = "test"
    problem = "TL-BR"
    # problem = "TR-BL"
    # problem = "BR-TL"
    # problem = "BL-TR"
    env = Game(game_width, game_width, problem, multiple_initial_states=False)

    use_option = False
    if use_option:
        # print("Selecting options...")
        # options = extract_options()
        # print("Option selection complete!")
        with open("options/run2/selected_options_2.pkl", "rb") as file:
            options = pickle.load(file)
        policy_agent = PolicyGuidedAgent(options=options)
        rnn = CustomRNN(21, hidden_size, (len(options)+len(env.get_actions())))
    else:
        rnn = QuantizedRNN(2*(game_width**2) + 3, hidden_size, 3)
        policy_agent = PolicyGuidedAgent()

                
#     wandb.init(
#     project=f"ComboGrid-Naive",
#     entity=None,
#     sync_tensorboard=True,
#     config={
#         "problem" : problem,
#         "hidden_size": hidden_size,
#         "use_option" : use_option,
#         "game_width" : game_width
#     },
#     name=f"{problem}-{game_width}-{hidden_size}-{use_option}",
#     monitor_gym=False,
#     save_code=False,
# )

    # shortest_trajectory_length = np.inf
    shortest_trajectory_length = 40
    best_trajectory = None
    best_model = None
    best_loss = None

    initial_state_trajectory_map = {}
    if problem != "test":
        for _ in range(400):
            for _ in range(1500):
                env = Game(game_width, game_width, problem, multiple_initial_states=True)
                shortest_trajectory_length = 40
                if env in initial_state_trajectory_map:
                    shortest_trajectory_length = len(initial_state_trajectory_map[env].get_trajectory())
                trajectory, _ = policy_agent.run(copy.deepcopy(env), rnn, length_cap=shortest_trajectory_length, verbose=False)

                if env not in initial_state_trajectory_map:
                    initial_state_trajectory_map[env] = trajectory
                elif len(trajectory.get_trajectory()) < len(initial_state_trajectory_map[env].get_trajectory()):
                    initial_state_trajectory_map[env] = trajectory
            loss = 0
            for _ in range(20):
                for state, trajectory in initial_state_trajectory_map.items():
                    print('Trajectory length: ', len(trajectory.get_trajectory()), state._x, state._y)
                    loss += rnn.train(trajectory, l1_coef=0)
            print(loss)
    else:

            for i in range(300):
                for _ in range(1000):
                    env = Game(game_width, game_width, problem, multiple_initial_states=False)
                    trajectory, _ = policy_agent.run(copy.deepcopy(env), rnn, length_cap=shortest_trajectory_length, verbose=False)

                    if len(trajectory.get_trajectory()) < shortest_trajectory_length:
                        shortest_trajectory_length = len(trajectory.get_trajectory())
                        best_trajectory = trajectory

                print('i: Trajectory length: ', i, len(best_trajectory.get_trajectory()))
                for seq in best_trajectory.get_trajectory():
                    print(seq[1], end=" ")
                print()
                for _ in range(30):
                    loss = rnn.train(best_trajectory)
                    print(loss)
                    wandb.log({"loss": loss, "shortest_trajectory_length" : shortest_trajectory_length})
                print()

    # wandb.finish()
    policy_agent._epsilon = 0.0
    env = Game(game_width, game_width, problem)
    trajectory = policy_agent.run(copy.deepcopy(env), rnn, greedy=True, length_cap=15, verbose=True)

    # for i in range(env._rows):
    #     for j in range(env._columns):
    #         env.set_initial_state(i, j)
    #         trajectory = policy_agent.run(copy.deepcopy(env), rnn, greedy=True, length_cap=15, verbose=True)
    #         print()

    torch.save(rnn.state_dict(), 'binary/run2/game-width' + str(game_width) + '-' + problem + '-noreg-' + str(hidden_size) + '-' + str(use_option) + '-1.pth')


if __name__ == "__main__":
    main()
