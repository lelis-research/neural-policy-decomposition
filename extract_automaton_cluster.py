from graphviz import Digraph
from sklearn.cluster import KMeans
import torch
import numpy as np
import copy
from itertools import product
from joblib import parallel_backend     

from combo import Game
from model import CustomRNN

import os

# Set the environment variable before running any parallel code
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

class Mode:
    def __init__(self, index, hiddens):
        self._index = index
        self._hiddens = []
        # for h in hiddens:
        #     self._hiddens.append(torch.tensor(copy.deepcopy(h)).unsqueeze(0))

    def __eq__(self, other):
        if isinstance(other, Mode):
            return self._index == other._index
        return False

    def __hash__(self):
        return hash((self._index))

    def __repr__(self):
        return str(self._index)
        
    def get_index(self):
        return str(self._index)
    
    def has_hidden(self, h):
        # print(any(torch.equal(t, h) for t in self._hiddens))
        return any(torch.equal(t, h) for t in self._hiddens)
    
class Automaton:
    def __init__(self, partitioner, hidden_states, model, mask=None):
        self.partitioner = partitioner
        self.init_h = hidden_states[0]
        self.model = model
        self.mask = mask

        self.partitioner.get_modes(hidden_states)
    
    def transition(self, env, h=None, mask=None):
        actions = []
        if h is None:
            h = self.init_h
        if mask is None:
            mask0 = torch.tensor((-1, -1, -1, -1), dtype=torch.int8).view(1, -1)
            mask1 = torch.tensor((-1, -1), dtype=torch.int8).view(1, -1)
        else:
            mask0 = mask[0]
            mask1 = mask[1]
        for i in range(12):
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            action_prob, h = self.model.masked_forward(x_tensor, h, mask0,mask1)
            predict = self.partitioner.predict_with_threshold(h.detach().numpy().astype(np.float32))
            if predict == -1:
                return True, actions
            action = torch.argmax(action_prob).item()
            env.apply_action(action)
            # print(action)
            actions.append(action)
        return True, actions



class Partitioner:
    def __init__(self, k):
        self._k = k
        self.kmeans = None

    def get_modes(self, hidden_states):
        h_states_numpy = []
        for h in hidden_states:
           h_states_numpy.append(h.squeeze().detach().numpy())
        self.kmeans = KMeans(n_clusters=self._k, random_state = 42)  # Fix seed to get same results
        with parallel_backend('loky', n_jobs=1):
            self.kmeans.fit(h_states_numpy)
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.astype(np.float32)
        distances = self.kmeans.transform(h_states_numpy)
        clusters = self.kmeans.labels_
        modes = []
        # Compute the maximum distance for each cluster
        max_distances = []
        for cluster in range(self.kmeans.n_clusters):
            # Extract the distances of the points assigned to the current cluster
            cluster_distances = distances[clusters == cluster, cluster]
            # Find the maximum distance within the cluster
            max_distances.append(np.max(cluster_distances))
        for cluster_idx in set(clusters):
            cluster_group = [hidden_states[i] for i in range(len(hidden_states)) if clusters[i] == cluster_idx]
            mode = Mode(cluster_idx, cluster_group)
            modes.append(mode)
        self.max_distances = max_distances
        return modes, max_distances
    
    # def get_all_possible_observs(self, size_grid=3, num_actions=3):
    #     observations = []
    #     for i in range(size_grid**2):
    #         for j in range(size_grid**2):
    #             if i!=j:
    #                 for x in range(num_actions):
    #                     agent_pos = np.zeros((size_grid**2), dtype=int)
    #                     goal_pos = np.zeros((size_grid**2), dtype=int)
    #                     last_action = np.zeros((num_actions), dtype=int)
                        
    #                     agent_pos[i] = 1
    #                     goal_pos[j] = 1
    #                     if x == 0:
    #                         observations.append(torch.tensor(np.concatenate((agent_pos.ravel(), last_action.ravel(), goal_pos.ravel())), dtype=torch.float32).view(1, -1))
    #                     last_action[x] = 1
    #                     observations.append(torch.tensor(np.concatenate((agent_pos.ravel(), last_action.ravel(), goal_pos.ravel())), dtype=torch.float32).view(1, -1))
    #     return observations


    # def create_T_mat(self, model, modes, hidden_states, mask=None):
    #     observations = self.get_all_possible_observs()
    #     t = {}
    #     h = None
    #     for mode in modes:
    #         t[mode] = {}
    #         N = [0 for _ in range(len(modes))]
    #         for obs in observations:
    #             for i in range(len(mode._hiddens)):
    #                 if mask:
    #                     _, h = model.masked_forward(obs, mode._hiddens[i], mask[0], mask[1])
    #                 else:
    #                     _, h = model(obs, mode._hiddens[i])
    #                 for mode1 in modes:
    #                     if mode != mode1 and mode1.has_hidden(h):
    #                         N[int(mode1.get_index())-1] += 1

    #             t[mode][tuple(obs.squeeze().detach().numpy())] = np.argmax(N)
    #     return modes, observations, t


    def predict_with_threshold(self, new_values, threshold=None):
    # Predict the closest cluster for each new point
        cluster_labels = self.kmeans.predict(new_values)
        
        if threshold is None:
            threshold = max(self.max_distances)
        threshold = round(threshold, 2)
        
        # Get the distances of new values to their closest cluster centers
        distances = np.round(np.min(self.kmeans.transform(new_values), axis=1), 2)
        # Only assign clusters if the distance is below the threshold
        cluster_labels[distances > threshold] = -1  # Assign -1 for outliers (unassigned)
        
        return cluster_labels


def test_all_masks(rnn, option, size=4):
    problems = ["BL-TR", "TR-BL", "BR-TL", "TL-BR"]
    numbers = [-1, 0, 1]
    all_masks = list(product(numbers, repeat=size))
    all_masks_2 = list(product(numbers, repeat=2))
    hidden_states = {}
    actions = {}
    for problem in problems:
        hidden_states[problem] = {}
        actions[problem] = {}
        env = Game(3,3,problem)
        env.reset()
        for mask_init in all_masks:
            hidden_states[problem][mask_init] = {}
            actions[problem][mask_init] = {}
            mask = torch.tensor(mask_init, dtype=torch.int8).view(1, -1)
            for mask_init_2 in all_masks_2:
                hidden_states[problem][mask_init][mask_init_2] = {}
                actions[problem][mask_init][mask_init_2] = {}
                mask2 = torch.tensor(mask_init_2, dtype=torch.int8).view(1, -1)    
                for i in range(3):
                    for j in range(3):
                        env.reset()
                        h = rnn.init_hidden()
                        env._matrix_unit = np.zeros((3, 3))
                        env._matrix_unit[i][j] = 1
                        act = option.transition(env=env, mask=[mask,mask2])
                        print(act)
                        actions[problem][mask_init][mask_init_2][(i,j)] = act
                        

    np.save("actions_data_4_2_v2.npy", actions, allow_pickle=True)


# def main():
#     problem = "TL-BR"
#     # problem = "TR-BL"
#     # problem = "BR-TL"
#     # problem = "BL-TR"

#     kmeans = Partitioner(4)
#     env = Game(3, 3, problem)
#     rnn = CustomRNN(21, 4, 3)
#     hidden_states = []
#     rnn.load_state_dict(torch.load('binary/game-width3-TL-BR-rnn-noreg-4-2-model.pth'))

#     h = rnn.init_hidden()
#     while not env.is_over():
#         hidden_states.append(h.squeeze().detach().numpy().astype(np.float32))
#         x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
#         prob_actions, h = rnn(x_tensor, h)
#         a = torch.argmax(prob_actions).item()
#         env.apply_action(a)
#     hidden_states.append(h.squeeze().detach().numpy().astype(np.float32))

#     # modes, max_distances = kmeans.get_modes(hidden_states)
#     # modes, obs, table = kmeans.create_T_mat(rnn, modes, hidden_states)
#     option = Automaton(kmeans, hidden_states, rnn, [(-1,0, -1, -1),(1,-1)]) 
#     test_all_masks(rnn,option)
#     # problems = ["BL-TR", "TR-BL", "BR-TL", "TL-BR"]
#     # for problem in problems:
#     #     env = Game(3, 3, problem)
#     #     print(problem)
#     #     for i in range(3):
#     #         for j in range(3):
#     #             env.reset()
#     #             print("\n",i,j)
#     #             env._matrix_unit = np.zeros((3, 3))
#     #             env._matrix_unit[i][j] = 1
#     #             option.transition(env=env)

#     # env = Game(3, 3, "BR-TL")
#     # h = rnn.init_hidden()
#     # print(hidden_states)
#     # for _ in range(12):
#     #     x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
#     #     print(table[modes[kmeans.predict_with_threshold(h.detach().numpy().astype(np.float32))[0]]][tuple(env.get_observation())])
#     #     prob_actions, h = rnn(x_tensor, h)
#     #     print(kmeans.predict_with_threshold(h.detach().numpy().astype(np.float32)))
#     #     a = torch.argmax(prob_actions).item()
#     #     env.apply_action(a)


# if __name__ == "__main__":
#     main()