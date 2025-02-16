from graphviz import Digraph
from sklearn.cluster import KMeans
import torch
import numpy as np
import copy
from itertools import product
from joblib import parallel_backend     

from envs.combogrid import ComboGridEnv
from models.model import CustomRNN

import os

# Set the environment variable before running any parallel code
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

class Mode:
    def __init__(self, index, center):
        self._index = index
        if not torch.is_tensor(center):
            center = torch.from_numpy(center).view(1, -1)
        self._center = center

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
    def copy(self):
        return Mode(self._index, self._center)
class Automaton:
    def __init__(self, partitioner, hidden_states, model, problem, mask=None, get_adj_dict=True):
        self.partitioner = partitioner
        self.init_h = hidden_states[0]
        self._hidden_states = hidden_states
        self.model = model
        self.mask = mask
        if mask is None:
            self.mask = [(-1, -1, -1, -1), (-1, -1)]
        self._problem = problem
        self._modes, _  = self.partitioner.get_modes(hidden_states)
        self._initial_mode = self._modes[self.partitioner.kmeans.predict(self.init_h.detach().numpy().astype(np.float32))[0]]
        self._adjacency_dict = {}
        if get_adj_dict:
            self._adjacency_dict = self.get_adj_dict()

    def get_adj_dict(self):
        env = ComboGridEnv(3, 3, self._problem)
        h = self.init_h
        adj_dict = {}
        current_mode = self._modes[self.partitioner.predict_with_threshold(h.detach().numpy().astype(np.float32))[0]]
        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            action_prob, h = self.model(x_tensor, h)
            predict = self.partitioner.predict_with_threshold(h.detach().numpy().astype(np.float32))[0]
            # h = self.modes[predict]._center
            if current_mode in adj_dict:
                adj_dict[current_mode].add(self._modes[predict])
            else:
                adj_dict[current_mode] = {self._modes[predict]}
            current_mode = self._modes[predict]
            action = torch.argmax(action_prob).item()
            env.apply_action(action)
        return adj_dict
    
    def add_adj_dict(self, adj_dict):
        for mode, connections in adj_dict.items():
            if mode not in self._adjacency_dict:
                self._adjacency_dict[mode] = set()
            for con in connections:
                self._adjacency_dict[mode].add(con)

    
    def transition(self, env, h=None, mask=None):
        self._modes_used = self._adjacency_dict.keys()
        actions = []
        if h is None:
            h = self.init_h
        current_mode = self._modes[self.partitioner.predict_with_threshold(h.detach().numpy().astype(np.float32))[0]]
        if mask is None:
            mask0 = torch.tensor(self.mask[0], dtype=torch.int8).view(1, -1)
            mask1 = torch.tensor(self.mask[1], dtype=torch.int8).view(1, -1)
        else:
            mask0 = mask[0]
            mask1 = mask[1]
        for i in range(12):
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            action_prob, h = self.model.masked_forward(x_tensor, h, mask0,mask1)
            predict = self.partitioner.predict_with_threshold(h.detach().numpy().astype(np.float32))[0]
            if predict == -1 or self._modes[predict] not in self._modes_used:
                return True, actions
            if current_mode == self._modes[2] and predict == 2:
                h = self._modes[0]._center 
            action = torch.argmax(action_prob).item()
            env.apply_action(action)
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
            # cluster_group = [hidden_states[i] for i in range(len(hidden_states)) if clusters[i] == cluster_idx]
            mode = Mode(cluster_idx, self.kmeans.cluster_centers_[cluster_idx])
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
        env = ComboGridEnv(3,3,problem)
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


def main():
    problem = "TL-BR"
    # problem = "TR-BL"
    # problem = "BR-TL"
    # problem = "BL-TR"

    kmeans = Partitioner(4)
    env = ComboGridEnv(3, 3, problem)
    rnn = CustomRNN(21, 4, 3)
    hidden_states = []
    rnn.load_state_dict(torch.load(f'binary/game-width3-{problem}-rnn-4-2-model.pth'))

    h = rnn.init_hidden()
    while not env.is_over():
        hidden_states.append(h)
        x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
        prob_actions, h = rnn(x_tensor, h)
        a = torch.argmax(prob_actions).item()
        env.apply_action(a)
    hidden_states.append(h)

    # modes, max_distances = kmeans.get_modes(hidden_states)
    # modes, obs, table = kmeans.create_T_mat(rnn, modes, hidden_states)
    option = Automaton(kmeans, hidden_states, rnn, problem) 
    print(option.adjacency_dict)
    exit()
    # test_all_masks(rnn,option)
    problems = ["BL-TR", "TR-BL", "BR-TL", "TL-BR"]
    for problem in problems:
        env = ComboGridEnv(3, 3, problem)
        print(problem)
        for i in range(3):
            for j in range(3):
                env.reset()
                print("\n",i,j)
                env._matrix_unit = np.zeros((3, 3))
                env._matrix_unit[i][j] = 1
                _, actions = option.transition(env=env, mask=[torch.tensor((-1, -1, 1, -1), dtype=torch.int8).view(1, -1), torch.tensor((-1, -1), dtype=torch.int8).view(1, -1)])
                print(actions)

#     # env = ComboGridEnv(3, 3, "BR-TL")
#     # h = rnn.init_hidden()
#     # print(hidden_states)
#     # for _ in range(12):
#     #     x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
#     #     print(table[modes[kmeans.predict_with_threshold(h.detach().numpy().astype(np.float32))[0]]][tuple(env.get_observation())])
#     #     prob_actions, h = rnn(x_tensor, h)
#     #     print(kmeans.predict_with_threshold(h.detach().numpy().astype(np.float32)))
#     #     a = torch.argmax(prob_actions).item()
#     #     env.apply_action(a)


if __name__ == "__main__":
    main()