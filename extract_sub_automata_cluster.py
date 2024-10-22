from sklearn.cluster import KMeans
import torch
import numpy as np
from itertools import combinations
from itertools import product

from extract_automaton_cluster import Automaton, Partitioner
from combo import Game
from model import CustomRNN

import heapq

class SearchNode:
    def __init__(self):
        # set of modes that will be considered while generating the children of the node
        self._open = []
        # the set of modes already considered in the sub-automata that the node represents
        self._closed = set()
        # adjancecy list storing the sub-automaton the node represents
        self._adjacency_list = {}

        self._g = 0

    def __lt__(self, other):
        return self._g < other._g
    
    def __hash__(self):
        return hash(tuple(self._adjacency_list.keys()))
    
    def __eq__(self, other):
        if self._adjacency_list.keys() != other._adjacency_list.keys():
            return False
        
        for key, row in self._adjacency_list.items():
            if row != other._adjacency_list[key]:
                return False
        return True

    def add_connection(self, parent, child):
        if child not in self._closed:
            self._open.append(child)
        
        if parent not in self._adjacency_list:
            self._adjacency_list[parent] = set()
        if child not in self._adjacency_list:
            self._adjacency_list[child] = set()

        self._adjacency_list[parent].add(child)

    def add_initial_mode(self, mode):
        self._open.append(mode)
        self._adjacency_list[mode] = set()

    def copy_for_child(self):
        new_node = SearchNode()
        new_node._g = self._g + 1
        for n in self._closed:
            new_node._closed.add(n)
        for n in self._open:
            new_node._closed.add(n)
        
        for mode, row in self._adjacency_list.items():
            copy_mode = mode.copy()
            new_node._adjacency_list[copy_mode] = set()
            for row_mode in row:
                new_node._adjacency_list[copy_mode].add(row_mode.copy())
        return new_node

class SubAutomataExtractor:
    def __init__(self, automaton, k, width = None):
        self._automaton = automaton
        self._width = width
        self._hidden_states = automaton._hidden_states
        self._k = k

    def generate_children(self, node):
        edges_open_nodes = []
        children = []

        if len(node._open) == 0:
            return children
        for o in node._open:
            current_mode_edges = []
            list_edges = list(self._automaton._adjacency_dict[o])
            if self._width is None:
                width = len(list_edges)
            else:
                width = self._width
            for i in range(0, width + 1):
                current_mode_edges = current_mode_edges + list(combinations(list_edges, i))
            edges_open_nodes.append(current_mode_edges)

        cross_product = list(product(*edges_open_nodes))
        for child_edges in cross_product:
            child = node.copy_for_child()
            for j in range(len(child_edges)):
                for child_mode in child_edges[j]:
                    child.add_connection(node._open[j], child_mode)
            children.append(child)
        return children
    
    def extract_sub_automata(self, print_images=False):
        mapping_names = None
        if print_images:
            mapping_names = self._automaton.get_mapping_names()
            self._automaton.print_image_with_mapping_names('images/full', mapping_names)

        root = SearchNode()
        root.add_initial_mode(self._automaton._initial_mode)
          # print('Initial Mode: ', self._automaton._initial_mode._state, self._automaton._initial_mode._action, self._automaton._initial_mode._h)

        closed = self.bfs(root)
        automata = []

        counter = 1
        for node in closed:
            automaton_from_node = Automaton(Partitioner(self._k), self._hidden_states, self._automaton.model, self._automaton._problem, get_adj_dict=False)
            # automaton_from_node._adjacency_list = node._adjacency_list
            automaton_from_node.add_adj_dict(node._adjacency_list)
            automata.append(automaton_from_node)

            if print_images:
                automaton_from_node.print_image_with_mapping_names('images/sub_' + str(counter), mapping_names)
                counter += 1
        return automata

    def bfs(self, root):
        open = []
        closed = set()
        heapq.heappush(open, root)
        closed.add(root)
        
        while len(open) > 0:
            # if len(open) % 10 == 0:
            #     print('Size of Open: ', len(open))

            node = heapq.heappop(open)
            children = self.generate_children(node)

            for c in children:
                if c not in closed:
                    heapq.heappush(open, c)
                    closed.add(c)

        return closed



def main():
    problem = "TL-BR"
    # problem = "TR-BL"
    # problem = "BR-TL"
    # problem = "BL-TR"

    kmeans = Partitioner(4)
    env = Game(3, 3, problem)
    rnn = CustomRNN(21, 4, 3)
    hidden_states = []
    rnn.load_state_dict(torch.load(f'binary/game-width3-{problem}-rnn-4-2-model.pth', weights_only=True))

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
    # print(option._adjacency_dict)

    sub = SubAutomataExtractor(option, 4)
    sub_automata = sub.extract_sub_automata()
    for i, s in enumerate(sub_automata):
        print(i, ":", s._adjacency_dict)

    env.reset()
    h = rnn.init_hidden()
    for i in range(12):
        x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
        prob_actions, h = rnn(x_tensor, h)
        flag = True
        if sub_automata[36].partitioner.predict_with_threshold(h.detach().numpy().astype(np.float32))[0] != -1:
            actions = sub_automata[36].transition(env)
            print(actions)
            print(env)
                

if __name__ == "__main__":
    main()