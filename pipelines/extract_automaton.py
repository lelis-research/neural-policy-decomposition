import copy
import torch

from graphviz import Digraph

class Mode:
    def __init__(self, state, h, action=None):
        self._state = state
        self._h = h
        self._action = action

    def __eq__(self, other):
        if isinstance(other, Mode):
            return self._state == other._state and self._action == other._action
        return False

    def __hash__(self):
        return hash((self._state, self._action))
    
    def __repr__(self):
        return str(self._state) + " " + str(self._action)
        # return str(self._state[4]) + " " + str(self._action)
        # return str(self._state) + " " + str(self._action) + " " + str(self._h)

    def copy(self):
        return Mode(self._state, self._h, self._action)

class Automaton:
    def __init__(self, modes, model, initial_mode, k):
        self._modes = modes
        self._model = model
        self._initial_mode = initial_mode
        self._k = k
        self._adjacency_list = {}
        self._modes_dict = {}
        # automata might represent an infinite loop; max_steps will break the program
        self._max_steps = 20
        for mode in self._modes:
            self._adjacency_list[mode] = set()
            self._modes_dict[mode] = mode

    def get_size(self):
        return len(self._adjacency_list)

    def add_connection(self, parent, child):
        # These two if statements are to certify that all objects representing the same mode
        # will have the same hidden state _h
        if parent in self._modes_dict:
            parent._h = self._modes_dict[parent]._h

        if child in self._modes_dict:
            child._h = self._modes_dict[child]._h

        self._adjacency_list[parent].add(child)

    def add_adjacency_list(self, a_list):
        for mode, connections in a_list.items():
            if mode not in self._adjacency_list:
                self._adjacency_list[mode] = set()
            for con in connections:
                self._adjacency_list[mode].add(con)
            self._modes_dict[mode] = mode

    def _get_mode_label(self, node, mapping_names):
        return str(node._state) + ", a" + str(node._action)
        # if node._action is None:
        #     return str(mapping_names[node._state]) + ", " + str(node._action)
        # else:
        #     return str(mapping_names[node._state]) + ", a" + str(node._action)
    
    def run(self, env):
        current_h = self._initial_mode._h
        current_mode = self._initial_mode
        partitioner = Partitioner(self._k)
        actions = []
        number_steps = 0
        while True:
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            prob_actions, next_h = self._model(x_tensor, current_h)

            a = torch.argmax(prob_actions).item()

            next_mode = Mode(partitioner.get_state_automaton(next_h), next_h, a)
            if next_mode not in self._adjacency_list[current_mode]:
                break
            current_mode = next_mode
            # not clear we should use next_h or the h that is stored in the automaton for next_mode
            current_h = self._modes_dict[next_mode]._h
            # print(self._modes_dict[next_mode]._state, self._modes_dict[next_mode]._action, self._modes_dict[next_mode]._h)
            # current_h = next_mode._h
            # if next_mode in self._modes_dict:
            # print('h from mode: ', self._modes_dict[next_mode]._h)
            env.apply_action(a)
            actions.append(a)

            if env.is_over():
                return True, actions

            number_steps += 1
            if number_steps >= self._max_steps:
                return False, None
        return True, actions

    def print_image(self, filename):
        dot = Digraph()
        mapping_names = {}
        counter = 1
        for node, edges in self._adjacency_list.items():
            if node._state not in mapping_names:
                mapping_names[node._state] = counter
                counter +=1
            dot.node(self._get_mode_label(node, mapping_names))
            for edge in edges:
                if edge._state not in mapping_names:
                    mapping_names[edge._state] = counter
                    counter +=1
                dot.edge(self._get_mode_label(node, mapping_names), self._get_mode_label(edge, mapping_names))
        dot.format = 'png'
        dot.render(filename, cleanup=True)

    def print_image_with_mapping_names(self, filename, mapping_names):
        dot = Digraph()
        counter = 1
        for node, edges in self._adjacency_list.items():
            dot.node(self._get_mode_label(node, mapping_names))
            for edge in edges:
                dot.edge(self._get_mode_label(node, mapping_names), self._get_mode_label(edge, mapping_names))
        dot.format = 'png'
        dot.render(filename, cleanup=True)

    def get_mapping_names(self):
        mapping_names = {}
        counter = 1
        for node, edges in self._adjacency_list.items():
            if node._state not in mapping_names:
                mapping_names[node._state] = counter
                counter +=1
        
            for edge in edges:
                if edge._state not in mapping_names:
                    mapping_names[edge._state] = counter
                    counter +=1
        return mapping_names

class Partitioner:
    def __init__(self, k):
        self._k = k

    def get_state_automaton(self, h):
        state_automaton = []
        for row in h:
            for v in row:
                state_automaton.append(self._find_partition(v.item()))
        return tuple(state_automaton)

    def _find_partition(self, v):
        partition_size = 2 / self._k  
        partition_index = int((v + 1) / partition_size)
        if partition_index == self._k: partition_index = self._k - 1
        return partition_index

class ExtractAutomaton:
    def __init__(self, k, model):
        self._k = k
        self._h = None
        self._modes = set()
        self._model = model
        self._partitioner = Partitioner(k)

    def _choose_action(self, env):
        if self._h == None:
            self._h = self._model.init_hidden()
            initial_mode = Mode(self._partitioner.get_state_automaton(self._h), self._h, None)
            if initial_mode not in self._modes:
                self._modes.add(initial_mode)
                    
        x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
        prob_actions, self._h = self._model(x_tensor, self._h)
        
        a = torch.argmax(prob_actions).item()

        mode = Mode(self._partitioner.get_state_automaton(self._h), self._h, a)

        if mode not in self._modes:
            self._modes.add(mode)

        return a
    
    def _rollout(self, env):
        observations = []
        while not env.is_over():
            observations.append(copy.deepcopy(env))
            a = self._choose_action(env)
            env.apply_action(a)
        self._h = None
        return observations
    
    def _add_initial_modes(self):
        # The following code is to be used if we want to consider 
        # a single inital mode to the automaton, the one the recurrent model uses

        # self._initial_modes = set()
        # h_init = self._model.init_hidden()
        # mode = Mode(self._partitioner.get_state_automaton(h_init), h_init)
        # self._modes.add(mode)
        # self._initial_modes.add(mode)

        # The following code is to be used if we want to consider all modes as initial
        # modes in the automaton. This option gives us more programs to search over. 
        self._initial_modes = set()
        initial_modes = []
        for mode in self._modes:
            initial_mode = Mode(mode._state, mode._h)
            initial_modes.append(initial_mode)
    
        for initial_mode in initial_modes:
            self._modes.add(initial_mode)
            self._initial_modes.add(initial_mode)
    
    def build_automata(self, env):
        observations = self._rollout(env)
        self._add_initial_modes()

        automata = []
        for mode in self._initial_modes:
            automata.append(self.build_automaton_from_observations(mode, observations))
        return automata
    
    def build_automaton_from_observations(self, initial_mode, observations):
        automaton = Automaton(self._modes, self._model, initial_mode, self._k)

        # first build initial structure of the automaton, which includes modes and transitions from training data
        h = self._model.init_hidden()
        current_mode = Mode(self._partitioner.get_state_automaton(h), h)

        for o in observations:
            x_tensor = torch.tensor(o.get_observation(), dtype=torch.float32).view(1, -1)
            actions, next_h = self._model(x_tensor, h)
            a = torch.argmax(actions).item()

            next_mode = Mode(self._partitioner.get_state_automaton(next_h), next_h, a)

            automaton.add_connection(current_mode, next_mode)

            h = next_h
            current_mode = next_mode

        #second include transitions of possible initial states, one for each different state observed in training data
        for o in observations:
            x_tensor = torch.tensor(o.get_observation(), dtype=torch.float32).view(1, -1)
            actions, next_h = self._model(x_tensor, initial_mode._h)
            a = torch.argmax(actions).item()
            
            # next mode might not exist from the rollout. If that is the case, we will skip this connection
            next_mode = Mode(self._partitioner.get_state_automaton(next_h), next_h, a)
            if next_mode in automaton._modes:
                automaton.add_connection(initial_mode, next_mode)

        return automaton

