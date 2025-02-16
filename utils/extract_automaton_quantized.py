from envs.combogrid import ComboGridEnv
from envs.combogrid_gym import ComboGridGym
from models.model import QuantizedRNN
import torch
import numpy as np
from itertools import combinations
from itertools import product
import heapq
import copy
from graphviz import Digraph

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
next_done = torch.zeros(1).to(device)

class Mode:
    def __init__(self, hidden, transitions, rewire=False):
        self._h = hidden
        self._transitions = transitions
        self.rewire = rewire
    
    def __eq__(self, other):
        if isinstance(other, Mode):
            return self._h == other._h
        return False

    def __hash__(self):
        return hash(self._h)
    
    def __repr__(self):
        return str(self._h)

    def copy(self):
        return Mode(self._h, self._transitions)

class Automaton:
    def __init__(self, model, problem):
        self._modes = {}
        self._adjacency_dict = {}
        self._problem = problem
        self._model = model
        self._initial_mode = Mode(tuple(self._model.init_hidden().clone().detach().numpy().flatten()), set())
        self._modes[tuple(self._model.init_hidden().clone().detach().numpy().flatten())] = self._initial_mode
        self._adjacency_dict[tuple(self._model.init_hidden().clone().detach().numpy().flatten())] = set()
    
    def get_size(self):
        return len(self._modes)

    def generate_modes(self, env_init=None, game_size=5):
        if env_init is None:
            env = ComboGridGym(game_size,game_size,self._problem, visitation_bonus=False)
            next_obs, _ = env.reset()
        else:
            env = copy.deepcopy(env_init)
        h = self._model.init_hidden()
        self._modes.setdefault(tuple(h.clone().detach().numpy().flatten()), Mode(tuple(self._model.init_hidden().clone().detach().numpy().flatten()), set()))
        self._adjacency_dict.setdefault(tuple(h.clone().detach().numpy().flatten()),set())
        is_over = False
        while not is_over:
            h_old = h
            x_tensor = torch.tensor(next_obs, dtype=torch.float32)
            action, logprob, _, value, h = self._model.get_action_and_value(x_tensor, h_old, next_done)
            h_val = tuple(h_old.clone().detach().numpy().flatten())
            if h_val not in self._modes:
                self._adjacency_dict[h_val] = set()
                mode = Mode(h_val, set())
                self._modes[h_val] = mode
            mode = self._modes[h_val]
            self._adjacency_dict[h_val].add(tuple(h.clone().detach().numpy().flatten()))
            mode._transitions.add(tuple(h.clone().detach().numpy().flatten()))
            next_obs, _, terminations, truncations, _ = env.step(action.cpu().numpy())
            if True in np.logical_or(terminations, truncations):
                is_over = True
        #last hidden state
        h_val = tuple(h.clone().detach().numpy().flatten())
        if h_val not in self._modes:
            self._adjacency_dict[h_val] = set()
            mode = Mode(h_val, set())
            self._modes[h_val] = mode

        self._initial_mode = self._modes[tuple(self._model.init_hidden().clone().detach().numpy().flatten())]
    
    def add_adj_dict(self, adj_dict):
        for mode, connection in adj_dict.items():
            if isinstance(mode, Mode):
                mode_h = mode._h
                rewire = mode.rewire
            else:
                mode_h = mode
                rewire=False
            if mode_h not in self._modes:
                self._modes[mode_h] = Mode(mode_h, set(), rewire)
                self._adjacency_dict[mode_h] = set()
            for con in connection:
                if isinstance(con, Mode):
                    con_h = con._h
                else:
                    con_h = con
                self._modes[mode_h]._transitions.add(con_h)
                self._adjacency_dict[mode_h].add(con_h)
                if con_h not in self._modes:
                    self._modes[con_h] = Mode(con_h, set())
                    self._adjacency_dict[con_h] = set()
    
    def transition(self, env_init, hidden_state=None, apply_actions=False, max_ep_len=50):
        if hidden_state is None:
             mode = self._initial_mode
             h = torch.tensor(mode._h).view(1, -1).unsqueeze(1)
        else: 
            h = hidden_state
            h_val = tuple(h.clone().detach().numpy().flatten())
            mode = self._modes[h_val]
        actions = []
        terminate = False
        ep_len = 0
        episode_over = False
        env = copy.deepcopy(env_init)
        while not terminate and ep_len < max_ep_len and not episode_over:
            x_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
            action, logprob, _, value, h = self._model.get_action_and_value(x_tensor, h, next_done)
            h_val = tuple(h.clone().detach().numpy().flatten())
            if len(mode._transitions) > 1:
                #TODO: Implement rewiring
                if h_val in mode._transitions:
                    mode = self._modes[h_val]
                    h = torch.tensor(mode._h).view(1, -1).unsqueeze(1)
                else:
                    terminate = True
            elif len(mode._transitions) == 1:
                    mode = self._modes[list(mode._transitions)[0]]
                    h = torch.tensor(mode._h).view(1, -1).unsqueeze(1)
            else:
                terminate = True
            if not terminate:
                next_obs, _, terminations, truncations, _ = env.step(action.cpu().numpy())
                if True in np.logical_or(terminations, truncations):
                    episode_over = True
                actions.append(action)
            ep_len += 1
        if ep_len >= max_ep_len:
            return False, actions
        if apply_actions:
            for a in actions:
                env_init.step(a.cpu().numpy())
        return True, actions
    
    def print_image(self, filename):
        dot = Digraph()
        mapping_names = {}
        counter = 1
        for node, edges in self._adjacency_dict.items():
            if node not in mapping_names:
                mapping_names[node] = counter
                counter +=1
            dot.node(str(node))
            for edge in edges:
                if edge not in mapping_names:
                    mapping_names[edge] = counter
                    counter +=1
                dot.edge(str(node), str(edge))
        dot.format = 'png'
        dot.render(filename, cleanup=True)
    


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
    def __init__(self, automaton, width = 1):
        self._automaton = automaton
        self._width = width


    def generate_children(self, node):
        edges_open_nodes = []
        children = []

        if len(node._open) == 0:
            return children
        for o in node._open:
            current_mode_edges = []
            list_edges = list(self._automaton._adjacency_dict[o._h])
            for i in range(0, self._width + 1):
                current_mode_edges = current_mode_edges + list(combinations(list_edges, i))
            edges_open_nodes.append(current_mode_edges)

        cross_product = list(product(*edges_open_nodes))
        for child_edges in cross_product:
            child = node.copy_for_child()
            for j in range(len(child_edges)):
                for child_mode in child_edges[j]:
                    child.add_connection(node._open[j], self._automaton._modes[child_mode])
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
            automaton_from_node = Automaton(self._automaton._model, self._automaton._problem)
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
    
    def rewire_sub_automata(self, automata):
        sub_automata_copy = copy.deepcopy(automata)
        for automaton in sub_automata_copy:
            sub_generated = [copy.deepcopy(automaton._adjacency_dict)]
            for mode, trans in automaton._adjacency_dict.items():
                if len(trans) > 1:
                    trans = list(trans)
                    for j in range(len(sub_generated)):
                        for i in range(len(trans)):
                            adj_dict = copy.deepcopy(sub_generated[j])
                            adj_dict.pop(mode)
                            adj_dict[Mode(mode,trans[i],True)] = {trans[i]}
                            sub_generated.append(copy.deepcopy(adj_dict))
            sub_generated = sub_generated[1:]
            for sub in sub_generated:
                auto = Automaton(automaton._model, automaton._problem)
                auto.add_adj_dict(sub)
                automata.append(auto)
        return automata

def main():
    # problem = "TL-BR"
    problem = "TR-BL"
    # problem = "BR-TL"
    # problem = "BL-TR"

    env = ComboGridEnv(5, 5, problem)
    rnn = QuantizedRNN(21, 4, 3)
    hidden_states = []
    rnn.load_state_dict(torch.load(f'binary/quantized-game-width' + str(3) + '-' + problem + '-rnn-noreg-' + str(4) + '-model.pth', weights_only=True))

    option = Automaton(rnn, problem) 
    option.generate_modes()

    sub = SubAutomataExtractor(option, 1)
    sub_automata = sub.extract_sub_automata()
    print(len(sub_automata))
    sub_automata_rewired = sub.rewire_sub_automata(copy.deepcopy(sub_automata))
    print(len(sub_automata_rewired))
    # option.print_image('images/full-' + problem + '-quantized')


if __name__ == "__main__":
    main()