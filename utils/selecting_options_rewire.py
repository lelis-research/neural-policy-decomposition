import copy
import os
import math
import torch
import pickle
import multiprocessing
import gc
from functools import partial
import random
from utils.extract_automaton_quantized import Automaton
from utils.extract_automaton_quantized import SubAutomataExtractor
from models.model_recurrent import GruAgent

from utils.extract_automaton import ExtractAutomaton
import gymnasium as gym
from envs.combogrid import ComboGridEnv
from envs.combogrid_gym import ComboGridGym
from models.model import QuantizedRNN


device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


import numpy as np
def make_env(problem, width=3, options=None):
    def thunk():
        env = ComboGridGym(rows=width, columns=width, problem=problem, options=options, random_initial=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class LevinLossAutomaton:
    def is_automaton_applicable(self, trajectory, actions, finished, j):
        """
        This function checks whether an automaton is applicable in a given state. 

        An automaton is applicable if the sequence of actions it produces matches
        the sequence of actions in the trajectory. Note that we do not consider an
        automaton if it has less than 2 actions, as it would be equivalent to a 
        primitive action. 
        """
        # print(f"Process {os.getpid()} computing square of {random.random()}")
        if not finished or len(actions) <= 1 or len(actions) + j > len(trajectory):
            return False
        
        for i in range(len(actions)):
            if actions[i] != trajectory[i + j][1]:
                return False
        return True

    def loss(self, automata, trajectory, number_actions, joint_problem_name_list, problem_automata):
        """
        This function implements the dynamic programming method from Alikhasi & Lelis (2024). 

        Note that the if-statement with the following code is in a different place. I believe there is
        a bug in the pseudocode of Alikhasi & Lelis (2024).

        M[j] = min(M[j - 1] + 1, M[j])
        """
        t = trajectory.get_trajectory()
        M = np.arange(len(t) + 1)

        for j in range(len(t) + 1):
            if j > 0:
                M[j] = min(M[j - 1] + 1, M[j])
            if j < len(t):
                for automaton in automata:
                    # the automaton being considered for selection cannot be evaluated on the trajectory
                    # generated by the "full automaton" (i.e., the one that attempts to mimick the neural model) 
                    # from which the automaton was extracted. 
                    if joint_problem_name_list[j] == problem_automata:
                        continue
                    finished, actions = automaton.transition(copy.deepcopy(t[j][0]))
                    # print(finished, actions)

                    if self.is_automaton_applicable(t, actions, finished, j):
                        M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)
        uniform_probability = (1/(len(automata) + number_actions)) 
        depth = len(t) + 1
        number_decisions = M[len(t)]

        # use the Levin loss in log space to avoid numerical issues
        log_depth = math.log(depth)
        log_uniform_probability = math.log(uniform_probability)
        return log_depth - number_decisions * log_uniform_probability

    def compute_loss(self, automata, problem_automaton, trajectories, number_actions):
        """
        This function computes the Levin loss of a set of automata. Each automaton in the set is 
        what we select as a set of options, according to Alikhasi & Lelis (2024). 

        The loss is computed for a set of trajectories, one for each training task. Instead of taking
        the average loss across all trajectories, in this function we stich all trajectories together
        forming one long trajectory. The function is implemented this way for debugging purposes. 
        Since an sub-automaton s extracted from automaton b cannot be evaluated in the trajectory
        b generated, this "leave one out" was more difficult to debug. Stiching all trajectories
        into a single one makes it easier (see chained_trajectory below). 

        We still do not evaluate an automaton on the data it was used to generate it. This is achieved
        with the vector joint_problem_name_list below, which is passed to the loss function. 
        """
        chained_trajectory = None
        joint_problem_name_list = []
        for problem, prob_trajectories in trajectories.items():
            for trajectory in prob_trajectories:
                if chained_trajectory is None:
                    chained_trajectory = copy.deepcopy(trajectory)
                else:
                    chained_trajectory._sequence = chained_trajectory._sequence + copy.deepcopy(trajectory._sequence)
                name_list = [problem for _ in range(len(trajectory._sequence))]
                joint_problem_name_list = joint_problem_name_list + name_list
        return self.loss(automata, chained_trajectory, number_actions, joint_problem_name_list, problem_automaton)


class Trajectory:
    def __init__(self):
        self._sequence = []

    def add_pair(self, state, action):
        self._sequence.append((state, action))
    
    def get_trajectory(self):
        return self._sequence

def _rollout(agent, env):
    _h = None
    def _choose_action(env, _h):
        if _h == None:
            _h = agent.init_hidden()
                    
        x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
        prob_actions, _h = agent(x_tensor, _h)
        
        a = torch.argmax(prob_actions).item()
        return a, _h
        
    traj = Trajectory()
    while not env.is_over():
        a, _h = _choose_action(env, _h)
        traj.add_pair(copy.deepcopy(env),a)
        env.apply_action(a)
    _h = None
    return traj

def extract_options(seed=1, game_width=5, problem=[]):
    """
    This is the function to perform the selection of sub-automata from the automaton extract from recurrent models.

    This code assumes that the models were already trained for each one of the problems specified in the list problems below.
    """
    problems = ["BL-TR", "TR-BL","TL-BR","BR-TL"]
    # problems = ["BR-TL"]
    width_automaton = 1
    sub_automata = {}
    complete_automata = []
    base_automata = {}
    for i in range(len(problems)):
        sub_automata[problems[i]] = []
        base_automata[problems[i]] = []
        print('Extracting from models.model ', problems[i])

        # env = ComboGridEnv(5, 5, problems[i])
        env = gym.vector.SyncVectorEnv(
        [make_env(problems[i], width=game_width)],
    )
        rnn = GruAgent(env,64, option_len=0, greedy=True)
        rnn.load_state_dict(torch.load(f'training_data/models/{seed}/{problems[i]}.pt'))
        rnn.eval()
        with open (f'training_data/trajectories/{seed}/{problems[i]}.pkl', 'rb') as f:
            trajectory = pickle.load(f)
        full_automaton = Automaton(rnn, problems[i])
        for traj in trajectory: 
            full_automaton.generate_modes(traj.get_trajectory()[0][0])
        base_automata[problems[i]] = base_automata[problems[i]] + [copy.deepcopy(full_automaton)]

        # extractor = SubAutomataExtractor(full_automaton, width_automaton)
        complete_automata = complete_automata + base_automata[problems[i]]
        counter = 1
        for automaton in base_automata[problems[i]]:
            # this will generate an image with the complete automaton extracted from each neural model
            # the images can be quite helpful for debugging purposes.
            # automaton.print_image('images/base-' + problems[i] + '-' + str(counter))
            counter += 1

            sub_extractor = SubAutomataExtractor(automaton, width_automaton)
            automata = sub_extractor.extract_sub_automata()
            sub_automata_rewired = sub_extractor.rewire_sub_automata(automata)
            sub_automata[problems[i]] = sub_automata[problems[i]] + copy.deepcopy(sub_automata_rewired)
    
        print('Extracted: ', len(sub_automata[problems[i]]), ' automata')


    # loading the trajectories from the trained policies
    trajectories = {}
    for problem in problems:
        number_actions = 3
        
        #Generating trajectories
        # trajectory = _rollout(rnn, env)

        #Load trajectories
        with open (f'training_data/trajectories/{seed}/{problem}.pkl', 'rb') as f:
            trajectory = pickle.load(f)

        trajectories[problem] = trajectory

    loss = LevinLossAutomaton()
    selected_automata = []
    previous_loss = None
    best_loss = None

    # computing the Levin loss without options, for reference
    loss_no_automata = loss.compute_loss([], '', trajectories, number_actions)
    print('Loss with no automata: ', loss_no_automata)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_automaton = None
        best_size = None
        list_automata = []

        for problem_automaton, automata in sub_automata.items():
            for automaton in automata:
                list_automata.append((selected_automata + [automaton], problem_automaton))

        fixed_args = {'trajectories': trajectories, 'number_actions': number_actions}
        partial_function = partial(loss.compute_loss, **fixed_args)
        with multiprocessing.Pool(processes=ncpus) as pool:  # Adjust the number of processes here
            losses = pool.starmap(partial_function, list_automata)

        losses_with_automaton = list(zip(losses, list_automata))

        for opt_loss, automaton in losses_with_automaton:
            if best_loss is None or opt_loss < best_loss:
                    best_loss = opt_loss
                    best_automaton = copy.deepcopy(automaton[0][-1])
                    best_size = automaton[0][-1].get_size()

                # The following statement ensures that we prefer smaller automaton in case
                # of ties in the Levin loss. The minus 0.01 is to avoid precision issues 
                # while detecting ties. 

            elif opt_loss - 0.01 < best_loss and automaton[0][-1].get_size() < best_size:
                best_loss = opt_loss
                best_automaton = copy.deepcopy(automaton[0][-1])
                best_size = automaton[0][-1].get_size()

        selected_automata.append(best_automaton)
        best_loss = loss.compute_loss(selected_automata, "", trajectories, number_actions)
        print("Levin loss of the current set: ", best_loss)
        gc.collect()

        # for problem_automaton, automata in sub_automata.items():
        #     for automaton in automata:

        #         levin_loss = loss.compute_loss(selected_automata + [automaton], problem_automaton, trajectories, number_actions)
        #         # print(levin_loss)

        #         if best_loss is None or levin_loss < best_loss:
        #             best_loss = levin_loss
        #             best_automaton = automaton
        #             best_size = automaton.get_size()
        #             # print(best_size, levin_loss < best_loss)
        #         # The following statement ensures that we prefer smaller automaton in case
        #         # of ties in the Levin loss. The minus 0.01 is to avoid precision issues 
        #         # while detecting ties. 

        #         elif levin_loss - 0.01 < best_loss and automaton.get_size() < best_size:
        #             best_loss = levin_loss
        #             best_automaton = automaton
        #             best_size = automaton.get_size()

        # # we recompute the Levin loss after the automaton is selected so that we can use 
        # # the loss on all trajectories as the stopping condition for selecting automata
        # selected_automata.append(best_automaton)
        # best_loss = loss.compute_loss(selected_automata, "", trajectories, number_actions)

        # print("Levin loss of the current set: ", best_loss)

    # remove the last automaton added
    selected_automata = selected_automata[0:len(selected_automata) - 1]
    try:
        os.mkdir(f"training_data/options{seed}")
    except:
        pass
    with open(f"training_data/options{seed}/selected_options.pkl", "wb") as file:
        pickle.dump(automata, file)
