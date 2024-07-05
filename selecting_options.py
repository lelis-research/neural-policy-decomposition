import copy
import math
import torch
from agent import PolicyGuidedAgent
from combo import Game
from extract_automaton import ExtractAutomaton
from extract_sub_automata import SubAutomataExtractor
from model import CustomRNN

import numpy as np

class LevinLossAutomaton:
    def is_automaton_applicable(self, trajectory, actions, finished, j):
        """
        This function checks whether an automaton is applicable in a given state. 

        An automaton is applicable if the sequence of actions it produces matches
        the sequence of actions in the trajectory. Note that we do not consider an
        automaton if it has less than 2 actions, as it would be equivalent to a 
        primitive action. 
        """
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
                    finished, actions = automaton.run(copy.deepcopy(t[j][0]))

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
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
            else:
                chained_trajectory._sequence = chained_trajectory._sequence + copy.deepcopy(trajectory._sequence)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list
        return self.loss(automata, chained_trajectory, number_actions, joint_problem_name_list, problem_automaton)

def main():
    """
    This is the function to perform the selection of sub-automata from the automaton extract from recurrent models.

    This code assumes that the models were already trained for each one of the problems specified in the list problems below.
    """
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
    partition_k = 5
    sub_automata = {}
    complete_automata = []
    for i in range(len(problems)):
        sub_automata[problems[i]] = []
        print('Extracting from model ', problems[i])

        env = Game(3, 3, problems[i])
        rnn = CustomRNN(27, 5, 3)
        rnn.load_state_dict(torch.load('binary/' + problems[i] + '-model.pth'))

        extractor = ExtractAutomaton(partition_k, rnn)
        full_automata = extractor.build_automata(env)
        complete_automata = complete_automata + full_automata
        counter = 1
        for automaton in full_automata:
            # this will generate an image with the complete automaton extracted from each neural model
            # the images can be quite helpful for debugging purposes.
            automaton.print_image('images/full-' + problems[i] + '-' + str(counter))
            counter += 1

            sub_extractor = SubAutomataExtractor(automaton, partition_k, width=1)
            automata = sub_extractor.extract_sub_automata()
            sub_automata[problems[i]] = sub_automata[problems[i]] + automata
    
        print('Extracted: ', len(sub_automata[problems[i]]), ' automata')

    # loading the trajectories from the trained policies
    trajectories = {}
    for problem in problems:
        env = Game(3, 3, problem)
        agent = PolicyGuidedAgent()
        rnn = CustomRNN(27, 5, 3)
        number_actions = 3
        
        rnn.load_state_dict(torch.load('binary/' + problem + '-model.pth'))

        trajectory = agent.run(env, rnn, greedy=True)
        trajectories[problem] = trajectory

    loss = LevinLossAutomaton()
    selected_automata = []
    previous_loss = None
    best_loss = None

    # computing the Levin loss without options, for reference
    loss_no_automata = loss.compute_loss([], '', trajectories, number_actions)
    print('Loss with no automata: ', loss_no_automata)

    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_automaton = None
        best_size = None

        for problem_automaton, automata in sub_automata.items():
            for automaton in automata:

                levin_loss = loss.compute_loss(selected_automata + [automaton], problem_automaton, trajectories, number_actions)

                if best_loss is None or levin_loss < best_loss:
                    best_loss = levin_loss
                    best_automaton = automaton
                    best_size = automaton.get_size()
                # The following statement ensures that we prefer smaller automaton in case
                # of ties in the Levin loss. The minus 0.01 is to avoid precision issues 
                # while detecting ties. 
                elif levin_loss - 0.01 < best_loss and automaton.get_size() < best_size:
                    best_loss = levin_loss
                    best_automaton = automaton
                    best_size = automaton.get_size()

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting automata
        selected_automata.append(best_automaton)
        best_loss = loss.compute_loss(selected_automata, "", trajectories, number_actions)

        print("Levin loss of the current set: ", best_loss)

    # remove the last automaton added
    selected_automata = selected_automata[0:len(selected_automata) - 1]

    counter = 1
    for automaton in selected_automata:
        automaton.print_image('images/best-' + str(counter))
        counter += 1

if __name__ == "__main__":
    main()