import random
import numpy as np
import math
import gc
import copy

class ComboGridEnv:
    def __init__(self, rows, columns, problem, partial_observability=True, multiple_initial_states=False, visitation_bonus=0):
        self._rows = rows
        self._columns = columns
        self._matrix_unit = np.zeros((rows, columns))
        self._matrix_structure = np.zeros((rows, columns))
        self._matrix_goal = np.zeros((rows, columns))
        self._partial_observability = partial_observability
        self._multiple_initial_states = multiple_initial_states
        self._last_action = None
        self._goal = None
        self._goals_reached = set()
        self._problem = problem
        self._visitation_bonus = visitation_bonus
        if self._visitation_bonus:
            self._state_visitation_count = {}

        self._problem_1 = "TL-BR" # initial location at top-left and goal at bottom-right
        self._problem_2 = "TR-BL" # initial location at top-right and goal at bottom-left
        self._problem_3 = "BR-TL" # initial location at bottom-right and goal at top-left
        self._problem_4 = "BL-TR" # initial location at bottom-left and goal at top-right
        self._problem_5 = "test"  # initial location in the middle and four goals

        self._set_initial_goal(problem)

        # state of current action sequence
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        self._state = []
        self._pattern_length = 3

        self._action_pattern = {}
        self._action_pattern[(0, 0, 1)] = 0
        self._action_pattern[(0, 1, 2)] = 1
        self._action_pattern[(2, 1, 0)] = 2
        self._action_pattern[(1, 0, 2)] = 3

    #TODO: doesn't work with test problem, will fix later
    def __eq__(self, other):
        if not isinstance(other, ComboGridEnv):
            return False
        return (
            self._rows == other._rows and
            self._columns == other._columns and
            np.array_equal(self._matrix_unit, other._matrix_unit) and
            np.array_equal(self._matrix_structure, other._matrix_structure) and
            np.array_equal(self._matrix_goal, other._matrix_goal) and
            self._partial_observability == other._partial_observability and
            self._multiple_initial_states == other._multiple_initial_states and
            self._x == other._x and
            self._y == other._y and
            self._x_goal == other._x_goal and
            self._y_goal == other._y_goal
        )

    #TODO: doesn't work with test problem, will fix later
    def __hash__(self):
        return hash((
            self._rows,
            self._columns,
            self._partial_observability,
            self._multiple_initial_states,
            self._x,
            self._y,
            self._x_goal,
            self._y_goal,
            tuple(self._matrix_unit.ravel()),
            tuple(self._matrix_structure.ravel()),
            tuple(self._matrix_goal.ravel())
        ))

    def _set_initial_goal(self, problem):
        if problem == self._problem_1:
            self._matrix_unit[0][0] = 1
            self._matrix_goal[self._rows - 1][self._columns - 1] = 1

            self._x_goal = self._rows - 1
            self._y_goal = self._columns - 1

            self._x = 0
            self._y = 0
        if problem == self._problem_2:
            self._matrix_unit[0][self._columns - 1] = 1
            self._matrix_goal[self._rows - 1][0] = 1

            self._x_goal = self._rows - 1
            self._y_goal = 0

            self._x = 0
            self._y = self._columns - 1
        if problem == self._problem_3:
            self._matrix_unit[self._rows - 1][self._columns - 1] = 1
            self._matrix_goal[0][0] = 1

            self._x_goal = 0
            self._y_goal = 0

            self._x = self._rows - 1
            self._y = self._columns - 1
        if problem == self._problem_4:
            self._matrix_unit[self._rows - 1][0] = 1
            self._matrix_goal[0][self._columns - 1] = 1

            self._x_goal = 0
            self._y_goal = self._columns - 1

            self._x = self._rows - 1
            self._y = 0
        
        if problem == self._problem_5:
            if self._rows % 2 == 0 or self._columns % 2 == 0:
                print("Rows and columns should be odd!")
                exit()
            center1 = int(self._rows / 2)
            center2 = int(self._columns / 2)
            self._goal = {
                    (0, center2),
                    (center1, 0),
                    (center1, self._columns - 1),
                    (self._rows - 1, center2),
                }
            self._matrix_unit[center1][center2] = 1
            for g in self._goal:
               self._matrix_goal[g[0]][g[1]] = 1 
               
            self._x = center1
            self._y = center2

            self._x_goal = None
            self._y_goal = None
        #TODO: Random initial state doesn't work for test problem for now, will fix later
        if self._multiple_initial_states:
            finished = False

            self._matrix_unit[self._x][self._y] = 0
            while not finished:
                self._x = random.randint(0, self._rows - 1)
                self._y = random.randint(0, self._columns - 1)

                if self._x != self._x_goal or self._y != self._y_goal:
                    self._matrix_unit[self._x][self._y] = 1
                    finished = True

    def set_initial_state(self, x, y):
        self._matrix_unit[self._x][self._y] = 0
        self._x = x
        self._y = y
        self._matrix_unit[self._x][self._y] = 1

    def __repr__(self) -> str:
        str_map = ""
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_unit[i][j] == 1:
                    str_map += " A "
                elif self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def reset(self):
        self._matrix_unit = np.zeros((self._rows, self._columns))
        self._matrix_structure = np.zeros((self._rows, self._columns))
        self._matrix_goal = np.zeros((self._rows, self._columns))
        self._last_action = None
        self._goal = None
        self._goals_reached = set()

        self._set_initial_goal(self._problem)

        self._state = []

        if self._visitation_bonus:
            for r in range(self._rows):
                for c in range(self._columns):
                    one_hot_matrix_state = np.zeros((self._rows, self._columns), dtype=int)
                    one_hot_matrix_state[r][c] = 1
                    self._state_visitation_count[tuple(one_hot_matrix_state.ravel())] = 0
            
            self._state_visitation_count[copy.deepcopy(tuple(self._matrix_unit.ravel()))] += 1

    def get_observation(self):
        if self._partial_observability:
            one_hot_matrix_state = np.zeros((3), dtype=int)
            # if self._last_action is not None:
            #     one_hot_matrix_state[self._last_action] = 1
        else:
            one_hot_matrix_state = np.zeros((self._pattern_length, self._pattern_length), dtype=int)
            for i, v in enumerate(self._state):
                one_hot_matrix_state[v][i] = 1
        # return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel()))
        return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel(), self._matrix_goal.ravel()))
        # return np.concatenate((self._matrix_unit.ravel(), self._matrix_goal.ravel()))
       
    def is_over(self):
        if isinstance(self._goal, set):
            return self._goal == self._goals_reached
        if self._matrix_goal[self._x][self._y] == 1:
            return True
        else:
            return False

    def get_actions(self):
        return [0, 1, 2]
   
    def apply_action(self, action):
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        reach_goal = False
        if self._partial_observability:
            self._last_action = action
        # each column in _state_matrix represents an action
        self._state.append(action)

        if len(self._state) == self._pattern_length:
            action_tuple = tuple(self._state)
            if action_tuple in self._action_pattern:
                # moving up
                if self._action_pattern[action_tuple] == 0:
                    if self._x - 1 >= 0 and self._matrix_structure[self._x - 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving down
                if self._action_pattern[action_tuple] == 1:
                    if self._x + 1 < self._matrix_unit.shape[0] and self._matrix_structure[self._x + 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x += 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving left
                if self._action_pattern[action_tuple] == 2:
                    if self._y - 1 >= 0 and self._matrix_structure[self._x][self._y - 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving right
                if self._action_pattern[action_tuple] == 3:
                    if self._y + 1 < self._matrix_unit.shape[1] and self._matrix_structure[self._x][self._y + 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y += 1
                        self._matrix_unit[self._x][self._y] = 1
                # adding the reached goal to the reached goal set, used for determining the termination of the episode        
                if isinstance(self._goal, set) and (self._x, self._y) in self._goal and (self._x, self._y) not in self._goals_reached:
                    self._goals_reached.add((self._x, self._y))
                    self._matrix_goal[self._x][self._y] = 0
                    reach_goal = True
            self._state = []
        if self._visitation_bonus:
            self._state_visitation_count[copy.deepcopy(tuple(self._matrix_unit.ravel()))] += 1
        return reach_goal

    def get_exploration_bonus(self):
        return 0.5 / (self._state_visitation_count[copy.deepcopy(tuple(self._matrix_unit.ravel()))] ** 0.5)