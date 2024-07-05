import random
import numpy as np

class Game:
    def __init__(self, rows, columns, problem):
        self._rows = rows
        self._columns = columns
        self._matrix_unit = np.zeros((rows, columns))
        self._matrix_structure = np.zeros((rows, columns))
        self._matrix_goal = np.zeros((rows, columns))

        self._problem_1 = "TL-BR" # initial location at top-left and goal at bottom-right
        self._problem_2 = "TR-BL" # initial location at top-right and goal at bottom-left
        self._problem_3 = "BR-TL" # initial location at bottom-right and goal at top-left
        self._problem_4 = "BL-TR" # initial location at bottom-left and goal at top-right

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

    def _set_initial_goal(self, problem):
        if problem == self._problem_1:
            self._matrix_unit[0][0] = 1
            self._matrix_goal[self._rows - 1][self._columns - 1] = 1

            self._x = 0
            self._y = 0
        if problem == self._problem_2:
            self._matrix_unit[0][self._columns - 1] = 1
            self._matrix_goal[self._rows - 1][0] = 1

            self._x = 0
            self._y = self._columns - 1
        if problem == self._problem_3:
            self._matrix_unit[self._rows - 1][self._columns - 1] = 1
            self._matrix_goal[0][0] = 1

            self._x = self._rows - 1
            self._y = self._columns - 1
        if problem == self._problem_4:
            self._matrix_unit[self._rows - 1][0] = 1
            self._matrix_goal[0][self._columns - 1] = 1

            self._x = self._rows - 1
            self._y = 0

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
    
    def get_observation(self):
        one_hot_matrix_state = np.zeros((self._pattern_length, self._pattern_length), dtype=int)
        for i, v in enumerate(self._state):
            one_hot_matrix_state[v][i] = 1
        # return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel()))
        return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel(), self._matrix_goal.ravel()))
       
    def is_over(self):
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
            self._state = []

# state = Game(3, 3)
# actions = state.get_actions()
# a = actions[random.randint(0, len(actions) - 1)]
# print(a)
# while not state.is_over():
#     actions = state.get_actions()
#     a = actions[random.randint(0, len(actions) - 1)]
#     state.apply_action(a)

#     print(state)
#     print(state._state)
#     print(state.get_observation())
#     print()
#     # print(state)