import torch
from agent import PolicyGuidedAgent
from combo import Game
from model import CustomRelu

import numpy as np


def main():
    # problem = "TL-BR"
    # problem = "TR-BL"
    # problem = "BR-TL"
    problem = "BL-TR"
    rnn = CustomRelu(18, 4, 3)
    rnn.load_state_dict(torch.load('binary/' + problem + '-relu-model.pth'))
    env = Game(3, 3, problem)

    with torch.no_grad():  
        column = rnn.in2hidden.weight[:, 8].clone() 
        for i in range(9):
            rnn.in2hidden.weight[:, i] = column

    print(rnn.in2hidden.weight)

    for i in range(3):
        for j in range(3):
            env._matrix_unit = np.zeros((3, 3))
            env._matrix_unit[i][j] = 1

            for _ in range(3):
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                prob_actions, _ = rnn.forward_and_return_hidden_logits(x_tensor)
                a = torch.argmax(prob_actions).item()
                print(a)
                env.apply_action(a)

if __name__ == "__main__":
    main()