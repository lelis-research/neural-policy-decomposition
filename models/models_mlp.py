import torch
import numpy as np
from torch import nn

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, hidden_size)
        self.in3output = nn.Linear(hidden_size, output_size)
        self.outsoftmax = nn.Softmax(dim=1)
        self.apply(self._weights_init_xavier)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0)
        self._criterion = nn.CrossEntropyLoss()
    
    def _weights_init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def print_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy()
                print(f"Weights of layer {name}:")
                print(np.array_str(weights, precision=3, suppress_small=True))

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.tanh(self.in2hidden(combined))
        output_1 = torch.tanh(self.in2output(combined))
        output_2 = self.in3output(output_1)
        output = self.outsoftmax(output_2)

        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def _l1_norm(self, lambda_l1):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return lambda_l1 * l1_norm
    
    def train(self, trajectory):
        h = self.init_hidden()
        step_loss = 0
        self._optimizer.zero_grad()
        for x, y in trajectory.get_trajectory():
            x_tensor = torch.tensor(x.get_observation(), dtype=torch.float32).view(1, -1)
            y_tensor = torch.tensor([y], dtype=torch.long) 
            
            output, h = self(x_tensor, h)
            step_loss += self._criterion(output, y_tensor)
            step_loss += self._l1_norm(0.01)
            
        step_loss.backward()    
        self._optimizer.step()

        return step_loss

class CustomRelu(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRelu, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size, hidden_size)
        self.in3output = nn.Linear(hidden_size, output_size)
        self.outsoftmax = nn.Softmax(dim=1)
        self.apply(self._weights_init_xavier)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0)
        self._criterion = nn.CrossEntropyLoss()
    
    def _weights_init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def print_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                values = param.data.cpu().numpy()
                param_type = 'Weights' if 'weight' in name else 'Bias'
                print(f"{param_type} of layer {name}:")
                print(np.array_str(values, precision=3, suppress_small=True))

    def forward(self, x):
        hidden_logits = self.in2hidden(x)
        hidden = torch.relu(hidden_logits)
        output_logits = self.in3output(hidden)
        output = self.outsoftmax(output_logits)

        return output
    
    def forward_and_return_hidden_logits(self, x):
        hidden_logits = self.in2hidden(x)
        hidden = torch.relu(hidden_logits)
        output_logits = self.in3output(hidden)
        output = self.outsoftmax(output_logits)

        return output, hidden_logits
    
    def masked_forward_and_return_output_logits(self, x, mask):
        hidden_logits = self.in2hidden(x)
        hidden = self.masked_neuron_operation(hidden_logits, mask)
        output_logits = self.in3output(hidden)
        output = self.outsoftmax(output_logits)

        return output, output_logits
    
    def masked_neuron_operation(self, logits, mask):
        """
        Apply a mask to neuron outputs in a layer.

        Parameters:
            x (torch.Tensor): The pre-activation outputs (linear outputs) from neurons.
            mask (torch.Tensor): The mask controlling the operation, where:
                                1 = pass the linear input
                                0 = pass zero,
                                -1 = compute ReLU as usual (part of the program).

        Returns:
            torch.Tensor: The post-masked outputs of the neurons.
        """
        relu_out = torch.relu(logits)
        output = torch.zeros_like(logits)
        output[mask == -1] = relu_out[mask == -1]
        output[mask == 1] = logits[mask == 1]

        return output

    def masked_forward(self, x, mask):
        hidden_logits = self.in2hidden(x)
        hidden = self.masked_neuron_operation(hidden_logits, mask)
        output_logits = self.in3output(hidden)
        output = self.outsoftmax(output_logits)

        return output
    
    def _l1_norm(self, lambda_l1):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return lambda_l1 * l1_norm
    
    def train(self, trajectory):
        step_loss = 0
        self._optimizer.zero_grad()
        for x, y in trajectory.get_trajectory():
            x_tensor = torch.tensor(x.get_observation(), dtype=torch.float32).view(1, -1)
            y_tensor = torch.tensor([y], dtype=torch.long) 
            
            output = self(x_tensor)
            step_loss += self._criterion(output, y_tensor)
            # step_loss += self._l1_norm(0.01)
            
        step_loss.backward()    
        self._optimizer.step()

        return step_loss
