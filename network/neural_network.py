from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, activation_function ='ReLU'):
        super().__init__()
        if activation_function == 'ReLU':
            self.sequential_stack = nn.Sequential(
                    nn.Linear(2000, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 5)
                    )
        elif activation_function == 'SiLU':
            self.sequential_stack = nn.Sequential(
                    nn.Linear(2000, 512),
                    nn.SiLU(),
                    nn.Linear(512, 512),
                    nn.SiLU(),
                    nn.Linear(512, 5)
                    )
        else:
            raise ValueError(f"Unknown activation_function: {activation_function}")

    def forward(self, x):
        logits = self.sequential_stack(x)
        return logits

