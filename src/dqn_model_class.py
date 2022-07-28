import torch
from torch import nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        #   Create layers
        self.fc1 = nn.Linear(in_features=2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)


    def forward(self, state):
        #   Implement forward pass
        state = torch.as_tensor(state, dtype = torch.float32)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        Q = self.out(state)

        return Q
