import torch
from dqn import ReplayBuffer, NN
from env import Env2048
import torch.optim as optim

device = torch.device("cuda:0")
env = Env2048()
model = NN().to(device)
model.load_state_dict(torch.load("model.pt"))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer(100000)