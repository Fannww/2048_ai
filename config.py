import torch
from env import Env2048
from dqn import NN, ReplayBuffer
import torch.optim as optim

episodes = 10
batch = 128
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.05
decay = 0.7

device = torch.device("cuda:0")
env = Env2048()
model = NN().to(device)
model.load_state_dict(torch.load("model.pt"))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer(100000)