import torch
from dqn import ReplayBuffer, NN
from env import Env2048
import torch.optim as optim

device = torch.device("cuda:0")
env = Env2048()
online_q = NN().to(device)
target_q = NN().to(device)
#online_q.load_state_dict(torch.load("model.pt"))
#target_q.load_state_dict(torch.load("model.pt"))
optimizer = optim.Adam(online_q.parameters(), lr=1e-3)
buffer = ReplayBuffer(100000)