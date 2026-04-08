import torch
from dqn import ReplayBuffer, NN
from env import Env2048
import torch.optim as optim

device = torch.device("cuda:0")
env = Env2048()
online_q = NN().to(device)
target_q = NN().to(device)
optimizer = optim.Adam(online_q.parameters(), lr=1e-3)
checkpoint = torch.load("model.pt")
online_q.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
target_q.load_state_dict(online_q.state_dict())
buffer = ReplayBuffer(131072)