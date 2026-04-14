import torch
from dqn import ReplayBuffer, NN
from env import Env2048
import torch.optim as optim
from prompt import resume

device = torch.device("cuda:0")
env = Env2048()
online_q = NN().to(device)
target_q = NN().to(device)
optimizer = optim.Adam(online_q.parameters(), lr=1e-3)
if resume:
    checkpoint = torch.load("checkpoint.pt", weights_only=False)
    online_q.load_state_dict(checkpoint["online_q"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    target_q.load_state_dict(checkpoint["target_q"])
    buffer = checkpoint["buffer"]
    epsilon = checkpoint["epsilon"]
else:
    buffer = ReplayBuffer(131072)
    epsilon = 1