import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

device = torch.device("cuda:0")
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)
        self.nonlinear = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), - 1)
        x = self.nonlinear(self.fc1(x))
        x = self.nonlinear(self.fc2(x))
        x = self.nonlinear(self.fc3(x))
        x = self.fc4(x)
        return x
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        return sample
    
    def __len__(self):
        return len(self.buffer)

def SelectAction(state, epsilon, model, return_tier_list=False):
    if random.random() < epsilon:
        return torch.randint(0, 4, (128, 1), device=device)
    with torch.no_grad():
        q_values = model(state.float())
        sorted_actions = torch.argsort(q_values, descending=True)
    return sorted_actions[:, 0].view(128, 1) if not return_tier_list else sorted_actions

def trainstep(buffer, model, optimizer, batch_size, gamma):
    sample = buffer.sample(batch_size)
    states = torch.stack([t[0] for t in sample])
    actions = torch.stack([t[1] for t in sample])
    rewards = torch.stack([t[2] for t in sample])
    next_states = torch.stack([t[3] for t in sample])
    dones = torch.stack([t[4] for t in sample])
    q_values = model(states.float())
    q_taken = q_values.gather(1, actions)

    next_q = model(next_states.float())
    max_next_q = next_q.max(1)[0]
    target = (rewards + gamma * max_next_q * (1 - dones.int())).view(128, 1)
    loss = nn.MSELoss()(q_taken, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


log_values = torch.zeros(4097, dtype=torch.int16, device=device)
for i in range(1, 13):
    log_values[2**i] = i
directions = np.array([[0, 1], [1, 0]])
def issafe(row, col, grid):
    mask = torch.full((128, 1), 1, device=device)
    for d1, d2 in directions:
        mask *= (grid[:, row, col] == grid[:, row + d1, col + d2]).view(128, 1)

    return mask
def weighted_sum_og(grid):
    weights = np.array([[32768, 16384, 8192, 4096], 
                       [256 ,512 ,1024 , 2048],
                       [128, 64, 32, 16],
                       [1, 2, 4, 8]])
    score = torch.zeros((128, 1), dtype=int, device=device)
    for i in range(4):
        for j in range(4):
            score += (weights[i, j] * grid[:, i, j]).view(128, 1)
    return score
def evaluate(grids):
    grids = grids.view(128, 4, 4)
    weighted_sum = (weighted_sum_og(grids))
    smoothness = torch.full((128,1), 0.0, device=device)
    for i in range(3):
        for j in range(3):
            mask = ((grids[:, i, j] != 0) * (grids[:, i + 1, j] != 0))
            smoothness -= (abs((log_values[grids[:, i, j]] - log_values[grids[:, i + 1, j]]) * mask)).view(128, 1)
            mask = ((grids[:, i, j] != 0) * (grids[:, i, j + 1] != 0))
            smoothness -= (abs((log_values[grids[:, i, j]] - log_values[grids[:, i, j + 1]]) * mask)).view(128, 1)

    max_tile = torch.full((128, 1), -1.0, device=device)
    empty_tiles = torch.full((128, 1), 0.0, device=device)
    for row in range(4):
        for col in range(4):
            vals = grids[:, row, col].view(128, 1)
            mask = vals == 0
            empty_tiles += 1 * mask
            mask = vals > max_tile
            max_tile *= ~mask
            max_tile += mask * vals
    empty_weight = torch.full((128, 1), 0.0, device=device)
    smoothness_weight = torch.full((128 ,1), 0.0, device=device) 
    mask = max_tile < 1024
    empty_weight += 4 * mask
    smoothness_weight += 0.2 * mask
    mask = (max_tile >= 1024) & (max_tile < 2048)
    empty_weight += 2.5 * mask
    smoothness_weight += 0.8 * mask
    mask = max_tile >= 2048
    empty_weight += 1.5 * mask
    smoothness_weight += 3 * mask
    safe = torch.full((128, 1), 0, device=device)
    mask = empty_tiles == 0
    safe += mask
    for i in range(3):
        for j in range(3):
            mask = issafe(i, j, grids)
            safe = (safe == 1) & (mask == 0)
    score = (((empty_tiles * max_tile * empty_weight) + (smoothness * (max_tile * smoothness_weight)) + (max_tile) + (weighted_sum)) * (~safe).int()).int()
    return score.view(128,)