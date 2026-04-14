import torch
import torch.nn as nn
import random
import params
from torchrl.modules import NoisyLinear

device = torch.device("cuda:0")
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.advantage = NoisyLinear(128, 4)
        self.value = NoisyLinear(128, 1)
        self.nonlinear = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), - 1)
        x = self.nonlinear(self.fc1(x))
        x = self.nonlinear(self.fc2(x))
        x = self.nonlinear(self.fc3(x))
        A = self.advantage(x)
        V = self.value(x)
        Q = V.view(params.batch, 1) + (A - A.mean(dim=1, keepdim=True)).view(params.batch, 4)
        return Q
class ReplayBuffer():
    def __init__(self, capacity):
        self.state = torch.empty((capacity, 16), dtype=torch.int, device=device)
        self.action = torch.empty((capacity), dtype=torch.int,  device=device)
        self.reward = torch.empty(capacity, dtype=torch.float,  device=device)
        self.next_state = torch.empty((capacity, 16), dtype=torch.int,  device=device)
        self.done = torch.empty(capacity, dtype=torch.bool,  device=device)
        self.capacity = capacity
        self.len = 0
    def push(self, states, actions, rewards, next_states, dones):
        if self.len >= self.capacity:
            self.len = 0
        idx = torch.arange(self.len, self.len + params.batch)
        self.state[idx] = states
        self.action[idx] = actions
        self.reward[idx] = rewards
        self.next_state[idx] = next_states
        self.done[idx] = dones
        self.len += params.batch

    def sample(self, batch_size):
        idxs = torch.randint(0, self.len, (params.batch,))
        states = self.state[idxs]
        actions = self.action[idxs]
        rewards = self.reward[idxs]
        next_states = self.next_state[idxs]
        dones = self.done[idxs]
        return (states, actions, rewards, next_states, dones), idxs
        
    def __len__(self):
        return self.len
    
vmask = torch.zeros(params.batch, 4, dtype=torch.bool, device=device)
rank = torch.arange(4, device=device).expand(params.batch, 4)
masked_rank = torch.empty(params.batch, 4, device=device)
def return_valid_move(states, actions, vmask=vmask):
    vmask.zero_()
    vmask[:, 0] = (((states[:, :-1, :] == states[:, 1:, :]) | (states[:, :-1, :] == 0)) & ~(states[:, 1:, :] == 0)).any(dim=(1, 2))
    vmask[:, 2] = (((states[:, :, :-1] == states[:, :, 1:]) | (states[:, :, :-1] == 0)) & ~(states[:, :, 1:] == 0)).any(dim=(1, 2))
    vmask[:, 1] = (((states[:, :-1, :] == states[:, 1:, :]) | (states[:, 1:, :] == 0)) & ~(states[:, :-1, :] == 0)).any(dim=(1, 2))
    vmask[:, 3] = (((states[:, :, :-1] == states[:, :, 1:]) | (states[:, :, 1:] == 0)) & ~(states[:, :, :-1] == 0)).any(dim=(1, 2))
    masked_rank.copy_(rank)
    masked_rank[~vmask] = 999
    umasked_rank = masked_rank.unsqueeze(1)
    valid_mask = ((actions.unsqueeze(-1) == umasked_rank).any(dim=-1)).view(params.batch, 4)
    valid_idx = valid_mask.float().argmax(dim=1)
    vactions = actions.gather(1, valid_idx.unsqueeze(1)).squeeze(1)
    
    return vactions.int()
def SelectAction(state, epsilon, model):
    israndom = False
    if random.random() < epsilon:
        israndom = True
        randoma = torch.stack([torch.randperm(4, device=device) for _ in range(params.batch)])
        randoma = return_valid_move(state.view(params.batch, 4, 4), randoma)
    if not israndom:    
        with torch.no_grad():
            q_values = model(state.float())
            sorted_actions = torch.argsort(q_values, descending=True)
            actions = return_valid_move(state.view(params.batch, 4, 4), sorted_actions)
    return actions if not israndom else randoma


def trainstep(buffer, online_q, target_q, optimizer, batch_size, gamma):
    sample, idxs = buffer.sample(batch_size)
    states = sample[0]
    actions = sample[1]
    rewards = sample[2]
    next_states = sample[3]
    dones = sample[4]
    q_values = online_q(states.float())
    q_taken = q_values.gather(1, actions.long().view(params.batch, 1))

    next_q = target_q(next_states.float())
    max_next_q = next_q.max(1)[0]
    target = (rewards + gamma * max_next_q * (1 - dones.int())).view(params.batch, 1)
    loss = nn.MSELoss(reduction='none')(q_taken, target.detach())
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def issafe(grid):
    mask = torch.full((params.batch, 1), 1, device=device)
    rmask = (grid[:, :, :-1] == grid[:, :, 1:])
    dmask = (grid[:, :-1, :] == grid[:, 1:, :])
    mask = rmask.any(dim=(1, 2)) | dmask.any(dim=(1, 2)) | (grid.any(dim=(1, 2)) == 0)

    return mask

weights = torch.tensor([[16., 15., 14., 13.], 
                    [9. ,10. ,11. , 12.],
                    [8., 7., 6., 5.],
                    [1., 2., 3., 4.]], device=device)
def evaluate(grids):
    grids = torch.where(grids == 0, 0, torch.log2(grids))
    grids = grids.view(params.batch, 4, 4)
    weighted_sum = (grids * weights).sum(dim=(1, 2)).view(params.batch, 1)
    smoothness = torch.full((params.batch,1), 0.0, device=device)
    mask = ((grids[:, :-1, :] != 0) & (grids[:, 1:, :] != 0))
    smoothness -= ((abs((grids[:, :-1, :] - grids[:, 1:, :]) * mask)).view(params.batch, 12)).sum(dim=1).unsqueeze(-1)
    mask = ((grids[:, :, :-1] != 0) & (grids[:, :, 1:] != 0))
    smoothness -= ((abs((grids[:, :, :-1] - grids[:, :, 1:]) * mask)).view(params.batch, 12)).sum(dim=1).unsqueeze(-1)

    empty_tiles = torch.full((params.batch, 1), 0.0, device=device)
    mask = (grids == 0).view(params.batch, -1)
    empty_tiles += (1 * mask).sum(dim=1).unsqueeze(-1)
    max_tile = grids.view(params.batch, 16).max(dim=1).values.unsqueeze(-1)
    empty_weight = torch.full((params.batch, 1), 0.0, device=device)
    smoothness_weight = torch.full((params.batch ,1), 0.0, device=device) 
    mask = max_tile < 10
    empty_weight += 4 * mask
    smoothness_weight += 0.2 * mask
    mask = (max_tile >= 10) & (max_tile < 11)
    empty_weight += 2.5 * mask
    smoothness_weight += 0.8 * mask
    mask = max_tile >= 11
    empty_weight += 1.5 * mask
    smoothness_weight += 3 * mask
    safe = torch.full((params.batch, 1), 0, device=device)
    mask = empty_tiles == 0
    safe += mask
    mask = (issafe(grids)).unsqueeze(-1)
    safe = (safe == 1) & (mask == 0)
    max_tile = 1.5 ** max_tile
    score = (((empty_tiles * max_tile * empty_weight) + (smoothness * (max_tile * smoothness_weight)) + (max_tile) + (weighted_sum)) * (~safe).int())

    return score.view(params.batch,)