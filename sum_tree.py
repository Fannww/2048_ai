import numpy as np
import torch
import params

device = torch.device('cuda:0')
class sum_tree:
    def __init__(self, capacity):
        self.tree = torch.zeros(2 * capacity - 1, device=device)
        self.write = 0
        self.capacity = capacity

    def add(self, value):
        
        write = (torch.arange(self.write, self.write + params.batch, device=device) % self.capacity)
        self.update(write, value)
        self.write = (self.write + params.batch) % self.capacity
        return write
    
    def update(self, idx, value):
        widx = idx + self.capacity - 1
        change = value - self.tree[widx]
        self.tree[widx] = value
        mask = torch.full_like(idx,  1, dtype=torch.bool, device=device)
        while True:
            widx[mask] =  (widx[mask] - 1) // 2
            mask = widx >= 0
            if not mask.any():
                break
            self.tree.scatter_add_(0, widx[mask], change[mask])

    def get_sum(self):
        return self.tree[0]
    
    def sample(self, samples, lens):
        idxs = torch.zeros(len(samples), dtype=int, device=device)
        i = 0
        for sample in samples:
            idx = 0
            while idx < (self.capacity - 1):
                left = self.tree[idx * 2 + 1]
                if sample <= left:
                    idx = idx * 2 + 1
                else:
                    idx = idx * 2 + 2
                    sample = sample - left
            idxs[i] = int(idx - (self.capacity - 1))
            if idxs[i] > lens:
                continue
            i += 1
        return idxs