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
        if lens == 31104:
            pass
        idxs = torch.zeros(len(samples), dtype=int, device=device)
        idx = torch.zeros(len(samples), dtype=int, device=device)
        ndone = torch.full((len(samples),), 1, dtype=torch.bool, device=device)
        nomorendone = torch.full((len(samples),), 1, dtype=torch.bool, device=device)
        while ndone.any():
            left = self.tree[idx * 2 + 1]
            lmask = samples <= left
            idx[lmask] = idx[lmask] * 2 + 1
            rmask = samples > left
            idx[rmask] = idx[rmask] * 2 + 2
            samples[rmask] = samples[rmask] - left[rmask]
            ndone[nomorendone] = idx[nomorendone] < (self.capacity - 1)
            idxs[~ndone ^ ~nomorendone if ndone.size(0) != 0 else None] = idx[~ndone ^ ~nomorendone if ndone.size(0) != 0 else None] - (self.capacity - 1)
            idx[~ndone if ndone.size(0) else None] = 0
            nomorendone = ndone.clone()
        nvmask = idxs > lens
        idxs[nvmask] = 0
        return idxs