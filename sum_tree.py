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
    
    def update(self, idx, values):
        widx = idx + self.capacity - 1
        change = values - self.tree[widx]
        self.tree[widx] = values

        while True:
            widx = (widx - 1) // 2
            if (widx == 0).all():
                self.tree[0] += change.sum()
                break
            self.tree.scatter_add_(0, widx, change)

    def get_sum(self):
        return self.tree[0]
    
    def sample(self, samples, lens):
        idxs = torch.zeros_like(samples, dtype=torch.long, device=device)

        max_depth = (self.capacity - 1).bit_length()
        for _ in range(max_depth):
            left = idxs * 2 + 1
            right = idxs * 2 + 2
            cond = samples <= self.tree[left]
            idxs = torch.where(cond, left, right)
            samples = torch.where(cond, samples, samples - self.tree[left])
        idxs.sub_(self.capacity - 1)
        idxs = torch.where(idxs > lens, 0, idxs)
        return idxs