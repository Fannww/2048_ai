import torch
import params
import torch.nn.functional as F

device = torch.device("cuda:0")
def make_grids():
    grids = torch.full((params.batch, 16), 0, device=device)
    probs = torch.ones(params.batch, 16, device=device)
    cords = torch.multinomial(probs, 2)
    probs = torch.tensor([[0.9, 0.1]] * params.batch, device=device)
    values = torch.multinomial(probs, 2, replacement=True)
    values = (values + 1) * 2
    batch_idx = torch.arange(params.batch, device=device).unsqueeze(1)
    grids[batch_idx, cords] = values

    return grids.int()
#action up=0 down=1 left=2 right=3
def step(grids, actions):
    scores = torch.full_like(actions, 0, device=device)
    oldg = (grids.clone()).view(-1, 4, 4)
    grids = grids.view(-1, 4, 4) 
    #actions masks
    up_mask = actions == 0
    down_mask = actions == 1
    right_mask = actions == 3
    grids[up_mask] = grids[up_mask].rot90(1, (1, 2))
    grids[down_mask] = grids[down_mask].rot90(3, (1, 2))
    grids[right_mask] = grids[right_mask].rot90(2, (1, 2))
    mask = grids != 0

    #move logic
    idxs = (mask.cumsum(dim=2) - 1).clamp(min=0) 
    out = torch.zeros_like(grids, device=device) 
    out.scatter_add_(2, idxs, grids) 
    m_mask = out[:, :, :-1] == out[:, :, 1:] 
    m_mask = F.pad(m_mask, (0, 1, 0, 0), mode='constant', value=0) if mask.any() else torch.zeros((1, 4, 4), device=device) 
    fm0, fm1 = m_mask[:, :, 0], m_mask[:, :, 1]
    fake_zero_mask = torch.stack([(fm0 == True) & (fm1 == True), (fm0 == False) & ((fm1 == True) & (m_mask[:, :, 2] == True))], dim=2)
    m_mask[:, :, 1:3].logical_xor_(fake_zero_mask)
    tt0_mask = m_mask.roll(1, 2) if mask.any() else None
    out[tt0_mask] = 0
    out[m_mask if mask.any() else None] *= 2
    scores = (out * m_mask).sum(dim=(1,2)).int()
    mask = out != 0
    idxs = mask.cumsum(2) - 1
    c_out = torch.zeros_like(out)
    c_out.scatter_add_(2, idxs.clamp(min=0), out)
    grids = c_out

    #rotate grids back
    grids[up_mask] = grids[up_mask].rot90(3, (1, 2))
    grids[down_mask] = grids[down_mask].rot90(1, (1, 2))
    grids[right_mask] = grids[right_mask].rot90(2, (1, 2))

    #add 2-4 value to random empty place :D to only those that moved ^_^
    moved = (oldg != grids).any(dim=(1, 2))
    grids = grids.view(actions.size(0), 16)
    empty = grids == 0
    probs = torch.rand_like(empty, dtype=torch.float32, device=device)
    probs[~empty] -= 1
    cols = (torch.argmax(probs, dim=1))[moved]
    probs = torch.tensor([[0.9, 0.1]] * actions.size(0), device=device)
    values = torch.multinomial(probs, 1, replacement=True)
    values = ((values + 1) * 2).int()
    grids[moved, cols] = values[moved].squeeze()

    return grids.int(), scores

