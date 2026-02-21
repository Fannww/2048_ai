import torch
import params

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
    oldg = (grids.clone()).view(-1, 4, 4)
    grids = grids.clone()
    grids = grids.view(-1, 4, 4)
    mask = grids != 0 #for all directions
    #left
    left_mask = actions == 2
    left_idxs = mask[left_mask].cumsum(dim=2) - 1 #can work for left and right
    left_out = torch.zeros_like(grids[left_mask]) 
    left_out.scatter_add_(2, left_idxs.clamp_(min=0), grids[left_mask]) #can work for left and right
    lxm_mask = left_out[:, :, :-1] == left_out[:, :, 1:] #will work for left can work for right if i flip it
    lxm_mask = torch.cat([lxm_mask, torch.full((grids[left_mask].size(0), 4, 1), 0, dtype=torch.bool, device=device)], dim=2) if lxm_mask.size(0) != 0 else torch.full((1, 4, 4), 0) #this will make it for left and i can just flip it for  right
    lxfake_zero_mask = torch.stack([(lxm_mask[:, :, 0] == True) & (lxm_mask[:, :, 1] == True), (lxm_mask[:, :, 0] == False) & ((lxm_mask[:, :, 1] == True) & (lxm_mask[:, :, 2] == True))], dim=2)#it should work for the fliped version of left
    lxm_mask[:, :, 1:3] = lxm_mask[:, :, 1:3] ^ lxfake_zero_mask#works for the fliped version
    ltt0_mask = lxm_mask.roll(shifts=1 ,dims=2) if left_mask.any() else None#once agian hmm idk ill haave to see how it works
    left_out[ltt0_mask] = 0
    left_out[lxm_mask if left_mask.any() else None] *= 2
    lmask = left_out != 0
    x_idxs = lmask.cumsum(dim=2) - 1
    cx_out = torch.zeros_like(left_out)
    cx_out.scatter_add_(2, x_idxs.clamp(min=0), left_out)
    grids[left_mask] = cx_out

    #right
    right_mask = actions == 3
    right_idxs = (mask[right_mask].cumsum(dim=2) - 1).clamp(min=0)
    radd_to_idx = (3 - right_idxs[:, :, 3]).unsqueeze(-1)
    right_idxs += radd_to_idx
    right_out = torch.zeros_like(grids[right_mask])
    right_out.scatter_add_(2, right_idxs, grids[right_mask])
    rxm_mask = right_out[:, :, :-1] == right_out[:, :, 1:]
    rxm_mask = torch.cat([torch.full((grids[right_mask].size(0), 4, 1), 0, dtype=torch.bool, device=device), rxm_mask], dim=2)
    frxm_mask = rxm_mask.flip((2))
    rx_fake_zero_mask = torch.stack([(frxm_mask[:, :, 0] == True) & (frxm_mask[:, :, 1] == True), (frxm_mask[:, :, 0] == False) & ((frxm_mask[:, :, 1] == True) & (frxm_mask[:, :, 2] == True))], dim=2)
    rxm_mask[:, :, 1:3] = rxm_mask[:, :, 1:3] ^ rx_fake_zero_mask.flip((2)) 
    rtt0_mask = rxm_mask.roll(-1, 2)
    right_out[rtt0_mask] = 0
    right_out[rxm_mask] *= 2
    rmask = right_out != 0
    right_idxs = (rmask.cumsum(dim=2) - 1).clamp(min=0)
    radd_to_idx = (3 - right_idxs[:, :, 3]).unsqueeze(-1)
    right_idxs += radd_to_idx
    cright_out = torch.zeros_like(right_out)
    cright_out.scatter_add_(2, right_idxs, right_out)
    grids[right_mask] = cright_out
    
    #up
    up_mask = actions == 0
    up_idxs = mask[up_mask].cumsum(dim=1) - 1 #can work for up and down
    up_out = torch.zeros_like(grids[up_mask])
    up_out.scatter_add_(1, up_idxs.clamp_(min=0), grids[up_mask]) #can work for up and down
    um_mask = up_out[:, :-1, :] == up_out[:, 1:, :] #will work for up can work for down if i flip it
    um_mask = torch.cat([um_mask, torch.full((grids[up_mask].size(0), 1, 4), 0, dtype=torch.bool, device=device)], dim=1) if um_mask.size(0) != 0 else torch.full((1, 4, 4), 0) #this will make it for up and i can just flip it for down 
    uyfake_zero_mask = torch.stack([(um_mask[:, 0, :] == True) & (um_mask[:, 1, :] == True), (um_mask[:, 0, :] == False) & ((um_mask[:, 1, :] == True) & (um_mask[:, 2, :] == True))], dim=1)#it should work for the fliped version of up
    um_mask[:, 1:3, :] = um_mask[:, 1:3, :] ^ uyfake_zero_mask
    utt0_mask = um_mask.roll(shifts=1, dims=1) if up_mask.any() else None
    up_out[utt0_mask] = 0
    up_out[um_mask if up_mask.any() else None] *= 2
    umask = up_out != 0
    up_idxs = umask.cumsum(dim=1) - 1 #can work for up and down
    cup_out = torch.zeros_like(up_out)
    cup_out.scatter_add_(1, up_idxs.clamp_(min=0), up_out)     
    grids[up_mask] = cup_out

    #down
    down_mask = actions == 1
    down_idxs = (mask[down_mask].cumsum(dim=1) - 1).clamp(min=0)
    dadd_to_idx = (3 - down_idxs[:, 3, :]).view(grids[down_mask].size(0), 1, 4)
    down_idxs += dadd_to_idx
    down_out = torch.zeros_like(grids[down_mask])
    down_out.scatter_add_(1, down_idxs, grids[down_mask])
    dm_mask = down_out[:, :-1, :] == down_out[:, 1:, :]
    dm_mask = torch.cat([torch.full((grids[down_mask].size(0), 1, 4), 0, device=device, dtype=torch.bool), dm_mask], dim=1) if dm_mask.size(0) != 0 else torch.full((1, 4, 4), 0)
    fdm_mask = dm_mask.flip(1)
    d_fake_zero_mask = torch.stack([(fdm_mask[:, 0, :] == True) & (fdm_mask[:, 1, :] == True), (fdm_mask[:, 0, :] == False) & ((fdm_mask[:, 1, :] == True) & (fdm_mask[:, 2, :] == True))], dim=1)
    dm_mask[:, 1:3, :] = dm_mask[:, 1:3, :] ^ d_fake_zero_mask.flip((1))
    dtt0_mask = dm_mask.roll(-1, 1)
    down_out[dtt0_mask if down_mask.any() else None] = 0
    down_out[dm_mask if down_mask.any() else None] *= 2
    dmask = down_out != 0
    down_idxs = (dmask.cumsum(dim=1) - 1).clamp(min=0)
    dadd_to_idx = (3 - down_idxs[:, 3, :]).view(grids[down_mask].size(0), 1, 4)
    down_idxs += dadd_to_idx
    cdown_out = torch.zeros_like(down_out)
    cdown_out.scatter_add_(1, down_idxs, down_out)
    grids[down_mask] = cdown_out

    #add 2-4 value to random empty place :D to only those that moved ^_^
    moved = (oldg != grids).any(dim=(1, 2))
    fgrids = grids.view(actions.size(0), 16)
    empty = fgrids == 0
    probs = torch.rand_like(empty, dtype=torch.float32, device=device)
    probs[~empty] -= 1
    cols = (torch.argmax(probs, dim=1))[moved]
    probs = torch.tensor([[0.9, 0.1]] * actions.size(0), device=device)
    values = torch.multinomial(probs, 1, replacement=True)
    values = ((values + 1) * 2).int()
    fgrids[moved, cols] = values[moved].squeeze()

    return fgrids.int()

