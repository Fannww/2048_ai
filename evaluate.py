import torch
from dqn import SelectAction, issafe
import setup
import params

def evaluate_model():
    state = setup.env.reset()
    done = torch.tensor(False, device=setup.device)
    while not done.all():
        action = SelectAction(state, 0, setup.online_q)
        next_state, _ = setup.env.step(action)
        done = ~(issafe(next_state.view(params.batch, 4, 4)))
        state = next_state
    print(state)
    return state.float().sum(dim=1).mean().item()
