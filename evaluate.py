import torch
from env import Env2048
from dqn import SelectAction
import setup

def evaluate_model():
    state = setup.env.reset()
    done = torch.tensor(False)
    while not done.any():
        action = SelectAction(state, 0, setup.online_q)
        next_state, _, done = setup.env.step(action)
        state = next_state
    return state.sum(dim=1).max().item()
