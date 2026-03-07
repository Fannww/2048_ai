import torch
from env import Env2048
from dqn import SelectAction, evaluate
import setup

def evaluate_model():
    state = setup.env.reset()
    done = torch.tensor(False)
    while not done.all():
        action = SelectAction(state, 0, setup.online_q)
        next_state = setup.env.step(action)
        _, done = evaluate(state)
        state = next_state
    print(state)
    return state.sum(dim=1).max().item()
