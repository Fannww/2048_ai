import torch
import params
import gym

device = torch.device("cuda:0")
class Env2048:
    def __init__(self, batch=params.batch):
        self.batch = batch
        self.boards = gym.make_grids()

    def reset(self):
        self.boards = gym.make_grids()
        return self.boards

    def step(self, actions):
        grids = gym.step(self.boards, actions)
        self.boards = grids
        return grids