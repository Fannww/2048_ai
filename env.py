import akioi_2048 as ak
import numpy as np
import torch

device = torch.device("cuda:0")
class Env2048:
    def __init__(self):
        self.boards = np.empty((128, 4, 4), dtype=int)

    def reset(self):
        self.boards = np.empty((128, 4, 4), dtype=int)
        for k in range(128):
            board = ak.init()
            for i in range(4):
                for j in range(4):
                    if board[i][j] < 0:
                        board[i][j] = 0
            self.boards[k] = board
        return torch.as_tensor(self.boards,device=device).view(128, 16)
    
    def step(self, actions):
        dones = []
        #action up=0 down=1 left=2 right=3
        for k, (board, action) in enumerate(zip(self.boards, actions.squeeze())):
            direction = [ak.Direction.Up, ak.Direction.Down, ak.Direction.Left, ak.Direction.Right][action]
            #print(f'board {i}')
            new_board, _, done = ak.step(board, direction)
            dones.append(False if done == ak.State.Continue else True )
            for i in range(4):
                for j in range(4):
                    if new_board[i][j] < 0:
                        new_board[i][j] = 0
            self.boards[k] = np.array(new_board)
        return torch.as_tensor(self.boards, device=device).view(128, 16), _, torch.as_tensor(dones)