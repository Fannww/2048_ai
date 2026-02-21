import akioi_2048 as ak
import pygame
pygame.init()
import copy
import sys
import time
import numpy as np
from dqn import NN, SelectAction
import torch
import gym
import params
params.batch = 1

model = NN()
model.load_state_dict(torch.load("model.pt"))
model.eval()
board = gym.make_grids()

#le screen
WIDTH, HEIGHT = 804, 804
screen = pygame.display.set_mode((WIDTH, HEIGHT)) #pygame.FULLSCREEN
pygame.display.set_caption("2048 AI")
clock = pygame.time.Clock()


#colors
white = (255, 255, 255)
grey = (105, 105, 105)
beige = (187, 173, 160)
color = {
            0: beige,
            2: (238, 228, 218),       # very light beige
            4: (237, 224, 200),       # light beige
            8: (242, 177, 121),       # soft orange
            16: (245, 149, 99),       # medium orange
            32: (246, 124, 95),       # deep orange
            64: (246, 94, 59),        # red-orange
            128: (237, 207, 114),     # light gold
            256: (237, 204, 97),      # brighter gold
            512: (237, 200, 80),      # yellow-gold
            1024: (237, 197, 63),     # deeper gold
            2048: (237, 194, 46),     # saturated golden yellow
            4096: (173, 183, 119),    # soft olive green
            8192: (170, 183, 102),    # slightly deeper olive
            16384: (164, 183, 79),    # green-gold
            32768: (161, 183, 63),    # olive gold
            65536: (158, 183, 47),    # rich olive
            131072: (155, 183, 31),   # deeper green-gold
    }



#backround
background = pygame.Surface((WIDTH, HEIGHT))
background.fill(beige)


#you lose screen
font = pygame.font.SysFont('Arial', 80)
text = font.render('Game Over!', True, (255, 0, 0))
text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
#score
score = 0
prevscore = 0


#draw function
font = pygame.font.SysFont('Arial', 100)
def draw(value, row, col):
    if value < 0:
        value = 0
    x = 6 + (200 * col)
    y = 6 + (200 * row)
    tile_size = 193
    rect = pygame.Rect(x, y, tile_size, tile_size)
    pygame.draw.rect(background, color[int(abs(value))], rect, border_radius=20)
    
    #value
    if value != 0:
        text = font.render(str(int(value)), True, (255, 255, 255))
        text_rect = text.get_rect(center=rect.center)
        background.blit(text, text_rect)
firstr = True
moved = False
moved_ai = False
Game_over = False
#running loop
running = True
last_time_ai = time.time()
delay = 0.01
while running:


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            moved = True
            if event.key == pygame.K_r:
                b_after = gym.make_grids()
                score = 0
                Game_over = False
            if event.key == pygame.K_w:
                b_after = gym.step(board, torch.tensor(0).unsqueeze(0))
            if event.key == pygame.K_a:
                b_after = gym.step(board, torch.tensor(2).unsqueeze(0))
            if event.key == pygame.K_s:
                b_after = gym.step(board, torch.tensor(1).unsqueeze(0))
            if event.key == pygame.K_d:
                b_after = gym.step(board, torch.tensor(3).unsqueeze(0))
            score += 0
            old_b = (board.to('cpu')).numpy()
            new_b = (b_after.to('cpu')).numpy() 
            moved = not np.array_equal(old_b, new_b)
            if moved:
                board = b_after
    #ai makes move
    if time.time() - last_time_ai == -1:#delay:
        i = 0
        action = SelectAction(torch.tensor(board, dtype=torch.float32).flatten().unsqueeze(0), 0, model)
        while not moved_ai and i < 4:
            if action[i] == 0:
                b_after, t_score, _ = ak.step(board.unsqueeze(0), ak.Direction.Up)
            if action[i] == 1:
                b_after, t_score, _ = ak.step(board.unsqueeze(0), ak.Direction.Down)
            if action[i] == 2:
                b_after, t_score, _ = ak.step(board.unsqueeze(0), ak.Direction.Left)
            if action[i] == 3:
                b_after, t_score, _ = ak.step(board.unsqueeze(0), ak.Direction.Right)
            score += t_score
            old_b = np.array(board)
            new_b = np.array(b_after)
            moved_ai = not np.array_equal(old_b, new_b)
            if moved_ai:
                board = b_after
            last_time_ai = time.time()
            i += 1



    grids = 0     
    if moved or moved_ai or firstr:
        #the lines
        for y in range(0, 804, 200):
            pygame.draw.line(background, grey, (0, y), (804, y), 50)

        for x in range(0, 804, 200):
            pygame.draw.line(background, grey, (x, 0), (x, 804), 50)


        #deseneaza fiecare tile din grid
        for i in range(16):
            row = i // 4
            col = i % 4
            if board[0, i] > 0:
                draw(board[0, i], row, col)
                grids += 1
            else:
                ettile = pygame.Rect(6 + (col * 200), 6 + (row * 200), 193, 193)
                pygame.draw.rect(background, beige, ettile, border_radius=20)
    #check if the grid is full/safe CUZ THIS AK BOARD CANT DO THAT
    full = True if grids == 16 else False
    if moved or moved_ai:
        if full:
            Game_over = True
            for i in range(16):
                if i < 12 and board[0, i] == board[0, i + 4]:
                    Game_over = False
                    break
                if (i % 4) < 3 and board[0, i] == board[0, i + 1]:
                    Game_over = False
                    break
        if Game_over == True:
            background.blit(text, text_rect)
    moved = False
    moved_ai = False

    
    firstr = False
    screen.blit(background, (0, 0))
    if prevscore != score:
        print(str(score))
    prevscore = score

    pygame.display.flip()
pygame.quit()
sys.exit()