from env import Env2048
from dqn import NN, ReplayBuffer, SelectAction, trainstep, evaluate
import torch
import torch.optim as optim


episodes = 10
batch = 128
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.05
decay = 0.7

device = torch.device("cuda:0")
env = Env2048()
model = NN().to(device)
model.load_state_dict(torch.load("model.pt"))
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer(100000)
for ep in range(episodes):
    print (ep)
    states = env.reset()
    done = torch.full((128, 1), 0, dtype=torch.bool, device=device)
    max_tile = 0
    while not done.all():

        action = SelectAction(states, epsilon, model)
        next_state, _, done = env.step(action)
        reward = evaluate(next_state)

        buffer.push(states, action, reward, next_state.float(), done.to(device=device))
        
        states = next_state.float().view(128, 16)

        if len(buffer) > batch:
            for _ in range(8):
                trainstep(buffer, model, optimizer, batch, gamma)
    max_tile = max((states.max().item()), max_tile)
    print(max_tile)
    epsilon = max(min_epsilon, epsilon * decay)

print('done training')
torch.save(model.state_dict(), "model.pt")