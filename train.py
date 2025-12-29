from dqn import SelectAction, trainstep, evaluate
import torch
import config


for ep in range(config.episodes):
    print (ep)
    states = config.env.reset()
    done = torch.full((config.batch, 1), 0, dtype=torch.bool, device=config.device)
    max_tile = 0
    while not done.all():

        action = SelectAction(states, epsilon, config.model)
        next_state, _, done = config.env.step(action)
        reward = evaluate(next_state)

        config.buffer.push(states, action, reward, next_state.float(), done.to(device=config.device))
        
        states = next_state.float().view(config.batch, 16)

        if len(config.buffer) > config.batch:
            for _ in range(8):
                trainstep(config.buffer, config.model, config.optimizer, config.batch, config.gamma)
    max_tile = max((states.max().item()), max_tile)
    print(max_tile)
    epsilon = max(config.min_epsilon, epsilon * config.decay)

print('done training')
torch.save(config.model.state_dict(), "model.pt")