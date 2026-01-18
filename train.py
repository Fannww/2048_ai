from dqn import SelectAction, trainstep, evaluate
import torch
import params
import setup

steps = 0
for ep in range(params.episodes):
    print (ep)
    states = setup.env.reset()
    done = torch.full((params.batch, 1), 0, dtype=torch.bool, device=setup.device)
    max_tile = 0
    while not done.all():

        action = SelectAction(states, params.epsilon, setup.online_q)
        next_state, _, done = setup.env.step(action)
        reward = evaluate(next_state)
        maxp = setup.buffer.maxpr()
        setup.buffer.push(states, action, reward, next_state, done.to(device=setup.device), torch.full((params.batch,), maxp, dtype=torch.float, device=setup.device))
        states = next_state.view(params.batch, 16)
        if len(setup.buffer) > params.batch:
            for _ in range(8):
                trainstep(setup.buffer, setup.online_q, setup.target_q, setup.optimizer, params.batch, params.gamma)
                steps += 1
                if steps % params.target_upddate == 0:
                    setup.target_q.load_state_dict(setup.online_q.state_dict())
    
    max_tile = max((states.max().item()), max_tile)
    print(max_tile)
    epsilon = max(params.min_epsilon, params.epsilon * params.decay)
print('done training')
torch.save(setup.model.state_dict(), "model.pt")