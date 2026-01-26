from dqn import SelectAction, trainstep, evaluate
import torch
import params
import setup
import matplotlib.pyplot as plt
from evaluate import evaluate_model
plt.ion()
max_score = []
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel("Episodes")
ax.set_ylabel("Max_score")
plt.show()

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
    
    m_score = evaluate_model()
    max_score.append(m_score)

    line.set_data(range(len(max_score)), max_score)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

    params.epsilon = max(params.min_epsilon, params.epsilon * params.decay)
print('done training')
torch.save(setup.online_q.state_dict(), "model.pt")