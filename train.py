from dqn import SelectAction, trainstep, issafe, evaluate
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
    while not done.all():

        action = SelectAction(states, params.epsilon, setup.online_q)
        old_state = states.clone()
        _, reward = setup.env.step(action)
        current_max = states.squeeze(-1).max(dim=1)[0]
        done = ~(issafe(states.view(params.batch, 4, 4)))
        evaluation = (evaluate(states) / 75) - 10
        norevaluation = torch.sigmoid(evaluation)
        reward = reward * norevaluation
        maxp = setup.buffer.maxpr()
        setup.buffer.push(old_state, action, reward, states, done.to(device=setup.device), torch.full((params.batch,), maxp, dtype=torch.float, device=setup.device))
        if len(setup.buffer) > params.batch:
            for _ in range(16):
                trainstep(setup.buffer, setup.online_q, setup.target_q, setup.optimizer, params.batch, params.gamma)
                steps += 1
                if steps % params.target_update == 0:
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
print((max_score))
torch.save({
    'model': setup.online_q.state_dict(),
    'optimizer': setup.optimizer.state_dict(),
}, "model.pt")