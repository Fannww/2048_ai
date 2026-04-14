from dqn import SelectAction, trainstep, issafe, evaluate
import torch
import params
import setup
import matplotlib.pyplot as plt
from evaluate import evaluate_model
from prompt import resume

if resume:
    checkpoint = torch.load("checkpoint.pt", weights_only=False)
plt.ion()
avg_score = [] if not resume else checkpoint["avg_score"]
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel("Episodes")
ax.set_ylabel("Max_score")
plt.show()

steps = 0
for ep in range(0 if not resume else (checkpoint["episode"] + 1), params.episodes):
    print (ep)
    states = setup.env.reset()
    done = torch.full((params.batch, 1), 0, dtype=torch.bool, device=setup.device)
    while not done.all():

        action = SelectAction(states, setup.epsilon, setup.online_q)
        old_state = states.clone()
        states, reward = setup.env.step(action)
        current_max = states.squeeze(-1).max(dim=1)[0]
        done = ~(issafe(states.view(params.batch, 4, 4)))
        evaluation = (evaluate(states) / 75) - 10
        norevaluation = torch.sigmoid(evaluation)
        reward = reward * (1 + norevaluation)
        setup.buffer.push(old_state, action, reward, states, done.to(device=setup.device))
        if len(setup.buffer) > params.batch:
            for _ in range(16):
                trainstep(setup.buffer, setup.online_q, setup.target_q, setup.optimizer, params.batch, params.gamma)
                steps += 1
                if steps % params.target_update == 0:
                    setup.target_q.load_state_dict(setup.online_q.state_dict())
        
    a_score = evaluate_model()
    avg_score.append(a_score)

    torch.save({
        "online_q": setup.online_q.state_dict(),
        "target_q": setup.target_q.state_dict(),
        "episode": ep,
        "epsilon": setup.epsilon,
        "buffer": setup.buffer,
        "optimizer": setup.optimizer.state_dict(),
        "avg_score": avg_score
        
    }, "checkpoint.pt")

    line.set_data(range(len(avg_score)), avg_score)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

    setup.epsilon = max(params.min_epsilon, setup.epsilon * params.decay)
print('done training')
print((avg_score))
torch.save({
        "online_q": setup.online_q.state_dict()
    }, "model_final.pt")