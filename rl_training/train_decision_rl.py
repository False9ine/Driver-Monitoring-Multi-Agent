import torch
import torch.nn as nn
import torch.optim as optim
import random

from passenger_safety_env import PassengerSafetyEnv
from dqn import DQN

env = PassengerSafetyEnv()
model = DQN()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

for episode in range(500):
    state = env.reset()
    total_reward = 0

    while True:
        state_t = torch.tensor(state, dtype=torch.float32)

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                action = torch.argmax(model(state_t)).item()

        next_state, reward, done = env.step(action)
        total_reward += reward

        target = reward
        if not done:
            with torch.no_grad():
                target += gamma * torch.max(
                    model(torch.tensor(next_state, dtype=torch.float32))
                ).item()

        q_vals = model(state_t)
        q_target = q_vals.clone()
        q_target[action] = target

        loss = loss_fn(q_vals, q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode} | Reward: {total_reward:.2f}")

torch.save(model.state_dict(), "rl_training/passenger_decision_dqn.pth")
print("✅ Passenger decision RL trained")