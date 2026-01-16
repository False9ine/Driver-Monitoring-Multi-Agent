import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from passenger_env import PassengerSafetyEnv

SEQ_DIR = "data/passenger_sequences"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# DQN NETWORK
# ---------------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# LOAD SEQUENCES
# ---------------------------
sequences = []
for file in os.listdir(SEQ_DIR):
    sequences.append(np.load(os.path.join(SEQ_DIR, file)))

print(f"Loaded {len(sequences)} sequences")

# ---------------------------
# TRAINING SETUP
# ---------------------------
model = DQN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

# ---------------------------
# TRAINING LOOP
# ---------------------------
for epoch in range(30):
    total_reward = 0

    for seq in sequences:
        env = PassengerSafetyEnv(seq)
        state = env.reset()

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)

            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            target = reward
            if next_state is not None:
                next_tensor = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
                target += gamma * torch.max(model(next_tensor)).item()

            q_values = model(state_tensor)
            target_vec = q_values.clone()
            target_vec[action] = target

            loss = loss_fn(q_values, target_vec.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Epoch {epoch+1} | Total Reward: {total_reward:.2f} | ε={epsilon:.2f}")

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save(model.state_dict(), "rl_training/passenger_dqn.pth")
print("✅ Passenger safety RL model trained & saved.")
