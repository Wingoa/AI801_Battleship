import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simplified Environment
class SimpleBattleshipEnv(gym.Env):
    def __init__(self):
        self.board_size = 10
        self.ship_cells = 7
        self.max_steps = 93
        self.action_space = gym.spaces.Discrete(self.board_size ** 2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.board_size ** 2,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.board_size ** 2, dtype=np.int32)
        self.hidden = np.zeros(self.board_size ** 2, dtype=np.int32)
        self.steps = 0
        self.hits = 0
        ship_positions = random.sample(range(self.board_size ** 2), self.ship_cells)
        for pos in ship_positions:
            self.hidden[pos] = 1
        return self.board.copy(), {}
    
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if self.board[action] != 0:  # Already picked
            reward = -15.0
        elif self.hidden[action] == 1:  # Hit
            reward = 25.0
            self.hits += 1
            self.board[action] = 1
        else:  # Miss
            reward = -0.5
            self.board[action] = -1

        self.steps += 1

        if self.hits == self.ship_cells:
            reward += 35.0  # Win bonus
            terminated = True
            
        elif self.steps >= self.max_steps:
            reward -= 2.5  # Time penalty
            truncated = True

        return self.board.copy(), reward, terminated, truncated, {}


# Simple FC Q-Network
class QNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.fc(x)

# Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = QNet(self.state_dim, self.action_dim).to(device)
        self.target_net = QNet(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05 # Lower minimum epsilon
        self.epsilon_decay = 0.9995 # Slower decay
        self.batch_size = 128
        self.replay_buffer = []
        self.buffer_size = 10000
        self.update_target_steps = 250
        self.step_count = 0

    def select_action(self, state):
        valid_actions = np.where(state == 0)[0]
        if random.random() < self.epsilon:
            return int(random.choice(valid_actions))
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = self.q_net(state_tensor)
                q_vals = q_vals.masked_fill(torch.tensor(state != 0).unsqueeze(0).to(device), float("-inf")) # Mask invalid actions
                return int(torch.argmax(q_vals).item())
            
    def store(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_vals = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Print the board every step (optional)
def print_board(board, size): # Pass 'size' as an argument
    symbols = {0: ".", 1: "X", -1: "o"}
    rows = [board[i*size:(i+1)*size] for i in range(size)]
    for row in rows:
        print(" ".join(symbols[val] for val in row))
    print()

#print_board(state)  # Add this after `state = next_state`

# --- Training Loop ---
env = SimpleBattleshipEnv()
agent = DQNAgent(env)

episodes = 5000
wins = 0

MIN_REPLAY_SIZE = 1000  

print("Initializing...")

for ep in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if len(agent.replay_buffer) >= MIN_REPLAY_SIZE:
        
        for _ in range(4):  # Do 4 gradient steps per episode
            agent.train()

        # Only decay epsilon after a certain number of episodes
        if ep > 500:  # Changed from 200 to 500
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

    if terminated:
        wins += 1

    if ep % 10 == 0 or ep == 1:        
        win_rate = wins / ep
        print(f"Episode {ep:4d} | Reward: {total_reward:6.1f} | Win Rate: {win_rate:.2%} | Epsilon: {agent.epsilon:.2f}")

print("\nTraining complete!")

# Print final board state
final_board, _ = env.reset()
print("Final Board State:")
print_board(final_board, env.board_size)

# Final win rate
final_win_rate = wins / episodes
print(f"Final Win Rate: {final_win_rate:.2%}")


# Save the model
torch.save(agent.q_net.state_dict(), "battleship_dqn.pth")
print("Model saved as battleship_dqn.pth")
