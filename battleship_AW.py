
#Not done yet, but made a little progress

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

from gymnasium.envs.registration import register, registry

env = BattleshipEnv()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BattleshipEnv(gym.Env):

    def __init__(self, max_steps=100):
        self.ships = {'Carrier': 5, 'Battleship': 4, 'Cruiser': 3, 'Submarine': 3, 'Destroyer': 2}
        self.max_steps = max_steps
        self.board_size = 10

        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int32)

        self.hidden_board = None
        self.obs_board = None
        self.total_ship_cells = sum(self.ships.values())
        self.ship_cells_remaining = self.total_ship_cells
        self.hits_found = 0
        self.shots_taken = 0
        self.steps_taken = 0
        self.row = 0
        self.col = 0
        self.hit_list = []

    def reset(self, *, seed=None, options=None):
        super().reset()

        self.hidden_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.obs_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.hits_found = 0
        self.steps_taken = 0
        self.shots_taken = 0
        self.ship_cells_remaining = self.total_ship_cells

        for ship_length in self.ships.values():
            placed = False
            while not placed:
                orientation = random.choice(['horizontal', 'vertical'])
                if orientation == 'horizontal':
                    row = random.randrange(0, self.board_size)
                    col = random.randrange(0, self.board_size - ship_length + 1)
                    if all(self.hidden_board[row][c] == 0 for c in range(col, col + ship_length)):
                        for c in range(col, col + ship_length):
                            self.hidden_board[row][c] = 1
                        placed = True
                else:
                    row = random.randrange(0, self.board_size - ship_length + 1)
                    col = random.randrange(0, self.board_size)
                    if all(self.hidden_board[r][col] == 0 for r in range(row, row + ship_length)):
                        for r in range(row, row + ship_length):
                            self.hidden_board[r][col] = 1
                        placed = True

        return np.array(self.obs_board, dtype=np.int32), {}

    def step(self, action):
        self.row = action // self.board_size
        self.col = action % self.board_size

        self.steps_taken += 1
        self.shots_taken += 1

        reward = 0.0
        done = False
        hit = False
        info = {}

        if self.obs_board[self.row][self.col] != 0:
            reward = -5  # penalize redundant guess
        else:
            if self.hidden_board[self.row][self.col] == 1:
                self.obs_board[self.row][self.col] = 1
                self.hits_found += 1
                self.ship_cells_remaining -= 1
                hit = True
                reward = 2.0
            else:
                self.obs_board[self.row][self.col] = -1
                reward = -1.0  # stronger penalty for miss

        if self.ship_cells_remaining == 0:
            done = True
            terminated = True
            truncated = False
            reward = 100
        elif self.steps_taken >= self.max_steps:
            done = True
            terminated = False
            truncated = True
        else:
            done = False
            terminated = False
            truncated = False

        obs_array = np.array(self.obs_board, dtype=np.int32)
        info = {'ships_remaining': self.ship_cells_remaining, 'hit': hit, 'shots_taken': self.shots_taken}

        return obs_array, reward, terminated, truncated, info


Transition = collections.namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):  # increased hidden size
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device='cpu',
        gamma=0.99,
        lr=1e-4,  # lowered learning rate
        batch_size=64,
        memory_capacity=50000,  # increased replay size
        target_update_freq=1000
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = target_update_freq
        self.steps_done = 0

        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state, eps_threshold):
        # Normalize state from {-1,0,1} -> {0,0.5,1} to help learning
        norm_state = (state + 1) / 2

        avail = np.where(state == 0)[0]  # unvisited cells

        if random.random() < eps_threshold:
            return int(random.choice(avail))

        state_v = torch.from_numpy(norm_state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(state_v).squeeze(0)

        q_vals_np = q_vals.cpu().numpy()
        q_vals_np[state != 0] = -np.inf

        return int(np.argmax(q_vals_np))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps_done += 1
        batch = self.memory.sample(self.batch_size)
        trans = Transition(*zip(*batch))

        states_np      = np.array(trans.state,      dtype=np.float32)
        next_states_np = np.array(trans.next_state, dtype=np.float32)
        actions_np     = np.array(trans.action,     dtype=np.int64)
        rewards_np     = np.array(trans.reward,     dtype=np.float32)
        dones_np       = np.array(trans.done,       dtype=np.float32)

        # Normalize states for learning
        states_np = (states_np + 1) / 2
        next_states_np = (next_states_np + 1) / 2

        states      = torch.from_numpy(states_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        actions     = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(rewards_np).unsqueeze(1).to(self.device)
        dones       = torch.from_numpy(dones_np).unsqueeze(1).to(self.device)

        q_values = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)  # gradient clipping
        self.optimizer.step()

        if self.steps_done % self.update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

def train(env_name='Battleship-v0', num_episodes=1000):
    env = BattleshipEnv()
    agent = DQNAgent(
        state_dim=np.prod(env.observation_space.shape),
        action_dim=env.action_space.n,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    eps_start, eps_end, eps_decay = 1.0, 0.05, 1000  # slightly higher eps_end for exploration
    total_rewards, wins, hits_to_win = [], [], []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = obs.flatten()
        ep_reward = 0.0
        ep_hits = 0
        done = False

        while not done:
            eps = eps_end + (eps_start - eps_end) * np.exp(-1.0 * agent.steps_done / eps_decay)
            action = agent.select_action(state, eps)

            obs2, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get('hit', False):
                ep_hits += 1

            state2 = obs2.flatten()

            agent.memory.push(state, action, reward, state2, done)
            state = state2
            ep_reward += reward
            last_info = info

            agent.update()

        total_rewards.append(ep_reward)
        ships_left = last_info.get('ships_remaining')
        wins.append(terminated)
        hits_to_win.append(ep_hits if terminated else None)

        if episode % 50 == 0:
            last_50_rewards = total_rewards[-50:]
            last_50_wins    = wins[-50:]
            last_50_hits    = [h for h in hits_to_win[-50:] if h is not None]

            avg_reward = np.mean(last_50_rewards)
            win_ratio  = sum(last_50_wins) / len(last_50_wins)
            avg_hits   = np.mean(last_50_hits) if last_50_hits else float('nan')

            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Win Ratio: {win_ratio:.2f} | "
                  f"Avg Hits-to-Win: {avg_hits:.2f}")

    recent = wins[-50:]
    win_ratio = sum(recent) / len(recent)
    print(f"Win Ratio (last {len(recent)} eps): {win_ratio:.2f}")

    overall_ratio = sum(wins) / len(wins)
    print(f"Overall Win Ratio after {len(wins)} eps: {overall_ratio:.2f}")

    env.close()
    return total_rewards, wins

if __name__ == '__main__':
    train()
