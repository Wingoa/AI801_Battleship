import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

from gymnasium.envs.registration import register, registry

if 'Battleship-v0' not in registry:
    register(
        id='Battleship-v0',
        entry_point='battleship4:BattleshipEnv'
    )

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_ships = {'Carrier': 5, 'Battleship': 4, 'Cruiser': 3, 'Submarine': 3, 'Destroyer': 2}

class BattleshipEnv(gym.Env):

    def __init__(self, board_size=5, ships=None, max_steps=None):
        super().__init__()
        self.ships = ships or default_ships
        self.board_size = board_size
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.max_steps = max_steps or 2 * self.board_size * self.board_size
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.total_ship_cells = sum(self.ships.values())
        self.hidden_board = None
        self.obs_board = None
        self.hits_found = 0
        self.shots_taken = 0
        self.steps_taken = 0
        self.ship_cells_remaining = 0
        self.ai_obs_board = None
        self.row = 0
        self.col = 0
        self.hit_list = []
        self.reset()

    def get_state(self):
        return np.array(self.obs_board, dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset()
        self.hidden_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.obs_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.hits_found = 0
        self.steps_taken = 0
        self.shots_taken = 0
        self.ship_cells_remaining = self.total_ship_cells
        self.ai_obs_board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.row = 0
        self.col = 0
        self.hit_list = []
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
        # Decode the action (integer 0-99) into grid coordinates
        row = action // self.board_size
        col = action % self.board_size

        self.steps_taken += 1
        self.shots_taken += 1

        reward = 0.0
        done = False
        hit = False
        info = {}

        # Check if the agent already shot here
        if self.obs_board[row][col] != 0:
            # This cell was already revealed (hit or miss before)
            reward = -5   # negative reward for redundant guess
        else:
            # This is a new cell being targeted
            if self.hidden_board[row][col] == 1:
                # It's a hit!
                self.obs_board[row][col] = 1  # Mark the cell as hit in the observation board
                self.hits_found += 1
                self.ship_cells_remaining -= 1
                hit = True
                reward = 2.0  # reward for a hit
            else:
                # It's a miss!
                self.obs_board[row][col] = -1  # Mark the cell as a miss in the observation board
                reward = -1  # Small penalty for a miss to encourage fewer moves

        # Check if all ships have been sunk
        if self.ship_cells_remaining == 0:
            done = True
            terminated = True
            truncated = False
            reward = 10
        elif self.steps_taken >= self.max_steps:
            done = True
            terminated = False  # not all ships were found so treat as failure
            truncated = True
        else:
            done = False
            terminated = False
            truncated = False

        obs_array = np.array(self.obs_board, dtype=np.int32)
        info = {'ships_remaining': self.ship_cells_remaining, 'hit': hit, 'shots_taken': self.shots_taken}
        return obs_array, reward, terminated, truncated, info

def train(env_name='Battleship-v0', num_episodes=1000):
    env = gym.make(env_name)
    agent = DQNAgent(
        state_dim=np.prod(env.observation_space.shape),
        action_dim=env.action_space.n,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    eps_start, eps_end, eps_decay = 1.0, 0.001, 1000
    total_rewards, wins, hits_to_win = [], [], []

    for episode in range(1, num_episodes + 1):
        # Gymnasium reset() returns (obs, info)
        obs, _ = env.reset()
        state = obs.flatten()
        ep_reward = 0.0
        ep_hits = 0
        done = False

        while not done:
            eps = eps_end + (eps_start - eps_end) * np.exp(-1.0 * agent.steps_done / eps_decay)
            action = agent.select_action(state, eps)

            # step() returns (obs, reward, terminated, truncated, info)
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

        # define a win as "all ships sunk"
        ships_left = last_info.get('ships_remaining')
        shots_taken = last_info.get('shots_taken')

        print ("shots_taken:", shots_taken)
        print ("ships_left:", ships_left)
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

    recent = wins[-50:]  # last 50 episodes (or fewer if <50 run)
    win_ratio = sum(recent) / len(recent)  # avoids division by zero
    print(f"Win Ratio (last {len(recent)} eps): {win_ratio:.2f}")

    overall_ratio = sum(wins) / len(wins)
    print(f"Overall Win Ratio after {len(wins)} eps: {overall_ratio:.2f}")

    env.close()
    return total_rewards, wins

if __name__ == '__main__':
    train()