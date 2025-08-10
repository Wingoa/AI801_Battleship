import random
import collections
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

class BattleshipEnv(gym.Env):

    def __init__(self, max_steps=100):

        # Use the default ships and sizes based on the board game
        self.ships = {'Carrier': 5, 'Battleship': 4, 'Cruiser': 3, 'Submarine': 3, 'Destroyer': 2}
        #self.ships = {'Carrier': 5} # for testing 1 ship
        self.max_steps = max_steps
        # Define the board size which will be squared - 10x10 grid
        self.board_size = 10

        # Action: Discrete 100 (integer 0-99 corresponding to a grid cell)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)

        # Observation: 10x10 grid of {0, 1, -1} values (unknown, hit, miss)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int32)

        # Initialize state variables

        # Board for actual positions of ships
        self.hidden_board = None

        # Board the agent observes (hits/misses)
        self.obs_board = None

        # Total number of cells occupied by ships.  Could be static value but in case we change the sizes, sum is best
        # Set remaining cells to total ship cells initially
        self.total_ship_cells = sum(self.ships.values())
        self.ship_cells_remaining = self.total_ship_cells

        # Will be used to record the number of hits (ship cells) found so far and how many shots taken each round
        self.hits_found = 0
        self.shots_taken = 0

        # Will be used to record the number of steps taken in current episode
        self.steps_taken = 0

        # Rows and Cols - will use to track potential next targets after a hit
        self.row = 0
        self.col = 0
        self.hit_list = []

    # Reset the board for a new game
    def reset(self, *, seed=None, options=None):

        super().reset()

        # Create empty boards, one hidden for the actual placement and one for the observation of agent moves
        self.hidden_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.obs_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.hits_found = 0
        self.steps_taken = 0
        self.shots_taken = 0
        self.ship_cells_remaining = self.total_ship_cells

        # Loop through random placement of each ship vertically or horizontally
        # If there is a clash on the placement due to overlapping cell(s) then
        # attempt the placement again in another location and possible different orientation

        for ship_length in self.ships.values():
            placed = False
            while not placed:
                # Random ship orientation of horizontal or vertical
                orientation = random.choice(['horizontal', 'vertical'])
                if orientation == 'horizontal':
                    row = random.randrange(0, self.board_size)
                    col = random.randrange(0, self.board_size - ship_length + 1)
                    # Check if this group of cells is free for placement based on the size of the ship
                    if all(self.hidden_board[row][c] == 0 for c in range(col, col + ship_length)):
                        # Place the ship by marking each cell with 1's
                        for c in range(col, col + ship_length):
                            self.hidden_board[row][c] = 1
                        placed = True
                else:  # vertical
                    row = random.randrange(0, self.board_size - ship_length + 1)
                    col = random.randrange(0, self.board_size)
                    if all(self.hidden_board[r][col] == 0 for r in range(row, row + ship_length)):
                        for r in range(row, row + ship_length):
                            self.hidden_board[r][col] = 1
                        placed = True


        # Initial obs_board now looks like 10 iterations of [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]... for the 10x10 board
        # (all cells unknown = 0)
        # hidden_board will be 0's and 1's based on ship placement

        return np.array(self.obs_board, dtype=np.int32), {}

    # Execute the agent firing at a cell
    def step(self, action):
        # Decode the action (integer 0-99) into grid coordinates

        self.row = action // self.board_size
        self.col = action % self.board_size

        self.steps_taken += 1
        self.shots_taken += 1

        reward = 0.0
        done = False
        hit = False
        info = {}

        # Check if the agent already shot here
        # With new logic should never revisit
        if self.obs_board[self.row][self.col] != 0:
            # This cell was already revealed (hit or miss before)
            # Penalize repeat move
            #reward = -0.5  # negative reward for redundant guess
            reward = -5   # negative reward for redundant guess
        else:
            # This is a new cell being targeted
            if self.hidden_board[self.row][self.col] == 1:
                # It's a hit!
                self.obs_board[self.row][self.col] = 1  # Mark the cell as hit in the observation board
                self.hits_found += 1
                self.ship_cells_remaining -=1
                hit = True
                reward = 2.0  # reward for a hit
            else:
                # It's a miss!
                self.obs_board[self.row][self.col] = -1  # Mark the cell as a miss in the observation board
                reward = -.5  # Small penalty for a miss to encourage fewer moves

        # Check if all ships have been sunk
        if self.ship_cells_remaining == 0:
            # All ship cells have been hit – game was a success
            done = True
            terminated = True
            truncated = False
            reward = 100
        elif self.steps_taken >= self.max_steps:
            # Reached max allowed steps without finding all ships
            done = True
            terminated = False  # not all ships were found so treat as failure due to timeout
            truncated = True
        else:
            done = False
            terminated = False
            truncated = False

        # Prepare observation array to return
        obs_array = np.array(self.obs_board, dtype=np.int32)
        info = {'ships_remaining': self.ship_cells_remaining, 'hit': hit, 'shots_taken': self.shots_taken,}
        # Return observation, reward, termination flags, and info (Gymnasium API)
        return obs_array, reward, terminated, truncated, info

# Transition tuple for replay memory
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
    def __init__(self, state_dim, action_dim, hidden_dim=128):
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
        lr=1e-3,
        batch_size=64,
        memory_capacity=10000,
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
        # state: numpy array shape [100]
        avail = np.where(state == 0)[0]  # indices of unvisited cells

        # Exploration: uniform among available
        if random.random() < eps_threshold:
            return int(random.choice(avail))

        # Mask out visited cells in Q-values and remove the revisited cells issue

        state_v = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(state_v).squeeze(0)  # shape [100]

        # Set Q-values of visited cells to -∞ so they never get picked
        q_vals_np = q_vals.cpu().numpy()
        q_vals_np[state != 0] = -np.inf

        return int(np.argmax(q_vals_np))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps_done += 1
        batch = self.memory.sample(self.batch_size)
        trans = Transition(*zip(*batch))

        # 1) stack into contiguous NumPy arrays
        states_np      = np.array(trans.state,      dtype=np.float32)
        next_states_np = np.array(trans.next_state, dtype=np.float32)
        actions_np     = np.array(trans.action,     dtype=np.int64)
        rewards_np     = np.array(trans.reward,     dtype=np.float32)
        dones_np       = np.array(trans.done,       dtype=np.float32)

        # 2) one‐shot conversion to torch.Tensors
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
        self.optimizer.step()

        if self.steps_done % self.update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

def train(env_name='Battleship-v0', num_episodes=1000):
    env = gym.make(env_name)
    agent = DQNAgent(
        state_dim=np.prod(env.observation_space.shape),
        action_dim=env.action_space.n,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    eps_start, eps_end, eps_decay = 1.0, 0.01, 1000
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

        '''
        if episode % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            win_ratio = np.mean(wins[-50:])
            print(
                f"Episode {episode:4d} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Win Ratio: {win_ratio:.2f}"
            )
        '''

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
