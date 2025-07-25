import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BattleshipEnv(gym.Env):

    def __init__(self, max_steps=None):

        if max_steps is None:
            self.max_steps = 100

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
        self.total_ship_cells = sum(self.ships.values())

        # Will be used to record the number of hits (ship cells) found so far
        self.hits_found = 0

        # Will be used to record the number of steps taken in current episode
        self.steps_taken = 0

        # Rows and Cols - will use to track potential next targets after a hit

        self.row = 0
        self.col = 0
        self.hit_list = []

    # Reset the board for a new game
    def reset(self):

        super().reset()

        # Create empty boards, one hidden for the actual placement and one for the observation of agent moves
        self.hidden_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.obs_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.hits_found = 0
        self.steps_taken = 0

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

        #print ("row ", self.row, " col ", self.col)

        self.steps_taken += 1
        reward = 0.0
        done = False
        info = {}

        # Check if the agent already shot here
        if self.obs_board[self.row][self.col] != 0:
            # This cell was already revealed (hit or miss before)
            # Penalize repeat move
            reward = -0.5  # negative reward for redundant guess
        else:
            # This is a new cell being targeted
            if self.hidden_board[self.row][self.col] == 1:
                # It's a hit!
                self.obs_board[self.row][self.col] = 1  # Mark the cell as hit in the observation board
                self.hits_found += 1
                reward = 5.0  # reward for a hit
                #print ("Hit ****** at: ", [self.row],[self.col])
            else:
                # It's a miss!
                self.obs_board[self.row][self.col] = -1  # Mark the cell as a miss in the observation board
                reward = -0.1  # Small penalty for a miss to encourage fewer moves

        #print ("Hits found: ", self.hits_found)
        # Check if all ships have been sunk
        if self.hits_found == self.total_ship_cells:
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
        # Return observation, reward, termination flags, and info (Gymnasium API)
        return obs_array, reward, terminated, truncated, info

# Q-Learning Agent

class BattleshipAgent:

    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            discount_factor: float,
            epsilon: float,
            epsilon_min: float,
            epsilon_decay: float):

        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = {}

        # Total number of possible actions (10x10 grid has 100 cells)
        self.num_actions = 100

    def choose_action(self, state):

       # Ensure the state is in the Q-table (should be), if not, initialize with 0's
        if state not in self.Q:
            self.Q[state] = [0.0] * self.num_actions

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose a random action from 0 to 99
            action = random.randrange(self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value for this state
            best_value = max(self.Q[state])
            action = self.Q[state].index(best_value)

        return action

    def update(
            self,
            state,
            action: int,
            reward: float,
            next_state: int,
            done: bool):

        # Ensure the current state is in Q-table (should already be, from choose_action, but double-check to be safe)
        if state not in self.Q:
            self.Q[state] = [0.0] * self.num_actions

        # Ensure the next state is in the Q-table
        if not done and next_state not in self.Q:
            self.Q[next_state] = [0.0] * self.num_actions

        # Calculate the maximum Q-value for the next state (if next_state is terminal, this remains 0)
        if done:
           max_future_q = 0.0
        else:
           max_future_q = max(self.Q[next_state])

        # Current Q-value for state and action
        current_q = self.Q[state][action]

        # Q-learning formula: update towards reward + discounted max future value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.Q[state][action] = new_q

    def decay_epsilon(self):

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

# Main program

if __name__ == "__main__":

    env = BattleshipEnv(max_steps=100)
    agent = BattleshipAgent(
        env,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9999)

    num_episodes = 2000
    wins = 0
    win_rates = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = tuple(obs.flatten())
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state = tuple(next_obs.flatten())
            #print ("Next state: ", next_state)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        if terminated and reward > 0:
            wins += 1

        win_rates.append(wins / episode)
        agent.decay_epsilon()

        if episode % 100 == 0:
            print(
                f"Episode {episode:5d} | "
                f"Reward {total_reward:6.2f} | "
                f"WinRate {wins/episode:.3f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )


# Print the first 10 states and their Q-values
for i, (state, q_vals) in enumerate(agent.Q.items()):
    if i >= 10:
        break
    print(f"State #{i+1}:")
    print(state)              # the flattened-board tuple
    print("Q-values:", q_vals)
    print("-" * 60)


import matplotlib.pyplot as plt

# Create a list of episode numbers
episodes = list(range(1, num_episodes + 1))

# Plot the win rate
plt.figure(figsize=(12, 6))
plt.plot(episodes, win_rates, label='Win Rate', color='blue')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Battleship Agent Performance Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

