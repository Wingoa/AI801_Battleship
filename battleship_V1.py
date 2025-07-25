import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Import F for relu and other functional ops
import math
from collections import deque, namedtuple

# ------------------ Device Configuration -------------------
# For using GPU or CPU based. I'm using my mac so no GPU. Planning to try my PC later
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Environment Definition ---------------------
class BattleshipEnv(gym.Env):
    def __init__(self, max_steps=None):
        if max_steps is None:
            self.max_steps = 100

        self.ships = {'Carrier': 5, 'Battleship': 4, 'Cruiser': 3, 'Submarine': 3, 'Destroyer': 2}
        self.max_steps = max_steps
        self.board_size = 10

        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int32)

        self.hidden_board = None
        self.obs_board = None
        self.total_ship_cells = sum(self.ships.values())
        self.hits_found = 0
        self.steps_taken = 0
        self.row = 0
        self.col = 0
        self.hit_list = [] # Not currently used by the DQN agent, but kept for potential future heuristic agents

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Pass seed to super().reset()

        self.hidden_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.obs_board = [[0] * self.board_size for _ in range(self.board_size)]
        self.hits_found = 0
        self.steps_taken = 0

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
                else:  # vertical
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
        reward = 0.0
        done = False
        info = {}

        if self.obs_board[self.row][self.col] != 0:
            reward = -0.5
        else:
            if self.hidden_board[self.row][self.col] == 1:
                self.obs_board[self.row][self.col] = 1
                self.hits_found += 1
                reward = 5.0
            else:
                self.obs_board[self.row][self.col] = -1
                reward = -0.1

        if self.hits_found == self.total_ship_cells:
            done = True
            terminated = True
            truncated = False
            reward = 100.0 # Large positive reward for winning
        elif self.steps_taken >= self.max_steps:
            done = True
            terminated = False
            truncated = True # Game truncated due to timeout
            reward = -50.0 # Significant negative reward for timeout/loss
        else:
            done = False
            terminated = False
            truncated = False

        obs_array = np.array(self.obs_board, dtype=np.int32)
        return obs_array, reward, terminated, truncated, info

# --- Dueling Q-Network (with CNN for grid-like observations) ---
class DuelingQNetwork(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(DuelingQNetwork, self).__init__()
        # Assuming observation_shape is (board_size, board_size) e.g., (10, 10)
        self.board_size = observation_shape[0]

        # Convolutional Layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened features after conv layers
        # Input: (N, 1, board_size, board_size)
        # Output conv1: (N, 32, board_size, board_size)
        # Output conv2: (N, 64, board_size, board_size)
        conv_output_size = 64 * self.board_size * self.board_size

        # Dueling Network Architecture
        # Value Stream
        self.fc_value1 = nn.Linear(conv_output_size, 256)
        self.fc_value2 = nn.Linear(256, 1) # Outputs a single scalar value

        # Advantage Stream
        self.fc_advantage1 = nn.Linear(conv_output_size, 256)
        self.fc_advantage2 = nn.Linear(256, num_actions) # Outputs advantage for each action

    def forward(self, x):
        # Add a channel dimension: (batch_size, 10, 10) -> (batch_size, 1, 10, 10)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten for FC layers

        # Value stream
        val = F.relu(self.fc_value1(x))
        val = self.fc_value2(val)

        # Advantage stream
        adv = F.relu(self.fc_advantage1(x))
        adv = self.fc_advantage2(adv)

        # Combine Value and Advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - Avg(A(s,:)))
        q_values = val + adv - adv.mean(1, keepdim=True)
        return q_values

# --- Prioritized Experience Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha   # Controls the level of prioritization
        self.beta = 0.4      # Controls the amount of importance sampling correction
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = self.max_priority  # New samples get max priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Calculate probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        total_len = len(self.buffer)
        weights = (total_len * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalize weights

        # Convert samples to batch format
        batch = Transition(*zip(*samples))
        state_batch = torch.stack(batch.state).float().to(device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).float().unsqueeze(1).to(device)
        next_state_batch = torch.stack(batch.next_state).float().to(device)
        done_batch = torch.tensor(batch.done).bool().unsqueeze(1).to(device)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, torch.from_numpy(weights).float().unsqueeze(1).to(device)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            # Add .detach() before .cpu().numpy() to move to CPU and then to numpy
            self.priorities[idx] = ((error.abs().detach().cpu().numpy() + 1e-5) ** self.alpha).item()
        self.max_priority = self.priorities.max()

    def __len__(self):
        return len(self.buffer)

# --------------- DQN Agent ---------------
class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        discount_factor: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_steps: int, # decay over a fixed number of steps
        replay_buffer_capacity: int,
        batch_size: int,
        target_update_frequency: int,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000 # Number of frames over which to anneal beta
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon_start - epsilon_min) / epsilon_decay_steps
        self.epsilon_decay_steps = epsilon_decay_steps

        self.per_beta = per_beta_start
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames

        self.num_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape

        self.policy_net = DuelingQNetwork(self.observation_shape, self.num_actions).to(device)
        self.target_net = DuelingQNetwork(self.observation_shape, self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none') # Use Huber loss for PER, reduction='none' for individual errors

        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_capacity, alpha=per_alpha)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.steps_done = 0

    def choose_action(self, state_obs):
        # state_obs is already a numpy array coming from env.step or env.reset
        state_tensor = torch.from_numpy(state_obs).float().unsqueeze(0).to(device)

        # Decay epsilon linearly over the specified steps
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - self.steps_done * self.epsilon_decay_rate
        else:
            self.epsilon = self.epsilon_min

        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0 # Not enough experiences to train yet, return 0 loss

        (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, weights = \
            self.replay_buffer.sample(self.batch_size)

        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: Select action using policy_net, evaluate with target_net
        with torch.no_grad():
            # Select best action from policy_net for next_state
            next_state_policy_q_values = self.policy_net(next_state_batch)
            next_actions = next_state_policy_q_values.argmax(1).unsqueeze(1)

            # Evaluate Q-values of selected actions using target_net
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            next_q_values[done_batch] = 0.0

        # Compute the expected Q values (target)
        expected_q_values = reward_batch + self.discount_factor * next_q_values

        # Calculate TD errors for PER
        # Ensure q_values and expected_q_values have the same shape for loss calculation
        td_errors = self.loss_fn(q_values, expected_q_values)
        self.replay_buffer.update_priorities(indices, td_errors)

        # Apply importance sampling weights to the loss
        loss = (td_errors * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True) # Optimized zero_grad
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Anneal beta for PER
        self.replay_buffer.beta = min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) * (self.steps_done / self.per_beta_frames))
        
        return loss.item() # Return the scalar loss value

# ------------------ Main Training Loop ------------------------
if __name__ == "__main__":
    env = BattleshipEnv(max_steps=100)
    
    # Hyperparameters for faster and better learning (Need to tune these)
    agent_params = {
        'learning_rate': 0.0005, # Play around with this
        'discount_factor': 0.99, # Close to 1 for more long-term rewards
        'epsilon_start': 1.0, # Start with a high exploration rate
        'epsilon_min': 0.01, # Minimum exploration rate
        'epsilon_decay_steps': 50000, # Decay epsilon over 50,000 steps
        'replay_buffer_capacity': 50000, # Increased capacity
        'batch_size': 64, # Slightly reduced batch size for CPU
        'target_update_frequency': 200, # Update target network less frequently
        'per_alpha': 0.7, # Higher alpha for stronger prioritization
        'per_beta_start': 0.5, # Start beta higher and anneal to 1.0
        'per_beta_frames': 1000000 # Anneal beta over a longer period
    }

    agent = DQNAgent(env, **agent_params)

    num_episodes = 10000 # Increased episodes (can go higher, e.g., 20k-50k if needed)
    wins = 0
    win_rates = []
    episode_rewards = []
    
    # Track overall steps for epsilon decay and beta annealing
    global_steps = 0
    
    # For logging
    episode_losses = []
    episode_lengths = []

    print("Starting training...")
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = torch.from_numpy(obs).float() # Start on CPU, will be moved to device inside choose_action/update
        done = False
        total_reward = 0
        steps_this_episode = 0
        current_episode_losses = []

        while not done:
            action = agent.choose_action(state.cpu().numpy()) # Agent expects numpy for choice
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps_this_episode += 1

            next_state = torch.from_numpy(next_obs).float() # Start on CPU

            # Store the transition in the replay buffer
            # States, actions, rewards, next_states, done flags will be moved to device by the buffer.
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Perform a training step after enough samples are collected
            if len(agent.replay_buffer) > agent.batch_size:
                loss_val = agent.update()
                current_episode_losses.append(loss_val)

            state = next_state
            global_steps += 1

        if terminated and reward == 100.0: # Explicitly check for the win reward
            wins += 1

        win_rates.append(wins / episode)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps_this_episode)
        
        if current_episode_losses:
            episode_losses.append(np.mean(current_episode_losses))
        else:
            episode_losses.append(0.0) # No updates happened this episode

        # Logging Feedback
        log_frequency = 100
        if episode <= 100: # Log more frequently at the very beginning
            log_frequency = 10
        elif episode <= 1000:
            log_frequency = 50
        
        if episode % log_frequency == 0:
            avg_reward_window = np.mean(episode_rewards[-log_frequency:])
            avg_win_rate_window = np.mean(win_rates[-log_frequency:])
            avg_loss_window = np.mean(episode_losses[-log_frequency:])
            avg_steps_window = np.mean(episode_lengths[-log_frequency:])

            print(
                f"Episode {episode:5d}/{num_episodes} | "
                f"Global Steps {global_steps:7d} | "
                f"Avg Rwd ({log_frequency}ep): {avg_reward_window:8.2f} | "
                f"Avg WinRate ({log_frequency}ep): {avg_win_rate_window:.3f} | "
                f"Avg Loss ({log_frequency}ep): {avg_loss_window:.4f} | "
                f"Avg Steps/Ep ({log_frequency}ep): {avg_steps_window:.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"PER Beta: {agent.replay_buffer.beta:.3f}"
            )

            # Might want to save the model periodically here. Once I get it working well
            # torch.save(agent.policy_net.state_dict(), f"battleship_dqn_ep{episode}.pth")

    print("\nTraining complete!")


    # ------------ Plotting Results ------------------

    import matplotlib.pyplot as plt

    # Plot the win rate
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, num_episodes + 1)), win_rates, label='Win Rate', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Battleship Agent Performance Over Time (Dueling Double DQN with PER)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot episode rewards
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, num_episodes + 1)), episode_rewards, label='Total Episode Reward', color='green', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Battleship Agent Total Reward Per Episode (Dueling Double DQN with PER)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot average loss (if applicable)
    if any(episode_losses): # Only plot if there was any loss recorded
        plt.figure(figsize=(12, 6))
        plt.plot(list(range(1, num_episodes + 1)), episode_losses, label='Average Episode Loss', color='red', alpha=0.6)
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Battleship Agent Average Loss Per Episode (Dueling Double DQN with PER)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot average steps per episode
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, num_episodes + 1)), episode_lengths, label='Steps Per Episode', color='purple', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Battleship Agent Steps Per Episode (Dueling Double DQN with PER)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()