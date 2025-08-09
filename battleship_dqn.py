import random
from collections import deque, namedtuple
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    def clear(self):
        self.memory = []
        self.position = 0

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
        lr=0.0001,
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
        # state: numpy array shape [100] for 10x10 but dynamic now

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

        # stack into contiguous NumPy arrays
        states_np      = np.array(trans.state,      dtype=np.float32)
        next_states_np = np.array(trans.next_state, dtype=np.float32)
        actions_np     = np.array(trans.action,     dtype=np.int64)
        rewards_np     = np.array(trans.reward,     dtype=np.float32)
        dones_np       = np.array(trans.done,       dtype=np.float32)

        # one‐shot conversion to torch.Tensors
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