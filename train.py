# train.py

import torch
import numpy as np
from battleship_env import BattleshipEnv, default_ships
from battleship_dqn import DQNAgent   # adjust import to your agent
from collections import namedtuple

def train_varying_ships(
    board_size=5,
    episodes_per_setting=200,
    start_ships=1,
    eps_start=1.0,
    eps_final=0.01,
    eps_decay=5000
):
    ship_items = list(default_ships.items())
    max_ships = len(ship_items)

    agent = DQNAgent(
        state_dim=board_size * board_size,
        action_dim=board_size * board_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    global_step = 0

    for n_ships in range(start_ships, max_ships + 1):
        ships_subset = dict(ship_items[:n_ships])
        env = BattleshipEnv(board_size=board_size, ships=ships_subset)

        print(f"\n=== Training: {board_size}×{board_size}, {n_ships} ship(s) ===")

        for ep in range(1, episodes_per_setting + 1):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                eps = eps_final + (eps_start - eps_final) * np.exp(-global_step / eps_decay)
                action = agent.select_action(obs.flatten(), eps)
                obs2, reward, term, trunc, info = env.step(action)
                done = term or trunc

                agent.memory.push(obs.flatten(), action, reward, obs2.flatten(), done)
                agent.update()

                obs = obs2
                ep_reward += reward
                global_step += 1

            if ep % 50 == 0:
                print(f"Ships={n_ships} | Ep {ep:3d} | Reward {ep_reward:.1f}")

        env.close()

    return agent

if __name__ == "__main__":
    # Kick off the ship‐count curriculum on a 10×10 board
    trained_agent = train_varying_ships()