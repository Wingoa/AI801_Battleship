# train.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from battleship_env import BattleshipEnv, default_ships
from battleship_dqn import DQNAgent
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
    global_episode = 0

    shots_to_win = []
    win_episodes = []

    for n_ships in range(start_ships, max_ships + 1):
        ships_subset = dict(ship_items[:n_ships])
        env = BattleshipEnv(board_size=board_size, ships=ships_subset)

        print(f"\n=== Training: {board_size}×{board_size}, {n_ships} ship(s) ===")

        for ep in range(1, episodes_per_setting + 1):
            global_episode += 1
            obs, _ = env.reset()
            done = False
            shots = 0
            ep_reward = 0.0
            last_reward = 0.0
            terminated = False

            while not done:
                eps = eps_final + (eps_start - eps_final) * np.exp(-global_step / eps_decay)
                action = agent.select_action(obs.flatten(), eps)
                obs2, reward, term, trunc, info = env.step(action)
                done = term or trunc

                # push and learn

                agent.memory.push(obs.flatten(), action, reward, obs2.flatten(), done)
                agent.update()

                obs = obs2
                shots += 1
                global_step += 1
                last_reward = reward
                ep_reward += reward
                global_step += 1
                terminated = term

            # --- If this episode was a win, record shots & episode idx ---

            if terminated:
                shots_to_win.append(shots)
                win_episodes.append(global_episode)

            if ep % 50 == 0:
                avg_shots = np.mean(shots_to_win) if shots_to_win else float('nan')
                win_ratio = len(shots_to_win) / (global_episode)  # wins ÷ total eps so far
                print(f" Ships={n_ships} | Ep {ep:3d} | Avg Shots/Win {avg_shots:.1f} | Win% {win_ratio:.2f} | Reward {ep_reward:.1f}")
                #print(f"Ships={n_ships} | Ep {ep:3d} | Reward {ep_reward:.1f}")

        env.close()

        # After training is complete, compute & plot cumulative average shots to win
        # Numpy could have named this function something a lot better IMO

    '''
        if shots_to_win:
            cumlative_avg_shots = np.cumsum(shots_to_win) / np.arange(1, len(shots_to_win) + 1)

            plt.figure(figsize=(10, 5))
            plt.plot(win_episodes, cumlative_avg_shots, linewidth=2, label='Cumulative Avg Shots to Win')
            plt.xlabel('Global Episode Number (only wins shown)')
            plt.ylabel('Average Shots per Win')
            plt.title('Agent Learning Curve: Shots-to-Win Over Training')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No wins recorded—nothing to plot.")
    '''

    return agent

def main():
    trained_agent = None

    # Loop through training on board sizes 5-10
    for size in range(5, 11):  # 5,6,7,8,9,10
        print(f"\n>>> Starting curriculum training on {size}×{size} board <<<")

        # Try increasing episodes as size of the grid expands
        if size in (5, 6):
            episodes_per_setting = 200
        elif size in (7, 8):
            episodes_per_setting = 300
        else:
            episodes_per_setting = 500

        # Call the training agent with ever increasing board sizes
        trained_agent = train_varying_ships(
            board_size=size,
            episodes_per_setting=episodes_per_setting,
            start_ships=1,
            eps_start=1.0,
            eps_final=0.01,
            eps_decay=5000
        )

    return trained_agent

if __name__ == "__main__":
    # Kick off the training
    agent = main()

