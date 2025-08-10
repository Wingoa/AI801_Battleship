import numpy as np
import pygame
import random
from battleship_dqn import DQNAgent
from battleship_env import BattleshipEnv  

pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Battleship vs AI")
clock = pygame.time.Clock()

# Load AI
env = BattleshipEnv(board_size=6) #This has to match what the model was trained on
env.reset()
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)
agent.load("trained_agent.pth")

def draw_board():
    if env.obs_board is None:
        return
    cell_size = 40
    margin = 20
    # Draw user's board (left)
    for row in range(env.board_size):
        for col in range(env.board_size):
            cell_value = env.obs_board[row][col]
            color = (0, 0, 0)  # Unknown
            if cell_value == 1:
                color = (255, 0, 0)  # Hit - red
            elif cell_value == -1:
                color = (0, 0, 255)  # Miss - blue
            pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, (255, 255, 255), (col * cell_size, row * cell_size, cell_size, cell_size), 1)

    if last_ai_move is not None:
        r, c = last_ai_move
        pygame.draw.rect(screen, (255, 255, 0), (c * cell_size, r * cell_size, cell_size, cell_size), 3)

    # Draw AI's board (right)
    if env.ai_obs_board is not None:
        for row in range(env.board_size):
            for col in range(env.board_size):
                cell_value = env.ai_obs_board[row][col]
                color = (0, 0, 0)
                if cell_value == 1:
                    color = (255, 0, 0)
                elif cell_value == -1:
                    color = (0, 0, 255)
                x_offset = env.board_size * cell_size + margin
                pygame.draw.rect(screen, color, (x_offset + col * cell_size, row * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (255, 255, 255), (x_offset + col * cell_size, row * cell_size, cell_size, cell_size), 1)

    # Draw labels
    font = pygame.font.SysFont(None, 32)
    user_label = font.render('Your Board', True, (255,255,255))
    ai_label = font.render('AI Board', True, (255,255,255))
    screen.blit(user_label, (10, env.board_size * cell_size + 5))
    screen.blit(ai_label, (env.board_size * cell_size + margin + 10, env.board_size * cell_size + 5))


def get_valid_user_actions(board):
    valid_actions = []
    for r in range(env.board_size):
        for c in range(env.board_size):
            if board[r][c] == 0:  # unshot cell
                valid_actions.append(r * env.board_size + c)
    return valid_actions


running = True
user_turn = True
game_over = False
winner = None
pending_ai_move = False
last_ai_move = None
env.ai_obs_board = [[0 for _ in range(env.board_size)] for _ in range(env.board_size)]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if user_turn and not game_over:
        valid_actions = get_valid_user_actions(env.obs_board)
        if valid_actions:
            action = random.choice(valid_actions)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                game_over = True
                winner = "User"
            user_turn = False
            pygame.time.delay(500)
            pending_ai_move = True
        else:
            game_over = True
            winner = "AI"

    if pending_ai_move and not game_over:
        state = env.get_state()
        state = state.flatten() if hasattr(state, 'flatten') else np.array(state).flatten()
        action = agent.select_action(state, eps_threshold=0.0)
        ai_row = action // env.board_size
        ai_col = action % env.board_size
        last_ai_move = (ai_row, ai_col)

        # Avoid repeated shots
        if env.ai_obs_board[ai_row][ai_col] != 0:
            available = [(r, c) for r in range(env.board_size) for c in range(env.board_size) if env.ai_obs_board[r][c] == 0]
            if available:
                ai_row, ai_col = available[0]
                last_ai_move = (ai_row, ai_col)
            else:
                game_over = True
                winner = "User"

        # Update AI board manually
        if env.hidden_board[ai_row][ai_col] == 1:
            env.ai_obs_board[ai_row][ai_col] = 1  # Hit
        else:
            env.ai_obs_board[ai_row][ai_col] = -1  # Miss

        # Simple AI win check
        hits = sum(row.count(1) for row in env.ai_obs_board)
        total_ship_cells = sum(row.count(1) for row in env.hidden_board)
        if hits >= total_ship_cells:
            game_over = True
            winner = "AI"

        user_turn = True
        pending_ai_move = False

    screen.fill((0, 0, 0))
    draw_board()

    if game_over:
        font = pygame.font.SysFont(None, 48)
        if winner == "AI":
            msg = font.render("AI Wins!", True, (255, 0, 0))
        else:
            msg = font.render("You Win!", True, (0, 255, 0))
        screen.blit(msg, (screen.get_width()//2 - msg.get_width()//2,
                          screen.get_height()//2 - msg.get_height()//2))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
