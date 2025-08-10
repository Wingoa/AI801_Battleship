import numpy as np
import pygame
import random
from battleship_dqn import DQNAgent
from battleship_env import BattleshipEnv  

pygame.init()
screen = pygame.display.set_mode((625, 600))
pygame.display.set_caption("Battleship: User vs AI")
clock = pygame.time.Clock()
font_large = pygame.font.SysFont(None, 48)
font_small = pygame.font.SysFont(None, 32)

# Load AI
env = BattleshipEnv(board_size=6)
env.reset()
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)
agent.load("trained_agent.pth")

# --- Load Images ---
cell_size = 50
try:
    # Load and scale images to the correct cell size
    water_img = pygame.image.load("assets/water.png").convert_alpha()
    water_img = pygame.transform.scale(water_img, (cell_size, cell_size))

    ai_hit_img = pygame.image.load("assets/ai_hit.png").convert_alpha()
    ai_hit_img = pygame.transform.scale(ai_hit_img, (cell_size, cell_size))

    user_hit_img = pygame.image.load("assets/user_hit.png").convert_alpha()
    user_hit_img = pygame.transform.scale(user_hit_img, (cell_size, cell_size))

    user_miss_img = pygame.image.load("assets/user_miss.png").convert_alpha()
    user_miss_img = pygame.transform.scale(user_miss_img, (cell_size, cell_size))

    ai_miss_img = pygame.image.load("assets/ai_miss.png").convert_alpha()
    ai_miss_img = pygame.transform.scale(ai_miss_img, (cell_size, cell_size))

    ship_img = pygame.image.load("assets/ship.png").convert_alpha()
    ship_img = pygame.transform.scale(ship_img, (cell_size, cell_size))

except pygame.error as e:
    print(f"Error loading images: {e}")
    pygame.quit()
    exit()

def draw_board():

    if not isinstance(env, BattleshipEnv) or env.obs_board is None:
        return

    cell_size = 50
    margin = 20

    # Draw User Board (left)
    for row in range(env.board_size):
        for col in range(env.board_size):
            x, y = col * cell_size, row * cell_size
            screen.blit(water_img, (x, y))          # Water background
            if env.hidden_board[row][col] == 1:     # Your ship visible
                screen.blit(ship_img, (x, y))

            cell_value = env.obs_board[row][col]
            if cell_value == 1:                     # User hits
                screen.blit(user_hit_img, (x, y))
            elif cell_value == -1:                  # User misses
                screen.blit(user_miss_img, (x, y))

            # Draw Cell borders
            pygame.draw.rect(screen, (0, 0, 0), (x, y, cell_size, cell_size), 1)

    # Draw AI Board (right)
    x_offset = env.board_size * cell_size + margin
    for row in range(env.board_size):
        for col in range(env.board_size):
            x, y = x_offset + col * cell_size, row * cell_size
            screen.blit(water_img, (x, y))         # Water background

            cell_value = env.ai_obs_board[row][col]
            if cell_value == 1:                    # AI hits
                screen.blit(ai_hit_img, (x, y))
            elif cell_value == -1:                 # AI misses
                screen.blit(ai_miss_img, (x, y))

            # Draw Cell borders
            pygame.draw.rect(screen, (0, 0, 0), (x, y, cell_size, cell_size), 1)

    # Draw labels
    font = pygame.font.Font(None, 32)
    user_label = font.render('Your Board', True, (255,255,255))
    ai_label = font.render('AI Board', True, (255,255,255))
    screen.blit(user_label, (10, env.board_size * cell_size + 5))
    screen.blit(ai_label, (x_offset + 10, env.board_size * cell_size + 5))

    # Draw legend at the bottom
    legend_items = [
        (water_img, "Water"),
        (user_hit_img, "User Hit"),
        (user_miss_img, "User Miss"),
        (ai_hit_img, "AI Hit"),
        (ai_miss_img, "AI Miss"),
        (ship_img, "Ship")
    ]

    legend_x = 10
    legend_y = env.board_size * cell_size + 50
    legend_margin = 8  
    for img, label in legend_items:
        screen.blit(img, (legend_x, legend_y))
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, (255, 255, 255))
        text_y = legend_y + legend_margin
        screen.blit(text, (legend_x + cell_size + 5, text_y))
        legend_y += cell_size

def get_valid_user_actions(board):
    valid_actions = []
    for r in range(env.board_size):
        for c in range(env.board_size):
            if board[r][c] == 0:
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
        elif event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_SPACE:
                env.reset()
                env.ai_obs_board = [[0 for _ in range(env.board_size)] for _ in range(env.board_size)]
                user_turn = True
                game_over = False
                winner = None
                pending_ai_move = False
                last_ai_move = None

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

        if env.ai_obs_board[ai_row][ai_col] != 0:
            available = [(r, c) for r in range(env.board_size) for c in range(env.board_size) if env.ai_obs_board[r][c] == 0]
            if available:
                ai_row, ai_col = available[0]
                last_ai_move = (ai_row, ai_col)
            else:
                game_over = True
                winner = "User"

        if env.hidden_board[ai_row][ai_col] == 1:
            env.ai_obs_board[ai_row][ai_col] = 1
        else:
            env.ai_obs_board[ai_row][ai_col] = -1

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
        msg_text = "AI Wins!" if winner == "AI" else "You Win!"
        msg_color = (255, 0, 0) if winner == "AI" else (0, 255, 0)
        msg = font_large.render(msg_text, True, msg_color)
        msg2 = font_small.render("Press SPACE to Restart", True, (255, 255, 255))

        # Center both messages under the boards
        msg_rect = msg.get_rect(center=(screen.get_width() // 2, env.board_size * cell_size + 100))
        msg2_rect = msg2.get_rect(center=(screen.get_width() // 2, env.board_size * cell_size + 150))

        # Draw black background behind both messages
        padding = 20
        bg_rect = msg_rect.union(msg2_rect).inflate(padding, padding)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect)

        # Draw messages
        screen.blit(msg, msg_rect)
        screen.blit(msg2, msg2_rect)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()