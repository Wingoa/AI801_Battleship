import numpy as np
import pygame
import random
from battleship_dqn import DQNAgent
from battleship_env import BattleshipEnv  
import sys
import os

pygame.init()
screen = pygame.display.set_mode((625, 600))
pygame.display.set_caption("Battleship: User vs AI")
clock = pygame.time.Clock()
font_large = pygame.font.SysFont(None, 48)
font_small = pygame.font.SysFont(None, 32)
ships_dict = {'Battleship': 4, 'Cruiser': 3}

# Helper to find assets in PyInstaller bundle or source
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Set window icon with rounded corners
try:
    icon_img = pygame.image.load(resource_path("assets/battleship_icon.png")).convert_alpha()
    icon_img = pygame.transform.smoothscale(icon_img, (32, 32))

    mask = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=4)

    icon_img.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.display.set_icon(icon_img)
except Exception as e:
    print(f"Warning: Unable to load icon image. {e}")
    pass

# Load Images
cell_size = 50
try:
    # Load and scale images to the correct cell size
    logo_img = pygame.image.load(resource_path("assets/battleship_logo.png")).convert_alpha()
    logo_img = pygame.transform.smoothscale(logo_img, (400, 200))

    water_img = pygame.image.load(resource_path("assets/water.png")).convert_alpha()
    water_img = pygame.transform.scale(water_img, (cell_size, cell_size))

    ai_hit_img = pygame.image.load(resource_path("assets/ai_hit.png")).convert_alpha()
    ai_hit_img = pygame.transform.scale(ai_hit_img, (cell_size, cell_size))

    user_hit_img = pygame.image.load(resource_path("assets/user_hit.png")).convert_alpha()
    user_hit_img = pygame.transform.scale(user_hit_img, (cell_size, cell_size))

    user_miss_img = pygame.image.load(resource_path("assets/user_miss.png")).convert_alpha()
    user_miss_img = pygame.transform.scale(user_miss_img, (cell_size, cell_size))

    ai_miss_img = pygame.image.load(resource_path("assets/ai_miss.png")).convert_alpha()
    ai_miss_img = pygame.transform.scale(ai_miss_img, (cell_size, cell_size))

    ship_img = pygame.image.load(resource_path("assets/ship.png")).convert_alpha()
    ship_img = pygame.transform.scale(ship_img, (cell_size, cell_size))

except pygame.error as e:
    print(f"Error loading images: {e}")
    pygame.quit()
    exit()


# Load environment just to get a ship layout for the user
env = BattleshipEnv(board_size=6) # This will need to be changed to 10 for the newest version of the model
env.ships = ships_dict
env.reset()

# Determine board/action dims for AI model 
state_dim = env.board_size * env.board_size
action_dim = env.board_size * env.board_size
agent = DQNAgent(state_dim, action_dim)
#agent.load(resource_path("trained_agent.pth")) # This will load in the model that is generated from training
agent.load(resource_path("6x6_trained_agent.pth"))  # Load the 6x6 model


# Create AI hidden ships
def place_ships(board_size, ships_dict):
    board = [[0] * board_size for _ in range(board_size)]
    for length in ships_dict.values():
        placed = False
        while not placed:
            orientation = random.choice(['horizontal', 'vertical'])
            if orientation == 'horizontal':
                row = random.randrange(0, board_size)
                col = random.randrange(0, board_size - length + 1)
                if all(board[row][c] == 0 for c in range(col, col + length)):
                    for c in range(col, col + length):
                        board[row][c] = 1
                    placed = True
            else:
                row = random.randrange(0, board_size - length + 1)
                col = random.randrange(0, board_size)
                if all(board[r][col] == 0 for r in range(row, row + length)):
                    for r in range(row, row + length):
                        board[r][col] = 1
                    placed = True
    return board

# AI's ships are those in env.hidden_board
ai_hidden = [row[:] for row in env.hidden_board]

# User's ships we generate separately
user_hidden = place_ships(env.board_size, ships_dict)

# Reinitialize guess boards
env.obs_board = [[0] * env.board_size for _ in range(env.board_size)]       # user guesses at AI
env.ai_obs_board = [[0] * env.board_size for _ in range(env.board_size)]    # AI guesses at user

ai_ship_cells_remaining = sum(sum(row) for row in ai_hidden)
user_ship_cells_remaining = sum(sum(row) for row in user_hidden)

def draw_start_screen():
    screen.fill((10, 20, 40))
    if logo_img:
        logo_rect = logo_img.get_rect(center=(screen.get_width() // 2, 180))
        screen.blit(logo_img, logo_rect)
    prompt_font = pygame.font.SysFont(None, 36)
    prompt = prompt_font.render("Press any button to continue", True, (200, 200, 200))
    prompt_rect = prompt.get_rect(center=(screen.get_width() // 2, 350))
    screen.blit(prompt, prompt_rect)
    pygame.display.flip()

def draw_board():

    margin = 20

    # User Board (left): shows user's ships + AI shots (env.ai_obs_board)
    for row in range(env.board_size):
        for col in range(env.board_size):
            x, y = col * cell_size, row * cell_size
            screen.blit(water_img, (x, y))
            if user_hidden[row][col] == 1:
                screen.blit(ship_img, (x, y))
            val = env.ai_obs_board[row][col]
            if val == 1:
                screen.blit(ai_hit_img, (x, y))
            elif val == -1:
                screen.blit(ai_miss_img, (x, y))
            pygame.draw.rect(screen, (0, 0, 0), (x, y, cell_size, cell_size), 1)

    # AI Board (right): shows only user shots (env.obs_board) no ships revealed
    x_offset = env.board_size * cell_size + margin
    for row in range(env.board_size):
        for col in range(env.board_size):
            x, y = x_offset + col * cell_size, row * cell_size
            screen.blit(water_img, (x, y))
            val = env.obs_board[row][col]
            if val == 1:
                screen.blit(user_hit_img, (x, y))
            elif val == -1:
                screen.blit(user_miss_img, (x, y))
            pygame.draw.rect(screen, (0, 0, 0), (x, y, cell_size, cell_size), 1)

    # Labels
    label_font = pygame.font.Font(None, 32)
    screen.blit(label_font.render('Your Board', True, (255,255,255)), (10, env.board_size * cell_size + 5))
    screen.blit(label_font.render('AI Board', True, (255,255,255)), (x_offset + 10, env.board_size * cell_size + 5))

    # Legend
    legend_items = [
        (water_img, "Water"),
        (user_hit_img, "Your Hit"),
        (user_miss_img, "Your Miss"),
        (ai_hit_img, "AI Hit"),
        (ai_miss_img, "AI Miss"),
        (ship_img, "Your Ship")
    ]
    legend_x = 10
    legend_y = env.board_size * cell_size + 50
    for img, label in legend_items:
        screen.blit(img, (legend_x, legend_y))
        lf = pygame.font.Font(None, 24)
        text = lf.render(label, True, (255, 255, 255))
        screen.blit(text, (legend_x + cell_size + 5, legend_y + 8))
        legend_y += cell_size

def get_valid_user_actions(board):
    valid_actions = []
    for r in range(env.board_size):
        for c in range(env.board_size):
            if board[r][c] == 0:
                valid_actions.append(r * env.board_size + c)
    return valid_actions


# Game State Machine 
game_state = "start" 
running = True
user_turn = True 
game_over = False
winner = None
last_ai_move = None


while running:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
            break

        if game_state == "start":
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                # Start the game
                game_state = "playing"
                # Reset game state
                env.reset()
                ai_hidden = [row[:] for row in env.hidden_board]
                user_hidden = place_ships(env.board_size, getattr(env, 'ships', {'Battleship': 4, 'Cruiser': 3}))
                env.obs_board = [[0] * env.board_size for _ in range(env.board_size)]
                env.ai_obs_board = [[0] * env.board_size for _ in range(env.board_size)]
                ai_ship_cells_remaining = sum(sum(r) for r in ai_hidden)
                user_ship_cells_remaining = sum(sum(r) for r in user_hidden)
                user_turn = True
                game_over = False
                winner = None
                last_ai_move = None
        elif game_state == "playing":
            if event.type == pygame.KEYDOWN and game_over:
                # Restart the game
                if event.key == pygame.K_SPACE:
                    env.reset()
                    ai_hidden = [row[:] for row in env.hidden_board]
                    user_hidden = place_ships(env.board_size, getattr(env, 'ships', {'Battleship': 4, 'Cruiser': 3}))
                    env.obs_board = [[0] * env.board_size for _ in range(env.board_size)]
                    env.ai_obs_board = [[0] * env.board_size for _ in range(env.board_size)]
                    ai_ship_cells_remaining = sum(sum(r) for r in ai_hidden)
                    user_ship_cells_remaining = sum(sum(r) for r in user_hidden)
                    user_turn = True
                    game_over = False
                    winner = None
                    last_ai_move = None
                # User pressed ESC
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

            # USER INPUT MODE (commented out for now)
            '''
            elif user_turn and not pending_ai_move and not game_over:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos
                    col = mouse_x // cell_size
                    row = mouse_y // cell_size
                    if 0 <= row < env.board_size and 0 <= col < env.board_size:
                        if env.obs_board[row][col] == 0:
                            action = row * env.board_size + col
                            obs, reward, term, trunc, info = env.step(action)
                            if term or trunc:
                                game_over = True
                                winner = "User"
                            pending_ai_move = True
                            pygame.time.delay(200)
            '''
    if game_state == "start":
        draw_start_screen()
        clock.tick(30)
        continue

    # TURN HANDLING
    if not game_over:
        if user_turn:
            # User random move selecting from unknown AI cells
            valid_actions = get_valid_user_actions(env.obs_board)
            if valid_actions:
                action = random.choice(valid_actions)
                r = action // env.board_size
                c = action % env.board_size
                if ai_hidden[r][c] == 1:
                    env.obs_board[r][c] = 1
                    ai_ship_cells_remaining -= 1
                    if ai_ship_cells_remaining == 0:
                        game_over = True
                        winner = "User"
                else:
                    env.obs_board[r][c] = -1
                user_turn = False  # switch to AI
                pygame.time.delay(140)
            else:
                # No moves left -> AI wins by default
                game_over = True
                winner = "AI"
        else:
            # AI move using its own observation board
            state = np.array(env.ai_obs_board, dtype=np.float32).flatten()
            # Mask: choose among zeros
            action = agent.select_action(state, eps_threshold=0.0)
            r = action // env.board_size
            c = action % env.board_size
            if env.ai_obs_board[r][c] == 0:  # safety
                if user_hidden[r][c] == 1:
                    env.ai_obs_board[r][c] = 1
                    user_ship_cells_remaining -= 1
                    if user_ship_cells_remaining == 0:
                        game_over = True
                        winner = "AI"
                else:
                    env.ai_obs_board[r][c] = -1
            last_ai_move = (r, c)
            if not game_over:
                user_turn = True
            pygame.time.delay(140)

    screen.fill((0, 0, 0))
    draw_board()

    if game_over:
        msg_text = "AI Wins!" if winner == "AI" else "You Win!"
        msg_color = (255, 0, 0) if winner == "AI" else (0, 255, 0)
        msg = font_large.render(msg_text, True, msg_color)
        msg2 = font_small.render("Press SPACE to Restart", True, (255, 255, 255))
        msg3 = font_small.render("Press ESC to Quit", True, (255, 255, 255))

        # Center both messages under the boards
        msg_rect = msg.get_rect(center=(screen.get_width() // 2, env.board_size * cell_size + 100))
        msg2_rect = msg2.get_rect(center=(screen.get_width() // 2, env.board_size * cell_size + 150))
        msg3_rect = msg3.get_rect(center=(screen.get_width() // 2, env.board_size * cell_size + 200))

        # Draw black background behind both messages
        padding = 20
        bg_rect = msg_rect.union(msg2_rect).inflate(padding, padding)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect)

        # Draw messages
        screen.blit(msg, msg_rect)
        screen.blit(msg2, msg2_rect)
        screen.blit(msg3, msg3_rect)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()