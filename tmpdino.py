import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAVITY = 0.6
JUMP_STRENGTH = -10
OBSTACLE_SPEED = 5
GROUND_HEIGHT = HEIGHT - 50

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game")

# Dino
dino = pygame.Rect(50, GROUND_HEIGHT, 40, 40)
velocity_y = 0

# Obstacle list
obstacles = []
spawn_time = 0  # Track last spawn time

clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and dino.y >= GROUND_HEIGHT:
                velocity_y = JUMP_STRENGTH

    # Apply gravity
    velocity_y += GRAVITY
    dino.y += velocity_y
    if dino.y >= GROUND_HEIGHT:
        dino.y = GROUND_HEIGHT

    # Spawn obstacles at random intervals
    current_time = pygame.time.get_ticks()
    if current_time - spawn_time > random.randint(2000, 10000):  # Random delay (2-10sec)
        obstacle_width = random.randint(20, 40)
        obstacle_height = 40
        obstacles.append(pygame.Rect(WIDTH, GROUND_HEIGHT, obstacle_width, obstacle_height))
        spawn_time = current_time  # Reset spawn timer

    # Move and remove obstacles
    for obs in obstacles[:]:
        obs.x -= OBSTACLE_SPEED
        if obs.x + obs.width < 0:  # Remove off-screen obstacles
            obstacles.remove(obs)

    # Collision detection
    for obs in obstacles:
        if dino.colliderect(obs):
            print("Game Over!")  # Handle game over (restart, stop, etc.)
            running = False

    # Draw Dino and Obstacles
    pygame.draw.rect(screen, BLACK, dino)
    for obs in obstacles:
        pygame.draw.rect(screen, BLACK, obs)

    pygame.display.update()
    clock.tick(30)

pygame.quit()
