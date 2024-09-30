import os
import random
import numpy as np
import pygame
import sys
from dataclasses import dataclass
from PIL import Image
from agent import RandomAgent


# 💆‍ Need to integrate the step function so that the agent can play the game.
# Fixed Action being None ✅
# Need to add in state representation ✅
# Get reward working using the scoring change ✅
# Improve gameplay to make more challenging.
# Need to change the way the platforms are spawning to make it more difficult.

@dataclass
class GameConfig:
    WIDTH: int = 325
    HEIGHT: int = 488
    GRAVITY: int = 1
    JUMP_STRENGTH: int = -16
    HORIZONTAL_VEL: int = 6
    MAX_HORIZONTAL_VEL: int = 6
    FRICTION: float = 0.8

    # << ----------------------- PLATFORMS
    MIN_VERT_PLATFORM_SPACING: int = 80
    MAX_VERT_PLATFORM_SPACING: int = 120
    MIN_HORIZ_PLATFORM_SPACING: int = 40
    MAX_HORIZ_PLATFORM_SPACING: int = 60

    # Platform difficulty progression parameters
    DIFFICULTY_INCREASE_RATE: float = 0.01  # Rate at which platform spacing increases with score


    FPS: int = 60  # Add FPS to the config


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.image = pygame.image.load(os.path.join('images', 'sprite.png')).convert_alpha()
        self.rect = self.image.get_rect(center=(self.config.WIDTH // 2, self.config.HEIGHT - 150))
        self.gravity = 0
        self.horizontal_vel = 0
        self.on_platform = False
        self.player_alive = True
        self.score = 0
        self.highest_y = self.rect.y
        self.prev_y = self.rect.y  # Initialize prev_y for scrolling logic

    def jump(self):
        self.gravity = self.config.JUMP_STRENGTH

    def apply_gravity(self):
        self.gravity += self.config.GRAVITY
        self.rect.y += self.gravity

        if self.rect.bottom >= self.config.HEIGHT:
            self.player_alive = False

    def update(self, platforms, action):
        self.player_input(action)
        self.apply_gravity()
        self.apply_move()
        self.check_platform_collision(platforms)
        reward = self.scroll_screen(platforms)
        return reward

    def scroll_screen(self, platforms):
        score_change = 0  # For scoring we set to zero first
        if self.rect.top < self.config.HEIGHT // 2:
            y_change = self.prev_y - self.rect.y

            if y_change > 0:
                for platform in platforms:
                    platform.scroll_down(y_change)

                if self.rect.y < self.highest_y:
                    score_change = (self.highest_y - self.rect.y) / 10  # Scoring
                    self.score += score_change
                    self.highest_y = self.rect.y

        self.prev_y = self.rect.y

        return score_change  # Return our change in score for reward

    def player_input(self, action):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and self.on_platform:
            self.jump()
        elif keys[pygame.K_RIGHT] or action == 'RIGHT':
            self.horizontal_vel = self.config.HORIZONTAL_VEL
        elif keys[pygame.K_LEFT] or action == 'LEFT':
            self.horizontal_vel = -self.config.HORIZONTAL_VEL

        if self.horizontal_vel > self.config.MAX_HORIZONTAL_VEL:
            self.horizontal_vel = self.config.MAX_HORIZONTAL_VEL

        if self.horizontal_vel < -self.config.MAX_HORIZONTAL_VEL:
            self.horizontal_vel = -self.config.MAX_HORIZONTAL_VEL

    def apply_move(self):
        self.rect.x += self.horizontal_vel
        if self.rect.left > self.config.WIDTH:
            self.rect.right = 0
        if self.rect.right < 0:
            self.rect.left = self.config.WIDTH

        self.horizontal_vel *= self.config.FRICTION
        if abs(self.horizontal_vel) < 0.2:
            self.horizontal_vel = 0

    def check_platform_collision(self, platforms):
        self.on_platform = False
        if self.gravity > 0:
            for platform in platforms:
                if self.rect.bottom <= platform.rect.top + 10 and \
                        self.rect.right > platform.rect.left and \
                        self.rect.left < platform.rect.right:
                    if self.rect.bottom + self.gravity >= platform.rect.top:
                        self.rect.bottom = platform.rect.top
                        self.gravity = 0
                        self.on_platform = True
                        self.jump()
                        break


def initialize_game():
    pygame.init()
    screen = pygame.display.set_mode((325, 488))
    pygame.display.set_caption("DOODLE JUMP RL")
    return screen


class DoodleJumpEnv:
    def __init__(self):
        self.config = GameConfig()
        self.done = False

    def reset(self):
        self.done = False
        self.player = Player()
        # self.all_sprites = pygame.sprite.Group()
        # self.all_sprites.add(self.player)
        self.all_platforms = pygame.sprite.Group()
        self.spawn_initial_platforms(4)
        self.player.jump()

        return None

    def step(self, screen, action=None):

        reward = self.player.update(self.all_platforms, action)
        self.spawn_new_platforms()
        screen.fill((255, 255, 255))
        self.all_platforms.draw(screen)

        screen.blit(self.player.image, self.player.rect)

        state = self.get_rgb_screen(screen)  # Capture the current screen state

        self.done = False

        return state, reward, self.done

    def get_rgb_screen(self, screen):
        """Method to return RGB version of the screen."""
        screen_rgb = pygame.surfarray.array3d(screen)  # Capture screen
        screen_rgb = np.transpose(screen_rgb, (1, 0, 2))  # Convert array to match PIL's format (w, h, RGB)
        image = Image.fromarray(screen_rgb)   # Convert NumPy array to a PIL Image
        # image.show()  # Use this to examine
        return image

    def spawn_initial_platforms(self, num):
        """Spawns a fixed number of platforms at the start of the game, with controlled vertical spacing."""
        last_platform_y = self.config.HEIGHT - 50  # Start the first platform near the bottom of the screen

        for _ in range(num):
            platform = self.create_platform()

            # Change vertical spacing over time to increase difficulty by increasing max distance
            vert_spacing = self.config.MAX_VERT_PLATFORM_SPACING + \
                           int(self.player.score * self.config.DIFFICULTY_INCREASE_RATE * 100)

            # Ensure vertical spacing between platforms is within the defined range
            platform.rect.y = last_platform_y - random.randint(self.config.MIN_VERT_PLATFORM_SPACING,
                                                               vert_spacing)

            # Adjust horizontal position (stay within screen bounds)
            platform.rect.x = random.randint(0, self.config.WIDTH - platform.rect.width)

            last_platform_y = platform.rect.y

            self.all_platforms.add(platform)

    def create_platform(self):
        """Create a new platform at a random horizontal position."""
        platform = Platform(type='normal')

        # Ensure the platform spawns within the screen bounds
        platform.rect.x = random.randint(0, self.config.WIDTH - platform.rect.width)
        return platform

    def spawn_new_platforms(self):
        """Spawns new platforms when the player moves upward, ensuring proper vertical distance."""
        top_platform = min(self.all_platforms, key=lambda platform: platform.rect.y, default=None)

        if top_platform and top_platform.rect.y > 100:
            last_platform_y = top_platform.rect.y  # Start from the highest platform's position

            for _ in range(2):  # Add 2 new platforms
                platform = self.create_platform()

                # Change vertical spacing over time to increase difficulty by increasing max distance
                vert_spacing = self.config.MAX_VERT_PLATFORM_SPACING + \
                               int(self.player.score * self.config.DIFFICULTY_INCREASE_RATE * 100)

                # Ensure vertical spacing between platforms is within the defined range
                platform.rect.y = last_platform_y - random.randint(self.config.MIN_VERT_PLATFORM_SPACING,
                                                                   vert_spacing)

                # Adjust horizontal position (stay within screen bounds)
                platform.rect.x = random.randint(0, self.config.WIDTH - platform.rect.width)

                last_platform_y = platform.rect.y

                self.all_platforms.add(platform)

    def draw_score(self, screen):
        font = pygame.font.Font(None, 36)  # Create a font object
        score_text = font.render(f"Score: {int(self.player.score)}", True, (0, 0, 0))  # Render score text
        screen.blit(score_text, (10, 10))  # Blit (draw) the score at the top-left corner


class Platform(pygame.sprite.Sprite):
    def __init__(self, type):
        super().__init__()
        self.config = GameConfig()

        if type == 'normal':
            self.image = pygame.image.load(os.path.join('images', 'platform.png')).convert_alpha()

        self.rect = self.image.get_rect()
        self.image = pygame.transform.scale(self.image, (self.rect.width * 0.66, self.rect.height // 2))
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, self.config.WIDTH - self.rect.width)
        self.rect.y = random.randint(100, self.config.HEIGHT - self.rect.height)

    def scroll_down(self, val):
        self.rect.y += val
        if self.rect.y > self.config.HEIGHT:
            self.kill()


def main(agent_play=False):
    screen = initialize_game()
    env = DoodleJumpEnv()
    if agent_play:
        agent = RandomAgent()
    state = env.reset()

    clock = pygame.time.Clock()  # Add clock for FPS control

    running = True
    game_active = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_active:
            # Handle agent play or human play
            if agent_play:
                action = agent.choose_action()  # Agent chooses an action
            else:
                action = None  # Human play mode, no predefined action

            state, reward, done = env.step(screen, action)

            # Check player alive
            if not env.player.player_alive:
                game_active = False

        else:
            # Clear screen with a white background
            screen.fill((255, 255, 255))
            env.draw_score(screen)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:  # If we press space_bar we restart
                game_active = True
                env.reset()

        pygame.display.flip()  # Display changes on screen

        # Limit to 60 FPS
        clock.tick(GameConfig().FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main(agent_play=False)
