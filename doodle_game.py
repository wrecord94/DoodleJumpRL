import os
import random
import numpy as np
import pygame
import sys
from dataclasses import dataclass
from PIL import Image
from agent import RandomAgent
import cv2


# 💆‍ Need to integrate the step function so that the agent can play the game.
# Fixed Action being None ✅
# Need to add in state representation ✅
# Get reward working using the scoring change ✅
# Improve gameplay to make more challenging ✅
# Need to change the way the platforms are spawning to make it more difficult. ✅

# Fix bug 🐛 where the jump is influenced by the scrolling.
# Change the platform spawning to the most simple possible implementation then build from there?

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
        y_change = self.prev_y - self.rect.y  # Determine how much the player has moved up

        # Only scroll platforms down if the player has moved upward
        if y_change > 0 and self.rect.top < self.config.HEIGHT // 2:
            for platform in platforms:
                platform.scroll_down(y_change)

            if self.rect.y < self.highest_y:
                score_change = (
                                           self.highest_y - self.rect.y) / 10  # Update the score based on the player's highest point
                self.score += score_change
                self.highest_y = self.rect.y

        self.prev_y = self.rect.y

        return score_change  # Return the change in score for reward

    def player_input(self, action):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and self.on_platform:
            self.jump()
        elif keys[pygame.K_RIGHT] or action == 'RIGHT' or action == 1:
            self.horizontal_vel = self.config.HORIZONTAL_VEL
        elif keys[pygame.K_LEFT] or action == 'LEFT' or action == 2:
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

    def reset(self, screen):
        self.done = False
        self.player = Player()
        self.all_platforms = pygame.sprite.Group()
        self.spawn_initial_platforms(4)
        self.player.jump()

        state = self.get_rgb_screen(screen)

        return state

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
        screen_rgb = np.transpose(screen_rgb, (1, 0, 2))  # Convert array to match (width, height, RGB)
        processed_img = self._pre_process(screen_rgb)
        return processed_img  # Return as NumPy array

    def _pre_process(self, img: np.ndarray) -> np.ndarray:
        """
        Pre-processes an image to grayscale and resizes it.

        Args:
            img: The input RGB image to preprocess.

        Returns:
            A processed image in grayscale, resized to (84, 84).
        """
        shape = (1, 84, 84)
        # Convert to grayscale directly using OpenCV
        new_frame = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Resize to (84, 84)
        resized_screen = cv2.resize(new_frame, (shape[1], shape[2]), interpolation=cv2.INTER_AREA)
        # Rescale to [0, 1]
        new_obs = np.array(resized_screen, dtype=np.float32).reshape(shape) / 255.0
        return new_obs

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
