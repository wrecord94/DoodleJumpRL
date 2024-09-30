import sys
import pygame
from agent import RandomAgent
from doodle_game import initialize_game, DoodleJumpEnv, GameConfig


def main(agent_play=False):
    screen = initialize_game()
    env = DoodleJumpEnv()
    if agent_play:
        agent = RandomAgent()
    state = env.reset(screen)

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

            next_state, reward, done = env.step(screen, action)

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