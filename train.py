import sys

import pygame

from agent import RandomAgent
from dqn_agent import DQNAgent
from doodle_game import initialize_game, DoodleJumpEnv, GameConfig


def main(agent_play=False, training_bool=False):
    step_count = 0  # Keep track of steps
    training_bool = training_bool  # Triggers agent to store transitions and learn

    screen = initialize_game()
    env = DoodleJumpEnv()
    if agent_play:
        # Instantiate our agent with the args specified above
        agent = DQNAgent(gamma=0.95, epsilon=1, lr=0.0001, input_dims=(1, 84, 84), n_actions=3, mem_size=50000,
                         eps_min=0.01, batch_size=32, replace=1000, max_steps=10_000,
                         env_name=f"DoodleJumpDQN")

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
            if agent_play is True:
                action = agent.choose_action(state)  # Agent chooses an action
            else:
                action = None  # Human play mode, no predefined action

            next_state, reward, done = env.step(screen, action)

            # Check player alive
            if not env.player.player_alive:
                game_active = False

            # If training
            if training_bool:
                agent.store_transition(state, action, reward, next_state, done)
                agent.learn(step_n=step_count)

            state = next_state  # Move state to next state
            step_count += 1  # Increment step counter

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
    main(agent_play=True, training_bool=True)
