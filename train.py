import sys
import torch as T
import numpy as np
import pygame
from agent import RandomAgent
from dqn_agent import DQNAgent
from doodle_game import initialize_game, DoodleJumpEnv, GameConfig
import matplotlib.pyplot as plt
import os
# os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Run Pygame without display on macOS


# To do:
# > Get the agent to successfully learn...
# > Need to set-up a proper loop with logging of scores.
# > Need to fix the loop so it restarts the game for the agent ✅


def plot_training_data(log_interval_steps, avg_score_steps, best_scores, epsilon_values, figure_file):
    # Create a figure with 3 subplots, arranged vertically
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Average Score on the first subplot
    axs[0].plot(range(log_interval_steps, len(avg_score_steps) * log_interval_steps + 1,
                      log_interval_steps),
                avg_score_steps, color='tab:blue', label='Average Score')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Score')
    axs[0].set_title('Average Scores over Steps')
    axs[0].legend()

    # Plot Best Scores on the second subplot
    axs[1].plot(
        range(log_interval_steps, len(best_scores) * log_interval_steps + 1, log_interval_steps),
        best_scores, color='tab:orange', label='Best Score')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Best Score')
    axs[1].set_title('Best Scores during Training')

    # Plot Epsilon on the third subplot
    axs[2].plot(
        range(log_interval_steps, len(epsilon_values) * log_interval_steps + 1, log_interval_steps),
        epsilon_values, color='tab:green', label='Epsilon')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Epsilon Value')
    axs[2].set_title('Epsilon Decay over Steps')
    axs[2].legend()

    # Adjust layout to make sure everything fits well
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(figure_file)


def main(agent_play=False, training_bool=False):

    max_steps = 100_000
    log_interval_steps = 1000

    # Initialize empty variables for plotting and data saved at the end of training.
    step_count = 0  # Keep track of steps
    game_count = 0
    best_score = 0
    best_av_score = 0
    scores, epsilon_values, avg_score_steps, best_scores, eps_history = [], [], [], [], []

    # Bool to handle watching a trained model (not performing learning)
    training_bool = training_bool  # Triggers agent to store transitions and learn

    # Plotting and file names
    figure_file = 'training_plots/' + 'DQN' + '.png'
    model_file = 'trained_policies/' + 'DQN'

    screen = initialize_game()  # Get screen
    env = DoodleJumpEnv()  # Create game/env instance

    if agent_play:
        # Instantiate our agent with the args specified above
        agent = DQNAgent(gamma=0.95, epsilon=1, lr=0.0001, input_dims=(1, 84, 84), n_actions=3, mem_size=50000,
                         eps_min=0.01, batch_size=32, replace=1000, max_steps=10_000,
                         env_name=f"DoodleJumpDQN")

    state = env.reset(screen)  # << -------------------------------------------------- Reset the env

    clock = pygame.time.Clock()  # Add clock for FPS control

    running = True
    game_active = True


    while running and step_count < max_steps:  # While our step_count is less than the max steps we play games

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

            # Check if we reached the logging step interval
            if training_bool and step_count % log_interval_steps == 0:  # << ----------------------- Log training data
                if len(scores) > 0:
                    avg_score = round(np.mean(scores[-500:]), 2)
                else:
                    avg_score = 0  # Set default average score when no scores are available
                avg_score_steps.append(avg_score)
                best_scores.append(best_score)  # Track the current best score

                # Track the epsilon value
                epsilon_values.append(round(agent.epsilon, 2))

                # Save model if the average score improves
                if avg_score > best_av_score:
                    T.save(agent.q_target.state_dict(), model_file)

                    best_av_score = avg_score  # Handle best average score

                eps_history.append(agent.epsilon)

                # Print statement to track performance
                print(f"Steps:{step_count}: Best score: {best_score}, Average_score: {avg_score}, "
                      f"Epsilon: {round(agent.epsilon, 2)}")


        else:

            if training_bool:
                # At the end of a game we add the score to the array of scores
                final_score = env.player.score
                scores.append(final_score)
                if final_score > best_score:  # Handles best scores
                    best_score = final_score

            # Clear screen with a white background
            screen.fill((255, 255, 255))
            env.draw_score(screen)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] or agent_play:  # Press space_bar to restart OR agent version restarts automatically
                game_active = True
                env.reset(screen)

        pygame.display.flip()  # Display changes on screen

        # Limit to 60 FPS
        clock.tick(GameConfig().FPS)

    # << ------------------------------------------------------------------------------  Plot training data
    plot_training_data(log_interval_steps, avg_score_steps, best_scores, epsilon_values, figure_file)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main(agent_play=True, training_bool=True)
