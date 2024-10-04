# Doodle Jump with Deep Q-Learning 🤖

## Overview

This project implements a **Doodle Jump clone** using Pygame, enhanced with a **Deep Q-Learning algorithm** that allows an agent to learn how to play the game.

---

## Features

- Classic Doodle Jump gameplay mechanics
- Deep Q-Learning for self-learning AI
- Adjustable training parameters
- A Relay Buffer which stores transitions from the game as memories


## How It Works

**Game Environment:** The Doodle Jump game is created using Pygame, where the player controls a character that jumps on platforms.

**Deep Q-Learning:** The agent observes the game state via a grayscale downsampled view of the game-screen.

This view is then passed through a **neural network structure** which outputs q-values based on the available actions.

The agent will be able to therefore choose the 'optimal action' given it's state.

The **Q-values are updated using experience replay** where every 1000 frames 32 random transitions are sampled in a batch and given the rewards received and actions taken the q-values are updated accordingly using Bellman's Update rule.

The agent balances **exploration (trying new actions) and exploitation** using an exponential decaying value of epsilon which will hit the minimum value after 50% of the training period.

Training the Agent

The agent is trained over multiple episodes, where each episode consists of the agent playing the game until a terminal state is reached (falling off the screen).

Hyperparameters such as learning rate, discount factor, and exploration strategy can be modified in the configuration file.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wrecord94/doodle-jump-dql.git
   cd doodle-jump-dql

2. Install dependencies
    ```bash
   pip install pygame numpy matplotlib pytorch


## Usage

- To play the game yourself:
    ```bash
    python test.py

- To watch the agent train:
  ```bash
    python test.py

