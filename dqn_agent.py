from dqnetwork import *
from replay_memory import ReplayBuffer
import numpy as np
import torch as T

class DQNAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, max_steps, eps_min=0.1,
                 replace=1000, algo=None, env_name=None):
        """Initialize the DQN agent with the provided configuration."""
        self.env_name = env_name
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.replace_target_cnt = replace
        self.max_steps = max_steps
        self.learn_step_counter = 0  # Counter for tracking learning steps
        self.action_space = [i for i in range(n_actions)]  # Define the action space

        # Initialize the experience replay buffer
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # Initialize Q-networks: Convolutional or Linear based on input dimensions
        self.q_online = self._build_network()
        self.q_target = self._build_network()

    def _build_network(self):
        """Create a network based on input dimensions (CNN for 3D, Linear for 2D)."""
        if len(self.input_dims) == 3:
            return ConvolutionalDeepNN(lr=self.lr, n_actions=self.n_actions, input_dims=self.input_dims)
        else:
            return LinearDeepNN(lr=self.lr, n_actions=self.n_actions, input_dims=self.input_dims)

    def choose_action(self, observation):
        """Use an epsilon-greedy strategy to select the next action."""
        if np.random.random() > self.epsilon:
            state = T.tensor(observation[np.newaxis, :], dtype=T.float, device=self.q_online.device)
            q_values = self.q_online.forward(state)
            action = T.argmax(q_values).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        """Store the experience tuple in the replay buffer."""
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """Retrieve a batch of experiences from the replay buffer and convert them to tensors."""
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_online.device)
        rewards = T.tensor(reward).to(self.q_online.device)
        dones = T.tensor(done).to(self.q_online.device)
        actions = T.tensor(action).to(self.q_online.device)
        new_states = T.tensor(new_state).to(self.q_online.device)
        return states, actions, rewards, new_states, dones

    def replace_target_network(self):
        """Synchronize target network weights with the online network at fixed intervals."""
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def _reduce_epsilon_exponential(self, step_n):
        """Exponentially decrease epsilon based on training progress."""
        decay_factor = -np.log(self.eps_min / 1.0) / (0.75 * self.max_steps)
        if step_n < 0.5 * self.max_steps:
            self.epsilon = 1.0 * np.exp(-decay_factor * step_n)
        else:
            self.epsilon = self.eps_min
        self.epsilon = max(self.eps_min, self.epsilon)

    def learn(self, step_n):
        """Perform a learning step using a batch of experiences from memory."""
        if self.memory.mem_cntr < self.batch_size:
            return

        # Zero the gradients in the optimizer
        self.q_online.optimizer.zero_grad()

        # Replace target network weights if needed
        self.replace_target_network()

        # Sample experiences from memory
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        # Compute predicted Q-values for current states
        q_pred = self.q_online.forward(states)[indices, actions]

        # Compute the target Q-values for the next states using the target network
        q_next = self.q_target.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0  # No future rewards if the episode is done

        # Apply the Bellman equation to compute the target values
        q_target = rewards + self.gamma * q_next

        # Calculate the loss between the predicted and target Q-values
        loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)

        # Backpropagate the loss and update the network weights
        loss.backward()
        self.q_online.optimizer.step()

        # Increment the learning step counter
        self.learn_step_counter += 1

        # Adjust the epsilon value for exploration-exploitation balance
        self._reduce_epsilon_exponential(step_n)
