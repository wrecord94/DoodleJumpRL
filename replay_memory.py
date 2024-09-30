import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  # Set the maximum size to our parameter value.
        self.mem_cntr = 0  # Memory size
        # We are going to store each part of the transition in a separate numpy array.
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        # self.transition_input_shape = transition_input_shape  # Store the input shape

    def store_transition(self, state, action, reward, state_, done):
        """Input: current state, action, reward, new state, done flag as input."""
        # print(f"STORE TRANSITION HAS INPUTS:\nstate(shape)={state.shape}\naction={action}\nreward={reward}"
        #       f"\nstate_next(shape)={state_.shape}\ndone={done}")
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1    # Increment out memory size tracking variable.

    def sample_buffer(self, batch_size):
        """Returns a random sample from our memory deques and removes these samples from the memory."""
        # If we haven't filled up memory we still want to be able to sample so take min here.
        max_mem = min(self.mem_cntr, self.mem_size)
        # Selects a batch_size number of indices at random from our memory_size.
        batch = np.random.choice(max_mem, batch_size, replace=False)  # Samples removed using replace=False
        # Grab the relevant items from our each of the arrays.
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        # print(f"Returning {sample_states, sample_actions, sample_rewards, sample_states_next, sample_terminal}")
        return states, actions, rewards, states_, terminal

    def print_everything_helper_(self):
        print(f"ReplayBuffer memory size {self.mem_size}")
        print(f"ReplayBuffer memory count {self.mem_cntr}")
        # print(f"Transition_input_shape {self.}")
        print(f" > state_memory Deque is: {len(self.state_memory)}")
        print(f" > new_state_memory Deque is: {len(self.new_state_memory)}")
        print(f" > action_memory Deque is: {len(self.action_memory)}")
        print(f" > reward_memory Deque is: {len(self.reward_memory)}")
        print(f" > terminal_memory Deque is: {len(self.terminal_memory)}")
        print(f" > State_memory Deque is: {len(self.reward_memory)}")


# << =============================================================================================================== >>
# << ============================================== TEST CODE BELOW ================================================ >>
# << =============================================================================================================== >>

# state = np.array(np.zeros(shape=(210, 160, 1)))
# observation, reward, terminated, truncated, info = state, 0.0, False, False, {'lives': 5, 'episode_frame_number': 4,
#                                                                               'frame_number': 4}
# TEST = observation, reward, terminated, truncated, info
#
# buffer = ReplayBuffer(max_size=2000, transition_input_shape=(210, 160, 1))
# buffer.store_transition(state=observation, action=0, reward=reward, state_next=state, done=terminated)
# buffer.print_everything_helper_()
# states, actions, rewards, states_next, terminal = buffer.sample_buffer(batch_size=1)