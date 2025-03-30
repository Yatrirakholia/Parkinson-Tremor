import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_states = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))
    
    def add(self, state, action, next_state, reward, done):
        """Stores experience in the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """Samples a batch of experiences randomly."""
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idxs], 
            self.actions[idxs], 
            self.next_states[idxs], 
            self.rewards[idxs], 
            self.dones[idxs]
        )
