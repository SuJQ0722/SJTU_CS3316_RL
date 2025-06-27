import numpy as np
import torch
from gymnasium import spaces

class ReplayBuffer:
    """
    An effective Replay Buffer for storing and sampling transitions. 
    """
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device:str = 'cpu'
            ):
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.obs_dtype = observation_space.dtype
        self.action_dim = action_space.n 
        
        # pre-allocate memory
        self.observations = np.zeros((buffer_size, *self.obs_shape), dtype=self.obs_dtype)
        self.next_observations = np.zeros((buffer_size, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        self.pos = 0
        self.full = False
        self.device = device

    def add(self, obs, next_obs, action, reward, done):
        """ Add a transition to the replay buffer.
        Args:
            obs (np.ndarray): The observation at the current time step.
            next_obs (np.ndarray): The observation at the next time step.
            action (int): The action taken at the current time step.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended after taking the action.
        """
        self.observations[self.pos] = np.array(obs, dtype=self.obs_dtype).copy()
        self.next_observations[self.pos] = np.array(next_obs, dtype=self.obs_dtype).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        
    def sample(self, batch_size: int):
        """ Sample a batch of transitions from the replay buffer.
        Args:
            batch_size (int): The number of transitions to sample.
        Returns:
            dict: A dictionary containing sampled observations, next observations, actions, rewards, and dones.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, upper_bound, size=batch_size)
        batch = {
            'observations': torch.tensor(self.observations[indices], dtype=torch.float32, device=self.device),
            'next_observations': torch.tensor(self.next_observations[indices], dtype=torch.float32, device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device),
            'rewards': torch.tensor(self.rewards[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device)
        }
        return batch


    def __len__(self):
        """ Return the current size of the buffer. """
        return self.buffer_size if self.full else self.pos