# buffer.py --> MRQ Replay Buffer implementation
# @citation: buffer.py adapted from https://github.com/facebookresearch/MRQ/tree/main.
"""
Implements a replay buffer supporting:
  • fixed‐size circular storage of transitions
  • optional LAP sampling
  • frame‐history stacking for pixel or vector observations
  • multi‐step returns (horizon) and masking of incomplete trajectories
"""

from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, max_action: float, pixel_obs: bool, device: torch.device, history: int = 1, 
    horizon: int = 1, max_size: int = 1e6,  batch_size: int = 256, prioritized: bool = True, initial_priority: float = 1, normalize_actions: bool = True,
    ):
        self.max_size = int(max_size)
        self.batch_size = batch_size

        self.obs_shape = obs_shape 
        self.obs_dtype = torch.uint8 if pixel_obs else torch.float

        # Size of state given to network
        self.state_shape = [obs_shape[0] * history]  # Channels or obs dim.
        if pixel_obs:
            self.state_shape += [obs_shape[1], obs_shape[2]]  # Image size.
        self.num_channels = obs_shape[0] 

        self.device = device

        # Store obs on GPU if they are sufficient small.
        # revised to run on cpu too
        if torch.cuda.is_available():
            memory, _ = torch.cuda.mem_get_info()
        else:
            memory = 8 * 1024**3
        obs_space = np.prod((self.max_size, *self.obs_shape)) * 1 if pixel_obs else 4
        ard_space = self.max_size * (action_dim + 2) * 4
        if obs_space + ard_space < memory:
            self.storage_device = self.device
        else:
            self.storage_device = torch.device("cpu")

        self.action_dim = action_dim
        self.action_scale = max_action if normalize_actions else 1.0

        # Tracking
        self.ind, self.size = 0, 0
        self.ep_timesteps = 0
        self.env_terminates = (False)

        self.history = history
        self.state_ind = np.zeros((self.max_size, self.history), dtype=np.int32)  
        self.next_ind = np.zeros((self.max_size, self.history), dtype=np.int32)  

        self.history_queue = deque(maxlen=self.history)
         # Initialize with self.ind=0.
        for _ in range(self.history): 
            self.history_queue.append(0)

        # Multi-step
        self.horizon = horizon

        # LAP
        self.prioritized = prioritized
        self.priority = (torch.empty(self.max_size, device=self.device) if self.prioritized else [] )
        self.max_priority = initial_priority

        # Sampling mask, used to hide states that we don't want to sample, either due to truncation or replacing states in the horizon.
        self.mask = torch.zeros(self.max_size, device=self.device if prioritized else torch.device("cpu"))

        # Actual storage
        self.obs = torch.zeros((self.max_size, *self.obs_shape), device=self.storage_device, dtype=self.obs_dtype)
        self.action_reward_notdone = torch.zeros((self.max_size, action_dim + 2), device=self.device, dtype=torch.float)

    # Extract the most recent obs from the state that includes history.
    def extract_obs(self, state: np.array):
        return torch.tensor(state[-self.num_channels :].reshape(self.obs_shape), dtype=self.obs_dtype, device=self.storage_device)

    # Used to map discrete actions to one hot or normalize continuous actions.
    def one_hot_or_normalize(self, action):
        if isinstance(action, int):
            one_hot_action = torch.zeros(self.action_dim, device=self.device)
            one_hot_action[action] = 1
            return one_hot_action
        return torch.tensor(action / self.action_scale, dtype=torch.float, device=self.device)

    def add(self, state: np.array, action, next_state: np.array, reward: float, terminated: bool, truncated: bool ):
        self.obs[self.ind] = self.extract_obs(state)
        self.action_reward_notdone[self.ind, 0] = reward
        self.action_reward_notdone[self.ind, 1] = 1.0 - terminated
        self.action_reward_notdone[self.ind, 2:] = self.one_hot_or_normalize(action)

        if self.prioritized:
            self.priority[self.ind] = self.max_priority

        # Tracking
        self.size = max(self.size, self.ind + 1)
        self.ep_timesteps += 1
        if terminated:
            self.env_terminates = True

        # Masking
        self.mask[(self.ind + self.history - 1) % self.max_size] = 0
         # Allow states that have a completed horizon to be sampled
        if (self.ep_timesteps > self.horizon): 
            self.mask[(self.ind - self.horizon) % self.max_size] = 1

        # History
        next_ind = (self.ind + 1) % self.max_size
         # Track last x=history obs for the state.
        self.state_ind[self.ind] = np.array(self.history_queue, dtype=np.int32) 
        self.history_queue.append(next_ind) 
        self.next_ind[self.ind] = np.array(self.history_queue, dtype=np.int32)
        self.ind = next_ind

        if terminated or truncated:
            self.terminal(next_state, truncated)

    def terminal(self, state: np.array, truncated: bool):
        self.obs[self.ind] = self.extract_obs(state)

        self.mask[(self.ind + self.history - 1) % self.max_size] = 0
        past_ind = (self.ind - np.arange(min(self.ep_timesteps, self.horizon)) - 1) % self.max_size
         # Mask out truncated subtrajectories.
        self.mask[past_ind] = (0 if truncated else 1 ) 

        self.ind = (self.ind + 1) % self.max_size
        self.ep_timesteps = 0

        # Reset queue
        for _ in range(self.history):
            self.history_queue.append(self.ind)

    def sample_ind(self):
        if self.prioritized:
            csum = torch.cumsum(self.priority * self.mask, 0)
            self.sampled_ind = (torch.searchsorted(csum, torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]).cpu().data.numpy())
        else:
            nz = torch.nonzero(self.mask).reshape(-1)
            self.sampled_ind = np.random.randint(nz.shape[0], size=self.batch_size)
            self.sampled_ind = nz[self.sampled_ind]
        return self.sampled_ind

    def sample(self, horizon: int, include_intermediate: bool = False):
        ind = self.sample_ind()
        ind = (ind.reshape(-1, 1) + np.arange(horizon).reshape(1, -1)) % self.max_size

        ard = self.action_reward_notdone[ind]

        # Sample subtrajectory (with horizon dimension) for unrolling dynamics.
        if include_intermediate:
            # Group (state, next_state) to speed up CPU -> GPU transfer.
            state_ind = np.concatenate([self.state_ind[ind], self.next_ind[ind[:, -1].reshape(-1, 1)]], 1 )
            both_state = ( self.obs[state_ind].reshape(self.batch_size, -1, *self.state_shape).to(self.device).type(torch.float))
            state = both_state[:, :-1] 
            next_state = both_state[ :, 1:]
            action = ard[:, :, 2:]  

        # Sample at specific horizon (used for multistep rewards).
        else:
            state_ind = np.concatenate([self.state_ind[ind[:, 0].reshape(-1, 1)], self.next_ind[ind[:, -1].reshape(-1, 1)] ], 1)
            both_state = (self.obs[state_ind].reshape(self.batch_size, 2, *self.state_shape).to(self.device).type(torch.float))
            state = both_state[:, 0]
            next_state = both_state[:, 1] 
            action = ard[:, 0, 2:] 
        return (state, action, next_state, ard[:, :, 0].unsqueeze(-1), ard[:, :, 1].unsqueeze(-1) )

    def update_priority(self, priority: torch.Tensor):
        self.priority[self.sampled_ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reward_scale(self, eps: float = 1e-8):
        return float(self.action_reward_notdone[: self.size, 0].abs().mean().clamp(min=eps))