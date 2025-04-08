
import copy
import dataclasses
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import buffer
from hyperparams import Hyperparameters
import models
from utils import maybe_augment_state, realign, masked_mse, multi_step_reward, shift_aug
import utils
from two_hot import TwoHot
from losses import compute_encoder_loss

class Agent:
    def __init__(self, 
        obs_shape: tuple, 
        action_dim: int, 
        max_action: float, 
        pixel_obs: bool, 
        discrete: bool,
        device: torch.device, 
        history: int=1, 
        hp: Dict={}
    ):
        self.name = 'MR.Q'

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        # if discrete, we divide params by 2 
        if discrete:
            self.exploration_noise *= 0.5
            self.noise_clip *= 0.5
            self.target_policy_noise *= 0.5

        #set up replay buffer
        self.replay_buffer = buffer.ReplayBuffer(
            obs_shape, action_dim, max_action, pixel_obs, self.device,
            history, max(self.enc_horizon, self.Q_horizon), self.buffer_size, self.batch_size,
            self.prioritized, initial_priority=self.min_priority
        )

        # ------------------ Encoder ------------------#
        self.encoder = models.Encoder(
            obs_channels_or_dim = obs_shape[0] * history,  # same as old "state_dim"
            action_size         = action_dim, 
            pixel_based         = pixel_obs, 
            bins_for_reward     = self.num_bins, 
            latent_dim_state    = self.zs_dim, 
            latent_dim_action   = self.za_dim, 
            latent_dim_joint    = self.zsa_dim, 
            hidden_dim          = self.enc_hdim, 
            activation_name     = self.enc_activ
        ).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd
        )
        self.encoder_target = copy.deepcopy(self.encoder)

        # ------------------ Policy ------------------#
        self.policy = models.Policy(
            output_dim        = action_dim, 
            is_discrete       = discrete, 
            gumbel_temperature= self.gumbel_tau, 
            latent_dim_state  = self.zs_dim, 
            hidden_dim        = self.policy_hdim, 
            activation_name   = self.policy_activ
        ).to(self.device)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.policy_lr, weight_decay=self.policy_wd
        )
        self.policy_target = copy.deepcopy(self.policy)

        # ------------------ Value ------------------#
        self.value = models.Value(
            input_dim      = self.zsa_dim,    
            hidden_dim     = self.value_hdim, 
            activation_name= self.value_activ
        ).to(self.device)
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=self.value_lr, weight_decay=self.value_wd
        )
        self.value_target = copy.deepcopy(self.value)

        # Used by reward prediction
        self.two_hot = TwoHot(self.device, self.lower, self.upper, self.num_bins)
        
        #make this a helper function 
        self.gammas = torch.zeros(1, self.Q_horizon, 1, device=self.device)
        discount = 1
        for t in range(self.Q_horizon):
            self.gammas[:, t] = discount
            discount *= self.discount
        self.discount = discount

        # Environment properties
        self.pixel_obs = pixel_obs
        self.state_shape = self.replay_buffer.state_shape
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action

        # Tracked values 
        self.reward_scale, self.target_reward_scale = 1, 0
        self.training_steps = 0


    def select_action(self, state: np.array, use_exploration: bool=True):
        """
        Called by your environment loop to pick an action given the current state.
        """
        # Random action if buffer isn't large enough yet
        if self.replay_buffer.size < self.buffer_size_before_training and use_exploration:
            return None  # environment will sample random action (built into main)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float, device=self.device)
            state_t = state_t.reshape(-1, *self.state_shape)
            # encode 
            zs = self.encoder.encode_observation(state_t)
            # policy
            action = self.policy.select_action(zs)

            if use_exploration:
                #add exploration noise
                action += torch.randn_like(action) * self.exploration_noise

            if self.discrete:
                #arg for discrete
                return int(action.argmax().item())
            else:
                #clip to [-1, 1] then scale by max_action for continuous
                return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action


    def train(self):
        """
        Single training iteration
        """
        if self.replay_buffer.size <= self.buffer_size_before_training:
            return

        self.training_steps += 1


        if (self.training_steps - 1) % self.target_update_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()

            #train the encoder for 'target_update_freq' steps
            for _ in range(self.target_update_freq):
                state, action, next_state, reward, not_done = self.replay_buffer.sample(
                    self.enc_horizon, include_intermediate=True
                )
                state, next_state = maybe_augment_state(
                    state, next_state, self.pixel_obs, self.pixel_augs
                )
                #using smaller trajectories? 
                self.train_encoder(
                    state, action, next_state, reward, not_done, 
                    self.replay_buffer.env_terminates
                )

        # WE KNOW THIS 
        state, action, next_state, reward, not_done = self.replay_buffer.sample(
            self.Q_horizon, include_intermediate=False
        )
        state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)

        #multi -step returns
        reward, not_done = multi_step_reward(reward, not_done, self.gammas)

        #train Q and policy 
        Q, Q_target = self.train_rl(
            state, action, next_state, reward, not_done, 
            self.reward_scale, self.target_reward_scale
        )

        #priotized buffer thing 
        if self.prioritized:
            priority = (Q - Q_target.expand(-1,2)).abs().max(dim=1).values
            priority = priority.clamp(min=self.min_priority).pow(self.alpha)
            self.replay_buffer.update_priority(priority)


    def train_encoder(self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor,
        reward: torch.Tensor, 
        not_done: torch.Tensor, 
        env_terminates: bool
    ):
        """
        Encoder 
        """
        batch_size = state.shape[0]

        with torch.no_grad():
            #flatten out horizon dimension
            ns_flat = next_state.reshape(batch_size * self.enc_horizon, *self.state_shape)
            target_zs = self.encoder_target.encode_observation(ns_flat)
            target_zs = target_zs.reshape(batch_size, self.enc_horizon, -1)

        # Our initial predicted embedding: state[:, 0] is the first observation in the sub-trajectory
        pred_zs = self.encoder.encode_observation(state[:, 0])

        #compute loss 
        encoder_loss = compute_encoder_loss(self.enc_horizon, pred_zs, target_zs, 
            action, reward, self.encoder, self.two_hot, env_terminates, not_done, self.dyn_weight, 
            self.reward_weight, self.done_weight )
        
        #update 
        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()


    def train_rl(self, 
                 state: torch.Tensor, 
                 action: torch.Tensor, 
                 next_state: torch.Tensor,
                 reward: torch.Tensor, 
                 not_done: torch.Tensor,
                 reward_scale: float, 
                 target_reward_scale: float):
        """
        Training the Q-value networks and the policy (TD3 style).
        """
        # ------------------------ Target computation ------------------------
        with torch.no_grad():
            next_zs = self.encoder_target.encode_observation(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            noisy_action = self.policy_target.select_action(next_zs) + noise
            next_action = realign(noisy_action, self.discrete)

            next_zsa = self.encoder_target.merge_state_action(next_zs, next_action)
            Q_target_val = self.value_target(next_zsa).min(dim=1, keepdim=True).values

            # y = r + gamma * Q_target
            Q_target = (reward + not_done * self.discount * Q_target_val * target_reward_scale) / reward_scale

            zs = self.encoder.encode_observation(state)
            zsa = self.encoder.merge_state_action(zs, action)
        
        # ------------------------ Current Q training ------------------------
        Q_current = self.value(zsa)
        value_loss = F.smooth_l1_loss(Q_current, Q_target.expand(-1,2))

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()

        # ------------------------ Policy training ------------------------
        policy_action, pre_activ = self.policy(zs)
        # re-encode (z, policy_action) - why???
        policy_zsa = self.encoder.merge_state_action(zs, policy_action)
        Q_policy = self.value(policy_zsa)
        policy_loss = -Q_policy.mean() + self.pre_activ_weight * pre_activ.pow(2).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        return Q_current, Q_target


    def save(self, save_folder: str):
        for key in [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]:
            torch.save(self.__dict__[key].state_dict(), f'{save_folder}/{key}.pt')

        vars_to_save = ['hp', 'reward_scale', 'target_reward_scale', 'training_steps']
        np.save(f'{save_folder}/agent_var.npy', {k: self.__dict__[k] for k in vars_to_save})

        self.replay_buffer.save(save_folder)

    def load(self, save_folder: str):
        for key in [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]:
            self.__dict__[key].load_state_dict(
                torch.load(f'{save_folder}/{key}.pt', weights_only=True)
            )

        var_dict = np.load(f'{save_folder}/agent_var.npy', allow_pickle=True).item()
        for k, v in var_dict.items():
            self.__dict__[k] = v

        self.replay_buffer.load(save_folder)