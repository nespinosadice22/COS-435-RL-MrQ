import copy
import dataclasses
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import buffer
import models
import utils
from hyperparams import Hyperparameters
from losses import (compute_encoder_loss, compute_policy_loss, compute_value_loss)
from two_hot import TwoHot
from utils import (masked_mse, maybe_augment_state, multi_step_reward, realign, shift_aug)

import wandb 
class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool,
        discrete: bool, device: torch.device, history: int = 1, hp: Dict = {}):
        #-----same as original-------
        self.name = "MR.Q"
        self.hp = Hyperparameters(**hp) 
        utils.set_instance_vars(self.hp, self)
        self.device = device

        # if discrete, we divide params by 2 
        if discrete:
            self.exploration_noise *= 0.5
            self.noise_clip *= 0.5
            self.target_policy_noise *= 0.5

        # set up replay buffer
        self.replay_buffer = buffer.ReplayBuffer(obs_shape, action_dim, max_action, pixel_obs, self.device, history, 
            max(self.enc_horizon, self.Q_horizon), self.buffer_size, self.batch_size, self.prioritized, initial_priority=self.min_priority,)

        # ------------------ encoder ------------------#
        self.encoder = models.Encoder(obs_channels_or_dim=obs_shape[0] * history, action_size=action_dim, pixel_based=pixel_obs, bins_for_reward=self.num_bins,
            latent_dim_state=self.zs_dim, latent_dim_action=self.za_dim, latent_dim_joint=self.zsa_dim, hidden_dim=self.enc_hdim, activation_name=self.enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd)
        self.encoder_target = copy.deepcopy(self.encoder)
        # ------------------ policy ------------------#
        self.policy = models.Policy(output_dim=action_dim, is_discrete=discrete, gumbel_temperature=self.gumbel_tau, latent_dim_state=self.zs_dim, 
            hidden_dim=self.policy_hdim, activation_name=self.policy_activ).to(self.device)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.policy_lr, weight_decay=self.policy_wd)
        self.policy_target = copy.deepcopy(self.policy)
        # ------------------ value net ------------------#
        self.value = models.Value(input_dim=self.zsa_dim, hidden_dim=self.value_hdim, activation_name=self.value_activ).to(self.device)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=self.value_lr, weight_decay=self.value_wd)
        self.value_target = copy.deepcopy(self.value)
        #-------------------REWARDS LOGIC-----------------# 
        self.two_hot = TwoHot(self.device, self.lower, self.upper, self.num_bins)
        self.gammas = torch.zeros(1, self.Q_horizon, 1, device=self.device)
        discount = 1
        for t in range(self.Q_horizon):
            self.gammas[:, t] = discount
            discount *= self.discount
        self.discount = discount
        self.reward_scale, self.target_reward_scale = 1, 0
        #-------------------ENVIRONMENT PROPERTIES-----------------# 
        self.pixel_obs = pixel_obs
        self.state_shape = self.replay_buffer.state_shape
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action
       
        self.training_steps = 0
        self.wandb_log_count = 0 

        self.plan_discount = 0.99 

    def select_action(self, state: np.array, use_exploration: bool = True):
        # Random action if buffer isn't large enough yet --> random action (done in main)
        if (self.replay_buffer.size < self.buffer_size_before_training and use_exploration):
            return None
        #otherwise...
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float, device=self.device)
            state_t = state_t.reshape(-1, *self.state_shape)
            # encode
            zs = self.encoder.encode_observation(state_t)
            # get a from policy
            action = self.policy.select_action(zs)
            # add exploration 
            if use_exploration:
                action += torch.randn_like(action) * self.exploration_noise
            # get action (and clip continuous)
            if self.discrete:
                return int(action.argmax()) 
            else:
                return (action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action)

    def train(self):
        if self.replay_buffer.size <= self.buffer_size_before_training:
            return
        self.training_steps += 1
        log_data = {}
        #--------------IF WE NEED TO TRAIN ENCODER------------------#
        if (self.training_steps - 1) % self.target_update_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()
            # train the encoder for target_update_freq steps
            encoder_losses = [] 
            for i in range(self.target_update_freq):
                state, action, next_state, reward, not_done = self.replay_buffer.sample(self.enc_horizon, include_intermediate=True)
                state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)
                encoder_loss = self.train_encoder(state, action, next_state, reward, not_done, self.replay_buffer.env_terminates)
                encoder_losses.append(encoder_loss)
            log_data["loss/encoder"] = torch.stack(encoder_losses).mean().item()
        #------------OTHERWISE, PRETTY STANDARD Q LEARNING UPDATE-------#
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.Q_horizon, include_intermediate=False)
        state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)
        reward, not_done = multi_step_reward(reward, not_done, self.gammas)
        # compute targets
        Q_target, zs, zsa = self.compute_targets(state, action, next_state, reward, not_done, self.reward_scale, self.target_reward_scale)
        # update value
        Q_current, value_loss = self.train_value(zsa, Q_target)
        # update policy
        Q_policy, policy_loss = self.train_policy(zs)

        # priotized buffer thing (didn't touch)
        if self.prioritized:
            priority = (Q_current - Q_target.expand(-1, 2)).abs().max(dim=1).values
            priority = priority.clamp(min=self.min_priority).pow(self.alpha)
            self.replay_buffer.update_priority(priority)

        if (self.training_steps - 1) % self.target_update_freq == 0:
            log_data["loss/value"]  = value_loss.item()
            log_data["loss/policy"] = policy_loss.item()
        return log_data 
   

    def train_encoder(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
        batch_size = state.shape[0]
        with torch.no_grad():
            # flatten out horizon dimension
            ns_flat = next_state.reshape(batch_size * self.enc_horizon, *self.state_shape)
            target_zs = self.encoder_target.encode_observation(ns_flat)
            target_zs = target_zs.reshape(batch_size, self.enc_horizon, -1)
        # encode the initial state 
        pred_zs = self.encoder.encode_observation(state[:, 0])
        # compute loss
        encoder_loss = compute_encoder_loss(self.enc_horizon, pred_zs, target_zs, action, reward, self.encoder,self.two_hot,
            env_terminates, not_done, self.dyn_weight, self.reward_weight, self.done_weight)
        # backward update
        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()
        return encoder_loss

    def get_next_action(self, action, target_policy_noise, noise_clip, next_zs, policy_target, discrete):
        noise = (torch.randn_like(action) * target_policy_noise).clamp(-noise_clip, noise_clip)
        next_action = self.policy_target.select_action(next_zs) + noise
        if discrete:
            return F.one_hot(next_action.argmax(1), next_action.shape[1]).float()
        else:
            return next_action.clamp(-1, 1)

    def compute_targets(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, not_done: torch.Tensor,
        reward_scale: float, target_reward_scale: float):
        with torch.no_grad():
            next_zs = self.encoder_target.encode_observation(next_state)
            next_action = self.get_next_action(action, self.target_policy_noise, self.noise_clip, next_zs, self.policy_target, self.discrete)
            next_zsa = self.encoder_target.merge_state_action(next_zs, next_action)
            Q_target_value = self.value_target(next_zsa).min(dim=1, keepdim=True).values
            Q_target = (reward + not_done * self.discount * Q_target_value * target_reward_scale) / reward_scale
            zs = self.encoder.encode_observation(state)
            zsa = self.encoder.merge_state_action(zs, action)
            return Q_target, zs, zsa

    def train_value(self, zsa, Q_target):
        Q_current = self.value(zsa)
        value_loss = compute_value_loss(Q_current, Q_target)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()
        return Q_current, value_loss

    def train_policy(self, zs):
        #get a from pi 
        a_pi, pre_activ = self.policy(zs)
        #get zsa embedding based on pi's a
        zsa_pi = self.encoder.merge_state_action(zs, a_pi)
        #without planning, we follow Mr.Q as normal, which means we get Q from value network
        if not self.use_planning: 
            Q_pi = self.value(zsa_pi) 
        #with planning 
        else: 
            #we get the next state, reward and termination signal from the ENCODER
            zs_next, reward_pred, not_done_pred = self.encoder.get_dynamics(zs, a_pi)
            #we need a scalar reward for this 
            reward_pred = self.two_hot.inverse(reward_pred)
            scaled_reward_pred = (reward_pred * self.target_reward_scale) / self.reward_scale
            #when we get the next a from pi(zs'), we detach gradient...
            with torch.no_grad(): 
                a_next, _ = self.policy(zs_next)
            #get zsa embedding based on that a_next
            zsa_next = self.encoder.merge_state_action(zs_next, a_next)
            #i think without a gradient??
            with torch.no_grad(): 
                #here we're calculating the term E(Q(s', a'))
                Q_expectation = self.value(zsa_next) 
            #Now we can approximate it as r + gamma * (not done) * E[Q(s', a')] 
            Q_pi = scaled_reward_pred + self.plan_discount * not_done_pred * Q_expectation 

        #get policy loss and do backward update the same
        policy_loss = compute_policy_loss(Q_pi, self.pre_activ_weight, pre_activ)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        return Q_pi, policy_loss

#taking min here but not above because of gradient? 
#torch's min returns (values, tuple) - take values? 
#Q_expectation = self.value(zsa_next).min(dim=1, keepdim=True).values 