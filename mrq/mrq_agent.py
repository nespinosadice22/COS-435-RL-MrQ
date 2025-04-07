import torch
import copy
import numpy as np

from hyperparams import Hyperparameters  # or whatever your HP dataclass is named
from trainer import MRQTrainer
from replay_buffer import ReplayBuffer  # your custom buffer
import utils

from models import Encoder, Policy, Value

class AgentCoordinator:
    """
    A high-level agent that coordinates:
      - The replay buffer
      - The networks & target networks
      - The MRQTrainer for all training updates
      - The logic for exploration, random actions, and target update frequencies
    """

    def __init__(
        self,
        obs_shape,
        action_dim,
        max_action,
        pixel_obs,
        discrete,
        device,
        history=1,
        hp_dict=None
    ):
        self.name = "MRQAgent_Rewritten"
        if hp_dict is None:
            hp_dict = {}
        self.hp = Hyperparameters(**hp_dict)
        self.device = device
        self.discrete = discrete
        self.max_action = max_action
        self.pixel_obs = pixel_obs

        # Possibly scale exploration params for discrete
        if self.discrete:
            self.hp.exploration_noise *= 0.5
            self.hp.noise_clip *= 0.5
            self.hp.target_policy_noise *= 0.5

        # Build replay buffer
        horizon_for_buffer = max(self.hp.enc_horizon, self.hp.Q_horizon)
        self.replay_buffer = ReplayBuffer(
            obs_shape, action_dim, max_action, pixel_obs, self.device,
            history, horizon_for_buffer, 
            self.hp.buffer_size, self.hp.batch_size,
            self.hp.prioritized, initial_priority=self.hp.min_priority
        )

        # Build networks
        #   note: obs_shape[0]*history => channels if pixel-based, or obs_dim if not
        input_channels_or_dim = obs_shape[0] * history
        self.encoder = Encoder(
            obs_channels_or_dim=input_channels_or_dim,
            action_size=action_dim,
            pixel_based=pixel_obs,
            bins_for_reward=self.hp.num_bins,
            latent_dim_state=self.hp.zs_dim,
            latent_dim_action=self.hp.za_dim,
            latent_dim_joint=self.hp.zsa_dim,
            hidden_dim=self.hp.enc_hdim,
            activation_name=self.hp.enc_activ
        ).to(device)

        self.policy = Policy(
            output_dim=action_dim,
            is_discrete=discrete,
            gumbel_temperature=self.hp.gumbel_tau,
            latent_dim_state=self.hp.zs_dim,
            hidden_dim=self.hp.policy_hdim,
            activation_name=self.hp.policy_activ
        ).to(device)

        self.value = Value(
            input_dim=self.hp.zsa_dim,
            hidden_dim=self.hp.value_hdim,
            activation_name=self.hp.value_activ
        ).to(device)

        # Make target copies
        self.encoder_target = copy.deepcopy(self.encoder)
        self.policy_target = copy.deepcopy(self.policy)
        self.value_target = copy.deepcopy(self.value)

        # Create optimizers
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.hp.enc_lr,
            weight_decay=self.hp.enc_wd
        )
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.hp.policy_lr,
            weight_decay=self.hp.policy_wd
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(),
            lr=self.hp.value_lr,
            weight_decay=self.hp.value_wd
        )

        # Build trainer
        self.trainer = MRQTrainer(
            encoder=self.encoder,
            policy=self.policy,
            value=self.value,
            encoder_target=self.encoder_target,
            policy_target=self.policy_target,
            value_target=self.value_target,
            enc_optimizer=self.encoder_optimizer,
            policy_optimizer=self.policy_optimizer,
            value_optimizer=self.value_optimizer,
            hp=self.hp,
            device=self.device,
            discrete=self.discrete
        )

        # Additional bookkeeping
        self.state_shape = self.replay_buffer.state_shape
        self.training_steps = 0
        self.reward_scale = 1.0
        self.target_reward_scale = 0.0

    def select_action(self, state, use_exploration=True):
        """
        Based on the original logic:
          - If buffer < buffer_size_before_training => return None => random
          - else encode state => get policy action => optionally add exploration noise => clamp
        """
        if self.replay_buffer.size < self.hp.buffer_size_before_training and use_exploration:
            return None  # random action from the environment

        with torch.no_grad():
            # shape the input => (1, history*C, H, W) if pixel, or (1, history*obs_dim)
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_t = state_t.view(-1, *self.state_shape)  # (1, ...)

            # encode
            encoded = self.encoder.encode_observation(state_t)
            # policy => (action_out, pre_activ)
            action_out, _ = self.policy(encoded)

            if use_exploration:
                # Add gaussian noise if continuous, or random perturbation if discrete
                noise = torch.randn_like(action_out) * self.hp.exploration_noise
                action_out = action_out + noise

            if self.discrete:
                # pick argmax
                final_action = int(action_out.argmax(dim=1).cpu().numpy()[0])
            else:
                # clamp in [-1,1], then scale by max_action
                final_action = action_out.clamp(-1, 1).cpu().numpy().flatten()
                final_action = final_action * self.max_action

            return final_action

    def train_step(self, env_terminates: bool):
        """
        One training iteration:
          - Possibly update target networks if (training_steps % target_update_freq == 0)
          - run multiple enc updates
          - run single RL update
          - handle priorities
        """
        if self.replay_buffer.size <= self.hp.buffer_size_before_training:
            return

        self.training_steps += 1

        # If it's time, update the target networks and train the encoder multiple times
        if (self.training_steps - 1) % self.hp.target_update_freq == 0:
            self._update_targets()
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()

            for _ in range(self.hp.target_update_freq):
                # Sample for the encoder: we want sub-trajectory with enc_horizon steps
                batch = self.replay_buffer.sample(self.hp.enc_horizon, include_intermediate=True)
                self._train_encoder_batch(batch, env_terminates)

        # Single RL update
        batch = self.replay_buffer.sample(self.hp.Q_horizon, include_intermediate=False)
        self._train_rl_batch(batch)

    def _update_targets(self):
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.value_target.load_state_dict(self.value.state_dict())

    def _train_encoder_batch(self, batch, env_terminates):
        state, action, next_state, reward, not_done = batch
        # Possibly do augmentations (if pixel, if hp.pixel_augs => you can call some augmentation)
        state, next_state = utils.maybe_augment_state(state, next_state, self.pixel_obs, self.hp.pixel_augs)

        # compute the target latents for each step in horizon
        encoder_target_zs = self.trainer.compute_encoder_targets(next_state)
        self.trainer.update_encoder_on_batch(
            state, action, next_state, reward, not_done, encoder_target_zs, env_terminates
        )

    def _train_rl_batch(self, batch):
        state, action, next_state, reward, not_done = batch
        state, next_state = utils.maybe_augment_state(state, next_state, self.pixel_obs, self.hp.pixel_augs)

        # multi-step reward for Q
        cum_reward, cum_not_done = self.trainer.discount_rewards(reward, not_done)

        # 1) critic
        c_loss, Q_current, Q_target = self.trainer.update_critic_on_batch(
            state, action, next_state,
            cum_reward.unsqueeze(-1),  # shape => [batch,1]
            cum_not_done.unsqueeze(-1),  # shape => [batch,1]
            reward_scale=self.reward_scale,
            target_reward_scale=self.target_reward_scale
        )

        # 2) policy
        p_loss = self.trainer.update_policy_on_batch(state)

        # 3) update priorities if prioritized
        if self.hp.prioritized:
            # Q_current => shape [batch,2], Q_target => shape [batch,1], so expand Q_target => [batch,2]
            priority_err = (Q_current - Q_target.expand(-1, 2)).abs().max(dim=1).values
            priority_clamped = priority_err.clamp(min=self.hp.min_priority).pow(self.hp.alpha)
            self.replay_buffer.update_priority(priority_clamped)

    def save(self, save_folder: str):
        """
        Save model weights, training variables, and replay buffer
        """
        torch.save(self.encoder.state_dict(), f"{save_folder}/encoder.pt")
        torch.save(self.encoder_target.state_dict(), f"{save_folder}/encoder_target.pt")
        torch.save(self.encoder_optimizer.state_dict(), f"{save_folder}/encoder_optimizer.pt")

        torch.save(self.policy.state_dict(), f"{save_folder}/policy.pt")
        torch.save(self.policy_target.state_dict(), f"{save_folder}/policy_target.pt")
        torch.save(self.policy_optimizer.state_dict(), f"{save_folder}/policy_optimizer.pt")

        torch.save(self.value.state_dict(), f"{save_folder}/value.pt")
        torch.save(self.value_target.state_dict(), f"{save_folder}/value_target.pt")
        torch.save(self.value_optimizer.state_dict(), f"{save_folder}/value_optimizer.pt")

        var_dict = dict(
            hp=self.hp,
            reward_scale=self.reward_scale,
            target_reward_scale=self.target_reward_scale,
            training_steps=self.training_steps
        )
        np.save(f"{save_folder}/agent_var.npy", var_dict, allow_pickle=True)

        self.replay_buffer.save(save_folder)

    def load(self, load_folder: str):
        """
        Load model weights, training variables, and replay buffer
        """
        self.encoder.load_state_dict(torch.load(f"{load_folder}/encoder.pt"))
        self.encoder_target.load_state_dict(torch.load(f"{load_folder}/encoder_target.pt"))
        self.encoder_optimizer.load_state_dict(torch.load(f"{load_folder}/encoder_optimizer.pt"))

        self.policy.load_state_dict(torch.load(f"{load_folder}/policy.pt"))
        self.policy_target.load_state_dict(torch.load(f"{load_folder}/policy_target.pt"))
        self.policy_optimizer.load_state_dict(torch.load(f"{load_folder}/policy_optimizer.pt"))

        self.value.load_state_dict(torch.load(f"{load_folder}/value.pt"))
        self.value_target.load_state_dict(torch.load(f"{load_folder}/value_target.pt"))
        self.value_optimizer.load_state_dict(torch.load(f"{load_folder}/value_optimizer.pt"))

        var_dict = np.load(f"{load_folder}/agent_var.npy", allow_pickle=True).item()
        self.hp = var_dict["hp"]
        self.reward_scale = var_dict["reward_scale"]
        self.target_reward_scale = var_dict["target_reward_scale"]
        self.training_steps = var_dict["training_steps"]

        self.replay_buffer.load(load_folder)