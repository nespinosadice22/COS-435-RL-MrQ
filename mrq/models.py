"""
Define neural net archs AND losses! Defining losses is different than MrQ implementationo
"""

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(layer: torch.nn.modules):
    # Initializes weights for convlutional or linear layers using a relu gain factor
    # Zeros out bias
    # This is their helper method, just rewritten for clarity
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        gain_factor = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(layer.weight.data, gain_factor)
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0.0)


def apply_layernorm_and_activation(
    x: torch.Tensor, activation_fn: Callable
) -> torch.Tensor:
    # Applies layer normalization to the last dimension of inputs, then applies the given activation function
    # This is their helper method, just rewritten for clarity
    normed_x = F.layer_norm(x, (x.shape[-1],))
    return activation_fn(normed_x)


# ---------------------------------------------------------------------------#
class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        activation_name: str = "elu",
    ):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_features)

        self.activation_fn = getattr(F, activation_name)
        self.apply(initialize_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = apply_layernorm_and_activation(self.layer1(inputs), self.activation_fn)
        x = apply_layernorm_and_activation(self.layer2(x), self.activation_fn)
        return self.layer3(x)


# ---------------------------------------------------------------------------#
class Encoder(nn.Module):
    # Encoder pixel observations or low-D states into a latent state.
    # Then predicts next state embedding, reward (using categorical bins) and the termination signal
    def __init__(
        self,
        obs_channels_or_dim: int,
        action_size: int,
        pixel_based: bool,
        bins_for_reward: int = 65,
        latent_dim_state: int = 512,
        latent_dim_action: int = 256,
        latent_dim_joint: int = 512,
        hidden_dim: int = 512,
        activation_name: str = "elu",
    ):
        super().__init__()

        self.pixel_based = pixel_based
        self.latent_dim_state = latent_dim_state
        self.bins_for_reward = bins_for_reward
        self.activation_fn = getattr(F, activation_name)

        # if pixel-based, we use CNNs
        if self.pixel_based:
            self.conv1 = nn.Conv2d(obs_channels_or_dim, 32, 3, stride=2)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
            self.linear_after_conv = nn.Linear(1568, latent_dim_state)
        # otherwise, MLP for vector states
        else:
            self.state_mlp = MLPBlock(
                in_features=obs_channels_or_dim,
                out_features=latent_dim_state,
                hidden_dim=hidden_dim,
                activation_name=activation_name,
            )

        # Action embedding (actions --> latent)
        self.action_encoder = nn.Linear(action_size, latent_dim_action)

        # Joint MLP: action + state embedding
        self.joint_mlp = MLPBlock(
            latent_dim_state + latent_dim_action,
            latent_dim_joint,
            hidden_dim,
            activation_name,
        )

        # Output head: next-latent, reward-bins and done-scalar
        self.predictor_head = nn.Linear(
            latent_dim_joint, bins_for_reward + latent_dim_state + 1
        )
        self.apply(initialize_weights)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        # raw obs to zs via cnn or mlp
        # this combines cnn_zs and mlp_zs from original code
        # pixels
        if self.pixel_based:
            normalized = (obs / 255.0) - 0.5
            h = self.activation_fn(self.conv1(normalized))
            h = self.activation_fn(self.conv2(h))
            h = self.activation_fn(self.conv3(h))
            h = self.activation_fn(self.conv4(h))
            # flatten needed?
            h = h.reshape(obs.shape[0], -1)
            return apply_layernorm_and_activation(
                self.linear_after_conv(h), self.activation_fn
            )
        # vector states
        else:
            return apply_layernorm_and_activation(
                self.state_mlp(obs), self.activation_fn
            )

    def merge_state_action(
        self, encoded_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # zs + raw action = zsa
        action_latent = self.activation_fn(self.action_encoder(action))
        merged_zsa_input = torch.cat([encoded_state, action_latent], dim=1)
        return self.joint_mlp(merged_zsa_input)

    def forward(self, encoded_state: torch.Tensor, action: torch.Tensor):
        # prediction
        joint_zsa_encoding = self.merge_state_action(encoded_state, action)
        prediction = self.predictor_head(joint_zsa_encoding)
        # output shape is [batch_size, bins_for_reward + latent_dim_state + 1]
        return prediction

    def predict_all(self, encoded_state: torch.Tensor, action: torch.Tensor):
        # returns done_logits, next_latent, reward_logits
        prediction = self.forward(encoded_state, action)
        done_pred = prediction[:, 0:1]
        next_state_pred = prediction[:, 1 : 1 + self.latent_dim_state]
        reward_pred = prediction[:, 1 + self.latent_dim_state :]
        return done_pred, next_state_pred, reward_pred

    #NEW to do planning - just wanna get plain dynamics 
    def get_dynamics(self, zs, a): 
        done_pred, next_state_pred, reward_pred = self.predict_all(zs, a) 
        not_done_pred = 1.0 - torch.sigmoid(done_pred)
        return next_state_pred, reward_pred, not_done_pred


# ---------------------------------------------------------------------------#
class Policy(nn.Module):
    # zs to next a. gumbel-softmax for discrete, continuous for tanh
    # returns (actions, pre_activation)

    def __init__(
        self,
        output_dim: int,
        is_discrete: bool,
        gumbel_temperature: float = 10.0,
        latent_dim_state: int = 512,
        hidden_dim: int = 512,
        activation_name: str = "relu",
    ):
        super().__init__()

        self.is_discrete = is_discrete
        self.policy_mlp = MLPBlock(
            latent_dim_state, output_dim, hidden_dim, activation_name
        )

        if self.is_discrete:
            self.action_activation = partial(F.gumbel_softmax, tau=gumbel_temperature)
        else:
            self.action_activation = torch.tanh

    def forward(self, encoded_state: torch.Tensor):
        # action + raw pre activation
        pre_activation = self.policy_mlp(encoded_state)
        action_out = self.action_activation(pre_activation)
        return action_out, pre_activation

    def select_action(self, encoded_state: torch.Tensor):
        # action
        action, _ = self.forward(encoded_state)
        return action


# ---------------------------------------------------------------------------#
class Value(nn.Module):
    # x2 Q networks, each an MLP.
    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 512, activation_name: str = "elu"
    ):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, inp_dim, out_dim, hid, activ):
                super().__init__()
                self.q_mlp = MLPBlock(inp_dim, hid, hid, activ)
                self.q_final = nn.Linear(hid, out_dim)
                self.act_fn = getattr(F, activ)
                self.apply(initialize_weights)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # MLP block -> activation + final linear
                temp = apply_layernorm_and_activation(self.q_mlp(x), self.act_fn)
                return self.q_final(temp)

        self.q1 = ValueNetwork(input_dim, 1, hidden_dim, activation_name)
        self.q2 = ValueNetwork(input_dim, 1, hidden_dim, activation_name)

    def forward(self, joint_encoding: torch.Tensor) -> torch.Tensor:
        # Returns a 2D tensor: [batch_size, 2] representing Q1 and Q2
        q1_val = self.q1(joint_encoding)
        q2_val = self.q2(joint_encoding)
        return torch.cat([q1_val, q2_val], dim=1)
