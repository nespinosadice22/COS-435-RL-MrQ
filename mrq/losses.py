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
from two_hot import TwoHot


def compute_dynamic_loss(next_zs_pred, target_zs, prev_not_done, dyn_weight):
    dynamic_loss = F.mse_loss(next_zs_pred, target_zs, reduction="none")
    dynamic_loss = (dynamic_loss * prev_not_done).mean()
    return dynamic_loss * dyn_weight


def compute_reward_loss(reward_pred, reward, prev_not_done, two_hot, reward_weight):
    reward_pred = F.log_softmax(reward_pred, dim=-1)
    # check okay to do this
    target = two_hot.two_hot_encode(reward)
    reward_loss = -(target * reward_pred).sum(-1, keepdim=True)
    reward_loss = (reward_loss * prev_not_done).mean()
    return reward_loss * reward_weight


def compute_done_loss(
    done_pred, not_done, prev_not_done, i, env_terminates, done_weight
):
    if not env_terminates:
        return 0.0
    else:
        target = 1.0 - not_done[:, i].reshape(-1, 1)
        d_loss = F.mse_loss(done_pred, target, reduction="none")
        d_loss = (d_loss * prev_not_done).mean()
        return d_loss * done_weight


def compute_encoder_loss(
    enc_horizon,
    pred_zs,
    target_zs,
    action,
    reward,
    encoder,
    two_hot,
    env_terminates,
    not_done,
    dyn_weight,
    reward_weight,
    done_weight,
):

    prev_not_done = 1.0  # mask states after termination
    encoder_loss = 0.0

    for i in range(enc_horizon):
        done_pred, next_zs_pred, rew_pred = encoder.predict_all(pred_zs, action[:, i])
        # loss terms
        dynamic_loss = compute_dynamic_loss(
            next_zs_pred, target_zs[:, i], prev_not_done, dyn_weight
        )
        reward_loss = compute_reward_loss(
            rew_pred, reward[:, i], prev_not_done, two_hot, reward_weight
        )
        done_loss = compute_done_loss(
            done_pred, not_done, prev_not_done, i, env_terminates, done_weight
        )

        encoder_loss += dynamic_loss + reward_loss + done_loss
        pred_zs = next_zs_pred
        prev_not_done = not_done[:, i].reshape(-1, 1) * prev_not_done

    return encoder_loss


def compute_value_loss(Q_current, Q_target):
    Q_target = Q_target.expand(-1, 2)
    # manual implementation of smooth_l1_loss
    diff = Q_current - Q_target
    abs_diff = (diff).abs()
    mask = (abs_diff < 1.0).float()
    value_loss = mask * 0.5 * (diff**2) + (1 - mask) * (abs_diff - 0.5)
    value_loss = value_loss.mean()
    return value_loss


def compute_policy_loss(Q_policy, pre_activ_weight, pre_activ):
    policy_loss = -Q_policy.mean() + pre_activ_weight * pre_activ.pow(2).mean()
    return policy_loss
