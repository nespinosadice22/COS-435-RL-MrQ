# utils.py --> various helper functions we use throughout 
# @citation: utils.py adapted from https://github.com/facebookresearch/MRQ/tree/main.
# we added a couple of our own too

import torch
import torch.nn.functional as F
import dataclasses 
import pprint
import numpy as np

def enforce_dataclass_type(dataclass: dataclasses.dataclass):
    for field in dataclasses.fields(dataclass):
        setattr(dataclass, field.name, field.type(getattr(dataclass, field.name)))

def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp):
        c.__dict__[field.name] = getattr(hp, field.name)

def realign(x, discrete: bool):
    return F.one_hot(x.argmax(1), x.shape[1]).float() if discrete else x.clamp(-1, 1)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction="none") * mask).mean()


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, gammas: torch.Tensor):
    return (reward * not_done * gammas).sum(1), not_done.prod(1)


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5:
            state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width), ], 0 )
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


# Random shift.
def shift_aug(image: torch.Tensor, pad: int = 4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), "replicate")
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace( -1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float )
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode="zeros", align_corners=False)


def enforce_types(obj):
    for name, val in vars(obj).items():
        if isinstance(val, (int, float, bool, str)):
            try:
                setattr(obj, name, type(val)(val))
            except Exception:
                pass

def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp):
        c.__dict__[field.name] = getattr(hp, field.name)


# Takes the formatted results and returns a dictionary of env -> (timesteps, seed).
def results_to_numpy(file: str = "../results/gym_results.txt"):
    results = {}

    for line in open(file):
        if "----" in line:
            continue
        if "Timestep" in line:
            continue
        if "Env:" in line:
            env = line.split(" ")[1][:-1]
            results[env] = []
        else:
            timestep = []
            for seed in line.split("\t")[1:]:
                if seed != "":
                    seed = seed.replace("\n", "")
                    timestep.append(float(seed))
            results[env].append(timestep)

    for k in results:
        results[k] = np.array(results[k])
        print(k, results[k].shape)

    return results

