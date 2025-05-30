# hyperparameters.py --> MRQ training hyperparameters
# @citation: hyperparameters.py adapted from https://github.com/facebookresearch/MRQ/tree/main.
"""
Defines a dataclass for MRQ's training hyperparameters.
"""
import dataclasses

@dataclasses.dataclass
class Hyperparameters:
    """
    All tunable hyperparameters for MRQ.
    """
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_freq: int = 250
    buffer_size_before_training: int = 10e3
    exploration_noise: float = 0.2

    # TD3 specific 
    target_policy_noise: float = 0.2
    noise_clip: float = 0.3

    # Encoder Loss
    dyn_weight: float = 1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Replay Buffer 
    prioritized: bool = True
    alpha: float = 0.4
    min_priority: float = 1
    enc_horizon: int = 5
    Q_horizon: int = 3

    # Encoder Model
    zs_dim: int = 512
    zsa_dim: int = 512
    za_dim: int = 256
    enc_hdim: int = 512
    enc_activ: str = "elu"
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4
    pixel_augs: bool = True

    # Value Model
    value_hdim: int = 512
    value_activ: str = "elu"
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20

    # Policy Model
    policy_hdim: int = 512
    policy_activ: str = "relu"
    policy_lr: float = 3e-4
    policy_wd: float = 1e-4
    gumbel_tau: float = 10
    pre_activ_weight: float = 1e-5

    # Reward encoding 
    num_bins: int = 65
    lower: float = -10
    upper: float = 10

    #planning ablation
    use_planning : bool = False
    plan_discount : float = 0.99

