'''
Working version of MRQ.py. Loss functions should be explicit (here or in models.py?)
Obviously not done 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from models import Encoder, Policy, Value


#at some point we can rename these
@dataclasses.dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_freq: int = 250

    # Exploration
    buffer_size_before_training: int = 10e3
    exploration_noise: float = 0.2

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.3

    # Encoder Loss
    dyn_weight: float = 1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Replay Buffer (LAP)
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
    enc_activ: str = 'elu'
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4
    pixel_augs: bool = True

    # Value Model
    value_hdim: int = 512
    value_activ: str = 'elu'
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20

    # Policy Model
    policy_hdim: int = 512
    policy_activ: str = 'relu'
    policy_lr: float = 3e-4
    policy_wd: float = 1e-4
    gumbel_tau: float = 10
    pre_activ_weight: float = 1e-5

    # Reward model
    num_bins: int = 65
    lower: float = -10
    upper: float = 10

    def __post_init__(self): utils.enforce_dataclass_type(self)

#------------------------------Two Hot Reward Encoding Helper--------------------------------------#
class TwoHotEncoder: 
    """
    Convert scalar reward --> two adjacent bins for classification
    Cross-entropy loss  
    """
    def __init__(self, lower_bound: float = -10.0, upper_bound: float = 10.0, num_bins: int = 65, device: str = "cuda"): 
        self.num_bins = num_bins 
        self.device = device 
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp

    def two_hot_encode(self, scalar_rewards: torch.Tensor) -> torch.Tensor: 
        """
        Converts scalar rewards to two hot. Returns (batch_size, num_bins) distribution
        This is their function rn 
        """
        diff = scalar_rewards - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (scalar_rewards - lower)/(upper - lower)

        two_hot = torch.zeros(scalar_rewards.shape[0], self.num_bins, device=self.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot
    
    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)


    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.two_hot_encode(target)
        return -(target * pred).sum(-1, keepdim=True)



#------------------------------All their helpers--------------------------------------#
def realign(x, discrete: bool):
    return F.one_hot(x.argmax(1), x.shape[1]).float() if discrete else x.clamp(-1,1)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, gammas: torch.Tensor):
    return (reward * not_done * gammas).sum(1), not_done.prod(1)


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


# Random shift.
def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)

#------------------------------Agent--------------------------------------#
class MrQAgent: 
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool,
        device: torch.device, history: int=1, hyerparams: dict = {}):
        
        self.name = 'MR.Q'
        self.device = device 
        self.is_discrete = is_discrete
        self.max_action = max_action
        self.action_dim = action_dim 
        self.pixel_obs = pixel_obs 
        self.history = history 

        self.config = Hyperparameters(**hyperparams)


        #Scale action noise since discrete actions are [0,1] and continuous actions are [-1,1].
        if discrete: 
            self.config.exploration_noise *= 0.5
            self.config.noise_clip *= 0.5
            self.config.target_policy_noise *= 0.5

        #initialize two hot reward bins 
        self.twohot = TwoHotBins(
            device=self.device,
            lower=self.config.lower,
            upper=self.config.upper,
            num_bins=self.config.num_bins
        )

        #create encoder 
        self.encoder = Encoder(
            obs_channels_or_dim=obs_shape[0]*history if pixel_obs else np.prod(obs_shape)*history,
            action_size = action_dim, 
            pixel_based = self.pixel_obs, 
            bins_for_reward = self.config.num_bins, 
            latent_dim_state = self.config.zs_dim, 
            latent_dim_action = self.config.za_dim, 
            latent_dim_joint = self.config.zsa_dim, 
            hidden_dim = self.config.enc_hdim, 
            activation_name = self.config.enc_activ
        ).to(self.device)
        

        #create policy 
        self.policy = Policy(
            output_dim=action_dim,
            is_discrete=discrete, 
            gumbel_temperature = self.config.gumbel_tau, 
            latent_dim_state = self.config.zs_dim, 
            hidden_dim = self.config.policy_hdim, 
            activation_name =self.config.policy_activ
        ).to(self.device) 

        
        #value net 
        self.value_net = Value(
            input_dim = self.config.zsa_dim, 
            hidden_dim = self.config.value_hdim
            activation_name = self.config.value_activ
        ).to(self.device)
        
        #make targets 
        self.encoder_target = copy.deepcopy(self.encoder)
        self.policy_target = copy.deepcopy(self.policy)
        self.value_target = copy.deepcopy(self.value_net)

        #optimizers 
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), 
            lr=self.config.enc_lr,
            weight_decay=self.config.enc_wd
        )
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.policy_lr,
            weight_decay=self.config.policy_wd
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(),
            lr=self.config.value_lr,
            weight_decay=self.config.value_wd
        )

        #create replay buffer 
        self.replay_buffer = buffer.ReplayBuffer(
            obs_shape, action_dim, max_action, pixel_obs, self.device,
            history, max(self.configs.enc_horizon, self.config.Q_horizon), self.config.buffer_size, self.config.batch_size,
            self.config.prioritized, initial_priority=self.config.min_priority)

        self.gammas = torch.zeros(1, self.config.Q_horizon, 1, device=self.device)

        #discount 
        discount = 1
        for t in range(self.config.Q_horizon):
            self.gammas[:,t] = discount
            discount *= self.config.discount
        self.discount_factor = discount

        # Environment properties
        self.pixel_obs = pixel_obs
        self.state_shape = self.config.replay_buffer.state_shape # This includes history, horizon, channels, etc.
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action

        # Tracked values
        self.reward_scale = 1.0 
        self.target_reward_scale = 0.0 
        self.training_steps = 0


    def select_action(self, obs_np:np.ndarray, use_exporation: bool=True): 
        """" 

        """
        

    def train on batch 

    def train_rl train_encoder 

    losses


    save 

    load 

