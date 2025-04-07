
import copy
import torch
import torch.nn.functional as F

from utils import masked_mse, multi_step_reward, maybe_augment_state, realign
from twohot import TwoHot


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

