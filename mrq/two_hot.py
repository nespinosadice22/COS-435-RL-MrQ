"""
currently the same - we should change!
"""

import torch
import torch.nn.functional as F


class TwoHot:
    """
    Convert scalar reward --> two adjacent bins for classification
    Cross-entropy loss
    """

    def __init__(
        self,
        device: str = "cuda",
        lower_bound: float = -10.0,
        upper_bound: float = 10.0,
        num_bins: int = 65,
    ):
        self.num_bins = num_bins
        self.device = device
        self.bins = torch.linspace(lower_bound, upper_bound, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1)  # symexp

    def two_hot_encode(self, scalar_rewards: torch.Tensor) -> torch.Tensor:
        """
        Converts scalar rewards to two hot. Returns (batch_size, num_bins) distribution
        This is their function rn
        """
        diff = scalar_rewards - self.bins.reshape(1, -1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind + 1).clamp(0, self.num_bins - 1)]
        weight = (scalar_rewards - lower) / (upper - lower)

        two_hot = torch.zeros(
            scalar_rewards.shape[0], self.num_bins, device=self.device
        )
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind + 1).clamp(0, self.num_bins), weight)
        return two_hot

    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.two_hot_encode(target)
        return -(target * pred).sum(-1, keepdim=True)
