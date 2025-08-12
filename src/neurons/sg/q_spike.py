import torch
from torch import Tensor

def q_spike_prob(v: Tensor, alpha: float, beta: float) -> Tensor:
    theta = alpha * v + beta
    return 0.5 * (1.0 - torch.cos(theta))

def bernoulli_spike(p: Tensor, training: bool) -> Tensor:
    if training:
        return torch.bernoulli(p)
    return (p > 0.5).float()
