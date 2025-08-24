import math
import torch
from torch import Tensor

def q_spike_prob(v: Tensor, alpha, beta,
                 k: float = 1.0,
                 theta_max: float = math.pi/2 - 1e-3) -> Tensor:
    phi   = alpha * v
    theta = beta + theta_max * torch.tanh(k * phi)
    p = 0.5 * (1.0 - torch.cos(theta))
    return p.clamp_(1e-5, 1 - 1e-5)

def bernoulli_spike(p: Tensor, training: bool) -> Tensor:
    if training:
        y_hard = torch.bernoulli(p)
        return p + (y_hard - p).detach()
    return (p > 0.5).float()