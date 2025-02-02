import torch
import torch.nn as nn
import numpy as np

class DynamicSpikeProbability(nn.Module):
    def __init__(self, base_alpha=1.0, tau_adapt=20.0):
        """
        :param base_alpha: Base alpha value for the sigmoid function
        :param tau_adapt: Time constant for the adaptation
        """
        super(DynamicSpikeProbability, self).__init__()
        self.base_alpha = base_alpha
        self.tau_adapt = tau_adapt
        self.register_buffer('adaptation', torch.zeros(1))

    def forward(self, x, spikes):
        """
        :param x: Input difference to the threshold voltage
        :param spikes: Spike activity of the neurons in the previous time step
        :return: Dynamic spike probability for the current time step
        """

        spikes = spikes.float()
        # New_Adaption = Old_Adaptation * exp(-dt/tau) + (Spike-Effect)
        # dt = 1
        self.adaptation = self.adaptation * torch.exp(torch.tensor(-1.0 / self.tau_adapt)) + spikes.mean()

        # effective_alpha is like a self-locking mechanism
        effective_alpha = self.base_alpha / (1.0 + self.adaptation)
        prob = 1 / (1 + torch.exp(-effective_alpha * x))
        return prob