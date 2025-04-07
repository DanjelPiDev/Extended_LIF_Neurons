import torch
import torch.nn as nn
import numpy as np


class DynamicSpikeProbability(nn.Module):
    def __init__(self, base_alpha=1.0, tau_adapt=20.0, batch_size=1, num_neurons=1):
        """
        :param base_alpha: Base alpha value for the sigmoid function
        :param tau_adapt: Time constant for the adaptation
        """
        super(DynamicSpikeProbability, self).__init__()
        self.base_alpha = base_alpha
        self.register_buffer("tau_adapt", torch.tensor(tau_adapt))
        self.register_buffer("adaptation", torch.zeros(batch_size, num_neurons))

    def forward(self, x, prev_spikes):
        """
        :param x: Input difference to the threshold voltage
        :param prev_spikes: Spike activity of the neurons in the previous time step
        :return: Dynamic spike probability for the current time step
        """
        exp_factor = torch.exp(-1.0 / self.tau_adapt)  # Works with tensor
        self.adaptation = self.adaptation * exp_factor + prev_spikes.float()
        effective_alpha = self.base_alpha / (1.0 + self.adaptation)
        return torch.sigmoid(effective_alpha * x), self.adaptation

    def resize(self, batch_size, num_neurons):
        self.adaptation = torch.zeros(batch_size, num_neurons, device=self.adaptation.device)

    def reset(self, batch_size=None):
        if batch_size is not None and batch_size != self.adaptation.shape[0]:
            self.adaptation = torch.zeros(batch_size, self.adaptation.shape[1], device=self.adaptation.device)
        else:
            self.adaptation.zero_()

