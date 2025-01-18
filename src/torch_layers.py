import torch
import torch.nn as nn
import numpy as np

from lif.lif_neuron import LIFNeuron
from lif.lif_neuron_group import LIFNeuronGroup
from abss.autoregressive_bernoulli_layer import AutoregressiveBernoulliLayer


class TorchLIFLayer(nn.Module):
    """
    PyTorch-compatible LIF layer that wraps the LIFNeuron class.
    """

    def __init__(self, num_neurons, V_th=1.0, V_reset=0.0, tau=20.0, dt=1.0):
        super(TorchLIFLayer, self).__init__()
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(V_th, V_reset, tau, dt) for _ in range(num_neurons)]

    def forward(self, input_current):
        """
        Forward pass for LIF neurons.

        :param input_current: Input tensor of shape (batch_size, num_neurons).
        :return: Spike tensor (binary) of shape (batch_size, num_neurons).
        """
        batch_size = input_current.size(0)
        output_spikes = []

        for i in range(batch_size):
            spikes = [neuron.step(I) for neuron, I in zip(self.neurons, input_current[i].tolist())]
            output_spikes.append(spikes)

        return torch.tensor(output_spikes, dtype=torch.float32)


class TorchLIFNeuronGroup(nn.Module):
    """
    A PyTorch wrapper for the LIFNeuronGroup class to integrate with PyTorch layers.
    """
    def __init__(self, num_neurons, V_th=1.0, V_reset=0.0, tau=20.0, dt=1.0,
                 eta=0.1, use_adaptive_threshold=True, noise_std=0.1,
                 stochastic=True, min_threshold=0.5, max_threshold=2.0):
        super(TorchLIFNeuronGroup, self).__init__()

        self.lif_group = LIFNeuronGroup(
            num_neurons=num_neurons, V_th=V_th, V_reset=V_reset, tau=tau, dt=dt,
            eta=eta, use_adaptive_threshold=use_adaptive_threshold,
            noise_std=noise_std, stochastic=stochastic,
            min_threshold=min_threshold, max_threshold=max_threshold
        )

    def forward(self, input_current):
        """
        Forward pass for the LIFNeuronGroup in PyTorch.

        :param input_current: Input tensor of shape (batch_size, num_neurons).
        :return: Spike tensor (binary) of shape (batch_size, num_neurons).
        """
        if not isinstance(input_current, torch.Tensor):
            raise ValueError("Input current must be a PyTorch tensor.")

        input_current_np = input_current.cpu().numpy()

        batch_size, num_neurons = input_current_np.shape
        output_spikes = []
        for i in range(batch_size):
            spikes = self.lif_group.step(input_current_np[i])
            output_spikes.append(spikes)

        return torch.tensor(output_spikes, dtype=torch.float32, device=input_current.device)


class TorchBernoulliLayer(nn.Module):
    """
    PyTorch-compatible Autoregressive Bernoulli spiking layer that wraps the AutoregressiveBernoulliLayer class.
    """

    def __init__(self, num_neurons, tau=20.0, dt=1.0, autoregressive_weight=0.5):
        super(TorchBernoulliLayer, self).__init__()
        self.num_neurons = num_neurons
        self.layer = AutoregressiveBernoulliLayer(num_neurons, tau, dt)
        self.autoregressive_weight = autoregressive_weight

    def forward(self, input_current):
        """
        Forward pass for Bernoulli spiking neurons.

        :param input_current: Input tensor of shape (batch_size, num_neurons).
        :return: Spike tensor (binary) of shape (batch_size, num_neurons).
        """
        batch_size = input_current.size(0)
        output_spikes = []

        for i in range(batch_size):
            spikes = self.layer.step(input_current[i].tolist(), self.autoregressive_weight)
            output_spikes.append(spikes)

        return torch.tensor(output_spikes, dtype=torch.float32)
