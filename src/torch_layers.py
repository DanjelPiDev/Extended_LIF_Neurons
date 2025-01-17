import torch
import torch.nn as nn
import numpy as np

from lif.lif_neuron import LIFNeuron
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
