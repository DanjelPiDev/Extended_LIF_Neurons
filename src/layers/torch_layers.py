import torch
import torch.nn as nn
import numpy as np

from lif.lif_neuron import LIFNeuron
from lif.lif_neuron_group import LIFNeuronGroup


@DeprecationWarning
class SingleLIFLayer(nn.Module):
    """
    Not recommended for large neuron counts, use LIFLayer (wrapper for the LIFNeuronGroup) instead.

    PyTorch-compatible LIF layer that wraps the LIFNeuron class.
    """

    def __init__(self, num_neurons, V_th=1.0, V_reset=0.0, tau=20.0, dt=1.0):
        super(SingleLIFLayer, self).__init__()
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


class LIFLayer(nn.Module):
    """
    A PyTorch wrapper for the LIFNeuronGroup class to integrate with PyTorch layers.
    """

    def __init__(self,
                 num_neurons,
                 V_th=1.0,
                 V_reset=0.0,
                 tau=20.0,
                 dt=1.0,
                 eta=0.1,
                 use_adaptive_threshold=True,
                 noise_std=0.1,
                 stochastic=True,
                 min_threshold=0.5,
                 max_threshold=2.0,
                 batch_size=1,
                 device="cuda",
                 spike_coding=None):
        super(LIFLayer, self).__init__()

        self.lif_group = LIFNeuronGroup(
            num_neurons=num_neurons,
            V_th=V_th,
            V_reset=V_reset,
            tau=tau,
            dt=dt,
            eta=eta,
            use_adaptive_threshold=use_adaptive_threshold,
            noise_std=noise_std,
            stochastic=stochastic,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            batch_size=batch_size,
            device=device
        )
        self.spike_coding = spike_coding

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LIFNeuronGroup in PyTorch.

        :param input_data: Input tensor of shape (batch_size, num_neurons) or (timesteps, batch_size, num_neurons).
        :return: Spike tensor (binary) of shape (batch_size, num_neurons) or (timesteps, batch_size, num_neurons).
        """
        if self.spike_coding == "rate":
            # Specific for rate coding, when e.g. using images as input
            input_data = (torch.rand_like(input_data) < (input_data / 255.0)).float()

        if len(input_data.shape) == 3:
            timesteps, batch_size, num_neurons = input_data.shape
            assert batch_size == self.lif_group.batch_size, \
                f"Batch size must match ({self.lif_group.batch_size})."
            assert num_neurons == self.lif_group.num_neurons, \
                f"Number of neurons must match ({self.lif_group.num_neurons})."

            output_spikes = []
            for t in range(timesteps):
                spikes = self.lif_group.step(input_data[t])
                output_spikes.append(spikes)

            return torch.stack(output_spikes, dim=0)
        else:
            assert input_data.shape == (self.lif_group.batch_size, self.lif_group.num_neurons), \
                f"Input tensor shape must match (batch_size={self.lif_group.batch_size}, num_neurons={self.lif_group.num_neurons})."

            return self.lif_group.step(input_data)
