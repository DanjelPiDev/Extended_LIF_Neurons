import torch
import torch.nn as nn
from torch import Tensor
from src.lif.lif_neuron_group import LIFNeuronGroup


class LIFLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lif_group = LIFNeuronGroup(**kwargs)

    def forward(self, input_seq: Tensor, external_modulation: Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Simulates the LIF neuron group over time.
        Input shape: (timesteps, batch_size, num_neurons)
        Output: (timesteps, batch_size, num_neurons), voltages
        """
        timesteps, batch_size, num_neurons = input_seq.shape

        # Resize internal buffers to match batch size
        self.lif_group.resize(batch_size)

        spike_trace = torch.zeros_like(input_seq, dtype=torch.bool)
        voltage_trace = torch.zeros_like(input_seq)

        for t in range(timesteps):
            ext_mod_t = external_modulation[t] if external_modulation is not None and external_modulation.ndim == 3 else external_modulation
            spikes = self.lif_group(input_seq[t], ext_mod_t)
            spike_trace[t] = spikes
            voltage_trace[t] = self.lif_group.V

        return spike_trace, voltage_trace

    def reset(self):
        self.lif_group.reset()