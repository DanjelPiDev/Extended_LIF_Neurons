# layers/torch_layers.py
import torch
import torch.nn as nn
from lif.lif_neuron_group import LIFNeuronGroup

class LIFLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lif_group = LIFNeuronGroup(**kwargs)

    def forward(self, input_seq: torch.Tensor, external_modulation: torch.Tensor = None, return_extras: bool = False):
        timesteps, batch_size, num_neurons = input_seq.shape
        self.lif_group.resize(batch_size, device=input_seq.device)

        spike_trace = torch.zeros_like(input_seq, dtype=torch.bool)
        voltage_trace = torch.zeros_like(input_seq)

        if return_extras and self.lif_group.dynamic_spike_probability is not None:
            adapt_trace = torch.zeros_like(input_seq, dtype=torch.float32)
            alpha_trace = torch.zeros_like(input_seq, dtype=torch.float32)
        else:
            adapt_trace = alpha_trace = None

        for t in range(timesteps):
            ext_mod_t = external_modulation[t] if external_modulation is not None and external_modulation.ndim == 3 else external_modulation
            spikes = self.lif_group(input_seq[t], ext_mod_t)
            spike_trace[t] = spikes
            voltage_trace[t] = self.lif_group.V

            if adapt_trace is not None:
                dsp = self.lif_group.dynamic_spike_probability
                adapt = dsp.adaptation
                adapt_trace[t] = adapt
                # alpha_eff = base_alpha / (1 + adaptation)
                alpha_eff = dsp.base_alpha / (1.0 + adapt).clamp_min(dsp.eps)
                alpha_trace[t] = alpha_eff

        if return_extras and adapt_trace is not None:
            return spike_trace, voltage_trace, {"adaptation_trace": adapt_trace, "alpha_eff_trace": alpha_trace}
        return spike_trace, voltage_trace

    def reset(self):
        self.lif_group.reset()
