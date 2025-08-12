# qlif_layers/torch_layers.py
import torch
import torch.nn as nn
from neurons.qlif import QLIF

class QLIFLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qlifs = QLIF(**kwargs)

    def forward(self, input_seq: torch.Tensor,
                external_modulation: torch.Tensor = None,
                return_extras: bool = False):
        T, B, N = input_seq.shape
        dev = input_seq.device
        self.qlifs.resize(B, device=dev)

        spike_trace   = torch.zeros_like(input_seq, dtype=torch.bool)
        voltage_trace = torch.zeros_like(input_seq)

        want_quantum_extras = return_extras and bool(self.qlifs.quantum_mode)
        want_dsp_extras     = return_extras and (self.qlifs.dynamic_spike_probability is not None)

        q_scale_trace = None
        adaptcur_trace = None
        adapt_trace = None
        alpha_trace = None

        if want_quantum_extras:
            q_scale_trace   = torch.zeros(T, B, N, device=dev)
            adaptcur_trace  = torch.zeros(T, B, N, device=dev)

        if want_dsp_extras:
            dsp = self.qlifs.dynamic_spike_probability
            adapt_trace = torch.zeros(T, B, N, device=dev)
            alpha_trace = torch.zeros(T, B, N, device=dev)
            eps = float(dsp.eps)
            base_alpha = float(dsp.base_alpha)

        for t in range(T):
            I_t  = input_seq[t]
            m_t  = external_modulation[t] if external_modulation is not None else None
            _ = self.qlifs(I_t, m_t)  # step

            spike_trace[t]   = self.qlifs.spikes
            voltage_trace[t] = self.qlifs.V

            if want_quantum_extras:
                q_like = self.qlifs._expand_like(self.qlifs.q_scale.to(dev), self.qlifs.V)
                q_scale_trace[t]  = q_like
                adaptcur_trace[t] = self.qlifs.adaptation_current

            if want_dsp_extras:
                dsp = self.qlifs.dynamic_spike_probability
                adapt = dsp.adaptation
                adapt_trace[t] = adapt
                alpha_eff = base_alpha / (1.0 + adapt).clamp_min(eps)
                alpha_trace[t] = alpha_eff

        if return_extras:
            extras = {}
            if want_quantum_extras:
                extras["q_scale_trace"] = q_scale_trace
                extras["adaptation_current_trace"] = adaptcur_trace
            if want_dsp_extras:
                extras["adaptation_trace"] = adapt_trace
                extras["alpha_eff_trace"]  = alpha_trace
            return spike_trace, voltage_trace, extras

        return spike_trace, voltage_trace

    def reset(self):
        self.qlifs.reset()
