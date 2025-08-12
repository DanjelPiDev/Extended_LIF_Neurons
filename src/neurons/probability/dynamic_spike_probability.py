import torch
import torch.nn as nn

class DynamicSpikeProbability(nn.Module):
    def __init__(self, base_alpha=1.0, tau_adapt=20.0,
                 min_alpha=0.05, max_alpha=5.0, eps=1e-6):
        super().__init__()
        self.base_alpha = float(base_alpha)
        self.register_buffer("tau_adapt", torch.tensor(float(tau_adapt)))
        self.register_buffer("adaptation", torch.zeros(1, 1))
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.eps = float(eps)

    @torch.no_grad()
    def resize(self, batch_size, num_neurons, device):
        self.adaptation = torch.zeros(batch_size, num_neurons, device=device)

    @torch.no_grad()
    def reset(self, batch_size=None, num_neurons=None, device=None):
        if batch_size is None:
            self.adaptation.zero_()
        else:
            if num_neurons is None:
                num_neurons = self.adaptation.shape[1]
            if device is None:
                device = self.adaptation.device
            self.adaptation = torch.zeros(batch_size, num_neurons, device=device)

    def forward(self, x, prev_spike_float, mod: torch.Tensor | None = None,
                mod_strength: float = 1.0, mod_mode: str = "none"):
        """
        x: delta = V - V_th, shape (B,N)
        prev_spike_float: spike values (float in [0,1]), shape (B,N)
        mod: optional Neuromodulation (B,N) for 'prob_slope'
        """
        exp_factor = torch.exp(-1.0 / self.tau_adapt)
        self.adaptation = self.adaptation * exp_factor + prev_spike_float
        alpha_eff = self.base_alpha / (1.0 + self.adaptation).clamp_min(self.eps)

        if mod_mode == "prob_slope" and mod is not None:
            alpha_eff = alpha_eff * (1.0 + float(mod_strength) * mod)

        alpha_eff = alpha_eff.clamp(self.min_alpha, self.max_alpha)
        return torch.sigmoid(alpha_eff * x), self.adaptation
