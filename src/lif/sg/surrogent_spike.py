from torch import nn

from src.lif.sg.spike_function import SpikeFunction


class SurrogateSpike(nn.Module):
    def forward(self, x, surrogate_gradient_function: str = "heaviside", alpha: float = 1.0):
        return SpikeFunction.apply(x, surrogate_gradient_function, alpha)
