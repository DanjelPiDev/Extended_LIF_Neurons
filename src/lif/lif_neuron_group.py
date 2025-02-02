import numpy as np
import torch

from lif.sg.spike_function import SpikeFunction
from lif.probability.dynamic_spike_probability import DynamicSpikeProbability


class LIFNeuronGroup:
    """
    A vectorized LIF neuron model for multiple neurons.
    Because the LIFNeuron is inefficient for large neuron counts.
    """
    def __init__(self,
                 num_neurons: int,
                 V_th: float = 1.0,
                 V_reset: float = 0.0,
                 tau: float = 20.0,
                 dt: float = 1.0,
                 eta: float = 0.1,
                 use_adaptive_threshold: bool = True,
                 noise_std: float = 0.1,
                 stochastic: bool = True,
                 min_threshold: float = 0.5,
                 max_threshold: float = 2.0,
                 batch_size: int = 1,
                 device: str = "cpu",
                 surrogate_gradient_function: str = "heaviside",
                 alpha: float = 1.0,
                 allow_dynamic_spike_probability: bool = False,
                 base_alpha: float = 2.0,
                 tau_adapt: float = 20.0,
                 neuromod_transform=None):
        """
        Initialize the LIF neuron group with its parameters.

        :param num_neurons: Number of neurons in the group.
        :param V_th: Initial threshold voltage for all neurons.
        :param V_reset: Reset voltage after a spike.
        :param tau: Membrane time constant, controlling decay rate.
        :param dt: Time step for updating the membrane potential.
        :param eta: Adaptation rate for the threshold voltage.
        :param noise_std: Standard deviation of Gaussian noise added to the membrane potential.
        :param stochastic: Whether to enable stochastic firing.
        :param min_threshold: Minimum threshold value.
        :param max_threshold: Maximum threshold value.
        :param batch_size: Batch size for the input data.
        :param device: Device to run the simulation on.
        :param surrogate_gradient_function: Surrogate gradient function for backpropagation.
        :param alpha: Parameter for the surrogate gradient function.
        :param allow_dynamic_spike_probability: Whether to allow dynamic spike probability, this takes the last spike into account. Works like a self-locking mechanism.
        :param base_alpha: Base alpha value for the dynamic sigmoid function.
        :param tau_adapt: Time constant for the adaptation.
        :param neuromod_transform: A function or module that takes an external modulation tensor (e.g. reward/error signal)
            and returns a transformed tensor (e.g. modulation factors in [0,1]).
            If None, a default sigmoid transformation will be applied.
        """
        if stochastic:
            assert noise_std > 0, "Noise standard deviation must be positive in stochastic mode."

        assert tau > 0.0, "Membrane time constant must be positive."
        assert num_neurons > 0, "Number of neurons must be positive."
        assert min_threshold > 0, "Minimum threshold must be positive."
        assert max_threshold > min_threshold, "Maximum threshold must be greater than the minimum threshold."
        assert dt > 0, "Time step (dt) must be positive."
        assert batch_size > 0, "Batch size must be positive."
        assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'."
        assert surrogate_gradient_function in ["heaviside", "fast_sigmoid", "gaussian", "arctan"], \
            "Surrogate gradient function must be one of 'heaviside', 'fast_sigmoid', 'gaussian', 'arctan'."
        assert alpha > 0, "Alpha must be positive."

        self.batch_size = batch_size
        self.device = device

        self.num_neurons = num_neurons
        self.V_th = torch.full((batch_size, num_neurons), V_th, device=device)
        self.V_reset = torch.tensor(V_reset, device=device)
        self.tau = torch.tensor(tau, device=device)
        self.dt = torch.tensor(dt, device=device)
        self.eta = torch.tensor(eta, device=device)
        self.V = torch.zeros((batch_size, num_neurons), device=device)
        self.spikes = torch.zeros((batch_size, num_neurons), dtype=torch.bool, device=device)
        self.use_adaptive_threshold = use_adaptive_threshold
        self.noise_std = torch.tensor(noise_std, device=device)
        self.stochastic = stochastic
        self.surrogate_gradient_function = surrogate_gradient_function
        self.alpha = torch.tensor(alpha, device=device)
        self.allow_dynamic_spike_probability = allow_dynamic_spike_probability
        self.dynamic_spike_probability = DynamicSpikeProbability(base_alpha=base_alpha, tau_adapt=tau_adapt).to(device) if allow_dynamic_spike_probability else None

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Dynamic forward pass variables
        self.adaptation_current = torch.zeros((batch_size, num_neurons), device=device)
        self.synaptic_efficiency = torch.ones((batch_size, num_neurons), device=device)
        self.neuromodulator = torch.ones((batch_size, num_neurons), device=device)

        # Parameters for updating dynamic variables.
        self.adaptation_decay = torch.tensor(0.9, device=device)    # Decay factor for adaptation current.
        self.spike_increase = torch.tensor(0.5, device=device)      # Increase in adaptation current upon spiking.
        self.depression_rate = torch.tensor(0.1, device=device)     # Synaptic depression factor.
        self.recovery_rate = torch.tensor(0.05, device=device)      # Rate at which synaptic efficiency recovers.

        self.neuromod_transform = neuromod_transform

    def reset_state(self, initial_threshold=1.0):
        self.V.fill_(0.0)
        self.V_th.fill_(initial_threshold)
        self.spikes.zero_()

        self.adaptation_current.fill_(0.0)
        self.synaptic_efficiency.fill_(1.0)
        self.neuromodulator.fill_(1.0)

    def step(self, I: torch.Tensor, external_modulation: torch.Tensor = None) -> torch.Tensor:
        """
        Simulate one time step for all neurons in the group.

        :param I: Tensor of input currents with shape (batch_size, num_neurons).
        :param external_modulation: Tensor of external neuromodulatory signals with shape
                                    (batch_size, num_neurons) or broadcastable shape.
                                    For example, this could encode a reward signal for dopamine modulation.
        :return: Spike tensor (binary) of shape (batch_size, num_neurons).
        """
        assert I.shape == (self.batch_size, self.num_neurons), \
            "Input current shape must match (batch_size, num_neurons)."

        if external_modulation is not None:
            # For instance, it can simulate a dopamine-like effect:
            #       - High reward --> high dopamine --> increased excitability or enhanced learning.
            #       - Low reward / negative error --> low dopamine --> reduced excitability.
            if self.neuromod_transform is not None:
                self.neuromodulator = self.neuromod_transform(external_modulation)
            else:
                self.neuromodulator = torch.sigmoid(external_modulation)

        noise = torch.normal(0, self.noise_std.item(), size=I.shape, device=self.device) if self.stochastic else torch.zeros_like(I)

        # Modify the input current with dynamic factors:
        # - Multiply by synaptic efficiency (depressed if previous spikes occurred)
        # - Add neuromodulatory effect (could boost or reduce excitability)
        # - Subtract adaptation current (reducing excitability after spiking)
        I_effective = I * self.synaptic_efficiency + self.neuromodulator - self.adaptation_current

        dV = (I_effective - self.V) / self.tau
        self.V += dV * self.dt + noise / self.V_th

        if self.stochastic:
            if not self.allow_dynamic_spike_probability:
                spike_prob = self.sigmoid(self.V - self.V_th)
            else:
                spike_prob = self.dynamic_spike_probability(self.V - self.V_th, self.spikes)
            self.spikes = torch.rand_like(self.V, device=self.device) < spike_prob
        else:
            self.spikes = SpikeFunction.apply(self.V - self.V_th, self.surrogate_gradient_function, self.alpha)

        self.V[self.spikes.bool()] = self.V_reset

        # Update the adaptation current: decay it and add an increment for neurons that spiked
        self.adaptation_current = self.adaptation_current * self.adaptation_decay + self.spike_increase * self.spikes.float()
        # Update synaptic efficiency: depress on spike and allow recovery towards 1
        self.synaptic_efficiency = self.synaptic_efficiency * (1 - self.depression_rate * self.spikes.float()) \
                                   + self.recovery_rate * (1 - self.synaptic_efficiency)

        if self.use_adaptive_threshold:
            self.V_th[self.spikes.bool()] += self.eta
            self.V_th[~self.spikes.bool()] -= self.eta * (self.V_th[~self.spikes.bool()] - 1.0)
        self.V_th = torch.clamp(self.V_th, self.min_threshold, self.max_threshold)

        return self.spikes

    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid function to calculate spike probability.

        :param x: Input tensor.
        :return: Probability tensor in the range [0, 1].
        """
        return 1 / (1 + torch.exp(-x))
