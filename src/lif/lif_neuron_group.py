import numpy as np
import torch


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
                 device: str = "cpu",):
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

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def step(self, I: torch.Tensor) -> torch.Tensor:
        """
        Simulate one time step for all neurons in the group.

        :param I: Input current (array of size `num_neurons`).
        :return: Array of spike statuses (True for neurons that fired, False otherwise).
        """
        assert I.shape == (self.batch_size, self.num_neurons), \
            "Input current shape must match (batch_size, num_neurons)."

        noise = torch.normal(0, self.noise_std.item(), size=I.shape, device=self.device) if self.stochastic else torch.zeros_like(I)

        dV = (I - self.V) / self.tau
        self.V += dV * self.dt + noise / self.V_th

        if self.stochastic:
            spike_prob = self.sigmoid(self.V - self.V_th)
            self.spikes = torch.rand_like(self.V, device=self.device) < spike_prob
        else:
            self.spikes = self.V >= self.V_th

        self.V[self.spikes] = self.V_reset

        if self.use_adaptive_threshold:
            # Increase threshold for spiking neurons
            self.V_th[self.spikes] += self.eta
            # Decay threshold
            self.V_th[~self.spikes] -= self.eta * (self.V_th[~self.spikes] - 1.0)

        # Clip threshold to the specified range
        self.V_th = torch.clamp(self.V_th, self.min_threshold, self.max_threshold)

        return self.spikes

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function to calculate spike probability.
        :param x: Input value.
        :return: Probability in the range [0, 1].
        """
        return 1 / (1 + np.exp(-x))
