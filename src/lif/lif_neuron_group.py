import numpy as np


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
                 max_threshold: float = 2.0):
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

        self.num_neurons = num_neurons
        self.V_th = np.full(num_neurons, V_th)
        self.V_reset = V_reset
        self.tau = tau
        self.dt = dt
        self.eta = eta
        self.V = np.zeros(num_neurons)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.use_adaptive_threshold = use_adaptive_threshold
        self.noise_std = noise_std
        self.stochastic = stochastic

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def step(self, I: np.ndarray) -> np.ndarray:
        """
        Simulate one time step for all neurons in the group.

        :param I: Input current (array of size `num_neurons`).
        :return: Array of spike statuses (True for neurons that fired, False otherwise).
        """
        assert I.shape[0] == self.num_neurons, "Input current must match the number of neurons."

        noise = np.random.normal(0, self.noise_std, size=self.num_neurons) if self.stochastic else 0.0

        dV = (I - self.V) / self.tau
        self.V += dV * self.dt + noise / self.V_th

        if self.stochastic:
            spike_prob = self.sigmoid(self.V - self.V_th)
            self.spikes = np.random.uniform(0, 1, size=self.num_neurons) < spike_prob
        else:
            self.spikes = self.V >= self.V_th

        self.V[self.spikes] = self.V_reset

        if self.use_adaptive_threshold:
            # Increase threshold for spiking neurons
            self.V_th[self.spikes] += self.eta
            # Decay threshold
            self.V_th[~self.spikes] -= self.eta * (self.V_th[~self.spikes] - 1.0)

        # Clip threshold to the specified range
        self.V_th = np.clip(self.V_th, self.min_threshold, self.max_threshold)

        return self.spikes

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function to calculate spike probability.
        :param x: Input value.
        :return: Probability in the range [0, 1].
        """
        return 1 / (1 + np.exp(-x))
