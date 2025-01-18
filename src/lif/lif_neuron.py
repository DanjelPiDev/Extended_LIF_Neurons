import numpy as np


@DeprecationWarning
class LIFNeuron:
    """
    A Leaky Integrate-and-Fire (LIF) neuron model.

    Simulates the behavior of a single neuron that integrates input current over time,
    leaks potential at a constant rate, and fires a spike when a threshold is reached.
    """

    def __init__(self,
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
        Initialize the LIF neuron with its parameters.

        :param V_th: Threshold voltage for firing a spike.
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
        assert min_threshold > 0, "Minimum threshold must be positive."
        assert max_threshold > min_threshold, "Maximum threshold must be greater than the minimum threshold."
        assert dt > 0, "Time step (dt) must be positive."

        self.V_th = V_th
        self.V_reset = V_reset
        self.tau = tau
        self.dt = dt
        self.eta = eta
        self.V = 0.0
        self.spike = False
        self.use_adaptive_threshold = use_adaptive_threshold
        self.noise_std = noise_std
        self.stochastic = stochastic

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def step(self, I) -> bool:
        """
        Simulate one time step of the LIF neuron.

        :param I: Input current to the neuron.
        :return: Spike status (True if neuron fired, False otherwise).
        """
        self.spike = False

        noise = np.random.normal(0, self.noise_std) if self.stochastic else 0.0

        dV = (I - self.V) / self.tau
        self.V += dV * self.dt + noise * (1.0 / self.V_th)

        if self.stochastic:
            # Compute the probability of spiking based on the membrane potential
            spike_prob = self.sigmoid(self.V - self.V_th)
            self.spike = np.random.uniform(0, 1) < spike_prob
        else:
            if self.V >= self.V_th:
                self.spike = True
                self.V = self.V_reset

        if self.spike and self.use_adaptive_threshold:
            self.V_th += self.eta
        elif self.use_adaptive_threshold:
            self.V_th -= self.eta * (self.V_th - 1.0)

        # Clip threshold to the specified range
        self.V_th = np.clip(self.V_th, self.min_threshold, self.max_threshold)

        return self.spike

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function to calculate spike probability.
        :param x: Input value.
        :return: Probability in the range [0, 1].
        """
        return 1 / (1 + np.exp(-x))

    def get_potential(self) -> float:
        """
        Get the current membrane potential of the neuron.

        :return: Current membrane potential (voltage).
        """
        return self.V

    def get_threshold(self) -> float:
        """
        Get the current threshold voltage of the neuron.

        :return: Current threshold voltage.
        """
        return self.V_th
