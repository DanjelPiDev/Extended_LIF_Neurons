import numpy as np


class AutoregressiveBernoulliLayer:
    """
    A layer of spiking neurons using an autoregressive Bernoulli spike sampling mechanism.

    Each neuron's spike probability is determined by a sigmoid transformation of its membrane potential,
    with past spikes influencing the membrane potential in an autoregressive manner.
    """
    def __init__(self, num_neurons, tau=20.0, dt=1.0):
        """
        Initialize the Autoregressive Bernoulli Layer.

        :param num_neurons: Number of neurons in the layer.
        :param tau: Membrane time constant, controlling the decay rate.
        :param dt: Time step for updating the membrane potential.
        """
        self.num_neurons = num_neurons
        self.tau = tau
        self.dt = dt
        self.V = np.zeros(num_neurons)  # Membrane potentials
        self.spikes = np.zeros(num_neurons, dtype=bool)  # Spike states

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function to map membrane potential to spike probability.

        :param x: Membrane potential or input value.
        :return: Sigmoid-transformed probability in the range [0, 1].
        """
        return 1 / (1 + np.exp(-x))

    def step(self, I, autoregressive_weight=0.5):
        """
        Simulate one time step for the layer with autoregressive feedback.

        :param I: Input current for each neuron (array of size num_neurons).
        :param autoregressive_weight: Weight of the influence of past spikes on the current potential.
        :return: Updated spike states for each neuron (boolean array).
        """
        dV = (I - self.V) / self.tau
        self.V += dV * self.dt + autoregressive_weight * self.spikes

        spike_prob = self.sigmoid(self.V)

        self.spikes = np.random.uniform(0, 1, size=self.num_neurons) < spike_prob

        # Reset potential for spiking neurons
        self.V[self.spikes] = 0.0
        return self.spikes