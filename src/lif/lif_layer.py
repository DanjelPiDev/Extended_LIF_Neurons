import numpy as np


class LIFLayer:
    def __init__(self, num_neurons: int, V_th: float = 1.0, V_reset: float = 0.0, tau: float = 20.0, dt: float = 1.0):
        """
        Initialize a layer of LIF neurons.

        :param num_neurons: Number of neurons in the layer.
        :param V_th: Threshold voltage.
        :param V_reset: Reset voltage.
        :param tau: Membrane time constant.
        :param dt: Time step.
        """
        self.num_neurons = num_neurons
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau = tau
        self.dt = dt
        self.V = np.zeros(num_neurons)  # Membrane potentials
        self.spike = np.zeros(num_neurons, dtype=bool)  # Spike status

    def step(self, I):
        """
        Simulate one time step of the LIF layer.

        :param I: Input current (array of size num_neurons).
        :return: Spike status for each neuron.
        """

        # Reset spike flags
        self.spike[:] = False

        dV = (I - self.V) / self.tau
        self.V += dV * self.dt

        spiking_neurons = self.V >= self.V_th
        self.spike[spiking_neurons] = True
        self.V[spiking_neurons] = self.V_reset

        return self.spike
