
class LIFNeuron:
    """
    A Leaky Integrate-and-Fire (LIF) neuron model.

    Simulates the behavior of a single neuron that integrates input current over time,
    leaks potential at a constant rate, and fires a spike when a threshold is reached.
    """
    def __init__(self, V_th: float = 1.0, V_reset: float = 0.0, tau: float = 20.0, dt: float = 1.0):
        """
        Initialize the LIF neuron with its parameters.

        :param V_th: Threshold voltage for firing a spike.
        :param V_reset: Reset voltage after a spike.
        :param tau: Membrane time constant, controlling decay rate.
        :param dt: Time step for updating the membrane potential.
        """
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau = tau
        self.dt = dt
        self.V = 0.0
        self.spike = False

    def step(self, I) -> bool:
        """
        Simulate one time step of the LIF neuron.

        :param I: Input current to the neuron.
        :return: Spike status (True if neuron fired, False otherwise).
        """
        self.spike = False
        dV = (I - self.V) / self.tau
        self.V += dV * self.dt

        if self.V >= self.V_th:
            self.spike = True
            self.V = self.V_reset

        return self.spike

    def get_potential(self) -> float:
        """
        Get the current membrane potential of the neuron.

        :return: Current membrane potential (voltage).
        """
        return self.V
