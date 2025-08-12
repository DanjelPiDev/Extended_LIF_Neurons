import math

import pennylane as qml
import torch
import torch.nn as nn

from neurons.sg.spike_function import SpikeFunction
from neurons.probability.dynamic_spike_probability import DynamicSpikeProbability
from neurons.sg.q_spike import q_spike_prob, bernoulli_spike


def get_surrogate_fn(name, alpha):
    if name == "heaviside":
        return lambda x: SpikeFunction.apply(x, "heaviside", alpha)
    elif name == "fast_sigmoid":
        return lambda x: SpikeFunction.apply(x, "fast_sigmoid", alpha)
    elif name == "gaussian":
        return lambda x: SpikeFunction.apply(x, "gaussian", alpha)
    elif name == "arctan":
        return lambda x: SpikeFunction.apply(x, "arctan", alpha)
    else:
        raise ValueError(f"Unknown surrogate gradient function: {name}")


class QLIF(nn.Module):
    """
    A vectorized QLIF neuron model for multiple neurons.
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
                 allow_dynamic_spike_probability: bool = True,
                 base_alpha: float = 2.0,
                 tau_adapt: float = 20.0,
                 adaptation_decay: float = 0.9,
                 spike_increase: float = 0.5,
                 depression_rate: float = 0.1,
                 recovery_rate: float = 0.05,
                 neuromod_transform=None,
                 neuromod_mode: str = "gain",
                 neuromod_strength: float = 1.0,
                 learnable_threshold: bool = True,
                 learnable_tau: bool = False,
                 learnable_eta: bool = False,
                 learnable_qscale: bool = True,
                 learnable_qbias: bool = True,
                 quantum_mode: bool = True,
                 quantum_threshold: float = 0.7,
                 quantum_leak: float = 0.1,
                 quantum_wire: int = 4,
                 ):
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
        :param adaptation_decay: Decay rate for the adaptation current.
        :param spike_increase: Increment for the adaptation current on spike.
        :param depression_rate: Rate of synaptic depression on spike.
        :param recovery_rate: Rate of synaptic recovery after spike.
        :param neuromod_transform: A function or module that takes an external modulation tensor (e.g. reward/error signal)
            and returns a transformed tensor (e.g. modulation factors in [0,1]).
            If None, a default sigmoid transformation will be applied.
        :param neuromod_mode: Mode of neuromodulation, can be "gain", "threshold", "prob_slope", or "off".
            - "gain": Modulates the input current by a gain factor.
            - "threshold": Modulates the threshold voltage.
            - "prob_slope": Modulates the spike probability slope.
            - "off": No neuromodulation.
        :param neuromod_strength: Strength of the neuromodulation, a scalar value
            that scales the modulation effect.
        :param learnable_threshold: Whether the threshold voltage should be learnable.
        :param learnable_tau: Whether the membrane time constant should be learnable.
        :param learnable_eta: Whether the adaptation rate should be learnable.
        :param learnable_qscale: Whether the quantum scale factor should be learnable.
        :param learnable_qbias: Whether the quantum bias should be learnable.
        :param quantum_mode: If True, the neuron group operates in quantum mode, which may affect how spikes are processed.
        :param quantum_threshold: Threshold for quantum spike firing.
        :param quantum_leak: Leakage factor for quantum spikes.
        :param quantum_wire: Number of quantum wires used in the quantum mode.
        """
        assert num_neurons > 0, "Number of neurons must be positive."

        if stochastic:
            assert noise_std > 0, "Noise standard deviation must be positive in stochastic mode."

        assert tau > 0.0, "Membrane time constant must be positive."
        assert min_threshold > 0, "Minimum threshold must be positive."
        assert max_threshold > min_threshold, "Maximum threshold must be greater than the minimum threshold."
        assert dt > 0, "Time step (dt) must be positive."
        assert batch_size > 0, "Batch size must be positive."
        assert device in ["cpu", "cuda"], "Device must be either 'torch.device('cpu')' or 'torch.device('cuda')'."
        assert surrogate_gradient_function in ["heaviside", "fast_sigmoid", "gaussian", "arctan"], \
            "Surrogate gradient function must be one of 'heaviside', 'fast_sigmoid', 'gaussian', 'arctan'."
        assert alpha > 0, "Alpha must be positive."
        assert adaptation_decay >= 0, "adaptation_decay must be non-negative."
        assert spike_increase >= 0, "spike_increase must be non-negative."
        assert 0 <= depression_rate <= 1, "depression_rate must be in [0, 1]."
        assert recovery_rate >= 0, "recovery_rate must be non-negative."
        assert quantum_threshold > 0, "quantum_threshold must be positive."
        assert quantum_leak >= 0, "quantum_leak must be non-negative."
        assert quantum_wire > 0, "quantum_wire must be a positive integer."

        super(QLIF, self).__init__()
        self.num_neurons = num_neurons

        shape = (1, num_neurons)
        self.V_th = (nn.Parameter(torch.full((num_neurons,), V_th))
                     if learnable_threshold else torch.full((num_neurons,), V_th))

        self.V_reset = V_reset
        self.dt = dt
        self.noise_std = noise_std
        self.stochastic = stochastic
        self.use_adaptive_threshold = use_adaptive_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_decay = adaptation_decay
        self.spike_increase = spike_increase
        self.depression_rate = depression_rate
        self.recovery_rate = recovery_rate

        self.allow_dynamic_spike_probability = allow_dynamic_spike_probability
        self.dynamic_spike_probability = DynamicSpikeProbability(
            base_alpha=base_alpha,
            tau_adapt=tau_adapt,
        ) if allow_dynamic_spike_probability else None

        self.neuromod_transform = neuromod_transform
        self.neuromod_mode = neuromod_mode
        self.neuromod_strength = float(neuromod_strength)
        self.surrogate_fn = get_surrogate_fn(surrogate_gradient_function, alpha)

        self.learnable_threshold = learnable_threshold
        self.learnable_tau = learnable_tau
        self.learnable_eta = learnable_eta
        self.quantum_mode = quantum_mode
        self.quantum_threshold = quantum_threshold
        self.quantum_leak = quantum_leak
        self.quantum_wire = quantum_wire

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(tau)))

        if learnable_eta:
            self.eta = nn.Parameter(torch.tensor(float(eta)))
        else:
            self.register_buffer("eta", torch.tensor(float(eta)))

        if learnable_qscale:
            self.q_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("q_scale", torch.tensor(1.0))

        if learnable_qbias:
            self.q_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("q_bias", torch.tensor(0.0))

        self.register_buffer("_qnode_ready", torch.tensor(0, dtype=torch.int8))

        self.register_buffer("V", torch.zeros(1, num_neurons))
        self.register_buffer("spikes", torch.zeros(1, num_neurons, dtype=torch.bool))
        self.register_buffer("adaptation_current", torch.zeros(1, num_neurons))
        self.register_buffer("synaptic_efficiency", torch.ones(1, num_neurons))
        self.register_buffer("neuromodulator", torch.ones(1, num_neurons))
        self.register_buffer("spike_values", torch.zeros(1, num_neurons))

        self.init_quantum(p0=0.02, slope0=0.10)

    def resize(self, batch_size: int, device: torch.device | None = None):
        dev = device if device is not None else self.V.device
        shape = (batch_size, self.num_neurons)
        self.V = torch.zeros(shape, device=dev)
        self.spikes = torch.zeros(shape, dtype=torch.bool, device=dev)
        self.adaptation_current = torch.zeros(shape, device=dev)
        self.synaptic_efficiency = torch.ones(shape, device=dev)
        self.neuromodulator = torch.ones(shape, device=dev)
        self.spike_values = torch.zeros(shape, device=dev)
        if self.dynamic_spike_probability:
            self.dynamic_spike_probability.resize(batch_size, self.num_neurons, dev)

    def reset(self):
        self.V.zero_()
        self.spikes.zero_()
        self.spike_values.zero_()
        self.adaptation_current.zero_()
        self.synaptic_efficiency.fill_(1.0)
        self.neuromodulator.fill_(1.0)
        if self.dynamic_spike_probability:
            self.dynamic_spike_probability.reset(self.V.shape[0],
                                                 self.num_neurons,
                                                 self.V.device)

    @staticmethod
    def _expand_like(x, ref):
        # x = (B,N)
        if x is None:
            return None
        if x.ndim == 0:  # scalar
            return x.view(1, 1).expand_as(ref)
        if x.ndim == 1:  # (N,)
            return x.view(1, -1).expand_as(ref)
        if x.shape != ref.shape:
            return x.expand_as(ref)
        return x

    def init_quantum(self, p0: float = 0.02, slope0: float = 0.10):
        # p0 in (0,1), slope0 ~ dp/d_delta bei delta=0
        eps = 1e-6
        # β = arccos(1 - 2 p0)
        beta = torch.acos(torch.clamp(1.0 - 2.0 * torch.tensor(p0), -1.0 + eps, 1.0 - eps))
        # α = 2*slope0 / sin(beta)
        alpha = (2.0 * torch.tensor(slope0)) / torch.clamp(torch.sin(beta), min=eps)

        with torch.no_grad():
            self.q_bias.fill_(float(beta + self.quantum_leak))
            self.q_scale.fill_(float(alpha))

    def _build_qnode_if_needed(self):
        if bool(self._qnode_ready.item()):
            return
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface='torch', diff_method="parameter-shift")
        def qnode(theta):
            # shape: M,
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        self._qnode = qnode
        self._qnode_ready.fill_(1)

    def _quantum_prob(self, delta: torch.Tensor, mod_for_prob=None):
        """
        delta: (B,N) = V - V_th_eff
        returns: p in [0,1], shape (B,N)
        """
        alpha = self.q_scale
        beta = self.q_bias - self.quantum_leak

        if self.neuromod_mode == "prob_slope" and (mod_for_prob is not None):
            alpha = alpha * (1.0 + self.neuromod_strength * mod_for_prob)

        p = q_spike_prob(delta, alpha=alpha, beta=beta)
        return p.clamp_(0.0, 1.0)

    def forward(self, I: torch.Tensor, external_modulation: torch.Tensor = None) -> torch.Tensor:
        """
        Simulate one time step for all neurons in the group.

        :param I: Tensor of input currents with shape (batch_size, num_neurons).
        :param external_modulation: Tensor of external neuromodulatory signals with shape
                                    (batch_size, num_neurons) or broadcastable shape.
                                    For example, this could encode a reward signal for dopamine modulation.
        :return: Spike tensor (binary) of shape (batch_size, num_neurons).
        """
        if I.shape != self.V.shape or I.device != self.V.device:
            self.resize(I.shape[0], device=I.device)

        if external_modulation is not None:
            mod = (self.neuromod_transform(external_modulation)
                   if self.neuromod_transform else torch.sigmoid(external_modulation))
            self.neuromodulator = mod.to(I.device)

        if self.stochastic:
            noise = torch.randn_like(I).mul_(self.noise_std)
        else:
            noise = None

        B, N = I.shape
        m = self._expand_like(self.neuromodulator.to(I.device), I)  # (B,N)
        V_th_eff = self.V_th.to(I.device).view(1, -1).expand(B, -1)

        # Neuromodulations-Modis
        match self.neuromod_mode:
            case "gain":
                # Gain: (1 + s*m) * I
                gain = 1.0 + self.neuromod_strength * m
                I_eff = gain * I * self.synaptic_efficiency - self.adaptation_current
                mod_for_prob = None
            case "threshold":
                # Threshold-Shift: V_th_eff += s*m
                V_th_eff = V_th_eff + self.neuromod_strength * m
                I_eff = I * self.synaptic_efficiency - self.adaptation_current
                mod_for_prob = None
            case "prob_slope":
                I_eff = I * self.synaptic_efficiency - self.adaptation_current
                mod_for_prob = m
            case "off" | None:
                I_eff = I * self.synaptic_efficiency - self.adaptation_current
                mod_for_prob = None
            case _:
                raise ValueError(f"Unknown neuromodulation mode: {self.neuromod_mode}")

        dV = (I_eff - self.V) / (
            self.tau if isinstance(self.tau, torch.Tensor) else torch.tensor(self.tau, device=I.device))
        self.V = self.V + dV * self.dt + (noise if noise is not None else 0.0)

        if self.quantum_mode:
            delta = self.V - V_th_eff
            p = self._quantum_prob(delta, mod_for_prob=mod_for_prob)
            if torch.isnan(p).any():
                raise RuntimeError("NaN in spike probabilities")

            if self.training:
                u = torch.rand_like(p)
                y_hard = (u < p).float()
                y = p + (y_hard - p).detach()  # STE
                self.spike_values = y
                self.spikes = y_hard.bool()
            else:
                self.spikes = (torch.rand_like(p) < p)
                self.spike_values = self.spikes.float()

            if self.allow_dynamic_spike_probability and self.dynamic_spike_probability is not None:
                _p, _ = self.dynamic_spike_probability(
                    delta,
                    prev_spike_float=self.spike_values,
                    mod=mod_for_prob,
                    mod_strength=self.neuromod_strength,
                    mod_mode=("prob_slope" if self.neuromod_mode == "prob_slope" else "off")
                )
        else:
            delta = self.V - V_th_eff

            if self.stochastic:
                if self.allow_dynamic_spike_probability:
                    p, _ = self.dynamic_spike_probability(
                        delta,
                        prev_spike_float=self.spike_values,
                        mod=mod_for_prob,
                        mod_strength=self.neuromod_strength,
                        mod_mode=("prob_slope" if self.neuromod_mode == "prob_slope" else "none")
                    )
                else:
                    p = torch.sigmoid(delta)

                if self.training:
                    y_hard = bernoulli_spike(p, training=True)
                    y = p + (y_hard - p).detach()
                    self.spike_values = y
                    self.spikes = y_hard.bool()
                else:
                    y_hard = bernoulli_spike(p, training=False)
                    self.spikes = y_hard.bool()
                    self.spike_values = y_hard
            else:
                y = self.surrogate_fn(delta)  # float, 0/1 forward with Surrogate-Gradient
                self.spike_values = y
                # Bool for the reset
                self.spikes = (delta >= 0)

        self.V.masked_fill_(self.spikes, self.V_reset)

        s_float = self.spike_values
        self.adaptation_current = self.adaptation_current * self.adaptation_decay + self.spike_increase * s_float
        self.synaptic_efficiency = (
                self.synaptic_efficiency * (1 - self.depression_rate * s_float) +
                self.recovery_rate * (1 - self.synaptic_efficiency)
        )

        if self.use_adaptive_threshold:
            if isinstance(self.V_th, nn.Parameter):
                with torch.no_grad():
                    self.V_th.clamp_(self.min_threshold, self.max_threshold)
            else:
                self.V_th.clamp_(self.min_threshold, self.max_threshold)

        return self.spikes

    def get_quantum_spikes(self, V, V_th, leak, threshold, dev=None):
        """
        Proof-of-concept for quantum LIF neuron group.

        V, V_th, leak, threshold = torch.tensor, shape (batch, num_neurons)
        :returns spikes (torch.BoolTensor), shape (batch, num_neurons)
        """
        batch_size, num_neurons = V.shape
        spikes = torch.zeros_like(V, dtype=torch.bool)

        if dev is None:
            dev = qml.device("default.qubit", wires=self.quantum_wire)

        for b in range(batch_size):
            for n in range(num_neurons):
                mem = V[b, n].item()
                vth = V_th[b, n].item() if isinstance(V_th, torch.Tensor) else V_th

                @qml.qnode(dev, interface='torch')
                def qc():
                    qml.RY(mem, wires=0)
                    qml.RY(-leak, wires=0)
                    return qml.expval(qml.PauliZ(0))

                z = qc()
                fire = float(z) < math.cos(threshold)
                spikes[b, n] = fire
        return spikes
