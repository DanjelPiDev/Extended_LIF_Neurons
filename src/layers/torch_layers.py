from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from lif.lif_neuron_group import LIFNeuronGroup
from lif.sg.spike_function import SpikeFunction


class LIFLayer(nn.Module):
    """
    A PyTorch wrapper for the LIFNeuronGroup class to integrate with PyTorch layers.
    """

    def __init__(self,
                 num_neurons,
                 V_th=1.0,
                 V_reset=0.0,
                 tau=20.0,
                 dt=1.0,
                 eta=0.1,
                 use_adaptive_threshold=True,
                 noise_std=0.1,
                 stochastic=True,
                 min_threshold=0.5,
                 max_threshold=2.0,
                 batch_size=1,
                 device=torch.device("cpu"),
                 spike_coding=None,
                 surrogate_gradient_function="heaviside",
                 alpha=1.0,
                 allow_dynamic_spike_probability=False,
                 base_alpha=2.0,
                 tau_adapt=20.0,
                 adaptation_decay: float = 0.9,
                 spike_increase: float = 0.5,
                 depression_rate: float = 0.1,
                 recovery_rate: float = 0.05,
                 neuromod_transform=None):
        """
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
        :param spike_coding: (Optional) Spike coding scheme.
        :param surrogate_gradient_function: Surrogate gradient function for backpropagation.
        :param alpha: Parameter for the surrogate gradient function.
        :param allow_dynamic_spike_probability: Whether to allow dynamic spike probability (uses internal state).
        :param base_alpha: Base alpha value for the dynamic sigmoid function.
        :param tau_adapt: Time constant for adaptation (used in dynamic spike probability).
        :param adaptation_decay: Decay rate for the adaptation current.
        :param spike_increase: Increment for the adaptation current on spike.
        :param depression_rate: Rate of synaptic depression on spike.
        :param recovery_rate: Rate of synaptic recovery after spike.
        :param neuromod_transform: A function or module that transforms an external modulation tensor
                                   (e.g., reward/error signal) into modulation factors.
                                   If None, a default sigmoid transformation will be applied.
        """
        super(LIFLayer, self).__init__()

        self.lif_group = LIFNeuronGroup(
            num_neurons=num_neurons,
            V_th=V_th,
            V_reset=V_reset,
            tau=tau,
            dt=dt,
            eta=eta,
            use_adaptive_threshold=use_adaptive_threshold,
            noise_std=noise_std,
            stochastic=stochastic,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            batch_size=batch_size,
            device=device,
            surrogate_gradient_function=surrogate_gradient_function,
            alpha=alpha,
            allow_dynamic_spike_probability=allow_dynamic_spike_probability,
            base_alpha=base_alpha,
            tau_adapt=tau_adapt,
            adaptation_decay=adaptation_decay,
            spike_increase=spike_increase,
            depression_rate=depression_rate,
            recovery_rate=recovery_rate,
            neuromod_transform=neuromod_transform
        )
        self.spike_coding = spike_coding

    def _apply_neuromod_transform(self, external_modulation):
        if self.lif_group.neuromod_transform is None:
            return torch.sigmoid(external_modulation)
        else:
            return self.lif_group.neuromod_transform(external_modulation)

    def _lif_step(
            self, I, V, V_th, adaptation_current, synaptic_efficiency,
            neuromodulator, external_modulation, prev_spikes, dynamic_adaptation
    ):
        assert I.ndim == 2, f"I should be (batch, neurons), got {I.shape}"
        if external_modulation is not None:
            assert external_modulation.ndim == 2, f"External mod should be (batch, neurons), got {external_modulation.shape}"
            neuromodulator = self._apply_neuromod_transform(external_modulation)

        # Compute effective current with noise
        noise = torch.randn_like(I) * self.lif_group.noise_std if self.lif_group.stochastic else 0
        I_effective = I * synaptic_efficiency + neuromodulator - adaptation_current
        dV = (I_effective - V) / self.lif_group.tau
        V = V + dV * self.lif_group.dt + noise

        # Compute spike probabilities (BEFORE determining spikes)
        if self.lif_group.stochastic:
            if self.lif_group.allow_dynamic_spike_probability:
                # Use PREVIOUS spikes for adaptation
                spike_prob, dynamic_adaptation = self.lif_group.dynamic_spike_probability(
                    V - V_th, prev_spikes
                )
            else:
                spike_prob = torch.sigmoid(V - V_th)
            spikes = torch.rand_like(V) < spike_prob
        else:
            spikes = SpikeFunction.apply(V - V_th, self.lif_group.surrogate_gradient_function, self.lif_group.alpha)

        # Reset membrane potential
        V = torch.where(spikes.bool(), self.lif_group.V_reset, V)

        # Update threshold, adaptation, etc. (same as before)
        if self.lif_group.use_adaptive_threshold:
            V_th = torch.where(spikes.bool(), V_th + self.lif_group.eta, V_th - self.lif_group.eta * (V_th - 1.0))
        V_th = torch.clamp(V_th, self.lif_group.min_threshold, self.lif_group.max_threshold)

        # Update adaptation and synaptic efficiency
        adaptation_current = adaptation_current * self.lif_group.adaptation_decay + self.lif_group.spike_increase * spikes.float()
        synaptic_efficiency = (
                synaptic_efficiency * (1 - self.lif_group.depression_rate * spikes.float()) +
                self.lif_group.recovery_rate * (1 - synaptic_efficiency)
        )

        return spikes, V, V_th, adaptation_current, synaptic_efficiency

    def forward(self, input_data: torch.Tensor, external_modulation: torch.Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Forward pass for the LIFNeuronGroup in PyTorch.

        :param input_data: Input tensor of shape (batch_size, num_neurons) or
                           (timesteps, batch_size, num_neurons).
        :param external_modulation: External modulation tensor (e.g., reward signal)
                                    with shape (batch_size, num_neurons) or broadcastable shape.
        :return: Spike tensor (binary) of shape (batch_size, num_neurons) or
                 (timesteps, batch_size, num_neurons).
        """
        timesteps, batch_size, num_neurons = input_data.shape

        # Preallocate spike tensor for all timesteps
        output_spikes = torch.zeros(
            timesteps, batch_size, num_neurons,
            device=self.lif_group.device, dtype=torch.bool
        )

        voltages = torch.zeros(
            timesteps, batch_size, num_neurons,
            device=self.lif_group.device
        )

        V = self.lif_group.V.clone()
        V_th = self.lif_group.V_th.clone()
        adaptation_current = self.lif_group.adaptation_current.clone()
        synaptic_efficiency = self.lif_group.synaptic_efficiency.clone()
        neuromodulator = self.lif_group.neuromodulator.clone()
        dynamic_adaptation = (
            self.lif_group.dynamic_spike_probability.adaptation.clone()
            if self.lif_group.allow_dynamic_spike_probability
            else None
        )

        prev_spikes = torch.zeros_like(V, dtype=torch.bool)

        # Vectorized simulation over time
        for t in range(timesteps):
            current_external_mod = (
                external_modulation[t]
                if (external_modulation is not None and external_modulation.ndim == 3)
                else external_modulation
            )

            spikes, V, V_th, adaptation_current, synaptic_efficiency = self._lif_step(
                I=input_data[t],
                V=V,
                V_th=V_th,
                adaptation_current=adaptation_current,
                synaptic_efficiency=synaptic_efficiency,
                neuromodulator=neuromodulator,
                external_modulation=current_external_mod,
                prev_spikes=prev_spikes,
                dynamic_adaptation=dynamic_adaptation
            )
            output_spikes[t] = spikes
            voltages[t] = V
            prev_spikes = spikes

        # Update the neuron group's internal states
        self.lif_group.V = V
        self.lif_group.V_th = V_th
        self.lif_group.adaptation_current = adaptation_current
        self.lif_group.synaptic_efficiency = synaptic_efficiency

        return output_spikes, voltages
