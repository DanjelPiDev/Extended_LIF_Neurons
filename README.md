# Extended LIF Neurons

### 🚀 Development Notes

> ⚡ I developed my own implementation of LIF neurons because
> existing libraries did not meet my specific requirements.
> I also implemented PyTorch-compatible layers for LIF neurons,
> enabling their integration into neural network models.

---

## Overview

This repository features an advanced implementation of LIF neurons. Building upon the classic LIF model, it incorporates
dynamic spike probability, adaptive thresholds, synaptic short-term plasticity, and neuromodulatory influences to more
accurately simulate the complex behavior of biological neurons. Key features include:

- **Membrane Potential Tracking**: Detailed monitoring of voltage dynamics over time.
- **Spike Generation**: Realistic spike output using both deterministic and stochastic mechanisms.
- **Dynamic Adaptation**: Mechanisms that adjust neuron excitability based on recent activity.
- **Neuromodulation**: Integration of external signals (e.g., reward signals) to modulate firing behavior.
- **PyTorch Integration**: Fully PyTorch-compatible layers enable seamless incorporation into larger neural network
  architectures.

This framework not only allows for comprehensive simulations and comparisons of spiking behaviors across different
neuron types but also facilitates the integration of biologically inspired dynamics into modern machine learning
workflows.

### Key Features

| Feature                   | 	Description                                                             |
|---------------------------|--------------------------------------------------------------------------|
| Adaptive Threshold        | 	Threshold increases after spikes, preventing runaway firing             |
| Stochastic Dynamics       | 	Probabilistic spiking with noise injection                              |
| Synaptic Plasticity       | 	Short-term depression and recovery of synaptic efficacy                 |
| Neuromodulation           | 	External signals (e.g., simulated dopamine) modulate excitability       |
| Dynamic Spike Probability | 	Self-limiting spiking based on recent activity (homeostatic adaptation) |

### Parameters

| Parameter	                           | Default	     | Description                                                                                                                                                                                           |
|--------------------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_neurons`	                       | `required`	  | Number of neurons in the group.                                                                                                                                                                       |
| `V_th`	                              | 1.0	         | Initial threshold voltage for all neurons.                                                                                                                                                            |
| ``V_reset``	                         | 0.0	         | Voltage to which the membrane potential is reset after a spike.                                                                                                                                       |
| ``tau``	                             | 20.0	        | Membrane time constant that controls the decay rate of the membrane potential.                                                                                                                        |
| ``dt``	                              | 1.0	         | Time step used for updating the membrane potential.                                                                                                                                                   |
| ``eta``	                             | 0.1	         | Adaptation rate for the threshold voltage (used in adaptive threshold updating).                                                                                                                      |
| ``use_adaptive_threshold``	          | True	        | If set to true, the threshold will adapt based on recent spiking activity.                                                                                                                            |
| ``noise_std``	                       | 0.1	         | Standard deviation of the Gaussian noise added to the membrane potential (used if stochastic firing is enabled).                                                                                      |
| ``stochastic``	                      | True	        | Enables stochastic firing. In stochastic mode, spikes are sampled based on a computed spike probability.                                                                                              |
| ``min_threshold``	                   | 0.5	         | Minimum allowable value for the threshold voltage.                                                                                                                                                    |
| ``max_threshold``	                   | 2.0	         | Maximum allowable value for the threshold voltage.                                                                                                                                                    |
| ``batch_size``	                      | 1	           | Batch size for input data processing.                                                                                                                                                                 |
| ``device``	                          | `cpu`	       | Device to run the simulation on (either `cpu` or `cuda`).                                                                                                                                             |
| ``surrogate_gradient_function``	     | `heaviside`	 | Name of the surrogate gradient function for backpropagation. Options include `heaviside`, `fast_sigmoid`, `gaussian`, and `arctan`.                                                                   |
| ``alpha``	                           | 1.0	         | Parameter for the surrogate gradient function.                                                                                                                                                        |
| ``allow_dynamic_spike_probability``	 | True	        | If true, enables dynamic spike probability computation that uses previous spike history (acts as a self-locking mechanism).                                                                           |
| ``base_alpha``	                      | 2.0	         | Base alpha value for the dynamic sigmoid function used in dynamic spike probability.                                                                                                                  |
| ``tau_adapt``	                       | 20.0	        | Time constant for the adaptation in the dynamic spike probability mechanism.                                                                                                                          |
| ``adaptation_decay``	                | 0.9	         | Decay rate for the adaptation current (how quickly the adaptation effect decays over time).                                                                                                           |
| ``spike_increase``	                  | 0.5	         | Increment added to the adaptation current each time a neuron spikes.                                                                                                                                  |
| ``depression_rate``	                 | 0.1	         | Rate at which synaptic efficiency is reduced (depressed) when a neuron spikes.                                                                                                                        |
| ``recovery_rate``	                   | 0.05	        | Rate at which synaptic efficiency recovers toward its baseline (typically 1) after being depressed.                                                                                                   |
| ``neuromod_transform``	              | None	        | A function or module that transforms an external modulation tensor (e.g. reward or error signal) into modulation factors (typically in [0, 1]). If None, a default sigmoid transformation is applied. |

### How It Works

#### Input Processing

- **I (Input Current):** The neuron receives a raw current, which is the primary drive.
- **External Modulation:** Optionally, an external signal (e.g., representing a reward or dopamine level) is provided.
  This signal is transformed (using a user-defined `neuromod_transform` or a default sigmoid) to produce a modulation
  factor that influences neuronal excitability.

#### Effective Input Calculation

The raw input is modified by internal dynamic factors:

- **Synaptic Efficiency:** Scales down the input if previous spikes have occurred (modeling synaptic depression).
- **Neuromodulator:** Adds a context-dependent boost (or reduction) to excitability.
- **Adaptation Current:** Subtracts from the input to account for refractory periods after spiking.

The effective input is computed as:

```math
I_{effective} = I * synaptic_{efficiency} + neuromodulator - adaptation_{current}
```

#### Membrane Potential Update

- The neuron's membrane potential ``V`` is updated based on the effective input, time constant tau, and any noise (if
  stochastic mode is enabled).
- When ``V`` exceeds the adaptive threshold ``V_th``, a spike is generated.
- In deterministic (non-stochastic) mode, a hard threshold is applied; in stochastic mode, a probability is computed (
  either static or dynamic) and a spike is sampled.

#### Spike Generation & Reset

- If a spike occurs, ``V`` is reset to ``V_reset``.
- The model then updates internal states:
- Adaptation Current: Increases for spiking neurons and decays over time.
- Synaptic Efficiency: Depresses upon spiking and recovers gradually.
- Adaptive Threshold (``V_th``): Adjusts (increases upon spiking and decays when inactive).

#### Results

<div align="center">
    <img src="./src/Images/neuron_activity_2.png" width="1000">
    <p>Figure 1: Example of LIF Neuron Activity (Different Modes)</p>
</div>

---

### How to Use

#### Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

or just use the following command inside your project:
```bash
pip install git+https://github.com/NullPointerExcy/Extended_LIF_Neurons
```

#### PyTorch Integration

This repository also includes PyTorch-compatible layers for
LIF neurons. Below is an example of using the
LIFLayer class with PyTorch:

```python
import torch
from layers.torch_layers import LIFLayer

# Initialize neuron group
lif = LIFLayer(
    num_neurons=128,
    V_th=1.5,
    tau=30.0,
    stochastic=True,
    noise_std=0.05,
    use_adaptive_threshold=True
).to("cuda")

# Simulate 100 timesteps
input_current = torch.randn(100, 1, 128).to("cuda")  # (timesteps, batch, neurons)
spikes, voltages = lif(input_current)

# Visualize
import matplotlib.pyplot as plt

plt.plot(voltages[:, 0, 0].cpu().numpy())  # First neuron's voltage
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential")
plt.show()
```

### Expected Behavior

| Mode                | 	Characteristics                                                                   |
|---------------------|------------------------------------------------------------------------------------|
| Deterministic	      | Regular spiking when input > threshold; reset to V_reset after spike               |
| Stochastic          | 	Irregular spiking, rate depends on noise_std and V-V_th difference                |
| Adaptive Threshold  | 	Spike rate decreases over time as threshold (V_th) rises                          |
| Dynamic Probability | 	Initial high spiking followed by self-stabilization due to adaptation             |
| Neuromodulation     | 	External signals boost/reduce excitability (e.g., increased spike rate on reward) |

---

### Biological Interpretation

This implementation captures three key biological phenomena:

- Leaky Integration: Membrane potential decays over time (tau)

- Refractoriness: Adaptation current reduces excitability post-spike

- Homeostasis: Dynamic spike probability prevents hyperactivity

---

### Performance Analysis

> Tested on NVIDIA RTX 4080 Super (16GB VRAM) with PyTorch 2.5.1+cu118

#### Memory Scaling

##### Complexity: 𝓞(n) (linear)

- 128 neurons → 0.2 MB
- 1 million neurons → 490 MB
- 16 million neurons → 15.7 GB

Memory usage grows linearly with neuron count, following the 𝓞(n) complexity:

```math
Memory (MB) ≈ 0.03 * num_neurons
```

#### Computation Time Scaling

##### Complexity: 𝓞(n) (linear with parallelization factor)

- 128 neurons - 8 million neurons: ~0.05–0.42s per 100 timesteps
- 16 million neurons: 2.27s per 100 timesteps

Despite apparent log-scale plot curvature, actual complexity is linear.

<div align="center">
    <img src="./src/Images/performance_scaling_v2.png" width="1000">
    <p>Figure 2: Plot of memory and time scaling (x and y are log-scaled), with up to 16 million neurons.</p>
</div>

---

### License

This project is licensed under the MIT License. Feel free to use and modify it for your research or personal projects.

