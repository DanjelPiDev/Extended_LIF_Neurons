import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from layers.torch_layers import LIFLayer


def generate_input(timesteps, batch_size, num_neurons, noise_level):
    base_signal = torch.cat([
        torch.zeros(50, batch_size, num_neurons),
        torch.linspace(0, 2, 100).repeat(batch_size, num_neurons, 1).permute(2, 0, 1),
        torch.zeros(50, batch_size, num_neurons)
    ])
    noise = torch.randn(timesteps, batch_size, num_neurons) * noise_level
    return base_signal + noise


def run_simulation(config, input_current, ext_mod):
    lif = LIFLayer(num_neurons=config["num_neurons"], **config["lif_args"])
    with torch.no_grad():
        spikes, voltages = lif(input_current, ext_mod)
    return spikes.squeeze().cpu().numpy(), voltages.squeeze().cpu().numpy(), input_current.squeeze().cpu().numpy()


def plot_results(spikes, voltages, input_current, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.plot(voltages, label="Membrane Potential", color='tab:blue')
    ax.plot(input_current, label="Input Current", linestyle='--', alpha=0.5, color='tab:orange')
    spike_times = np.where(spikes)[0]
    ax.scatter(spike_times, np.ones_like(spike_times)*1.5, marker='x', color='tab:red', label='Spikes')
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


st.title("LIF Neuron Simulator")
st.markdown("Visualize different LIF neuron modes with simulated input.")

mode = st.selectbox("Select Neuron Mode", [
    "Deterministic (Basic LIF)",
    "Stochastic (Constant Threshold)",
    "Adaptive Threshold",
    "Dynamic Spike Probability",
    "With Neuromodulation"
])

noise_level = st.slider("Input Noise Level", 0.0, 1.0, 0.2)
num_neurons = st.slider("Number of Neurons", 1, 10, 3)
timesteps = 200
batch_size = 1

device = "cpu"
input_current = generate_input(timesteps, batch_size, num_neurons, noise_level)
external_mod = torch.sin(torch.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1, 1)

configs = {
    "Deterministic (Basic LIF)": {
        "num_neurons": num_neurons,
        "lif_args": {
            "stochastic": False,
            "use_adaptive_threshold": False,
            "V_th": 1.0,
            "tau": 30.0,
            "device": device
        }
    },
    "Stochastic (Constant Threshold)": {
        "num_neurons": num_neurons,
        "lif_args": {
            "stochastic": True,
            "noise_std": 0.05,
            "V_th": 1.5,
            "tau": 30.0,
            "use_adaptive_threshold": False,
            "device": device
        }
    },
    "Adaptive Threshold": {
        "num_neurons": num_neurons,
        "lif_args": {
            "stochastic": False,
            "use_adaptive_threshold": True,
            "V_th": 1.0,
            "tau": 30.0,
            "eta": 0.15,
            "min_threshold": 0.8,
            "max_threshold": 2.0,
            "device": device
        }
    },
    "Dynamic Spike Probability": {
        "num_neurons": num_neurons,
        "lif_args": {
            "stochastic": True,
            "allow_dynamic_spike_probability": True,
            "V_th": 1.0,
            "tau": 30.0,
            "base_alpha": 5.0,
            "tau_adapt": 10.0,
            "device": device
        }
    },
    "With Neuromodulation": {
        "num_neurons": num_neurons,
        "lif_args": {
            "stochastic": False,
            "V_th": 1.0,
            "tau": 30.0,
            "neuromod_transform": lambda x: torch.sigmoid(2*x),
            "device": device
        }
    }
}

config = configs[mode]
mod_signal = external_mod if "Neuromod" in mode else None
spikes, voltages, inputs = run_simulation(config, input_current, mod_signal)
plot_results(spikes, voltages, inputs, mode)