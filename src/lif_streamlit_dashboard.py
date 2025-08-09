import streamlit as st
import torch
print(torch.__version__)
import numpy as np
import matplotlib.pyplot as plt
from layers.torch_layers import LIFLayer

# --------------------------
# helpers
# --------------------------
def generate_input(timesteps, batch_size, num_neurons, noise_level):
    pad_left  = timesteps // 4
    ramp_len  = timesteps // 2
    pad_right = timesteps - pad_left - ramp_len

    ramp = torch.linspace(0, 2, ramp_len)
    ramp = ramp.repeat(batch_size, num_neurons, 1).permute(2, 0, 1)

    base_signal = torch.cat([
        torch.zeros(pad_left,  batch_size, num_neurons),
        ramp,
        torch.zeros(pad_right, batch_size, num_neurons)
    ], dim=0)

    noise = torch.randn(timesteps, batch_size, num_neurons) * noise_level
    return base_signal + noise


def run_simulation(config, input_current, ext_mod, want_extras=False):
    lif = LIFLayer(num_neurons=config["num_neurons"], **config["lif_args"])
    lif.eval()
    with torch.no_grad():
        if want_extras:
            spikes, voltages, extras = lif(input_current, ext_mod, return_extras=True)
            extras_np = {k: v.squeeze().cpu().numpy() for k, v in extras.items()}
            return (spikes.squeeze().cpu().numpy(),
                    voltages.squeeze().cpu().numpy(),
                    input_current.squeeze().cpu().numpy(),
                    extras_np)
        else:
            spikes, voltages = lif(input_current, ext_mod)
            return (spikes.squeeze().cpu().numpy(),
                    voltages.squeeze().cpu().numpy(),
                    input_current.squeeze().cpu().numpy(),
                    None)


def plot_results(spikes, voltages, input_current, title, use_legend, use_quantum,
                 extras=None, show_adaptation=False, show_alpha=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    n_title = f"{'Quantum Mode: ' if use_quantum else ''}{title}"
    ax.set_title(n_title)
    ax.plot(voltages, label="Membrane Potential", color='tab:blue')
    ax.plot(input_current, label="Input Current", linestyle='--', alpha=0.5, color='tab:orange')
    spike_times = np.where(spikes)[0]
    ax.scatter(spike_times, np.ones_like(spike_times)*1.5, marker='x', color='tab:red', label='Spikes')
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    if use_legend:
        ax.legend()
    st.pyplot(fig)

    if extras is not None and (show_adaptation or show_alpha):
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.set_title("DynamicSpikeProbability traces")
        if show_adaptation and "adaptation_trace" in extras:
            ax2.plot(extras["adaptation_trace"], label="adaptation")
        if show_alpha and "alpha_eff_trace" in extras:
            ax2.plot(extras["alpha_eff_trace"], label="alpha_eff")
        ax2.set_xlabel("Time")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

# --------------------------
# UI
# --------------------------
st.set_page_config(layout="wide")
st.title("LIF Neuron Simulator")
st.markdown("Visualize different LIF neuron modes with simulated input.")

left, right = st.columns([1, 2], gap="large")

with left:
    spike_mode = st.selectbox(
        "Spike Decision Mode",
        ["Classic (Threshold/Probability)", "Quantum (Qubit Measurement)"]
    )
    use_quantum = spike_mode == "Quantum (Qubit Measurement)"

    mode = st.selectbox("Select Neuron Mode", [
        "Deterministic (Basic LIF)",
        "Stochastic (Constant Threshold)",
        "Adaptive Threshold",
        "Dynamic Spike Probability",
        "With Neuromodulation"
    ])

    if use_quantum:
        quantum_wire = st.slider("Number of Quantum Wires (Qubits)", 1, 10, 4)
        quantum_threshold = st.slider("Quantum Threshold", 0.0, 1.0, 0.7)
        quantum_leak = st.slider("Quantum Leak (RY rotation)", 0.0, 2.0, 0.1)
    else:
        quantum_wire = 4
        quantum_threshold = 0.7
        quantum_leak = 0.1

    available_neuromod_functions = [
        "lambda x: torch.sigmoid(2*x)",
        "lambda x: torch.tanh(x)",
        "lambda x: torch.relu(x)",
        "lambda x: torch.sigmoid(x) * torch.tanh(x)"
    ]
    want_neuromod_controls = (mode in ["With Neuromodulation", "Dynamic Spike Probability"])
    if want_neuromod_controls:
        neuromodulation_function = st.selectbox("Neuromodulation Transform", available_neuromod_functions)
        neuromodulation_function = eval(neuromodulation_function)
        neuromod_mode = st.selectbox("Neuromodulation Mode", ["gain", "threshold", "prob_slope", "off"], index=0)
        neuromod_strength = st.slider("Neuromodulation Strength", -2.0, 2.0, 1.0, 0.1)
    else:
        neuromodulation_function = None
        neuromod_mode = "off"
        neuromod_strength = 1.0

    # DSP controls (sichtbar bei DSP-Mode)
    if mode == "Dynamic Spike Probability":
        base_alpha = st.slider("base_alpha (sigmoid slope)", 0.1, 10.0, 5.0, 0.1)
        tau_adapt  = st.slider("tau_adapt (adaptation decay)", 1.0, 100.0, 10.0, 1.0)
        min_alpha  = st.slider("min_alpha (clamp)", 0.01, 5.0, 0.05, 0.01)
        max_alpha  = st.slider("max_alpha (clamp)", 0.5, 20.0, 5.0, 0.5)
    else:
        base_alpha, tau_adapt, min_alpha, max_alpha = 5.0, 10.0, 0.05, 5.0

    noise_level = st.slider("Input Noise Level", 0.0, 1.0, 0.2)
    num_neurons = st.slider("Number of Neurons", 1, 10, 3)
    timesteps = st.slider("Number of Timesteps", 5, 500, 200)
    V_th = st.slider("Voltage Threshold", 0.0, 5.0, 1.0)
    eta = st.slider("Adaptation Rate (for adaptive threshold)", 0.0, 1.0, 0.15)
    tau = st.slider("Membrane Time Constant (tau)", 1.0, 100.0, 30.0)

batch_size = 1
device = "cpu"

# Signals
input_current = generate_input(timesteps, batch_size, num_neurons, noise_level)
external_mod = torch.sin(torch.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1, 1)

# --------------------------
# Configs
# --------------------------
base_lif_args = {
    "V_th": V_th,
    "tau": tau,
    "device": device,
    "quantum_mode": use_quantum,
    "quantum_wire": quantum_wire,
    "quantum_threshold": quantum_threshold,
    "quantum_leak": quantum_leak,
}

configs = {
    "Deterministic (Basic LIF)": {
        "num_neurons": num_neurons,
        "lif_args": {
            **base_lif_args,
            "stochastic": False,
            "use_adaptive_threshold": False,
            "neuromod_transform": None,
            "neuromod_mode": "off",
        }
    },
    "Stochastic (Constant Threshold)": {
        "num_neurons": num_neurons,
        "lif_args": {
            **base_lif_args,
            "stochastic": True,
            "noise_std": 0.05,
            "use_adaptive_threshold": False,
            "neuromod_transform": None,
            "neuromod_mode": "off",
        }
    },
    "Adaptive Threshold": {
        "num_neurons": num_neurons,
        "lif_args": {
            **base_lif_args,
            "stochastic": False,
            "use_adaptive_threshold": True,
            "eta": eta,
            "min_threshold": 0.8,
            "max_threshold": 2.0,
            "neuromod_transform": None,
            "neuromod_mode": "off",
        }
    },
    "Dynamic Spike Probability": {
        "num_neurons": num_neurons,
        "lif_args": {
            **base_lif_args,
            "stochastic": True,
            "allow_dynamic_spike_probability": True,
            "base_alpha": base_alpha,
            "tau_adapt": tau_adapt,
            # "min_alpha": min_alpha,
            # "max_alpha": max_alpha,
            "neuromod_transform": neuromodulation_function,
            "neuromod_mode": neuromod_mode,
            "neuromod_strength": neuromod_strength,
        }
    },
    "With Neuromodulation": {
        "num_neurons": num_neurons,
        "lif_args": {
            **base_lif_args,
            "stochastic": False,
            "use_adaptive_threshold": False,
            "neuromod_transform": neuromodulation_function,
            "neuromod_mode": neuromod_mode,           # typ. "gain" oder "threshold"
            "neuromod_strength": neuromod_strength,
        }
    }
}

config = configs[mode]

if (mode == "With Neuromodulation" and neuromod_mode != "off") or \
        (mode == "Dynamic Spike Probability" and neuromod_mode == "prob_slope"):
    mod_signal = external_mod
else:
    mod_signal = None

want_extras = (mode == "Dynamic Spike Probability")
spikes, voltages, inputs, extras = run_simulation(config, input_current, mod_signal, want_extras=want_extras)

with right:
    use_legend = st.checkbox("Use Legend", value=True)
    show_adapt = st.checkbox("Show adaptation (DSP)", value=want_extras)
    show_alpha = st.checkbox("Show alpha_eff (DSP)", value=want_extras)
    plot_results(spikes, voltages, inputs, mode, use_legend, use_quantum,
                 extras=extras, show_adaptation=show_adapt, show_alpha=show_alpha)
