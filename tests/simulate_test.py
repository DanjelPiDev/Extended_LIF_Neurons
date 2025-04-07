import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.layers.torch_layers import LIFLayer


def visualize_neuron_modes():
    # Common parameters
    timesteps = 200
    num_neurons = 3
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test input (ramp + noise)
    input_current = torch.cat([
        torch.zeros(50, batch_size, num_neurons),
        torch.linspace(0, 2, 100).repeat(batch_size, num_neurons, 1).permute(2, 0, 1),
        torch.zeros(50, batch_size, num_neurons)
    ]).to(device) + torch.randn(200, batch_size, num_neurons).to(device) * 0.2

    # Create different configurations
    configs = {
        "Deterministic (Basic LIF)": {
            "stochastic": False,
            "use_adaptive_threshold": False,
            "V_th": 1.0,
            "tau": 30.0,
            "allow_dynamic_spike_probability": False
        },
        "Stochastic (Constant Threshold)": {
            "stochastic": True,
            "noise_std": 0.05,
            "V_th": 1.5,
            "tau": 30.0,
            "use_adaptive_threshold": False
        },
        "Adaptive Threshold": {
            "stochastic": False,
            "use_adaptive_threshold": True,
            "V_th": 1.0,
            "tau": 30.0,
            "eta": 0.15,
            "min_threshold": 0.8,
            "max_threshold": 2.0
        },
        "Dynamic Spike Prob": {
            "stochastic": True,
            "allow_dynamic_spike_probability": True,
            "V_th": 1.0,
            "tau": 30.0,
            "base_alpha": 5.0,
            "tau_adapt": 10.0
        },
        "With Neuromodulation": {
            "stochastic": False,
            "V_th": 1.0,
            "tau": 30.0,
            "neuromod_transform": lambda x: torch.sigmoid(2*x)  # Custom modulation
        }
    }

    plt.figure(figsize=(15, 10))
    plt.suptitle("LIF Neuron Activity in Different Modes")

    external_mod = torch.sin(torch.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1, 1).to(device)

    for idx, (name, params) in enumerate(configs.items(), 1):
        lif = LIFLayer(
            num_neurons=num_neurons,
            batch_size=batch_size,
            device=device,
            dt=1.0,
            **params
        ).to(device)

        with torch.no_grad():
            if "Neuromod" in name:
                spikes, voltages = lif(input_current, external_modulation=external_mod)
            else:
                spikes, voltages = lif(input_current)

        spikes_np = spikes.cpu().numpy().squeeze()
        voltages_np = voltages.cpu().numpy().squeeze()
        input_np = input_current.cpu().numpy().squeeze()

        plt.subplot(len(configs), 1, idx)
        plt.title(name)
        plt.plot(voltages_np, label='Membrane Potential', color='tab:blue')
        plt.plot(input_np, label='Input Current', linestyle='--', alpha=0.6, color='tab:orange')

        spike_times = np.where(spikes_np)[0]
        plt.scatter(spike_times, np.ones_like(spike_times)*1.5,
                    marker='x', color='tab:red', label='Spikes')

        if params.get('use_adaptive_threshold', False):
            threshold_np = lif.lif_group.V_th.cpu().numpy().squeeze()
            plt.plot(threshold_np, label='Threshold', color='tab:green', linestyle='--')

        plt.ylabel('Voltage/Input')
        plt.xlabel('Time (ms)')
        plt.grid(True)
        plt.ylim(-0.5, 2.5)

        if idx == 1:
            plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('neuron_activity_2.png')
    print("Plot saved to neuron_activity_2.png")
    plt.show()


if __name__ == "__main__":
    visualize_neuron_modes()
