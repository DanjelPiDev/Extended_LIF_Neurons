import time
import matplotlib
import matplotlib.pyplot as plt
import torch
from lif.lif_neuron_group import LIFNeuronGroup
import tracemalloc

matplotlib.use('TkAgg')


def test_performance(neuron_counts, timesteps, noise_std=0.0, use_adaptive_threshold=True, stochastic=True, batch_size=1, device="cuda"):
    """
    Test performance of LIFNeuronGroup for different neuron counts and configurations.

    :param neuron_counts: List of neuron counts to test.
    :param timesteps: Number of timesteps to simulate.
    :param noise_std: Standard deviation of noise.
    :param use_adaptive_threshold: Whether to use adaptive thresholds.
    :param stochastic: Whether to enable stochastic spiking.
    :param batch_size: Number of batches.
    :param device: Device to run the simulation ('cpu' or 'cuda').
    :return: List of tuples with runtime and peak memory usage for each neuron count.
    """
    runtimes = []

    for num_neurons in neuron_counts:
        print(f"Testing {num_neurons} neurons over {timesteps} timesteps...")

        # Initialize LIF neuron group
        neuron_group = LIFNeuronGroup(
            num_neurons=num_neurons,
            batch_size=batch_size,
            noise_std=noise_std,
            use_adaptive_threshold=use_adaptive_threshold,
            stochastic=stochastic,
            device=device
        )

        # Generate input currents
        inputs = torch.rand((timesteps, batch_size, num_neurons), device=device)

        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()

        # Simulate each timestep
        for t in range(timesteps):
            neuron_group.step(inputs[t])

        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime = end_time - start_time
        runtimes.append((runtime, peak_memory / 1e6))  # Convert memory usage to MB

        print(f"Runtime for {num_neurons} neurons: {runtime:.2f} seconds")
        print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")

    return runtimes


def plot_results(neuron_counts, results, title):
    """
    Plot runtime and memory usage results.

    :param neuron_counts: List of neuron counts tested.
    :param results: Results from the performance test (runtime and memory usage).
    :param title: Title for the plots.
    """
    runtimes, memory_usages = zip(*results)

    # Runtime plot
    plt.figure(figsize=(10, 6))
    plt.plot(neuron_counts, runtimes, marker='o', label='Runtime (seconds)')
    plt.title(f"{title} - Runtime")
    plt.xlabel('Number of Neurons')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_runtime.png')
    plt.show()

    # Memory usage plot
    plt.figure(figsize=(10, 6))
    plt.plot(neuron_counts, memory_usages, marker='o', label='Peak Memory (MB)', color='orange')
    plt.title(f"{title} - Memory Usage")
    plt.xlabel('Number of Neurons')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_memory.png')
    plt.show()


if __name__ == "__main__":
    # Test configurations
    neuron_counts = [10_000, 100_000, 1_000_000, 10_000_000]
    timesteps = 16
    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Performance tests
    print("Testing with Noise and Adaptive Thresholds Enabled")
    results = test_performance(neuron_counts, timesteps, noise_std=0.1, use_adaptive_threshold=True, batch_size=batch_size, device=device)
    plot_results(neuron_counts, results, "Performance with Noise and Adaptive Thresholds")

    print("Testing without Noise")
    results_no_noise = test_performance(neuron_counts, timesteps, noise_std=0.0, use_adaptive_threshold=True, stochastic=False, batch_size=batch_size, device=device)
    plot_results(neuron_counts, results_no_noise, "Performance without Noise")

    print("Testing without Adaptive Thresholds")
    results_no_adaptive = test_performance(neuron_counts, timesteps, noise_std=0.1, use_adaptive_threshold=False, batch_size=batch_size, device=device)
    plot_results(neuron_counts, results_no_adaptive, "Performance without Adaptive Thresholds")
