import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lif.lif_neuron_group import LIFNeuronGroup
import tracemalloc

matplotlib.use('TkAgg')


def test_performance(neuron_counts, timesteps, noise_std=0.0, use_adaptive_threshold=True):
    runtimes = []

    for num_neurons in neuron_counts:
        print(f"Testing {num_neurons} neurons over {timesteps} timesteps...")
        neuron_group = LIFNeuronGroup(
            num_neurons=num_neurons,
            noise_std=noise_std,
            use_adaptive_threshold=use_adaptive_threshold
        )
        inputs = np.random.uniform(0, 1, size=(timesteps, num_neurons))

        tracemalloc.start()

        start_time = time.time()
        for t in range(timesteps):
            neuron_group.step(inputs[t])
        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime = end_time - start_time

        # Convert memory to MB
        runtimes.append((runtime, peak_memory / 1e6))

        print(f"Runtime for {num_neurons} neurons: {runtime:.2f} seconds")
        print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")

    return runtimes


def plot_results(neuron_counts, results, title):
    runtimes, memory_usages = zip(*results)

    # Runtime plot for different neuron counts
    plt.figure(figsize=(10, 6))
    plt.plot(neuron_counts, runtimes, marker='o', label='Runtime (seconds)')
    plt.title(f"{title} - Runtime")
    plt.xlabel('Number of Neurons')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_runtime.png')

    # Memory usage plot for different neuron counts
    plt.figure(figsize=(10, 6))
    plt.plot(neuron_counts, memory_usages, marker='o', label='Peak Memory (MB)', color='orange')
    plt.title(f"{title} - Memory Usage")
    plt.xlabel('Number of Neurons')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_memory.png')


if __name__ == "__main__":
    neuron_counts = [10_000, 100_000, 1_000_000, 10_000_000]

    # Number of timesteps to simulate, 16 is a typical value (I guess)
    timesteps = 16

    print("Testing with Noise and Adaptive Thresholds Enabled")
    results = test_performance(neuron_counts, timesteps, noise_std=0.1, use_adaptive_threshold=True)
    plot_results(neuron_counts, results, "Performance with Noise and Adaptive Thresholds")

    print("Testing without Noise")
    results_no_noise = test_performance(neuron_counts, timesteps, noise_std=0.0, use_adaptive_threshold=True)
    plot_results(neuron_counts, results_no_noise, "Performance without Noise")

    print("Testing without Adaptive Thresholds")
    results_no_adaptive = test_performance(neuron_counts, timesteps, noise_std=0.1, use_adaptive_threshold=False)
    plot_results(neuron_counts, results_no_adaptive, "Performance without Adaptive Thresholds")
