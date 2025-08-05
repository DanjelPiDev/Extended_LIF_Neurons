from setuptools import setup, find_packages

setup(
    name="extended_lif_neurons",
    version="0.6.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "pennylane"
    ],
    description="Advanced LIF neuron model with dynamic spike probability, adaptive thresholds, "
                "synaptic plasticity, neuromodulation, and a hybrid quantum mode for biologically and "
                "quantum-inspired spiking behavior. Features include membrane potential tracking, "
                "deterministic/stochastic/quantum spike generation, adaptive excitability, "
                "and PyTorch integration.",
    author="DanjelPiDev",
    url="https://github.com/DanjelPiDev/Extended_LIF_Neurons",
)
