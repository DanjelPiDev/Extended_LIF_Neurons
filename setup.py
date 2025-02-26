from setuptools import setup, find_packages

setup(
    name="extended_lif_neurons",
    version="0.2.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
    description="This repository presents an advanced LIF neuron model with dynamic spike probability, adaptive "
                "thresholds, synaptic plasticity, and neuromodulation for more biologically realistic behavior. Key "
                "features include membrane potential tracking, stochastic and deterministic spike generation, "
                "adaptive excitability, and PyTorch integration.",
    author="NullPointerExcy",
    url="https://github.com/NullPointerExcy/Extended_LIF_Neurons",
)
