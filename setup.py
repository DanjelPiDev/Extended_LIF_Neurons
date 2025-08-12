from setuptools import setup, find_packages

setup(
    name="qlif_neurons",
    version="0.6.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "pennylane"
    ],
    description="Quantum-inspired Leaky Integrate-and-Fire (QLIF) neurons for PyTorch. "
                "Implements adaptive thresholds, dynamic spike probabilities, synaptic plasticity, "
                "and neuromodulation, with an optional qubit-based spike decision mechanism. "
                "Supports deterministic, stochastic, and quantum spike generation, "
                "membrane potential tracking, and seamless integration into spiking neural networks.",
    author="DanjelPiDev",
    url="https://github.com/DanjelPiDev/QLIF-Neurons",
)
