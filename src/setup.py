from setuptools import setup, find_packages

setup(
    name="snn_neurons",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
    description="A project implementing custom LIF Neurons and layers for Spiking Neural Networks (SNNs) in PyTorch.",
    author="NullPointerExcy",
    url="https://github.com/NullPointerExcy/snn_neurons",
)
