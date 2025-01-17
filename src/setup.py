from setuptools import setup, find_packages

setup(
    name='snn_neurons',
    version='0.1.0',
    description='Spiking Neural Network Neurons and Layers with Autorregresive Bernoulli Layer for PyTorch.',
    author='NullPointerExcy',
    url='https://github.com/NullPointerExcy/snn_neurons',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
