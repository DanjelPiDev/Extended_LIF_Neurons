import numpy as np


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Applies ReLU activation to the input.

        :param x: Input tensor (list or numpy array).
        :return: Activated output.
        """
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Computes the gradient of ReLU for backpropagation.

        :param grad_output: Gradient from the next layer.
        :return: Gradient to pass to the previous layer.
        """
        return (grad_output > 0).astype(float)
