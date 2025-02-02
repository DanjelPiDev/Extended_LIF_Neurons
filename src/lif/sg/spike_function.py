from typing import Optional

import torch


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, surrogate_gradient_function: str = "heaviside", alpha: float = 1.0):
        ctx.surrogate_gradient_function = surrogate_gradient_function.lower()
        ctx.alpha = alpha
        output = (input >= 0).float()
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        surrogate_gradient_function = ctx.surrogate_gradient_function
        func = getattr(SpikeFunction, surrogate_gradient_function)
        alpha = ctx.alpha
        grad_input = func(input, alpha) * grad_output
        return grad_input, None, None

    @staticmethod
    def heaviside(x):
        return 0.5 * (torch.sign(x) + 1)

    @staticmethod
    def fast_sigmoid(x, alpha: float = 1.0):
        return 1 / ((1 + alpha * x.abs()) ** 2)

    @staticmethod
    def gaussian(x, sigma: float = 1.0):
        return torch.exp(-((x ** 2) / (2 * sigma ** 2)))

    @staticmethod
    def arctan(x, alpha: float = 1.0):
        return 1 / (1 + (alpha * x) ** 2)
