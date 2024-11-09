from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class DiffGrad(Optimizer):
    r"""The DiffGrad optimizer [1].

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.99],
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["gp"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas

        # TODO: gradient.is_space RuntimeError

        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter

        m = state["m"]
        v = state["v"]
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)
        denom = mx.sqrt(v) + self.eps

        gp = state["gp"]

        # step not passed with state, needs self
        b1_adj = 1 - b1 ** self.state["step"]
        b2_adj = 1 - b2 ** self.state["step"]

        diff = mx.abs(gp - gradient)
        dfc = 1 / (1.0 + mx.exp(-diff))

        state["m"] = m
        state["v"] = v
        state["gp"] = gradient

        eta = lr * mx.sqrt(b2_adj) / b1_adj
        return parameter - eta * (m * dfc / denom)
