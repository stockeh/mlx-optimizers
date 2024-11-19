from typing import List

import mlx.core as mx
from mlx.optimizers import Optimizer

from .common import newton_schulz


class MARS(Optimizer):
    r"""Make vAriance Reduction Shine [1].

    ..
    """

    def __init__(
        self,
        learning_rate: float,
        betas: List[float] = [0.95, 0.99],
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        gamma: float = 0.025,
        is_approx: bool = True,
        mars_type: str = "mars-adamw",
        optimize_1d: bool = False,
        learning_rate_1d: float = 3e-3,  # TODO: use
        betas_1d: List[float] = [0.9, 0.95],
        weight_decay_1d: float = 0.0,
    ):
        super().__init__()

        assert mars_type in ["mars-adamw", "mars-lion", "mars-shampoo"], "MARS type not supported"

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.gamma = gamma
        self.is_approx = is_approx
        self.mars_type = mars_type
        self.optimize_1d = optimize_1d
        self.learning_rate_1d = learning_rate_1d
        self.betas_1d = betas_1d
        self.weight_decay_1d = weight_decay_1d

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["last_grad"] = mx.zeros_like(parameter)
        state["max_v"] = mx.zeros_like(parameter)
        state["first"] = True

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        mars_type = self.mars_type
        weight_decay = self.weight_decay
        weight_decay_1d = self.weight_decay_1d
        eps = self.eps
        m = state["m"]
        v = state["v"]
        last_grad = mx.where(not self.is_approx or state["first"], gradient, state["last_grad"])
        max_v = state["max_v"]

        update = 0
        is_grad_2d = gradient.ndim == 2
        if self.optimize_1d or is_grad_2d:
            c_t = gradient + self.gamma * (b1 / (1 - b1)) * (gradient - last_grad)
            c_t_norm = mx.linalg.norm(c_t)
            c_t = mx.where(c_t_norm > 1, c_t / c_t_norm, c_t)
            m = b1 * m + (1 - b1) * c_t
            if (mars_type == "mars-adamw") or (mars_type == "mars-shampoo" and not is_grad_2d):
                v = b2 * v + (1 - b2) * mx.square(c_t)
                bias_correction1 = 1 - mx.power(b1, self.state["step"])
                bias_correction2 = 1 - mx.power(b2, self.state["step"])
                denom = (
                    mx.sqrt(max_v)
                    if self.amsgrad and (max_v := mx.maximum(v, max_v))
                    else mx.sqrt(v)
                ) * (1 / mx.sqrt(bias_correction2)) + eps
                update = weight_decay * parameter + m / (denom * bias_correction1)
            elif mars_type == "mars-lion":
                update = weight_decay * parameter + mx.sign(m)
            elif mars_type == "mars-shampoo" and is_grad_2d:
                factor = max(1, gradient.shape[0] / gradient.shape[1]) ** 0.5
                update = (
                    newton_schulz((1 / (1 - b1)) * m, steps=5, eps=eps) * factor
                    + weight_decay * parameter
                )
        else:
            b1d, b2d = self.betas_1d
            m = b1d * m + (1 - b1d) * gradient
            v = b2d * v + (1 - b2d) * mx.square(gradient)
            bias_correction1 = 1 - mx.power(b1d, self.state["step"])
            bias_correction2 = 1 - mx.power(b2d, self.state["step"])
            denom = (
                mx.sqrt(max_v) if self.amsgrad and (max_v := mx.maximum(v, max_v)) else mx.sqrt(v)
            ) * (1 / mx.sqrt(bias_correction2)) + eps
            wd = weight_decay if self.optimize_1d else weight_decay_1d
            update = wd * parameter + m / (denom * bias_correction1)

        state["m"] = m
        state["v"] = v
        state["max_v"] = max_v

        if self.is_approx:
            state["last_grad"] = gradient
        state["first"] = False

        return parameter - lr * update
