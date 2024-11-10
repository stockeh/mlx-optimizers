from typing import Callable, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class MADGRAD(Optimizer):
    r"""Momentumized, Adaptive, Dual averaged GRADient [1].

    .. math::

        s_0 &= 0, v_0 = 0, x_0 = \theta \\
        \lambda_t &= \eta \sqrt{t + 1} \\
        s_{t+1} &= s_t + \lambda_t g_t \\
        v_{t+1} &= v_t + \lambda_t g_t^2 \\
        z_{t+1} &= x_0 - \frac{s_{t+1}}{\sqrt[3]{v_{t+1}} + \epsilon} \\
        \theta_{t+1} &= (1 - \beta) \theta_t + \beta z_{t+1}

    [1] Defazio, Aaron, and Samy Jelassi, 2022. A momentumized, adaptive, dual
    averaged gradient method. JMLR 23.144 (2022): 1-34.
    https://arxiv.org/abs/2101.11075
    https://github.com/facebookresearch/madgrad

    Args:
        learning_rate (float or callable): The learning rate :math:`\eta`.
        momentum (float, optional): The momentum coefficient :math:`\beta`. 
            Default: ``0.9``
        weight_decay (float, optional): The weight decay coefficient. 
            Default: ``0.0``
        eps (float, optional): The term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-6``
    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        state["v"] = mx.zeros_like(parameter)
        state["s"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        eps = self.eps
        lr = self.learning_rate.astype(gradient.dtype) + eps

        # cheat code because __init_single__ has grad (not param)
        if self.momentum != 0 and "x0" not in state:
            state["x0"] = parameter

        ck = 1 - self.momentum
        lamb = lr * mx.sqrt(self.state["step"])

        v = state["v"]
        s = state["s"]

        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter

        if self.momentum == 0:
            rms = mx.power(v, 1 / 3) + eps
            x0 = parameter + s / rms
        else:
            x0 = state["x0"]

        s = s + lamb * gradient
        v = v + lamb * mx.square(gradient)
        rms = mx.power(v, 1 / 3) + eps

        if eps == 0:
            rms = mx.where(rms != 0, rms, float("inf"))

        if self.momentum == 0:
            parameter = x0 - s / rms
        else:
            z = x0 - s / rms
            parameter = (1 - ck) * parameter + ck * z

        state["v"] = v
        state["s"] = s

        return parameter
