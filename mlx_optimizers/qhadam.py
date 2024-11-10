from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class QHAdam(Optimizer):
    r"""Quasi-Hyperbolic Adaptive Moment Estimation [1].

    .. math::

        g_{t+1} &= \beta_1 g_t + (1 - \beta_1) g_t \\
        \theta_{t+1} &= \theta_t - \eta \left[ (1 - \nu) g_t + \nu g_{t+1}\right]

    [1] Ma, Jerry, and Denis Yarats, 2019. Quasi-hyperbolic momentum 
    and Adam for deep learning. ICLR 2019.
    https://arxiv.org/abs/1810.06801
    https://github.com/facebookresearch/qhoptim/

    Args:
        learning_rate (float or callable): learning rate :math:`\eta`.
        betas (Tuple[float, float], optional): coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        nus (Tuple[float, float], optional): immediate discount factors
            used to estimate the gradient and its square :math:`(\nu_1, \nu_2)`. 
            Default: ``(1.0, 1.0)``
        weight_decay: weight decay. Default: ``0.0``
        decouple_weight_decay: whether to decouple weight decay from the
            optimization step. Default: ``False``
        eps: term added to the denominator to improve numerical stability.
            Default: ``1e-8``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
        nus: List[float] = [1.0, 1.0],
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.nus = nus
        self.weight_decay = weight_decay
        self.decouple_weight_decay = decouple_weight_decay
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["b1_w"] = 0.0
        state["b2_w"] = 0.0
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        nu1, nu2 = self.nus

        # TODO: gradient.is_space RuntimeError

        if self.weight_decay != 0:
            if self.decouple_weight_decay:
                parameter = parameter * (1 - lr * self.weight_decay)
            else:
                gradient = gradient + self.weight_decay * parameter

        b1_w = 1.0 + b1 * state["b1_w"]
        b2_w = 1.0 + b2 * state["b2_w"]
        m = state["m"]
        v = state["v"]

        b1_adj = 1 - 1.0 / b1_w
        b2_adj = 1 - 1.0 / b2_w
        gradient_sq = mx.square(gradient)

        m = m * b1_adj + (1 - b1_adj) * gradient
        v = v * b2_adj + (1 - b2_adj) * gradient_sq

        avg_grad = m * nu1
        if nu1 != 1.0:
            avg_grad = avg_grad + (1.0 - nu1) * gradient

        avg_grad_rms = v * nu2
        if nu2 != 1.0:
            avg_grad_rms = avg_grad_rms + (1.0 - nu2) * gradient_sq
        avg_grad_rms = mx.sqrt(avg_grad_rms)
        if self.eps != 0:
            avg_grad_rms = avg_grad_rms + self.eps

        state["b1_w"] = b1_w
        state["b2_w"] = b2_w
        state["m"] = m
        state["v"] = v

        return parameter - lr * (avg_grad / avg_grad_rms)
