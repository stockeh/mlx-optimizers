import mlx.core as mx

from mlx.optimizers import Optimizer
from typing import Callable, List, Union


class QHAdam(Optimizer):
    r"""Quasi-Hyperbolic Adam.

    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
        nus: List[float] = [1.0, 1.0],
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = False,
        eps: float = 1e-8
    ):
        super().__init__()

        self._maybe_schedule('learning_rate', learning_rate)
        self.betas = betas
        self.nus = nus
        self.weight_decay = weight_decay
        self.decouple_weight_decay = decouple_weight_decay
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state['b1_w'] = 0.0
        state['b2_w'] = 0.0
        state['m'] = mx.zeros_like(parameter)
        state['v'] = mx.zeros_like(parameter)

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

        b1_w = 1. + b1 * state['b1_w']
        b2_w = 1. + b2 * state['b2_w']
        m = state['m']
        v = state['v']

        b1_adj = 1 - 1. / b1_w
        b2_adj = 1 - 1. / b2_w
        gradient_sq = mx.square(gradient)

        m = m * b1_adj + (1 - b1_adj) * gradient
        v = v * b2_adj + (1 - b2_adj) * gradient_sq

        avg_grad = m * nu1
        if nu1 != 1.:
            avg_grad = avg_grad + (1. - nu1) * gradient

        avg_grad_rms = v * nu2
        if nu2 != 1.:
            avg_grad_rms = avg_grad_rms + (1. - nu2) * gradient_sq
        avg_grad_rms = mx.sqrt(avg_grad_rms)
        if self.eps != 0:
            avg_grad_rms = avg_grad_rms + self.eps

        state['b1_w'] = b1_w
        state['b2_w'] = b2_w
        state['m'] = m
        state['v'] = v

        return parameter - lr * (avg_grad / avg_grad_rms)