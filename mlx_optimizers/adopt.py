from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class ADOPT(Optimizer):
    r"""ADaptive gradient method with the OPTimal convergence rate [1].

    .. math::

        v_0 &= g_0^2, m_1 = g_1 / \max{\sqrt{v_0}, \epsilon} \\
        \theta_{t} &= \theta_{t-1} - \eta m_{t-1} \\
        v_{t} &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
        m_{t+1} &= \beta_1 m_{t} + (1 - \beta_1) (g_{t+1} / \max{\sqrt{v_t}, \epsilon})

    [1] Taniguchi, Shohei, et al., 2024. ADOPT: Modified Adam Can
    Converge with Any :math:`\beta_2` with the Optimal Rate. NeurIPS 2024.
    https://arxiv.org/abs/2411.02853
    https://github.com/iShohei220/adopt

    Args:
        learning_rate (float or callable): The learning rate :math:`\eta`.
        betas (List[float, float], optional): The coefficients
            :math:`(\beta_1, \beta_2)` used for computing running averages of the
            gradient and its square. Default: ``(0.9, 0.9999)``
        weight_decay (float, optional): The weight decay. Default: ``0.0``
        eps (float, optional): The term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-6``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.9999],
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = []
        state["c"] = 0

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        state["c"] += 1

        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter

        if state["c"] == 1:
            state["v"] = mx.square(gradient)
            return parameter

        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps

        m = state["m"]
        v = state["v"]
        denom = mx.maximum(mx.sqrt(v), eps)

        if state["c"] == 2:
            m = gradient / denom
        else:
            m = b1 * m + (1 - b1) * gradient / denom

        parameter = parameter - lr * m

        state["v"] = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m

        return parameter
