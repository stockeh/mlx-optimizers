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
        betas (Tuple[float, float], optional): The coefficients
            :math:`(\beta_1, \beta_2)` used for computing running averages of the
            gradient and its square. Default: ``(0.9, 0.9999)``
        eps (float, optional): The term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-6``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.9999],
        eps: float = 1e-6,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["first"] = True

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps

        m = state["m"]
        v = state["v"]

        if state["first"]:  # TODO: eval of self.step not allowed with mx.compile
            state["first"] = False
            v = mx.square(gradient)
            m = gradient / mx.maximum(mx.sqrt(v), eps)
        else:
            m = b1 * m + (1 - b1) * gradient / mx.maximum(mx.sqrt(v), eps)

        parameter = parameter - lr * m
        v = b2 * v + (1 - b2) * mx.square(parameter)

        state["m"] = m
        state["v"] = v

        return parameter
