from typing import Callable, List, Optional, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class ADOPT(Optimizer):
    r"""ADaptive gradient method with the OPTimal convergence rate [1].

    .. math::

        m_0 &= \mathbf{0}, \quad v_0 = g_0^2 \\
        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \text{clip} \left( \frac{g_t}{\text{max}(\sqrt{v_{t-1}, \epsilon})}, c_t\right) \\
        \theta_{t} &= \theta_{t-1} - \eta m_t \\
        v_{t} &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2

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
        decouple (bool, optional): AdamW if ``True``. Default: ``False``
        clip_lambda (callable, optional): The clipping function :math:`c_t` for the
            gradient. Set to ``None`` for previous behavior. Default: ``step**0.25``
        eps (float, optional): The term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-6``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.9999],
        weight_decay: float = 0.0,
        decouple: bool = False,
        clip_lambda: Optional[Callable[[mx.array], mx.array]] = lambda step: mx.power(step, 0.25),
        eps: float = 1e-6,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay
        self.decouple = decouple
        self.clip_lambda = clip_lambda
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["c"] = 0

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""

        weight_decay = self.weight_decay
        decouple = self.decouple

        if weight_decay != 0 and not decouple:
            gradient = gradient + weight_decay * parameter

        if state["c"] == 0:
            state["v"] = mx.square(gradient)
            state["c"] += 1
            return parameter

        lr = self.learning_rate.astype(gradient.dtype)

        if weight_decay != 0 and decouple:
            parameter = parameter - lr * weight_decay * parameter

        b1, b2 = self.betas

        m = state["m"]
        v = state["v"]
        denom = mx.maximum(mx.sqrt(v), self.eps)
        normed_grad = gradient / denom
        if self.clip_lambda is not None:
            clip = self.clip_lambda(self.step - 1)
            normed_grad = mx.clip(normed_grad, -clip, clip)

        m = b1 * m + (1 - b1) * normed_grad
        parameter = parameter - lr * m

        state["v"] = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m

        return parameter
