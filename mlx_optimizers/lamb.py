from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class Lamb(Optimizer):
    r"""The Lamb optimizer [1].

    [1]: You, Yang, et al., 2019. Large batch optimization for deep learning:
    Training bert in 76 minutes. https://arxiv.org/abs/1904.00962

    .. math::

        m_0 &= 0, v_0 = 0 \\
        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        m_t &= m_t / (1 - \beta_1^t) \\
        v_t &= v_t / (1 - \beta_2^t) \\
        r_t &= \frac{m_t}{\sqrt{v_t} + \epsilon} \\
        \theta_{t+1} &= \theta_t - \eta \frac{\phi(\|\theta_t\|)}{\|r_t + \lambda \theta_t\|} \left(r_t + \lambda \theta_t\right)

    Args:
        learning_rate (float or callable): The learning rate :math:`\eta`.
        betas (Tuple[float, float], optional): The coefficients
            :math:`(\beta_1, \beta_2)` used for computing running averages of the
            gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-6``
        weight_decay: weight decay (L2 penalty). Default: ``0.9``
        clamp_value: clamp weight_norm in (0,clamp_value)
            set to a high value to avoid it (e.g :math:`10e3`). Default: ``10``
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. Default: ``False``
        debias: debias adam by (1 - beta**step). Default: ``False``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
        weight_decay: float = 0.0,
        clamp_value: float = 10,
        eps: float = 1e-6,
        adam: bool = False,
        debias: bool = False,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay
        self.clamp_value = clamp_value
        self.eps = eps
        self.adam = adam
        self.debias = debias

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas

        m = state["m"]
        v = state["v"]

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)

        bias_correction = (
            (mx.sqrt(1 - mx.power(b2, self.state["step"])) / (1 - mx.power(b1, self.state["step"])))
            if self.debias
            else 1
        )

        eta = lr * bias_correction
        weight_norm = mx.clip(mx.linalg.norm(parameter), self.clamp_value, float("inf"))
        adam_step = m / (mx.sqrt(v) + self.eps)

        if self.weight_decay:
            adam_step = adam_step + self.weight_decay * parameter

        adam_norm = mx.linalg.norm(adam_step)
        trust_ratio = (
            1
            if self.adam  # TODO: or not mx.any(weight_norm) or not mx.any(adam_norm)
            else weight_norm / adam_norm
        )

        state["m"] = m
        state["v"] = v

        return parameter - eta * trust_ratio * adam_step
