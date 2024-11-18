from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class Lamb(Optimizer):
    r"""Layerwise Adaptive Large Batch Optimization [1].

    .. math::

        m_0 &= 0, v_0 = 0 \\
        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        mh_t &= m_t / (1 - \beta_1^t) \\
        vh_t &= v_t / (1 - \beta_2^t) \\
        r_t &= \frac{mh_t}{\sqrt{vh_t} + \epsilon} \\
        \theta_{t+1} &= \theta_t - \eta \frac{\phi(\|\theta_t\|)}{\|r_t + \lambda \theta_t\|} \left(r_t + \lambda \theta_t\right)

    [1] You, Yang, et al., 2019. Large Batch Optimization for Deep Learning: 
    Training BERT in 76 Minutes. 
    https://arxiv.org/abs/1904.00962 v5
    https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py

    Args:
        learning_rate (float or callable): learning rate :math:`\eta`.
        betas (Tuple[float, float], optional): coefficients
            :math:`(\beta_1, \beta_2)` used for computing running averages of the
            gradient and its square. Default: ``(0.9, 0.999)``
        weight_decay (float): weight decay. Default: ``0.0``
        eps (float, optional): term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-8``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
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

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas

        m = state["m"]
        v = state["v"]

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)

        m_hat = m / (1 - mx.power(b1, self.state["step"]))
        v_hat = v / (1 - mx.power(b2, self.state["step"]))

        update = m_hat / (mx.sqrt(v_hat) + self.eps)
        if self.weight_decay:
            update = update + self.weight_decay * parameter

        w_norm = mx.linalg.norm(parameter)
        g_norm = mx.linalg.norm(update)
        ratio = mx.where(
            mx.greater(w_norm, 0),
            mx.where(mx.greater(g_norm, 0), (w_norm / g_norm), 1),
            1,
        )

        state["m"] = m
        state["v"] = v

        return parameter - lr * ratio * update
