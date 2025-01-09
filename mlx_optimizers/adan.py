from typing import Callable, List, Optional, Union

import mlx.core as mx
from mlx.optimizers import Optimizer

class Adan(Optimizer):
    r"""The Adan optimizer [1]. In detail,

    [1]: Xie, Z., Wang, S., Hu, X., Huang, Y., Long, M., 2022. Adan: Adaptive 
    Nesterov momentum algorithm for faster training of deep learning models. 
    NeurIPS 2022.

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) (g_t - g_{t-1}) \\
        n_t &= \beta_3 n_{t-1} + (1 - \beta_3) (g_t + (1 - \beta_2)(g_t - g_{t-1}))^2 \\
        \eta_t &= \frac{\eta}{\sqrt{n_t + \epsilon_{\text{root}}} + \epsilon} \\
        w_{t+1} &= \frac{w_t - \eta_t (m_t + (1 - \beta_2)v_t)}{1 + \lambda \eta_t}

    Args:
        learning_rate (float or callable): The learning rate :math:`\eta`.
        betas (Tuple[float, float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2, \beta_3)` used for computing running averages
          of the gradient, gradient differences, and squared terms. Default: ``(0.9, 0.999, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the denominator
          to improve numerical stability. Default: ``1e-8``
        eps_root (float, optional): The term :math:`\epsilon_{\text{root}}` added
          inside the square root to improve numerical stability. Default: ``1e-8``
        weight_decay (float, optional): The strength of the weight decay
          regularization :math:`\lambda`. Default: ``0.0``
        bias_correction (bool, optional): If set to ``True``, bias correction is applied.
          Default: ``True``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.98, 0.92, 0.99],
        eps: float = 1e-8,
        eps_root: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction=False,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.eps_root = eps_root
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["n"] = mx.zeros_like(parameter)
        state["g_prev"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Adam parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        parameter = parameter * (1 - lr * self.weight_decay)
        b1, b2, b3 = self.betas
        eps = self.eps
        eps_root = self.eps_root
        bias_correction = self.bias_correction
        step = self.step

        m = state["m"]
        v = state["v"]
        n = state["n"]
        g_prev = state["g_prev"]

        diff = gradient - g_prev

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * diff
        n = b3 * n + (1 - b3) * mx.square(gradient + (1 - b2) * diff)

        state["m"] = m
        state["v"] = v
        state["n"] = n
        state["g_prev"] = gradient

        if bias_correction:
            denominator = mx.sqrt((n / (1 - b3**step)) + eps_root) + eps
            return (
                parameter
                - lr
                * ((m / (1 - b1**step)) + (1 - b2) * (v / (1 - b2**step)))
                / denominator
            )

        denominator = mx.sqrt(n + eps_root) + eps
        return parameter - lr * (m + (1 - b2) * v) / denominator