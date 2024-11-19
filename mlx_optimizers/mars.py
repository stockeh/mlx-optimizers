from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer
from mlx.utils import tree_map

from .common import newton_schulz


class MARS(Optimizer):
    r"""Make vAriance Reduction Shine [1].

    MARS combines two main components: (a) a scaled stochastic recursive momentum that 
    acts as a variance-reduced estimator of the full gradient, and (b) a preconditioner 
    to approximates the second-order Newton's method for better per-iteration complexity.

    This is based on the following preconditioned variance-reduced update rules:

    .. math::

        & \mathbf{m}_0 \gets 0, \quad \mathbf{x}_1 \gets \mathbf{x}_0 \\
        & \text{For } \, t = 1 \text{ to } n: \\
        & \quad \text{Sample } \xi_t \text{ and let } \mathbf{c}_t = \nabla f(\mathbf{x}_t, \xi_t) + \gamma_t \frac{\beta_1}{1 - \beta_1} \big( \nabla f(\mathbf{x}_t, \xi_t) - \nabla f(\mathbf{x}_{t-1}, \xi_t) \big) \\
        & \quad \text{If } \|\mathbf{c}_t\|_2 > 1, \text{ then } \tilde{\mathbf{c}}_t = \frac{\mathbf{c}_t}{\|\mathbf{c}_t\|_2}, \text{ else } \tilde{\mathbf{c}}_t = \mathbf{c}_t \\
        & \quad \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \tilde{\mathbf{c}}_t \\
        & \quad \mathbf{x}_{t+1} = \arg \min_{\mathbf{x}} \big\{ \eta_t \langle \mathbf{m}_t, \mathbf{x} \rangle + \frac{1}{2} \|\mathbf{x} - \mathbf{x}_t\|_{\mathbf{H}_t}^2 \big\}

    Hessian matrix approximations (``mars-adamw``, ``mars-lion``, ``mars-shampoo``):

    - ``mars-adamw``

    .. math::
        
        \mathbf{v}\_t &=\beta_2 \mathbf{v}\_{t-1}+(1-\beta_2) \big(\nabla f(\mathbf{x}\_t, \mathbf{\xi}\_t)\big)^2\\
        \mathbf{H}_t &:= \sqrt{\text{diag}\Big(\mathbf{v}_t\Big)}\cdot \frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}.

    - ``mars-lion``

    .. math::
    
        \mathbf{H}_t := \sqrt{\text{diag}(\mathbf{m}_t^2)}.


    - ``mars-shampoo`` (with Newton-Schulz iteration instead of SVD)

    .. math::
    
        \mathbf{U}\_t, &\mathbf{\Sigma}\_t, \mathbf{V}\_t = \text{SVD}(\mathbf{G}\_t),\\
        \mathbf{x}\_{t+1} &=\mathbf{x}\_t \eta_t\mathbf{U}_t\mathbf{V}\_t^\top.

        
    [1] Yuan, Huizhuo, Liu, Yifeng and Wu, Shuang and Zhou, Xun and Gu, Quanquan, 2024. 
    MARS: Unleashing the Power of Variance Reduction for Training Large Models.
    https://arxiv.org/abs/2411.10438
    https://github.com/AGI-Arena/MARS

    Args:
        learning_rate (float or callable): The MARS learning rate :math:`\eta`.
        betas (List[float], optional): The MARS coefficients :math:`(\beta_1, \beta_2)` 
            for exponential moving average. Default: ``[0.95, 0.99]``
        eps (float, optional): The term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-8``
        weight_decay (float, optional): The MARS weight decay. Default: ``0.0``
        amsgrad (bool, optional): Whether to use the AMSGrad variant. Default: ``False``
        gamma (float, optional): Scaling parameter that controls the strength of gradient 
            correction. Default: ``0.025``
        is_approx (bool, optional): Whether to use the approximate version. Default: ``True``
        mars_type (str, optional): The MARS type {mars-adamw, mars-lion, mars-shampoo}.
            Default: ``mars-adamw``
        optimize_1d (bool, optional): Whether MARS should optimize 1D parameters. 
            False, AdamW will be used for optimizing 1D parameters. Default: ``False``
        learning_rate_1d (float or callable): The learning rate for 1D parameters. Default: ``3e-3``
        betas_1d (List[float], optional): The coefficients for 1D parameters. Default: ``[0.9, 0.95]``
        weight_decay_1d (float, optional): The weight decay for 1D parameters. Default: ``0.1``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 3e-3,
        betas: List[float] = [0.95, 0.99],
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        gamma: float = 0.025,
        is_approx: bool = True,
        mars_type: str = "mars-adamw",
        optimize_1d: bool = False,
        learning_rate_1d: Union[float, Callable[[mx.array], mx.array]] = 3e-3,
        betas_1d: List[float] = [0.9, 0.95],
        weight_decay_1d: float = 0.1,
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
        self._maybe_schedule("learning_rate_1d", learning_rate_1d)
        self.betas_1d = betas_1d
        self.weight_decay_1d = weight_decay_1d

    @property
    def learning_rate_1d(self):
        return self.state["learning_rate_1d"]

    @learning_rate_1d.setter
    def learning_rate_1d(self, learning_rate_1d: Union[float, mx.array]):
        self.state["learning_rate_1d"] = mx.array(learning_rate_1d)

    def set_last_grad(self, gradients: dict):
        """Set the last gradient for each parameter"""
        assert self._initialized, "Must be initialized before setting"
        if not self.is_approx:

            def update_last(gradient, state):
                state["last_grad"] = gradient

            tree_map(update_last, gradients, self.state)

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["last_grad"] = mx.zeros_like(parameter)
        state["max_v"] = mx.zeros_like(parameter) if self.amsgrad else 0

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        mars_type = self.mars_type
        eps = self.eps
        m = state["m"]
        v = state["v"]
        last_grad = state["last_grad"]
        max_v = state["max_v"]

        update = 0
        is_grad_2d = gradient.ndim == 2
        if self.optimize_1d or is_grad_2d:
            lr = self.learning_rate.astype(gradient.dtype)
            b1, b2 = self.betas
            weight_decay = (  # no decay: bias, norms, and what looks like embedding/head
                self.weight_decay if (parameter.ndim > 1 and parameter.shape[0] < 1e4) else 0
            )
            # start
            c_t = gradient + self.gamma * (b1 / (1 - b1)) * (gradient - last_grad)
            c_t_norm = mx.linalg.norm(c_t)
            c_t = mx.where(c_t_norm > 1, c_t / c_t_norm, c_t)
            m = b1 * m + (1 - b1) * c_t
            if (mars_type == "mars-adamw") or (mars_type == "mars-shampoo" and not is_grad_2d):
                v = b2 * v + (1 - b2) * mx.square(c_t)
                bias_correction1 = 1 - mx.power(b1, self.state["step"])
                bias_correction2 = 1 - mx.power(b2, self.state["step"])
                denom = (
                    mx.where(
                        self.amsgrad and mx.any((max_v := mx.maximum(v, max_v))),
                        mx.sqrt(max_v),
                        mx.sqrt(v),
                    )
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
            lr = self.learning_rate_1d.astype(gradient.dtype)
            b1, b2 = self.betas_1d
            weight_decay = self.weight_decay_1d
            # start
            m = b1 * m + (1 - b1) * gradient
            v = b2 * v + (1 - b2) * mx.square(gradient)
            bias_correction1 = 1 - mx.power(b1, self.state["step"])
            bias_correction2 = 1 - mx.power(b2, self.state["step"])
            denom = (
                mx.where(
                    self.amsgrad and mx.any((max_v := mx.maximum(v, max_v))),
                    mx.sqrt(max_v),
                    mx.sqrt(v),
                )
            ) * (1 / mx.sqrt(bias_correction2)) + eps
            update = weight_decay * parameter + m / (denom * bias_correction1)

        state["m"] = m
        state["v"] = v
        state["max_v"] = max_v

        if self.is_approx:
            state["last_grad"] = gradient

        return parameter - lr * update
