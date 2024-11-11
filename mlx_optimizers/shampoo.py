from typing import Callable, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


def _matrix_power(matrix: mx.array, power: float, eps=1e-16) -> mx.array:
    u, s, vt = mx.linalg.svd(matrix, stream=mx.cpu)  # type: ignore
    # eps: needed to avoid runtime command buffer execution (MLX)
    return u @ mx.power(s + eps, power).diag() @ vt


class Shampoo(Optimizer):
    r"""Preconditioned Stochastic Tensor Optimization (general tensor case) [1].

    .. math::

        W_1 &= 0_{n_1 \times \dots \times n_k}; \forall i \in [k]: H_0^i = \epsilon I_{n_i}\\
        H_t^i &= H_{t-1}^i + G_t^{(i)}\\
        \tilde{G}_t &= \tilde{G}_t \times_i (H_t^i)^{-1/2k}\\
        W_{t+1} &= W_t - \eta \tilde{G}_t

    [1] Gupta, Vineet, Tomer Koren, and Yoram Singer, 2018. Shampoo: Preconditioned 
    stochastic tensor optimization. ICML 2018.
    https://arxiv.org/abs/1802.09568
    

    Args:
        learning_rate (float or callable): learning rate :math:`\eta`.
        momentum (float, optional): momentum factor. Default: ``0.00``
        weight_decay (float, optional): weight decay factor. Default: ``0.00``
        update_freq (int, optional): frequency of updating the preconditioner. Default: ``1``
        eps (float, optional): term :math:`\epsilon` added to the
            denominator to improve numerical stability. Default: ``1e-6``
    
    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        update_freq: int = 1,
        eps: float = 1e-4,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.update_freq = update_freq
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        if self.momentum != 0:
            state["buf"] = mx.zeros_like(parameter)
        for i, dim in enumerate(parameter.shape):
            state[f"precond_{i}"] = self.eps * mx.eye(dim)
            state[f"inv_precond_{i}"] = mx.zeros((dim, dim))
        state["dim_inds"] = list(range(parameter.ndim))
        state["update_step"] = 0

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        lr = self.learning_rate.astype(gradient.dtype)
        momentum = self.momentum

        d = state["dim_inds"]
        order = gradient.ndim
        original_size = gradient.shape
        if momentum != 0:
            if state["update_step"] == 0:
                state["buf"] = gradient
            gradient = (1 - momentum) * gradient + momentum * state["buf"]

        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter

        for i, dim in enumerate(gradient.shape):
            precond = state[f"precond_{i}"]
            inv_precond = state[f"inv_precond_{i}"]

            if i != 0:
                gradient = gradient.transpose([d[i]] + d[1:i] + [d[0]] + d[i + 1 :])
            transpose_size = gradient.shape
            gradient = gradient.reshape(dim, -1)

            gradient_t = gradient.T
            precond = precond + gradient @ gradient_t
            if state["update_step"] % self.update_freq == 0:
                inv_precond = _matrix_power(precond, -1 / order)

            if i == order - 1:  # finally
                gradient = gradient_t @ inv_precond
                gradient = gradient.reshape(original_size)
                state[f"precond_{i}"] = precond
                state[f"inv_precond_{i}"] = inv_precond
            else:
                gradient = inv_precond @ gradient
                gradient = gradient.reshape(transpose_size)

        if momentum != 0:
            state["buf"] = gradient
        state["update_step"] += 1

        gradient = gradient.reshape(original_size)
        return parameter - lr * gradient
