import mlx.core as mx

from mlx.optimizers import Optimizer
from typing import Callable, List, Union


class Lion(Optimizer):
    r"""The Lion optimizer [1].

    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.99],
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self._maybe_schedule('learning_rate', learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state['m'] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)

        # stepweight decay (AdamW)
        if self.weight_decay != 0:
            parameter = parameter * (1 - lr * self.weight_decay)

        m = state['m']
        b1, b2 = self.betas

        c = b1 * m + (1 - b1) * gradient
        state['m'] = b2 * m + (1 - b2) * gradient

        return parameter - lr * mx.sign(c)
