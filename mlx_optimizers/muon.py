from typing import Callable, Union

import mlx.core as mx
from mlx.optimizers import Adam, Optimizer


def zeropower_via_svd(G, steps=None) -> mx.array:
    U, S, V = mx.linalg.svd(G, stream=mx.cpu)  # type: ignore
    return U @ V.T


@mx.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7) -> mx.array:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(mx.bfloat16)
    X /= mx.linalg.norm(X) + eps  # ensure top singular value <= 1
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class Muon(Optimizer):
    r"""Muon - MomentUm Orthogonalized by Newton-schulz

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
        alternate_optimizer: Optimizer = Adam(1e-3),
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.backend = backend
        self.backend_steps = backend_steps
        self.alternate_optimizer = alternate_optimizer

        try:
            self.orthogonalize = {
                "svd": zeropower_via_svd,
                "newtonschulz5": zeropower_via_newtonschulz5,
            }[backend]
        except KeyError:
            raise ValueError(f"Unknown backend: {backend}")

    def init_single(self, parameter: mx.array, state: dict):
        if parameter.ndim != 2:
            return self.alternate_optimizer.init_single(parameter, state)
        state["muon_v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Apply Muon optimization update with Newton-Schulz orthogonalization."""

        if parameter.ndim != 2:  # TODO: find a better solution to flat parameters
            return self.alternate_optimizer.apply_single(gradient, parameter, state)

        if self.weight_decay != 0:
            gradient += self.weight_decay * parameter

        buf = state["muon_v"]
        buf = self.momentum * buf + gradient
        state["muon_v"] = buf

        gradient = (gradient + self.momentum * buf) if self.nesterov else buf
        gradient = self.orthogonalize(gradient, steps=self.backend_steps)
        scale = max(1, gradient.shape[0] / gradient.shape[1]) ** 0.5

        return parameter - self.learning_rate.astype(gradient.dtype) * scale * gradient
