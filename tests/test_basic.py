import mlx.core as mx
import pytest
from mlx.optimizers import AdamW

import mlx_optimizers as optim

from .common import ids


def rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 1 * (y - x**2) ** 2


def quadratic(xy):
    x, y = xy
    a = 1.0
    b = 1.0
    return (x**2) / a + (y**2) / b


def beale(xy):
    x, y = xy
    f = (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2
    return f


cases = [
    (rosenbrock, (-1.5, 2), (1, 1)),
    (quadratic, (5.0, -3), (0, 0)),
    (beale, (1.0, 0), (3, 0.5)),
]


optimizers = [
    (optim.QHAdam, {"learning_rate": 0.25}, 150),
    (optim.DiffGrad, {"learning_rate": 0.3}, 150),
    (optim.MADGRAD, {"learning_rate": 0.03}, 150),
    (optim.ADOPT, {"learning_rate": 0.17}, 150),
    (
        optim.Muon,  # using alternate for ndim < 2
        {
            "alternate_optimizer": AdamW(learning_rate=0.12, betas=[0.9, 0.99]),
        },
        700,
    ),
    (optim.Kron, {"learning_rate": 0.015, "precond_update_prob": 0.75}, 800),
    (optim.MARS, {"learning_rate": 0.1, "optimize_1d": True, "mars_type": "mars-adamw"}, 250),
    (optim.MARS, {"learning_rate_1d": 0.1, "weight_decay_1d": 0, "optimize_1d": False}, 150),
    # TODO: Lamb & Shampoo tests
]


@pytest.mark.parametrize("case", cases, ids=ids)
@pytest.mark.parametrize("optimizer_config", optimizers, ids=ids)
def test_benchmark_function(case, optimizer_config):
    func, initial_state, min_loc = case
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(**config)
    mx.random.seed(42)

    x = mx.array(initial_state)

    for _ in range(iterations):
        grad = mx.grad(func)(x)
        x = optimizer.apply_gradients({"o": grad}, {"o": x})["o"]

    min_loc = mx.array(min_loc)
    error = mx.sqrt(mx.sum((x - min_loc) ** 2)).item()
    assert mx.allclose(x, min_loc, atol=0.01), f"x=({x[0]:.4f},{x[1]:.4f}), {error=:.4f}"

    name = optimizer.__class__.__name__
    assert name in optimizer.__repr__()
