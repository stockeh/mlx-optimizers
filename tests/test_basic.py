import mlx.core as mx
import pytest

import mlx_optimizers as optim


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
    (rosenbrock, (1.5, 1.5), (1, 1)),
    (quadratic, (1.5, 1.5), (0, 0)),
    (beale, (1.5, 1.5), (3, 0.5)),
]


def ids(v):
    return f"{v[0].__name__, } {v[1:]}"


optimizers = [
    (optim.QHAdam, {"learning_rate": 0.25}, 300),
    (optim.DiffGrad, {"learning_rate": 0.3}, 300),
    (optim.MADGRAD, {"learning_rate": 0.03}, 300),
    (optim.ADOPT, {"learning_rate": 0.8}, 1200),
    # TODO: Muon & Lamb tests
]


@pytest.mark.parametrize("case", cases, ids=ids)
@pytest.mark.parametrize("optimizer_config", optimizers, ids=ids)
def test_benchmark_function(case, optimizer_config):
    func, initial_state, min_loc = case
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(**config)

    x = mx.array(initial_state)

    for _ in range(iterations):
        grad = mx.grad(func)(x)
        x = optimizer.apply_gradients({"o": grad}, {"o": x})["o"]

    min_loc = mx.array(min_loc)
    error = mx.sqrt(mx.sum((x - min_loc) ** 2)).item()
    assert mx.allclose(x, min_loc, atol=0.001), f"x=({x[0]:.4f},{x[1]:.4f}), {error=:.4f}"

    name = optimizer.__class__.__name__
    assert name in optimizer.__repr__()
