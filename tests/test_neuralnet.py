from functools import partial

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.optimizers import AdamW, clip_grad_norm

import mlx_optimizers as optim

from .common import MLP, ids


def generate_moons(n_samples: int = 100, noise: float = 0.2):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_linspace = mx.linspace(0, mx.pi, n_samples_out)
    inner_linspace = mx.linspace(0, mx.pi, n_samples_in)

    outer_circ_x = mx.cos(outer_linspace)
    outer_circ_y = mx.sin(outer_linspace)

    inner_circ_x = 1 - mx.cos(inner_linspace)
    inner_circ_y = 1 - mx.sin(inner_linspace) - 0.5

    X = mx.zeros((n_samples, 2))
    X[:n_samples_out, 0] = outer_circ_x
    X[:n_samples_out, 1] = outer_circ_y
    X[n_samples_out:, 0] = inner_circ_x
    X[n_samples_out:, 1] = inner_circ_y

    T = mx.concatenate(
        [mx.zeros(n_samples_out, dtype=mx.int16), mx.ones(n_samples_in, dtype=mx.int16)]
    )

    if noise > 0:
        X += mx.random.uniform(shape=(n_samples, 2)) * noise

    return X, T


optimizers = [
    (optim.QHAdam, {"learning_rate": 0.01}, 50),
    (optim.DiffGrad, {"learning_rate": 0.01}, 100),
    (
        optim.Muon,
        {"learning_rate": 0.01, "alternate_optimizer": AdamW(learning_rate=0.001)},
        100,
    ),
    (optim.MADGRAD, {"learning_rate": 0.01}, 50),
    (optim.ADOPT, {"learning_rate": 0.03}, 50),
    (optim.Lamb, {"learning_rate": 0.03}, 50),
    (optim.Shampoo, {"learning_rate": 0.03}, 50),
    (optim.Kron, {"learning_rate": 0.03}, 50),
    (optim.MARS, {"learning_rate": 0.03, "mars_type": "mars-adamw"}, 50),
    (optim.MARS, {"learning_rate": 0.03, "mars_type": "mars-lion"}, 50),
    (optim.MARS, {"learning_rate": 0.03, "mars_type": "mars-shampoo"}, 50),
    (optim.MARS, {"learning_rate": 0.03, "amsgrad": True}, 50),
]


@pytest.mark.parametrize("optimizer_config", optimizers, ids=ids)
def test_neuralnet(optimizer_config):
    mx.random.seed(42)

    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(**config)

    model = MLP()
    X, T = generate_moons()

    def eval_fn(X, T):
        return nn.losses.cross_entropy(model(X), T, reduction="mean")

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(X, T):
        train_step_fn = nn.value_and_grad(model, eval_fn)
        loss, grads = train_step_fn(X, T)
        grads, _ = clip_grad_norm(grads, 1)
        optimizer.update(model, grads)
        return loss

    losses = []
    for _ in range(iterations):
        loss = step(X, T)
        mx.eval(state)
        losses.append(loss.item())
        if loss < 0.2:
            break

    acc = mx.sum(mx.argmax(model(X), axis=1) == T) / T.shape[0]  # type: ignore
    assert losses[0] > 2 * losses[-1], f"Bad loss: loss={losses[-1]:.5f}, {acc=:.3f}"
    assert acc > 0.85, f"Bad Acc: loss={losses[-1]:.5f}, {acc=:.3f}"
