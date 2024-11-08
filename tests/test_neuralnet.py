from functools import partial

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_optimizers as optim


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


class MLP(nn.Module):
    def __init__(self, n_inputs: int = 2, n_hiddens_list: list = [10, 10], n_outputs: int = 2):
        super().__init__()
        assert len(n_hiddens_list) > 0 and all(n > 0 for n in n_hiddens_list)
        activation = nn.ReLU
        self.layers = [
            layer
            for ni, no in zip([n_inputs] + n_hiddens_list, n_hiddens_list)
            for layer in (nn.Linear(ni, no), activation())
        ] + [nn.Linear(n_hiddens_list[-1], n_outputs)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def ids(v):
    return f"{v[0].__name__, } {v[1:]}"


optimizers = [
    (optim.QHAdam, {"learning_rate": 0.01}, 500),
    (optim.DiffGrad, {"learning_rate": 0.01}, 500),
    (optim.Muon, {"learning_rate": 0.01}, 500),
    (optim.MADGRAD, {"learning_rate": 0.01}, 500),
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
    assert losses[0] > 2 * losses[-1], f"loss={losses[-1]:.5f}, {acc=:.3f}"
