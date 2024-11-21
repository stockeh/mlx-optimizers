from functools import partial

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import mlx_optimizers as optim

from .common import MLP

optimizers = [
    optim.QHAdam,
    optim.DiffGrad,
    optim.MADGRAD,
    optim.ADOPT,
    optim.Muon,
    optim.Lamb,
    optim.Kron,
    optim.MARS,
    # optim.Shampoo,
]


@pytest.mark.parametrize("optclass", optimizers)
def test_init_from_state(optclass):
    model = MLP()
    optimizer = optclass(learning_rate=3e-4)
    optimizer.init(model.trainable_parameters())

    # Flatten the state for serialization
    state = tree_flatten(optimizer.state)

    # Make a new optimizer and load the state
    optimizer = optclass(learning_rate=3e-4)
    optimizer.state = tree_unflatten(state)

    # This should work without any errors
    grads = model.trainable_parameters()
    optimizer.update(model, grads)


@pytest.mark.parametrize("optclass", optimizers)
def test_compiled_optimizer(optclass):
    model = nn.Linear(10, 10)
    x = mx.random.uniform(shape=(2, 10))

    mx.random.seed(42)
    optimizer = optclass(learning_rate=1e-2)

    orig_params = model.parameters()

    def uncompiled_loss(model, x):
        return model(x).sum()

    # Uncompiled version
    def uncompiled_step(x):
        _, grad = nn.value_and_grad(model, uncompiled_loss)(model, x)
        optimizer.update(model, grad)

    uncompiled_step(x)
    uncompiled_params = model.parameters()

    # Pure version
    def pure_loss(params, x):
        model.update(params)
        return model(x).sum()

    model.update(orig_params)
    mx.random.seed(42)
    optimizer = optclass(learning_rate=1e-2)

    @mx.compile
    def pure_step(params, opt_state, x):  # type: ignore
        grad = mx.grad(pure_loss)(params, x)
        optimizer.state = opt_state
        params = optimizer.apply_gradients(grad, params)
        return params, optimizer.state

    optimizer.init(model.parameters())
    pure_params, _ = pure_step(model.parameters(), optimizer.state, x)
    assert mx.allclose(pure_params["weight"], uncompiled_params["weight"])  # type: ignore
    assert mx.allclose(pure_params["bias"], uncompiled_params["bias"])  # type: ignore

    # Impure version
    def impure_loss(model, x):
        return model(x).sum()

    model.update(orig_params)
    mx.random.seed(42)
    optimizer = optclass(learning_rate=1e-2)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def impure_step(x):
        _, grad = nn.value_and_grad(model, impure_loss)(model, x)
        optimizer.update(model, grad)

    impure_step(x)
    impure_params = model.parameters()
    assert mx.allclose(impure_params["weight"], uncompiled_params["weight"])  # type: ignore
    assert mx.allclose(impure_params["bias"], uncompiled_params["bias"])  # type: ignore
