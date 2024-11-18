from functools import partial

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import mlx_optimizers as optim
from tests import MLP

optimizers = [
    optim.QHAdam,
    optim.DiffGrad,
    optim.MADGRAD,
    optim.ADOPT,
    optim.Muon,
    optim.Lamb,
    optim.Kron,
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

    def loss(model, x):  # type: ignore
        return model(x).sum()

    # Uncompiled version
    def step(x):  # type: ignore
        _, grad = nn.value_and_grad(model, loss)(model, x)
        optimizer.update(model, grad)

    step(x)
    print(optimizer.state)
    uncompiled_params = model.parameters()

    # Pure version
    def loss(params, x):  # type: ignore
        model.update(params)
        return model(x).sum()

    model.update(orig_params)
    mx.random.seed(42)
    optimizer = optclass(learning_rate=1e-2)

    @mx.compile
    def step(params, opt_state, x):  # type: ignore
        grad = mx.grad(loss)(params, x)
        optimizer.state = opt_state
        params = optimizer.apply_gradients(grad, params)
        return params, optimizer.state

    optimizer.init(model.parameters())
    pure_params, _ = step(model.parameters(), optimizer.state, x)
    assert mx.allclose(pure_params["weight"], uncompiled_params["weight"])  # type: ignore
    assert mx.allclose(pure_params["bias"], uncompiled_params["bias"])  # type: ignore

    # Impure version
    def loss(model, x):
        return model(x).sum()

    model.update(orig_params)
    mx.random.seed(42)
    optimizer = optclass(learning_rate=1e-2)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x):
        _, grad = nn.value_and_grad(model, loss)(model, x)
        optimizer.update(model, grad)

    step(x)
    print(optimizer.state)
    impure_params = model.parameters()
    assert mx.allclose(impure_params["weight"], uncompiled_params["weight"])  # type: ignore
    assert mx.allclose(impure_params["bias"], uncompiled_params["bias"])  # type: ignore
