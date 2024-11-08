import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from hyperopt import fmin, hp, tpe
from mlx.optimizers import SGD, Adam, AdamW, clip_grad_norm

import mlx_optimizers as optim

# CONSTANTS
ROSENBROCK_INITIAL = mx.array((-2.0, 2.0))
ROSENBROCK_MINIMUM = mx.array((1.0, 1.0))


def rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def execute_steps(func, initial_state, optimizer, n_steps):
    xy_t = mx.array(initial_state)
    steps = mx.zeros((n_steps + 1, 2))
    steps[0, :] = xy_t
    for i in range(1, n_steps + 1):
        grad = mx.grad(func)(xy_t)
        grad, _ = clip_grad_norm(grad, 1.0)
        xy_t = optimizer.apply_gradients({"o": grad}, {"o": xy_t})["o"]
        steps[i, :] = xy_t
    return steps


def objective_rosenbrock(params):
    optimizer_config = {"learning_rate": params["learning_rate"]}
    optimizer_config.update(params.get("optimizer_kwargs", {}))
    steps = execute_steps(
        rosenbrock, ROSENBROCK_INITIAL, params["optimizer"](**optimizer_config), params["steps"]
    )
    return (
        (steps[-1, 0] - ROSENBROCK_MINIMUM[0]) ** 2 + (steps[-1, 1] - ROSENBROCK_MINIMUM[1]) ** 2
    ).item()


def plot_rosenbrock(steps, optimizer_name, lr):
    X, Y = mx.meshgrid(mx.linspace(-2.5, 2, 250), mx.linspace(-1, 3, 250))
    Z = rosenbrock([X, Y])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.contour(X, Y, Z, levels=90, cmap="inferno", linewidths=0.5)
    ax.contourf(X, Y, Z, levels=90, cmap="binary")
    ax.plot(steps[:, 0], steps[:, 1], color="b", marker="x", lw=1, ms=3)
    ax.plot(steps[-1, 0], steps[-1, 1], color="r", marker="o", ms=6)
    ax.plot(ROSENBROCK_MINIMUM[0], ROSENBROCK_MINIMUM[1], c="k", marker="o", ms=6)
    ax.set_title(f"{optimizer_name} (lr={lr:.6}, n={len(steps)})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.tight_layout()
    fig.savefig(f"../docs/media/rosenbrock_{optimizer_name}.png", dpi=300, bbox_inches="tight")


def execute_experiments(optimizers, objective, func, plot_func, initial_state):
    for optimizer, lr_low, lr_hi, optimizer_kwargs in optimizers:
        space = {
            "optimizer": hp.choice("optimizer", [optimizer]),
            "learning_rate": hp.uniform("learning_rate", lr_low, lr_hi),
            "steps": 100,
            "optimizer_kwargs": optimizer_kwargs,  # additional kwargs
        }
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            rstate=np.random.default_rng(42),
        )
        if best:
            print(f"{optimizer.__name__}: {best['learning_rate']}")
            optimizer_instance = optimizer(learning_rate=best["learning_rate"], **optimizer_kwargs)
            steps = execute_steps(func, initial_state, optimizer_instance, space["steps"])
            plot_func(steps, optimizer.__name__, best["learning_rate"])


if __name__ == "__main__":
    optimizers = [
        # default
        (Adam, -1, 0.3, {}),
        (SGD, -1, 0.3, {"momentum": 0.9, "nesterov": True}),
        (AdamW, -1, 0.3, {}),
        # custom
        (optim.QHAdam, 0.01, 0.5, {}),
        (optim.DiffGrad, 0.01, 0.5, {}),
    ]
    execute_experiments(
        optimizers, objective_rosenbrock, rosenbrock, plot_rosenbrock, ROSENBROCK_INITIAL
    )
