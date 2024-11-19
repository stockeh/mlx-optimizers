from typing import List

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


def plot_rosenbrock(steps: List[mx.array], name: str, title: str = "", labels: List[str] = []):
    X, Y = mx.meshgrid(mx.linspace(-2.5, 2, 250), mx.linspace(-1, 3, 250))
    Z = rosenbrock([X, Y])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.contour(X, Y, Z, levels=90, cmap="inferno", linewidths=0.5)
    ax.contourf(X, Y, Z, levels=90, cmap="binary")

    c1 = "b" if len(steps) == 1 else None
    for i, st in enumerate(steps):
        m = ax.plot(
            st[:, 0],
            st[:, 1],
            color=c1,
            marker="x",
            lw=1,
            ms=3,
            label=labels[i] if labels else None,
        )
        ax.plot(st[-1, 0], st[-1, 1], marker="o", ms=4, color=m[0].get_color())
    ax.plot(ROSENBROCK_MINIMUM[0], ROSENBROCK_MINIMUM[1], c="k", marker="o", ms=6)

    if labels:
        ax.legend(fontsize=8, ncols=2, loc="upper right", columnspacing=0.8)
    if title:
        ax.set_title(f"{name} {title}")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.tight_layout()
    fig.savefig(f"../docs/src/_static/media/rosenbrock_{name}.png", dpi=300, bbox_inches="tight")


def execute_experiments(optimizers, objective, func, plot_func, initial_state):
    all_steps, all_labels = [], []
    for optimizer, lr_low, lr_hi, optimizer_kwargs in optimizers:
        space = {
            "optimizer": hp.choice("optimizer", [optimizer]),
            "learning_rate": hp.uniform("learning_rate", lr_low, lr_hi),
            "steps": 75,
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
            name = optimizer.__name__
            print(f"{name}: lr={best['learning_rate']:.6f}")
            optimizer_instance = optimizer(learning_rate=best["learning_rate"], **optimizer_kwargs)
            steps = execute_steps(func, initial_state, optimizer_instance, space["steps"])
            plot_func(
                [steps],
                name=name,
                title=f'(lr={best["learning_rate"]:.4f}, n={space["steps"]})',
            )
            if name not in ["AdamW", "SGD"]:
                all_steps.append(steps)
                all_labels.append(name)

    # plot_func(all_steps, name="all", labels=all_labels)


if __name__ == "__main__":
    mx.random.seed(42)
    optimizers = [
        # default
        (Adam, 0, 0.2, {}),
        (SGD, 0, 0.2, {"momentum": 0.9, "nesterov": True}),
        (AdamW, 0, 0.2, {}),
        # custom
        (optim.QHAdam, 0, 0.5, {}),
        (optim.DiffGrad, 0, 0.5, {}),
        (optim.MADGRAD, 0, 0.5, {}),
        (optim.ADOPT, 0, 0.25, {}),
        (optim.Lamb, 0, 0.25, {}),
        (optim.Muon, 0, 0.2, {"alternate_optimizer": AdamW(learning_rate=0.0842)}),  # fixed lr
        (optim.Shampoo, 0, 2, {}),
        (optim.Kron, 0, 0.5, {}),
        (optim.MARS, 0, 0.8, {"optimize_1d": True, "mars_type": "mars-adamw"}),
    ]
    execute_experiments(
        optimizers, objective_rosenbrock, rosenbrock, plot_rosenbrock, ROSENBROCK_INITIAL
    )
