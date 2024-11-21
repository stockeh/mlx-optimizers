import argparse

import matplotlib.pyplot as plt
import mlx.core as mx
from datasets import cifar10, mnist
from manager import Manager
from mlx.optimizers import Adam, AdamW, cosine_decay, join_schedules, linear_schedule
from models import Network

import mlx_optimizers as optim

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="dataset to use"
)
parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def get_cosine_schedule(max_lr, min_lr, n_warmup, decay_steps):
    learning_rate = join_schedules(
        [linear_schedule(min_lr, max_lr, n_warmup), cosine_decay(max_lr, decay_steps, min_lr)],
        [n_warmup],
    )
    return learning_rate


def get_optimizers(args):
    total_steps = 50_000 // args.batch_size * args.epochs
    n_warmup = int(total_steps * 0.10)  # % of total steps
    decay_steps = total_steps - n_warmup
    weight_decay = 1e-4
    learning_rate = get_cosine_schedule(6e-4, 1e-6, n_warmup, decay_steps)
    optimizers = [
        (
            Adam,
            {
                "learning_rate": learning_rate,
            },
        ),
        (
            AdamW,
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
        ),
        (
            optim.ADOPT,
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
        ),
        (
            optim.MARS,
            {
                "learning_rate": get_cosine_schedule(3e-3, 1e-6, n_warmup, decay_steps),
                "weight_decay": weight_decay,
                "learning_rate_1d": get_cosine_schedule(3e-3, 1e-6, n_warmup, decay_steps),
                "weight_decay_1d": weight_decay,
            },
        ),
    ]

    return optimizers


def plot_results(results, optimizers, args):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = ["#74add1", "#1730bd", "#1a9850", "#001c01"]

    for i, acc in enumerate(results):
        ax.plot(range(1, len(acc) + 1), acc, label=optimizers[i][0].__name__, lw=2, color=colors[i])

    ax.set_title(f"{args.dataset.upper()} (val)", loc="left")
    ax.set_xlabel("Epoch", fontsize="medium")
    ax.set_ylabel("Accuracy (%)", fontsize="medium")

    ax.legend(ncols=2, columnspacing=0.8, fontsize="medium")
    ax.grid(alpha=0.2)

    ax.set_ylim(90 if args.dataset == "mnist" else 70)
    acc_min, acc_max = ax.get_ylim()
    ax.set_yticks(mx.linspace(acc_min, acc_max, 5, dtype=mx.int8))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(
        f"../../docs/src/_static/media/compare-{args.dataset}-blank.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main(args):
    mx.random.seed(args.seed)
    if args.dataset == "mnist":
        train_data, test_data = mnist(args.batch_size)
    elif args.dataset == "cifar10":
        train_data, test_data = cifar10(args.batch_size)
    else:
        raise NotImplementedError(f"{args.dataset=} is not implemented.")
    x_shape = next(train_data)["image"].shape
    train_data.reset()

    model_config = {
        "n_inputs": x_shape[1:],
        "conv_layers_list": [
            {"filters": 32, "kernel_size": 3, "repeat": 2, "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "repeat": 2, "batch_norm": True},
            {"filters": 128, "kernel_size": 3, "repeat": 2, "batch_norm": True},
        ],
        "n_hiddens_list": [512],
        "n_outputs": 10,
        "dropout": 0.2,
    }

    optimizers = get_optimizers(args)

    results = []
    for optimizer, optimizer_kwargs in optimizers:
        mx.random.seed(args.seed)
        manager = Manager(Network(**model_config), optimizer(**optimizer_kwargs))  # type: ignore
        manager.train(train_data, val=test_data, epochs=args.epochs)
        results.append(100 * mx.array(manager.val_acc_trace))

    plot_results(results, optimizers, args)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.Device(mx.cpu))
    main(args)
