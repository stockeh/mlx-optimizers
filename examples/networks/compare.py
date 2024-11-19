import argparse

import matplotlib.pyplot as plt
import mlx.core as mx
from datasets import cifar10, mnist
from manager import Manager
from mlx.optimizers import Adam, cosine_decay, join_schedules, linear_schedule
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


def get_optimizers(args):
    n_warmup = 50000 // args.batch_size * 6
    decay_steps = n_warmup // 6 * args.epochs
    max_lr = 1e-4
    min_lr = 1e-7
    learning_rate = join_schedules(
        [linear_schedule(min_lr, max_lr, n_warmup), cosine_decay(max_lr, decay_steps, min_lr)],
        [n_warmup],
    )

    optimizers = [
        (Adam, {"learning_rate": learning_rate}),
        (optim.ADOPT, {"learning_rate": learning_rate}),
        (optim.MARS, {"learning_rate": learning_rate}),
        (optim.Muon, {"learning_rate": learning_rate}),
    ]
    return optimizers


def plot_results(results, optimizers, args):
    #! plotting
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#74add1", "#4575b4", "#1a9850", "#00441b"]
    for i, r in enumerate(results):
        ax.plot(range(1, len(r) + 1), r, label=optimizers[i][0].__name__, lw=2, color=colors[i])

    ax.set_title(f"{args.dataset.upper()}", loc="left")
    ax.legend(ncols=2, columnspacing=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    num_ticks = 5
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(mx.linspace(y_min, y_max, num_ticks, dtype=mx.int8))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        f"../../docs/src/_static/media/compare-{args.dataset}-blank.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main(args):

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
            {"filters": 32, "kernel_size": 3, "repeat": 1, "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "repeat": 1, "batch_norm": True},
        ],
        "n_hiddens_list": [256],
        "n_outputs": 10,
        "dropout": 0.0,
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
