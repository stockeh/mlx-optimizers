import argparse
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from datasets import cifar10, mnist
from mlx.optimizers import Optimizer, clip_grad_norm
from models import Network
from tqdm import tqdm

import mlx_optimizers as optim

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="dataset to use"
)
parser.add_argument("-b", "--batch_size", type=int, default=256, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=25, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


class Manager:
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

        self.train_error_trace: list[float] = []
        self.train_acc_trace: list[float] = []
        self.val_error_trace: list[float] = []
        self.val_acc_trace: list[float] = []

    def eval_fn(self, X, T):
        Y = self.model(X)
        loss = nn.losses.cross_entropy(Y, T, reduction="mean")
        correct = mx.sum(mx.argmax(Y, axis=1) == T)
        return loss, correct

    def train(self, train, val=None, epochs: int = 10):
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(X, T):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            (loss, correct), grads = train_step_fn(X, T)
            grads, _ = clip_grad_norm(grads, max_norm=1.0)
            self.optimizer.update(self.model, grads)
            return loss, correct

        epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        for _ in epoch_bar:
            self.model.train()
            train.reset()
            total_loss, total_correct, n = 0, 0, 0
            for batch in train:
                X, T = mx.array(batch["image"]), mx.array(batch["label"])
                loss, correct = step(X, T)
                mx.eval(state)

                total_loss += loss.item() * X.shape[0]
                total_correct += int(correct)
                n += X.shape[0]

            avg_train_loss = total_loss / n
            avg_train_acc = total_correct / n

            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)

            postfix = {"train_loss": f"{avg_train_loss:.3f}", "train_acc": f"{avg_train_acc:.3f}"}

            if val is not None:  # eval on validation data
                avg_val_loss, avg_val_acc = self.evaluate(val)
                self.val_error_trace.append(avg_val_loss)
                self.val_acc_trace.append(avg_val_acc)
                postfix.update({"val_loss": f"{avg_val_loss:.3f}", "val_acc": f"{avg_val_acc:.3f}"})

            epoch_bar.set_postfix(postfix)

    def evaluate(self, test):
        self.model.eval()
        test.reset()
        total_loss, total_correct, n = 0, 0, 0
        for batch in test:
            X, T = mx.array(batch["image"]), mx.array(batch["label"])
            loss, correct = self.eval_fn(X, T)

            total_loss += loss.item() * X.shape[0]
            total_correct += int(correct)
            n += X.shape[0]

        avg_loss = total_loss / n
        avg_acc = total_correct / n

        return avg_loss, avg_acc


def main(args):
    mx.random.seed(args.seed)

    if args.dataset == "mnist":
        train_data, test_data = mnist(args.batch_size)
    elif args.dataset == "cifar10":
        train_data, test_data = cifar10(args.batch_size)
    else:
        raise NotImplementedError(f"{args.dataset=} is not implemented.")
    n_inputs = next(train_data)["image"].shape[1:]
    train_data.reset()

    model_config = {
        "n_inputs": n_inputs,
        "conv_layers_list": [{"filters": 16, "kernel_size": 3}, {"filters": 32, "kernel_size": 3}],
        "n_hiddens_list": [128],
        "n_outputs": 10,
    }

    model = Network(**model_config)
    optimizer = optim.DiffGrad(learning_rate=args.lr)

    manager = Manager(model, optimizer)
    manager.train(train_data, val=test_data, epochs=args.epochs)

    #! plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    lw = 2
    ax.plot(manager.train_error_trace, label="train", color="#1b9e77", lw=lw)
    ax.plot(manager.val_error_trace, label="val", color="#d95f02", lw=lw)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.Device(mx.cpu))
    main(args)
