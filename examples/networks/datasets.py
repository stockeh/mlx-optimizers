import numpy as np
from mlx.data.datasets import load_cifar10, load_mnist


def mnist(batch_size, img_size=(28, 28), root=None):
    def normalize(x):  # normalize to [0,1]
        return x.astype("float32") / 255.0

    full_batch = batch_size == -1

    # iterator over training set
    train = load_mnist(root=root, train=True)
    tr_iter = (
        train.shuffle()
        .to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(len(train) if full_batch else batch_size)
    )
    if not full_batch:
        tr_iter = tr_iter.prefetch(4, 4)

    # iterator over test set
    test = load_mnist(root=root, train=False)
    test_iter = (
        test.to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(len(test) if full_batch else batch_size)
    )
    return tr_iter, test_iter


def cifar10(batch_size, img_size=(32, 32), root=None):
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 3))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((1, 1, 3))

    def normalize(x):  # z-score normalize
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    full_batch = batch_size == -1

    # iterator over training set
    train = load_cifar10(root=root, train=True)
    tr_iter = (
        train.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(len(train) if full_batch else batch_size)
    )
    if not full_batch:
        tr_iter = tr_iter.prefetch(4, 4)

    # iterator over test set
    test = load_cifar10(root=root, train=False)
    test_iter = (
        test.to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(len(test) if full_batch else batch_size)
    )

    return tr_iter, test_iter


if __name__ == "__main__":
    batch_size, img_size = 32, (28, 28)
    tr_iter, test_iter = mnist(batch_size=batch_size, img_size=img_size)

    B, H, W, C = batch_size, img_size[0], img_size[1], 1
    print(f"Batch size: {B}, Channels: {C}, Height: {H}, Width: {W}")

    batch_tr_iter = next(tr_iter)
    assert batch_tr_iter["image"].shape == (B, H, W, C), "Wrong training set size"
    assert batch_tr_iter["label"].shape == (batch_size,), "Wrong training set size"

    batch_test_iter = next(test_iter)
    assert batch_test_iter["image"].shape == (B, H, W, C), "Wrong training set size"
    assert batch_test_iter["label"].shape == (batch_size,), "Wrong training set size"
