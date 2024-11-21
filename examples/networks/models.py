import importlib
from typing import List, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_activation(activation_f: str) -> Type:
    package_name = "mlx.nn.layers.activations"
    module = importlib.import_module(package_name)

    activations = [getattr(module, attr) for attr in dir(module)]
    activations = [
        cls for cls in activations if isinstance(cls, type) and issubclass(cls, nn.Module)
    ]
    names = [cls.__name__.lower() for cls in activations]

    try:
        index = names.index(activation_f.lower())
        return activations[index]
    except ValueError:
        raise NotImplementedError(f"get_activation: {activation_f=} is not yet implemented.")


def compute_padding(
    input_size: tuple, kernel_size: int | tuple, stride: int | tuple = 1, dilation: int | tuple = 1
) -> Tuple[int, int]:
    if len(input_size) == 2:
        input_size = (*input_size, 1)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    input_h, input_w, _ = input_size
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # Compute the effective kernel size after dilation
    effective_kernel_h = (kernel_h - 1) * dilation_h + 1
    effective_kernel_w = (kernel_w - 1) * dilation_w + 1

    # Compute the padding needed for same convolution
    pad_h = int(max((input_h - 1) * stride_h + effective_kernel_h - input_h, 0))
    pad_w = int(max((input_w - 1) * stride_w + effective_kernel_w - input_w, 0))

    # Compute the padding for each side
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return (pad_top, pad_left)


class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @property
    def num_params(self):
        return sum(x.size for k, x in tree_flatten(self.parameters()))

    @property
    def shapes(self):
        return tree_map(lambda x: x.shape, self.parameters())

    def summary(self):
        print(self)
        print(f"Number of parameters: {self.num_params}")

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError("Subclass must implement this method")


class MLP(Base):
    def __init__(
        self,
        n_inputs: int,
        n_hiddens_list: Union[List, int],
        n_outputs: int,
        activation_f: str = "tanh",
    ):
        super().__init__()

        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]

        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        activation = get_activation(activation_f)

        self.layers = []
        ni = n_inputs
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                self.layers.append(nn.Linear(ni, n_units))
                self.layers.append(activation())
                ni = n_units
        self.layers.append(nn.Linear(ni, n_outputs))

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x


class Network(Base):
    """Fully Connected / Convolutional Neural Network

    Args:
        n_inputs (Union[List[int], Tuple[int], mx.array]): Input shape
        n_outputs (int): Number of output classes
        conv_layers_list (List[dict], optional): List of convolutional layers. Defaults to [].
        n_hiddens_list (Union[List, int], optional): List of hidden units. Defaults to 0.
        activation_f (str, optional): Activation function. Defaults to "relu".
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    conv_layers_list dict keys:
        filters: int
        kernel_size: int
        stride: int
        dilation: int
        padding: int
        bias: bool
        batch_norm: bool
        repeat: int

    """

    def __init__(
        self,
        n_inputs: Union[List[int], Tuple[int], mx.array],
        n_outputs: int,
        conv_layers_list: List[dict] = [],
        n_hiddens_list: Union[List, int] = 0,
        activation_f: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]

        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        activation = get_activation(activation_f)

        ni = mx.array(n_inputs)
        self.conv = []
        if conv_layers_list:
            for conv_layer in conv_layers_list:
                n_channels = int(ni[-1])

                padding = conv_layer.get(
                    "padding",
                    compute_padding(  # same padding
                        tuple(ni),
                        conv_layer["kernel_size"],
                        conv_layer.get("stride", 1),
                        conv_layer.get("dilation", 1),
                    ),
                )

                self.conv.extend(
                    [
                        layer
                        for i in range(conv_layer.get("repeat", 1))
                        for layer in [
                            nn.Conv2d(
                                n_channels if i == 0 else conv_layer["filters"],
                                conv_layer["filters"],
                                conv_layer["kernel_size"],
                                stride=conv_layer.get("stride", 1),
                                padding=padding,
                                dilation=conv_layer.get("dilation", 1),
                                bias=conv_layer.get("bias", True),
                            ),
                            activation(),
                        ]
                        + (
                            [nn.BatchNorm(conv_layer["filters"])]
                            if conv_layer.get("batch_norm")
                            else []
                        )
                    ]
                    + [nn.MaxPool2d(2, stride=2)]
                )
                if dropout > 0:
                    self.conv.append(nn.Dropout(dropout))
                ni = mx.concatenate([ni[:-1] // 2, mx.array([conv_layer["filters"]])])

        ni = int(mx.prod(ni))
        self.fcn = []
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                self.fcn.append(nn.Linear(ni, n_units))
                self.fcn.append(activation())
                if dropout > 0:
                    self.fcn.append(nn.Dropout(dropout))
                ni = n_units
        self.output = nn.Linear(ni, n_outputs)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.conv:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.fcn:
            x = layer(x)
        return self.output(x)


if __name__ == "__main__":
    x = mx.random.normal(shape=(4, 10))
    model = MLP(n_inputs=x.shape[1], n_hiddens_list=[10, 10], n_outputs=2, activation_f="relu")
    model.summary()
    print("mlp output shape:", model(x).shape)

    x = mx.random.normal(shape=(4, 32, 32, 3))
    model = Network(
        n_inputs=x.shape[1:],
        n_outputs=10,
        conv_layers_list=[{"filters": 8, "kernel_size": 3}, {"filters": 16, "kernel_size": 3}],
        n_hiddens_list=[32, 10],
        activation_f="gelu",
    )
    model.summary()
    print("cnn output shape:", model(x).shape)
