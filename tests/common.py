import mlx.nn as nn


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
