<p align="center">
  <img src="docs/src/_static/dark-mode-logo.svg#gh-dark-mode-only" alt="logo">
  <img src="docs/src/_static/light-mode-logo.svg#gh-light-mode-only" alt="logo">
</p>

# 

[**Documentation**](https://stockeh.github.io/mlx-optimizers/build/html/index.html) |
[**Install**](#install) |
[**Usage**](#usage) |
[**Examples**](#examples) |
[**Contributing**](#contributing)

[![ci](https://github.com/stockeh/mlx-optimizers/workflows/Main/badge.svg)](https://pypi.org/project/mlx-optimizers/)
[![PyPI](https://img.shields.io/pypi/v/mlx-optimizers)](https://pypi.org/project/mlx-optimizers/)

A library to experiment with new optimization algorithms in [MLX](https://github.com/ml-explore/mlx). 

- **Diverse Exploration**: includes proven and experimental optimizers like DiffGrad, QHAdam, and Muon ([docs](https://stockeh.github.io/mlx-optimizers/build/html/optimizers.html)).
- **Easy Integration**: fully compatible with MLX for straightforward experimentation and downstream adoption.
- **Benchmark Examples**: enables quick testing on classic optimization and machine learning tasks.

The design of mlx-optmizers is largely inspired by [pytorch-optmizer](https://github.com/jettify/pytorch-optimizer/tree/master).

## Install

The reccomended way to install mlx-optimizers is to install the latest stable release through [PyPi](https://pypi.org/project/mlx-optimizers/):

```bash
pip install mlx-optimizers
```

To install mlx-optimizers from source, first clone [the repository](https://github.com/stockeh/mlx-optimizers.git):

```bash
git clone https://github.com/stockeh/mlx-optimizers.git
cd mlx-optimizers
```
Then run

```bash
pip install -e .
```

## Usage

There are a variety of optimizers to choose from (see [docs](https://stockeh.github.io/mlx-optimizers/build/html/optimizers.html)). Each of these inherit the [`mx.optimizers`](https://ml-explore.github.io/mlx/build/html/python/optimizers.html) class from MLX, so the core functionality remains the same. We can simply use the optimizer as follows:

```python
import mlx_optimizers as optim

#... model, grads, etc.
optimizer = optim.DiffGrad(learning_rate=0.001)
optimizer.update(model, grads)
```

## Examples

The [examples](examples) folder offers a non-exhaustive set of demonstrative use cases for mlx-optimizers. This includes classic optimization benchmarks on the Rosenbrock function and training a simple neural net classifier on MNIST.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ff6430a-dfad-4879-ae39-ec76e2645f21#gh-dark-mode-only" alt="logo" width="45%">
  <img src="https://github.com/user-attachments/assets/505a5985-b56c-4eb5-a971-4fc345fb37ad#gh-dark-mode-only" alt="mnist" width="45%">
  <img src="https://github.com/user-attachments/assets/111f79f2-257e-4a27-8667-3831d233c3b8#gh-light-mode-only" alt="logo" width="45%">
  <img src="https://github.com/user-attachments/assets/89616bbc-e9ca-4806-882a-873b5bd8c681#gh-light-mode-only" alt="mnist" width="45%">
</p>

## Contributing

Interested in adding a new optimizer? Start with verifying it is not already implemented or in development, then open a new [feature request](https://github.com/stockeh/mlx-optimizers/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=)! If you spot a bug, please open a [bug report](https://github.com/stockeh/mlx-optimizers/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=). 

Developer? See our [contributing](.github/CONTRIBUTING.md) guide.
