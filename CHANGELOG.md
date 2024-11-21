# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- optim: MARS
- optim: common file for repeated ops (e.g., newton_schulz)
- examples: compare optimizers on mnist/cifar10 

### Fixed
- optim: ADOPT correctly implemented (exp_avg_sq and g_0)

## [v0.3.0](https://github.com/stockeh/mlx-optimizers/releases/tag/v0.3.0) - 2024-11-18

### Added
- optim: Kron PSGD
- tests: functional compilation (uncompiled, pure, and impure) and init from state

### Fixed
- optim: Lamb, closer to algorithm with updated scaling function

## [v0.2.0](https://github.com/stockeh/mlx-optimizers/releases/tag/v0.2.0) - 2024-11-11

### Added
- optim: Shampoo (general tensor case)

### Fixed
- optim: docstrings for exisiting methods

## [v0.1.0](https://github.com/stockeh/mlx-optimizers/releases/tag/v0.1.0) - 2024-11-09

### Added
- optim: QHAdam, Muon, MADGRAD, Lamb, DiffGrad
- tests: rosenbrock, quadratic, beale
- tests: neural network loss comparison