<p align="center">
  <img src="docs/src/_static/dark-mode-logo.svg#gh-dark-mode-only" alt="logo">
</p>
<p align="center">
  <img src="docs/src/_static/light-mode-logo.svg#gh-light-mode-only" alt="logo">
</p>

# 

A library to experiment with new optimization algorithms in [MLX](https://github.com/ml-explore/mlx).

```python
import mlx_optimizers as optim

#... model, grads, etc.
optimizer = optim.DiffGrad(learning_rate=0.001)
optimizer.update(model, grads)
```

Coming to pip soon! :tada: 
