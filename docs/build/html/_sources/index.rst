mlx-optimizers - |version| documentation
========================================

A library to experiment with new optimization algorithms in `MLX <https://github.com/ml-explore/mlx>`_. 

* **Diverse Exploration**: includes proven and experimental optimizers like DiffGrad, QHAdam, and others.
* **Easy Integration**: compatible with MLX for straightforward experimentation and downstream adoption.
* **Benchmark Examples**: enables quick testing on classic optimization and machine learning tasks.

See a **full list** of optimizers in the :doc:`API Reference <optimizers>`.

.. code-block:: python
   :caption: Example Usage

      import mlx_optimizers as optim

      #... model, grads, etc.
      optimizer = optim.DiffGrad(learning_rate=0.001)
      optimizer.update(model, grads)

.. image:: https://github.com/user-attachments/assets/2ff6430a-dfad-4879-ae39-ec76e2645f21
   :alt: graph
   :align: center
   :class: desktop-width only-dark

.. image:: https://github.com/user-attachments/assets/111f79f2-257e-4a27-8667-3831d233c3b8
   :alt: graph
   :align: center
   :class: desktop-width only-light

.. toctree::
   :caption: Install
   :maxdepth: 1

   install

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   optimizers

.. toctree::
   :caption: Development
   :maxdepth: 1

   License <https://raw.githubusercontent.com/stockeh/mlx-optimizers/refs/heads/main/LICENSE>
   GitHub Repository <https://github.com/stockeh/mlx-optimizers>