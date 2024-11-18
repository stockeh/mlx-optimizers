import random
import string
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


def flat_exponential_schedule(
    max_prob: float = 1.0,
    min_prob: float = 0.03,
    decay: float = 0.001,
    flat_start: float = 250,
) -> Callable:
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def schedule(step):
        """Exponential anneal with flat start."""
        step = mx.array(step, dtype=mx.float32)
        prob = max_prob * mx.exp(-decay * (step - flat_start))
        prob = mx.clip(prob, min_prob, max_prob)
        return prob

    return schedule


class Kron(Optimizer):
    """Kronecker-Factored Preconditioned Stochastic Gradient Descent [1].

    PSGD is a second-order optimizer that uses Hessian- or whitening-based :math:`(gg^T)`
    preconditioners and Lie groups to improve convergence. Kron uses Kronecker-factored
    preconditioners for tensors of any number dimension.

    [1] Xi-Lin Li, 2015. Preconditioned Stochastic Gradient Descent.
    https://arxiv.org/abs/1512.04202
    https://github.com/lixilinx/psgd_torch

    Args:
        learning_rate (float or callable): the learning rate.
        b1 (float, optional): coefficient used for computing running averages of the
            gradient. Default: ``0.9``
        weight_decay (float, optional): weight decay factor. Default: ``0.0``
        precond_update_prob (float or callable, optional): probability of updating the
            preconditioner. Default: ``None`` (flat exponential schedule)
        max_size_triangular (int, optional):  maximum size for dim's preconditioner to be
            triangular. Default: ``8192``
        min_ndim_triangular (int, optional): minimum number of dimensions a layer needs
            to have triangular preconditioners. Default: ``2``
        memory_save_mode (str, optional): (None, 'one_diag', or 'all_diag'). None: set all
            preconditioners to be triangular, 'one_diag': sets the largest or last dim to
            be diagonal per layer, and 'all_diag': sets all preconditioners to be diagonal.
            Default: ``None``
        momentum_into_precond_update (bool, optional): whether to use momentum in
            preconditioner update. Default: ``True``
    ..
    """

    rng = random.Random(5318008)

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        b1: float = 0.9,
        weight_decay: float = 0.0,
        precond_update_prob: Optional[Union[float, Callable[[mx.array], mx.array]]] = None,
        max_size_triangular: int = 8192,
        min_ndim_triangular: int = 2,
        memory_save_mode: Optional[str] = None,
        momentum_into_precond_update: bool = True,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.b1 = b1
        self.weight_decay = weight_decay
        self.precond_update_prob = precond_update_prob

        self.max_size_triangular = max_size_triangular
        self.min_ndim_triangular = min_ndim_triangular
        self.memory_save_mode = memory_save_mode
        self.momentum_into_precond_update = momentum_into_precond_update
        self.precond_learning_rate = 0.1  # hardcode
        self.precond_init_scale = 1.0  # hardcode

        self._tiny = 1e-8
        self.trust_region = lambda x: 0.1 * mx.sign(x) * mx.log(mx.abs(x) + 1) + 0.9 * mx.tanh(x)

        self._maybe_schedule("do_update_balance", self._do_update_balance)

    def _do_update_balance(self, step):
        precond_update_prob = (
            flat_exponential_schedule()
            if self.precond_update_prob is None
            else self.precond_update_prob
        )
        if callable(precond_update_prob):
            precond_update_prob = precond_update_prob(step)  # type: ignore
        do_update = self.rng.random() < precond_update_prob  # type: ignore
        balance = self.rng.random() < 0.01 and do_update
        return mx.array((do_update, balance))

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["momentum_buffer"] = mx.zeros_like(parameter)
        state["Q"], state["exprs"] = init_Q_exprs(
            parameter,
            self.precond_init_scale,
            self.max_size_triangular,
            self.min_ndim_triangular,
            self.memory_save_mode,
        )

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs a single optimization step, updating :math:`m` and :math:`v`"""
        lr = self.learning_rate.astype(gradient.dtype)
        b1 = self.b1

        state["momentum_buffer"] = b1 * state["momentum_buffer"] + (1 - b1) * gradient
        debiased_momentum = state["momentum_buffer"] / (1 - b1 ** self.state["step"])
        do_update, balance = self.state["do_update_balance"]

        if gradient.ndim > 1:
            for i, (balanced, original) in enumerate(zip(_balance_Q(state["Q"]), state["Q"])):
                state["Q"][i] = mx.where(balance, balanced, original)

        for i, (balanced, original) in enumerate(
            zip(
                _update_precond(
                    state["Q"],
                    state["exprs"],
                    mx.random.normal(debiased_momentum.shape, dtype=debiased_momentum.dtype),
                    debiased_momentum if self.momentum_into_precond_update else gradient,
                    self.precond_learning_rate,
                    self._tiny,
                ),
                state["Q"],
            )
        ):
            state["Q"][i] = mx.where(do_update, balanced, original)

        pre_grad = _precond_grad(state["Q"], state["exprs"], debiased_momentum)
        pre_grad = mx.clip(self.trust_region(pre_grad / 1.5) * 1.5, -2, 2)

        if self.weight_decay != 0 and parameter.ndim > 1:
            pre_grad = pre_grad + self.weight_decay * parameter

        return parameter - lr * pre_grad


def init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * mx.ones_like(t)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = mx.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[int(rev_sorted_dims[0])] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'one_diag', 'all_diag']"
            )

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * mx.ones(size, dtype=dtype))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * mx.eye(size, dtype=dtype))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                )
                piece2 = "".join(
                    [(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))]
                )
                subscripts = piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P

    return [Q, (exprA, exprGs, exprP)]


def _balance_Q(Q):
    norms = mx.stack([mx.linalg.norm(q, float("inf")) for q in Q])
    geometric_mean = norms.prod() ** (1 / len(Q))
    return [q * (geometric_mean / norms[i]) for i, q in enumerate(Q)]


@mx.compile
def _calc_A_and_conjB(exprA, G, Q, V):
    A = mx.einsum(exprA, *Q, G)
    order = G.ndim
    p = list(range(order))
    conjB = mx.transpose(mx.conj(V), p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = (
            conjB / q
            if q.ndim < 2
            # TODO: better to use `solve_triangular(q, conjB, upper=True, left=False)`
            else conjB @ mx.linalg.inv(q, stream=mx.cpu)  # type: ignore
        )
        j = order - 1
        if i < j:
            conjB = mx.swapaxes(conjB, i, j)
    return A, conjB


def _q_terms(exprGs, A, conjB):
    terms = []
    for exprG in exprGs:
        term1 = mx.einsum(exprG, A, mx.conj(A))
        term2 = mx.einsum(exprG, mx.conj(conjB), conjB)
        terms.append((term1, term2))
    return terms


def H(a):
    return mx.conj(mx.transpose(a))


def imax(a):
    return mx.max(a), mx.argmax(a)


def _lb(A, max_abs):
    A = A / max_abs
    aa = mx.real(A * mx.conj(A))
    vcol, i = imax(mx.sum(aa, axis=0))
    vrow, j = imax(mx.sum(aa, axis=1))

    return mx.where(
        vcol > vrow,
        max_abs * mx.linalg.norm(((x := mx.conj(A[:, i]) @ A) / mx.linalg.norm(x)) @ H(A)),
        max_abs * mx.linalg.norm(H(x := A @ mx.conj(A[j])) @ (x / mx.linalg.norm(x))),
    )


def _norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = mx.linalg.norm(A, float("inf"))
    return mx.where(max_abs > 0, _lb(A, max_abs), max_abs)


def _update_precond(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    A, conjB = _calc_A_and_conjB(exprA, G, Q, V)
    terms = _q_terms(exprGs, A, conjB)

    updated_Q = []
    for q, (term1, term2) in zip(Q, terms):
        tmp = term1 - term2
        tmp *= step
        if q.ndim < 2:
            tmp *= q
            tmp /= mx.linalg.norm(term1 + term2, float("inf")) + tiny
            q_updated = q - tmp
        else:
            tmp = mx.triu(tmp, k=0)
            tmp /= _norm_lower_bound(term1 + term2) + tiny
            q_updated = q - (tmp @ q)

        updated_Q.append(q_updated)

    return updated_Q


@mx.compile
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return mx.einsum(exprs[-1], *[mx.conj(q) for q in Q], *Q, G)
