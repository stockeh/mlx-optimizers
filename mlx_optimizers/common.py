import mlx.core as mx


@mx.compile
def newton_schulz(G, steps=5, eps=1e-7) -> mx.array:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(mx.bfloat16) / (mx.linalg.norm(G) + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.astype(G.dtype)


def numel(x: mx.array) -> int:
    return int(mx.prod(mx.array(x.shape)))
