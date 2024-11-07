from mlx_optimizers import hello


def test_hello():
    assert hello.simple_array().tolist() == [1, 2, 3]
