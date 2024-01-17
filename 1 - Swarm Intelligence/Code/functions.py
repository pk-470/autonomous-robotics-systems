import numpy as np


def rosenbrock(x, y):
    a = 0
    b = 100
    return np.square(a - x) + b * np.square(y - np.square(x))


def rastrigin(*x):
    n = len(x)
    return 10 * n + np.sum(
        [np.square(x_i) - 10 * np.cos(2 * np.pi * x_i) for x_i in x], axis=0
    )


# Gradients


def rosenbrock_grad(x, y):
    b = 100
    return np.asarray(
        (2 * x + 4 * b * x**3 - 4 * x * y * b, 2 * b * y - 2 * b * x**2)
    )


def rastrigin_grad(*x):
    return np.asarray([2 * x_i + 20 * np.pi * np.sin(2 * np.pi * x_i) for x_i in x])
