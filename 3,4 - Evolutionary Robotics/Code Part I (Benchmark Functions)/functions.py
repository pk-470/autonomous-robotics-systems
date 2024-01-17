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
