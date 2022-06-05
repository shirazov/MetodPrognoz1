import numpy as np
from matrix_np_api import *


def bubble_max_row(m, k):
    """Replace m[col] row with the one of the underlying rows with the modulo greatest first element.
    :param m: numpy matrix
    :param k: number of the step in forward trace
    :return: None. Function changes the matrix structure.
    """
    ind = k + np.argmax(np.abs(m[k:, k]))
    if ind != k:
        m[k, :], m[ind, :] = np.copy(m[ind, :]), np.copy(m[k, :])


def solve_gauss(m):
    """Solve linear equations system with gaussian method.
    :param m: numpy matrix
    :return: None
    """
    n = m.shape[0]
    # forward trace
    for k in range(n - 1):
        bubble_max_row(m, k)
        for i in range(k + 1, n):
            # modify row
            frac = m[i, k] / m[k, k]
            m[i, :] -= m[k, :] * frac

    # check modified system for nonsingularity
    if is_singular(m):
        print('The system has infinite number of answers...')
        return

    # backward trace
    x = np.matrix([0.0 for i in range(n)]).T
    for k in range(n - 1, -1, -1):
        x[k, 0] = (m[k, -1] - m[k, k:n] * x[k:n, 0]) / m[k, k]

    # Display results
    display_results(x)


def is_singular(m):
    """Check matrix for nonsingularity.
    :param m: matrix (list of lists)
    :return: True if system is nonsingular
    """
    return np.any(np.diag(m) == 0)


m = read_matrix('matrix10x10.txt')
solve_gauss(m)