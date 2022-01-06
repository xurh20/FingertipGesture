from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import numpy as np

def a_dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Advanced DTW
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    # jrange = range(c)
    # for i in range(r):
    #     if not isinf(w):
    #         jrange = range(max(0, i - w), min(c, i + w + 1))
    #     for j in jrange:
    #         min_list = [D0[i, j]]
    #         for k in range(1, warp + 1):
    #             i_k = min(i + k, r)
    #             j_k = min(j + k, c)
    #             min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
    #         D1[i, j] += min(min_list)
    for i in range(r):
        for j in range(c):
            if (i == 0):
                if (j == 0):
                    D1[i][j] = C[i][j]
                else:
                    D1[i][j] = D1[i][j - 1] + C[i][j]
            elif (j == 0):
                D1[i][j] = min(C[i][j], D1[i - 1][j])
            else:
                D1[i][j] = min(D1[i][j - 1] + C[i][j], D1[i - 1][j])
    # if len(x) == 1:
    #     path = zeros(len(y)).astype(int), np.array(range(len(y))).astype(int)
    # elif len(y) == 1:
    #     path = np.array(range(len(x))).astype(int), zeros(len(x)).astype(int)
    path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > -1):
        # tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if (i > 0 and j > -1):
            if (D[i + 1][j + 1] == D[i][j + 1]):
                tb = 1
            else:
                tb = 2
        elif (i == 0):
            tb = 2
        else:
            tb = 1
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p).astype(int), array(q).astype(int)
