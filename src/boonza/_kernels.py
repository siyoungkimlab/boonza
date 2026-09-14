"""numba kernels.  Imported lazily so ``import boonza`` does not pay for numba."""

import numpy as np
from numba import njit


@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def components(n, bi, bj):
    """Connected components of a graph; labels are numbered in order of first atom."""
    parent = np.arange(n)
    for k in range(bi.shape[0]):
        a = _find(parent, bi[k])
        b = _find(parent, bj[k])
        if a < b:
            parent[b] = a
        elif b < a:
            parent[a] = b
    labels = np.empty(n, np.int64)
    root_label = np.full(n, -1, np.int64)
    nlab = 0
    for x in range(n):
        r = _find(parent, x)
        if root_label[r] < 0:
            root_label[r] = nlab
            nlab += 1
        labels[x] = root_label[r]
    return labels, nlab
