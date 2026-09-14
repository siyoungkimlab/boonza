"""Bond-graph helpers: connected components and CSR adjacency."""

from __future__ import annotations

import numpy as np


def connected_components(n: int, bi: np.ndarray, bj: np.ndarray) -> tuple[np.ndarray, int]:
    """Label each atom with its fragment; fragments are numbered by their first atom."""
    if n == 0:
        return np.empty(0, np.int64), 0
    from ._kernels import components

    labels, nlab = components(n, bi.astype(np.int64), bj.astype(np.int64))
    return labels, int(nlab)


def csr(n: int, keys: np.ndarray):
    """Group row indices by ``keys`` (values in 0..n-1).

    Returns (offsets, order): the rows with key k are ``order[offsets[k]:offsets[k+1]]``,
    in their original order.
    """
    order = np.argsort(keys, kind="stable")
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(keys, minlength=n), out=offsets[1:])
    return offsets, order


def unique_in_order(a: np.ndarray) -> np.ndarray:
    """Distinct values of ``a`` in order of first appearance."""
    if len(a) == 0:
        return a[:0].copy()
    _, first = np.unique(a, return_index=True)
    return a[np.sort(first)]
