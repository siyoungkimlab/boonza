"""Periodic-boundary geometry and fast distances.

Boxes are (3, 3) arrays with the cell vectors as rows (Å); ``None`` or all
zeros means no periodicity, and ``(a, b, c, alpha, beta, gamma)`` is also
accepted.  Orthorhombic boxes use the direct minimum-image rule.  Triclinic
boxes are reduced in fractional coordinates and then the neighboring images
are checked, which gives the true minimum image for any cell shape.
Kernels are numba, parallel over the first set of positions.

    d = distances(a, b, box)                 # (n, m) matrix
    i, j, d = capped_distances(a, b, 5.0, box)
    theta = angles(p0, p1, p2, box)          # radians
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .spatial import _build


def as_box(box) -> np.ndarray | None:
    """(3, 3) cell vectors from a box or ``(a, b, c, alpha, beta, gamma)``; None if not periodic."""
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64)
    if box.shape == (6,):
        from .io.pdb import cell_from_lengths_angles

        box = cell_from_lengths_angles(*box)
    box = box.reshape(3, 3)
    return box if box.any() else None


def _prepare(box):
    box = as_box(box)
    if box is None:
        return np.zeros((3, 3)), np.zeros((3, 3)), False, True
    ortho = not np.any(box - np.diag(np.diag(box)))
    return box, np.linalg.inv(box), True, ortho


def _xyz(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 3)


@njit(cache=True)
def _mic(dx, dy, dz, box, inv, ortho):
    """Minimum-image version of the vector (dx, dy, dz)."""
    if ortho:
        lx, ly, lz = box[0, 0], box[1, 1], box[2, 2]
        return dx - lx * np.rint(dx / lx), dy - ly * np.rint(dy / ly), dz - lz * np.rint(dz / lz)
    na = np.rint(dx * inv[0, 0] + dy * inv[1, 0] + dz * inv[2, 0])
    nb = np.rint(dx * inv[0, 1] + dy * inv[1, 1] + dz * inv[2, 1])
    nc = np.rint(dx * inv[0, 2] + dy * inv[1, 2] + dz * inv[2, 2])
    dx -= na * box[0, 0] + nb * box[1, 0] + nc * box[2, 0]
    dy -= na * box[0, 1] + nb * box[1, 1] + nc * box[2, 1]
    dz -= na * box[0, 2] + nb * box[1, 2] + nc * box[2, 2]
    bx, by, bz = dx, dy, dz
    best = dx * dx + dy * dy + dz * dz
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                ex = dx + i * box[0, 0] + j * box[1, 0] + k * box[2, 0]
                ey = dy + i * box[0, 1] + j * box[1, 1] + k * box[2, 1]
                ez = dz + i * box[0, 2] + j * box[1, 2] + k * box[2, 2]
                e2 = ex * ex + ey * ey + ez * ez
                if e2 < best:
                    best, bx, by, bz = e2, ex, ey, ez
    return bx, by, bz


@njit(parallel=True, cache=True)
def _matrix(a, b, box, inv, periodic, ortho):
    out = np.empty((a.shape[0], b.shape[0]))
    for i in prange(a.shape[0]):
        for j in range(b.shape[0]):
            dx, dy, dz = b[j, 0] - a[i, 0], b[j, 1] - a[i, 1], b[j, 2] - a[i, 2]
            if periodic:
                dx, dy, dz = _mic(dx, dy, dz, box, inv, ortho)
            out[i, j] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return out


@njit(parallel=True, cache=True)
def _condensed(a, box, inv, periodic, ortho):
    n = a.shape[0]
    out = np.empty(n * (n - 1) // 2)
    for i in prange(n):
        base = i * (2 * n - i - 1) // 2 - i - 1
        for j in range(i + 1, n):
            dx, dy, dz = a[j, 0] - a[i, 0], a[j, 1] - a[i, 1], a[j, 2] - a[i, 2]
            if periodic:
                dx, dy, dz = _mic(dx, dy, dz, box, inv, ortho)
            out[base + j] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return out


@njit(parallel=True, cache=True)
def _vectors(v, box, inv, ortho):
    out = np.empty_like(v)
    for i in prange(v.shape[0]):
        out[i, 0], out[i, 1], out[i, 2] = _mic(v[i, 0], v[i, 1], v[i, 2], box, inv, ortho)
    return out


def minimum_image(vectors, box=None) -> np.ndarray:
    """Shortest periodic images of displacement vectors, shape (n, 3)."""
    v = _xyz(vectors)
    box, inv, periodic, ortho = _prepare(box)
    return _vectors(v, box, inv, ortho) if periodic else v.copy()


def distances(a, b, box=None) -> np.ndarray:
    """All distances between two sets of positions: an (n, m) matrix."""
    box, inv, periodic, ortho = _prepare(box)
    return _matrix(_xyz(a), _xyz(b), box, inv, periodic, ortho)


def self_distances(a, box=None) -> np.ndarray:
    """Distances between all pairs i < j of one set, in condensed (scipy pdist) order."""
    box, inv, periodic, ortho = _prepare(box)
    return _condensed(_xyz(a), box, inv, periodic, ortho)


def paired_distances(a, b, box=None) -> np.ndarray:
    """Distance between a[k] and b[k] for every k (bond lengths)."""
    return np.linalg.norm(minimum_image(_xyz(b) - _xyz(a), box), axis=1)


def angles(a, b, c, box=None) -> np.ndarray:
    """Angle a-b-c in radians for every triple."""
    u = minimum_image(_xyz(a) - _xyz(b), box)
    v = minimum_image(_xyz(c) - _xyz(b), box)
    cos = (u * v).sum(1) / np.sqrt((u * u).sum(1) * (v * v).sum(1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def dihedrals(a, b, c, d, box=None) -> np.ndarray:
    """Dihedral a-b-c-d in radians, in (-pi, pi], IUPAC sign convention."""
    b1 = minimum_image(_xyz(b) - _xyz(a), box)
    b2 = minimum_image(_xyz(c) - _xyz(b), box)
    b3 = minimum_image(_xyz(d) - _xyz(c), box)
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    y = np.linalg.norm(b2, axis=1) * (b1 * n2).sum(1)
    return np.arctan2(y, (n1 * n2).sum(1))


@njit(parallel=True, cache=True)
def _grid_pass(qcell, qpos, tpos, start, sidx, box, inv, ortho, r2, dims, self_pairs,
               offsets, oi, oj, od, fill):  # fmt: skip
    """Pairs within r on a periodic cell grid: count (fill=False) or record them."""
    for i in prange(qpos.shape[0]):
        base = offsets[i] if fill else 0
        cnt = 0
        for da in range(-1, 2):
            a = (qcell[i, 0] + da) % dims[0]
            for db in range(-1, 2):
                b = (qcell[i, 1] + db) % dims[1]
                for dc in range(-1, 2):
                    c = (qcell[i, 2] + dc) % dims[2]
                    key = (a * dims[1] + b) * dims[2] + c
                    for t in range(start[key], start[key + 1]):
                        j = sidx[t]
                        if self_pairs and j <= i:
                            continue
                        dx, dy, dz = _mic(tpos[j, 0] - qpos[i, 0], tpos[j, 1] - qpos[i, 1],
                                          tpos[j, 2] - qpos[i, 2], box, inv, ortho)  # fmt: skip
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 <= r2:
                            if fill:
                                oi[base + cnt] = i
                                oj[base + cnt] = j
                                od[base + cnt] = d2
                            cnt += 1
        if not fill:
            offsets[i + 1] = cnt


def periodic_pairs(a, b, r: float, box):
    """Pairs within ``r`` under periodic boundaries, on a cell grid aligned with the box.

    ``b=None`` gives pairs i < j within ``a``.  Returns (i, j, squared
    distance) sorted by (i, j), or None when the box is too small for three
    cells of height ``r`` along each cell vector.
    """
    box = np.asarray(box, dtype=np.float64).reshape(3, 3)
    inv = np.linalg.inv(box)
    heights = abs(np.linalg.det(box)) / np.linalg.norm(
        np.cross(box[[1, 2, 0]], box[[2, 0, 1]]), axis=1
    )
    dims = np.floor(heights / r).astype(np.int64)
    if (dims < 3).any():
        return None
    a = _xyz(a)
    target = a if b is None else _xyz(b)
    limit = 8 * len(target) + 1000
    if dims.prod() > limit:  # coarser cells are still exact; they just hold more atoms
        dims = np.maximum(3, (dims / (dims.prod() / limit) ** (1 / 3)).astype(np.int64))

    def cells(p):
        f = p @ inv
        c = ((f - np.floor(f)) * dims).astype(np.int64)
        return np.minimum(c, dims - 1)

    tcell = cells(target)
    keys = (tcell[:, 0] * dims[1] + tcell[:, 1]) * dims[2] + tcell[:, 2]
    sidx = np.argsort(keys, kind="stable")
    start = np.zeros(dims.prod() + 1, np.int64)
    np.cumsum(np.bincount(keys, minlength=dims.prod()), out=start[1:])
    qcell = tcell if b is None else cells(a)
    ortho = not np.any(box - np.diag(np.diag(box)))
    offsets = np.zeros(len(a) + 1, np.int64)
    empty = np.empty(0, np.int64)
    args = (qcell, a, target, start, sidx, box, inv, ortho, float(r) ** 2, dims, b is None)
    _grid_pass(*args, offsets, empty, empty, np.empty(0), False)
    np.cumsum(offsets, out=offsets)
    m = int(offsets[-1])
    oi, oj, od = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m)
    _grid_pass(*args, offsets, oi, oj, od, True)
    order = np.argsort((oi << 32) | oj, kind="stable")
    return oi[order], oj[order], od[order]


@njit(parallel=True, cache=True)
def _cross_pass(q, lo, size, dims, start, spos, sidx, r2, shifts, offsets, oi, oj, od, fill):
    for i in prange(q.shape[0]):
        base = offsets[i] if fill else 0
        cnt = 0
        for s in range(shifts.shape[0]):
            x, y, z = q[i, 0] + shifts[s, 0], q[i, 1] + shifts[s, 1], q[i, 2] + shifts[s, 2]
            cx = int(np.floor((x - lo[0]) / size))
            cy = int(np.floor((y - lo[1]) / size))
            cz = int(np.floor((z - lo[2]) / size))
            for u in range(max(cx - 1, 0), min(cx + 2, dims[0])):
                for v in range(max(cy - 1, 0), min(cy + 2, dims[1])):
                    for w in range(max(cz - 1, 0), min(cz + 2, dims[2])):
                        key = (u * dims[1] + v) * dims[2] + w
                        for t in range(start[key], start[key + 1]):
                            dx, dy, dz = x - spos[t, 0], y - spos[t, 1], z - spos[t, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 <= r2:
                                if fill:
                                    oi[base + cnt] = i
                                    oj[base + cnt] = sidx[t]
                                    od[base + cnt] = d2
                                cnt += 1
        if not fill:
            offsets[i + 1] = cnt


def capped_distances(a, b, cutoff: float, box=None):
    """Pairs (i of a, j of b) within ``cutoff``: returns (i, j, distance) sorted by (i, j).

    Periodic searches consider the 27 nearest images, exact while the cutoff
    is below half the smallest box width.
    """
    a, b = _xyz(a), _xyz(b)
    empty = np.empty(0, np.int64)
    if len(a) == 0 or len(b) == 0:
        return empty, empty.copy(), np.empty(0)
    box = as_box(box)
    if box is not None:
        found = periodic_pairs(a, b, float(cutoff), box)
        if found is not None:
            return found[0], found[1], np.sqrt(found[2])
        inv = np.linalg.inv(box)
        a = a - np.floor(a @ inv) @ box
        b = b - np.floor(b @ inv) @ box
        grid = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        shifts = np.ascontiguousarray(grid @ box)
    else:
        shifts = np.zeros((1, 3))
    lo, hi = b.min(axis=0), b.max(axis=0)
    extent = np.maximum(hi - lo, 1e-3)
    size = max(float(cutoff), float(np.prod(extent) / len(b)) ** (1 / 3), 1e-3)
    while True:
        dims = (extent / size).astype(np.int64) + 1
        if dims.prod() <= 8 * len(b) + 1000:
            break
        size *= 1.5
    start, spos, sidx = _build(b, lo, size, dims)
    r2 = float(cutoff) ** 2
    offsets = np.zeros(len(a) + 1, np.int64)
    _cross_pass(a, lo, size, dims, start, spos, sidx, r2, shifts, offsets,
                empty, empty, np.empty(0), False)  # fmt: skip
    np.cumsum(offsets, out=offsets)
    m = int(offsets[-1])
    oi, oj, od = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m)
    _cross_pass(a, lo, size, dims, start, spos, sidx, r2, shifts, offsets, oi, oj, od, True)
    keys = (oi << 32) | oj
    order = np.lexsort((od, keys))
    first = np.ones(m, bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    order = order[first]
    return oi[order], oj[order], np.sqrt(od[order])
