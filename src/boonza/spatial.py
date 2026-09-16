"""Distance searches on a cell list: within-cutoff tests and nearest neighbors.

Distances are computed in float32 like msys, so atoms right at a selection
cutoff are classified the same way.  Periodic searches take an orthorhombic
box and test the 27 nearest images.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _build(tpos, lo, size, dims):
    n = tpos.shape[0]
    ncell = dims[0] * dims[1] * dims[2]
    cell = np.empty(n, np.int64)
    counts = np.zeros(ncell + 1, np.int64)
    for i in range(n):
        c = np.empty(3, np.int64)
        for d in range(3):
            v = int((tpos[i, d] - lo[d]) / size)
            c[d] = min(max(v, 0), dims[d] - 1)
        key = (c[0] * dims[1] + c[1]) * dims[2] + c[2]
        cell[i] = key
        counts[key + 1] += 1
    for k in range(ncell):
        counts[k + 1] += counts[k]
    fill = counts[:-1].copy()
    spos = np.empty_like(tpos)
    sidx = np.empty(n, np.int64)
    for i in range(n):
        j = fill[cell[i]]
        fill[cell[i]] += 1
        spos[j] = tpos[i]
        sidx[j] = i
    return counts, spos, sidx


@njit(cache=True)
def _query(qpos, lo, hi, size, dims, start, spos, r, box, nimg, first_hit):
    """Smallest squared distance from each query point to a target (inf if none nearby)."""
    nq = qpos.shape[0]
    out = np.full(nq, np.inf, np.float32)
    r2 = r * r
    for q in range(nq):
        best = np.float32(np.inf)
        done = False
        for ii in range(-nimg, nimg + 1):
            x = qpos[q, 0] + box[0] * np.float32(ii)
            if x < lo[0] - r or x > hi[0] + r:
                continue
            for jj in range(-nimg, nimg + 1):
                y = qpos[q, 1] + box[1] * np.float32(jj)
                if y < lo[1] - r or y > hi[1] + r:
                    continue
                for kk in range(-nimg, nimg + 1):
                    z = qpos[q, 2] + box[2] * np.float32(kk)
                    if z < lo[2] - r or z > hi[2] + r:
                        continue
                    cx = int(np.floor((x - lo[0]) / size))
                    cy = int(np.floor((y - lo[1]) / size))
                    cz = int(np.floor((z - lo[2]) / size))
                    for a in range(max(cx - 1, 0), min(cx + 2, dims[0])):
                        for b in range(max(cy - 1, 0), min(cy + 2, dims[1])):
                            for c in range(max(cz - 1, 0), min(cz + 2, dims[2])):
                                key = (a * dims[1] + b) * dims[2] + c
                                for t in range(start[key], start[key + 1]):
                                    dx = x - spos[t, 0]
                                    dy = y - spos[t, 1]
                                    dz = z - spos[t, 2]
                                    d2 = dx * dx + dy * dy + dz * dz
                                    if d2 < best:
                                        best = d2
                                        if first_hit and best <= r2:
                                            done = True
                                            break
                                if done:
                                    break
                            if done:
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
            if done:
                break
        out[q] = best
    return out


@njit(parallel=True, cache=True)
def _count(qpos, lo, hi, size, dims, start, spos, r, box, nimg):
    """How many targets lie within ``r`` of each query point."""
    nq = qpos.shape[0]
    out = np.zeros(nq, np.int64)
    r2 = r * r
    for q in prange(nq):
        total = 0
        for ii in range(-nimg, nimg + 1):
            x = qpos[q, 0] + box[0] * np.float32(ii)
            if x < lo[0] - r or x > hi[0] + r:
                continue
            for jj in range(-nimg, nimg + 1):
                y = qpos[q, 1] + box[1] * np.float32(jj)
                if y < lo[1] - r or y > hi[1] + r:
                    continue
                for kk in range(-nimg, nimg + 1):
                    z = qpos[q, 2] + box[2] * np.float32(kk)
                    if z < lo[2] - r or z > hi[2] + r:
                        continue
                    cx = int(np.floor((x - lo[0]) / size))
                    cy = int(np.floor((y - lo[1]) / size))
                    cz = int(np.floor((z - lo[2]) / size))
                    for a in range(max(cx - 1, 0), min(cx + 2, dims[0])):
                        for b in range(max(cy - 1, 0), min(cy + 2, dims[1])):
                            for c in range(max(cz - 1, 0), min(cz + 2, dims[2])):
                                key = (a * dims[1] + b) * dims[2] + c
                                for t in range(start[key], start[key + 1]):
                                    dx = x - spos[t, 0]
                                    dy = y - spos[t, 1]
                                    dz = z - spos[t, 2]
                                    if dx * dx + dy * dy + dz * dz <= r2:
                                        total += 1
        out[q] = total
    return out


def _cell_list(target, r):
    """The cell list a query walks: (lo, hi, size, dims, start, spos)."""
    lo, hi = target.min(axis=0), target.max(axis=0)
    extent = np.maximum(hi - lo, np.float32(1e-3))
    size = max(float(r), float(np.prod(extent) / len(target)) ** (1 / 3), 1e-3)
    while True:
        dims = (extent / size).astype(np.int64) + 1
        if dims.prod() <= 8 * len(target) + 1000:
            break
        size *= 1.5
    start, spos, _ = _build(target, lo, np.float32(size), dims)
    return lo, hi, np.float32(size), dims, start, spos


def count_within(query, target, r, cell=None) -> np.ndarray:
    """Per query point, how many target points lie within ``r``.

    One integer per query point and no pair list, so memory does not grow with
    the number of pairs found.  With ``cell``, the 27 nearest images are
    searched, which counts a target twice if the box is narrower than ``2 r``.
    """
    query = np.ascontiguousarray(query, dtype=np.float32).reshape(-1, 3)
    target = np.ascontiguousarray(target, dtype=np.float32).reshape(-1, 3)
    if not len(query) or not len(target):
        return np.zeros(len(query), np.int64)
    lo, hi, size, dims, start, spos = _cell_list(target, float(r))
    if cell is None:
        box, nimg = np.zeros(3, np.float32), 0
    else:
        box, nimg = _box(cell), 1
    return _count(query, lo, hi, size, dims, start, spos, np.float32(r), box, nimg)


def _box(cell) -> np.ndarray:
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    lengths = np.sqrt((cell * cell).sum(axis=1))
    if (lengths == 0).any():
        raise ValueError("periodic search needs a unit cell with nonzero dimensions")
    if np.abs(cell - np.diag(np.diag(cell))).max() > 0:
        raise NotImplementedError("periodic search needs a cell aligned with x, y, z")
    return lengths.astype(np.float32)


def min_dist2(query, target, r, cell=None, first_hit=False) -> np.ndarray:
    """Per query point, the smallest squared distance (float32) to any target point.

    Only targets within ``r`` are guaranteed to be found; farther ones give
    either their distance or inf.  ``first_hit`` stops at the first target
    within ``r`` (enough for within tests).
    """
    query = np.ascontiguousarray(query, dtype=np.float32).reshape(-1, 3)
    target = np.ascontiguousarray(target, dtype=np.float32).reshape(-1, 3)
    if len(target) == 0 or len(query) == 0:
        return np.full(len(query), np.inf, np.float32)
    r = np.float32(r)
    lo, hi = target.min(axis=0), target.max(axis=0)
    extent = np.maximum(hi - lo, np.float32(1e-3))
    size = max(float(r), float(np.prod(extent) / len(target)) ** (1 / 3), 1e-3)
    while True:
        dims = (extent / size).astype(np.int64) + 1
        if dims.prod() <= 8 * len(target) + 1000:
            break
        size *= 1.5
    size = np.float32(size)
    start, spos, _ = _build(target, lo, size, dims)
    if cell is None:
        box, nimg = np.zeros(3, np.float32), 0
    else:
        box, nimg = _box(cell), 1
    return _query(query, lo, hi, size, dims, start, spos, r, box, nimg, first_hit)


def within(query, target, r, cell=None) -> np.ndarray:
    """Boolean mask: query points within ``r`` (inclusive) of any target point."""
    d2 = min_dist2(query, target, r, cell, first_hit=True)
    return d2 <= np.float32(r) * np.float32(r)


def nearest(query, query_ids, target, k, cell=None) -> np.ndarray:
    """The ``k`` query ids closest to any target point, sorted; ties go to lower ids."""
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if len(target) == 0:
        raise ValueError("no atoms in target selection")
    if len(query_ids) <= k:
        return np.sort(query_ids)
    r = 2.5
    while True:
        d2 = min_dist2(query, target, r, cell)
        hit = d2 <= np.float32(r) * np.float32(r)
        if hit.sum() >= k:
            break
        r *= 1.5
    ids, d2 = query_ids[hit], d2[hit]
    return np.sort(ids[np.lexsort((ids, d2))[:k]])


@njit(parallel=True, cache=True)
def _pair_pass(pos, lo, size, dims, start, spos, sidx, r2, shifts, offsets, oi, oj, od, fill):
    """Count (fill=False, into offsets[i+1]) or record (fill=True) pairs i<j within r."""
    n = pos.shape[0]
    for i in prange(n):
        base = offsets[i] if fill else 0
        cnt = 0
        for s in range(shifts.shape[0]):
            x = pos[i, 0] + shifts[s, 0]
            y = pos[i, 1] + shifts[s, 1]
            z = pos[i, 2] + shifts[s, 2]
            cx = int(np.floor((x - lo[0]) / size))
            cy = int(np.floor((y - lo[1]) / size))
            cz = int(np.floor((z - lo[2]) / size))
            for a in range(max(cx - 1, 0), min(cx + 2, dims[0])):
                for b in range(max(cy - 1, 0), min(cy + 2, dims[1])):
                    for c in range(max(cz - 1, 0), min(cz + 2, dims[2])):
                        key = (a * dims[1] + b) * dims[2] + c
                        for t in range(start[key], start[key + 1]):
                            j = sidx[t]
                            if j <= i:
                                continue
                            dx = x - spos[t, 0]
                            dy = y - spos[t, 1]
                            dz = z - spos[t, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 <= r2:
                                if fill:
                                    oi[base + cnt] = i
                                    oj[base + cnt] = j
                                    od[base + cnt] = d2
                                cnt += 1
        if not fill:
            offsets[i + 1] = cnt


def pairs_within(pos, r: float, cell=None):
    """All pairs ``i < j`` closer than or equal to ``r``: returns (i, j, d2) sorted by (i, j).

    With ``cell`` (rows are box vectors), distances use the nearest periodic
    image among the 27 neighboring cells, which is exact while ``r`` is below
    half the smallest box width.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    empty = np.empty(0, np.int64)
    if n < 2:
        return empty, empty.copy(), np.empty(0)
    if cell is not None:
        from .pbc import periodic_pairs

        found = periodic_pairs(pos, None, float(r), cell)
        if found is not None:  # the box holds at least 3 cells per direction
            return found
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        frac = pos @ np.linalg.inv(cell)
        pos = pos - np.floor(frac) @ cell
        grid = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
        shifts = np.ascontiguousarray(grid @ cell, dtype=np.float64)
    else:
        shifts = np.zeros((1, 3))
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    extent = np.maximum(hi - lo, 1e-3)
    size = max(float(r), float(np.prod(extent) / n) ** (1 / 3), 1e-3)
    while True:
        dims = (extent / size).astype(np.int64) + 1
        if dims.prod() <= 8 * n + 1000:
            break
        size *= 1.5
    start, spos, sidx = _build(pos, lo, size, dims)
    r2 = float(r) * float(r)
    offsets = np.zeros(n + 1, np.int64)
    _pair_pass(pos, lo, size, dims, start, spos, sidx, r2, shifts, offsets,
               empty, empty, np.empty(0), False)  # fmt: skip
    np.cumsum(offsets, out=offsets)
    m = int(offsets[-1])
    oi, oj, od = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m)
    _pair_pass(pos, lo, size, dims, start, spos, sidx, r2, shifts, offsets, oi, oj, od, True)
    keys = (oi << 32) | oj
    if cell is not None:  # a pair can be found through several images in small boxes
        order = np.lexsort((od, keys))
        first = np.ones(m, bool)
        first[1:] = keys[order][1:] != keys[order][:-1]
        order = order[first]
    else:
        order = np.argsort(keys, kind="stable")
    return oi[order], oj[order], od[order]
