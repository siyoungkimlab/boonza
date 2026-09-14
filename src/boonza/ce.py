"""Structure-based alignment by Combinatorial Extension (CE).

A port of the CE implementation in Biopython (``Bio.PDB.cealign``), which
adapts PyMOL's cealign by Jason Vertrees: Shindyalov & Bourne (1998),
"Protein structure alignment by incremental combinatorial extension (CE) of
the optimal path", Protein Eng. 11, 739-747.  CE aligns two chains from their
C-alpha geometry alone, so it works for remote homologs where sequence
alignment fails.

    idx_ref, idx_mob, rms, rotation, translation = ce_align(ref_xyz, mobile_xyz)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

_MAX_PATHS = 20
_D0, _D1 = -3.0, -4.0

_Z_TO_P = np.array([
    1.0, 9.20e-01, 8.41e-01, 7.64e-01, 6.89e-01, 6.17e-01, 5.49e-01, 4.84e-01, 4.24e-01, 3.68e-01,
    3.17e-01, 2.71e-01, 2.30e-01, 1.94e-01, 1.62e-01, 1.34e-01, 1.10e-01, 8.91e-02, 7.19e-02,
    5.74e-02, 4.55e-02, 3.57e-02, 2.78e-02, 2.14e-02, 1.64e-02, 1.24e-02, 9.32e-03, 6.93e-03,
    5.11e-03, 3.73e-03, 2.70e-03, 1.94e-03, 1.37e-03, 9.67e-04, 6.74e-04, 4.65e-04, 3.18e-04,
    2.16e-04, 1.45e-04, 9.62e-05, 6.33e-05, 4.13e-05, 2.67e-05, 1.71e-05, 1.08e-05, 6.80e-06,
    4.22e-06, 2.60e-06, 1.59e-06, 9.58e-07, 5.73e-07, 3.40e-07, 1.99e-07, 1.16e-07, 6.66e-08,
    3.80e-08, 2.14e-08, 1.20e-08, 6.63e-09, 3.64e-09, 1.97e-09, 1.06e-09, 5.65e-10, 2.98e-10,
    1.55e-10, 8.03e-11, 4.11e-11, 2.08e-11, 1.05e-11, 5.20e-12, 2.56e-12, 1.25e-12, 6.02e-13,
    2.88e-13, 1.36e-13, 6.38e-14, 2.96e-14, 1.36e-14, 6.19e-15, 2.79e-15, 1.24e-15, 5.50e-16,
    2.40e-16, 1.04e-16, 4.46e-17, 1.90e-17, 7.97e-18, 3.32e-18, 1.37e-18, 5.58e-19, 2.26e-19,
    9.03e-20, 3.58e-20, 1.40e-20, 5.46e-21, 2.10e-21, 7.99e-22, 3.02e-22, 1.13e-22, 4.16e-23,
    1.52e-23, 5.52e-24, 1.98e-24, 7.05e-25, 2.48e-25, 8.64e-26, 2.98e-26, 1.02e-26, 3.44e-27,
    1.15e-27, 3.82e-28, 1.25e-28, 4.08e-29, 1.31e-29, 4.18e-30, 1.32e-30, 4.12e-31, 1.27e-31,
    3.90e-32, 1.18e-32, 3.55e-33, 1.06e-33, 3.11e-34, 9.06e-35, 2.61e-35, 7.47e-36, 2.11e-36,
    5.91e-37, 1.64e-37, 4.50e-38, 1.22e-38, 3.29e-39, 8.77e-40, 2.31e-40, 6.05e-41, 1.56e-41,
    4.00e-42, 1.02e-42, 2.55e-43, 6.33e-44, 1.56e-44, 3.80e-45, 9.16e-46, 2.19e-46, 5.17e-47,
    1.21e-47, 2.81e-48, 6.45e-49, 1.46e-49, 3.30e-50])  # fmt: skip
_P_TO_Z = np.array([
    0.00, 0.73, 1.24, 1.64, 1.99, 2.30, 2.58, 2.83, 3.07, 3.29, 3.50, 3.70, 3.89, 4.07, 4.25,
    4.42, 4.58, 4.74, 4.89, 5.04, 5.19, 5.33, 5.46, 5.60, 5.73, 5.86, 5.99, 6.11, 6.23, 6.35,
    6.47, 6.58, 6.70, 6.81, 6.92, 7.02, 7.13, 7.24, 7.34, 7.44, 7.54, 7.64, 7.74, 7.84, 7.93,
    8.03, 8.12, 8.21, 8.30, 8.40, 8.49, 8.57, 8.66, 8.75, 8.84, 8.92, 9.01, 9.09, 9.17, 9.25,
    9.34, 9.42, 9.50, 9.58, 9.66, 9.73, 9.81, 9.89, 9.97, 10.04, 10.12, 10.19, 10.27, 10.34,
    10.41, 10.49, 10.56, 10.63, 10.70, 10.77, 10.84, 10.91, 10.98, 11.05, 11.12, 11.19, 11.26,
    11.32, 11.39, 11.46, 11.52, 11.59, 11.66, 11.72, 11.79, 11.85, 11.91, 11.98, 12.04, 12.10,
    12.17, 12.23, 12.29, 12.35, 12.42, 12.48, 12.54, 12.60, 12.66, 12.72, 12.78, 12.84, 12.90,
    12.96, 13.02, 13.07, 13.13, 13.19, 13.25, 13.31, 13.36, 13.42, 13.48, 13.53, 13.59, 13.65,
    13.70, 13.76, 13.81, 13.87, 13.92, 13.98, 14.03, 14.09, 14.14, 14.19, 14.25, 14.30, 14.35,
    14.41, 14.46, 14.51, 14.57, 14.62, 14.67, 14.72, 14.77, 14.83, 14.88, 14.93])  # fmt: skip
_SIM_AVG = [2.54, 2.51, 2.72, 3.01, 3.31, 3.61, 3.90, 4.19, 4.47, 4.74,
            4.99, 5.22, 5.46, 5.70, 5.94, 6.13, 6.36, 6.52, 6.68, 6.91]  # fmt: skip
_SIM_SD = [1.33, 0.88, 0.73, 0.71, 0.74, 0.80, 0.86, 0.92, 0.98, 1.04,
           1.08, 1.10, 1.15, 1.19, 1.23, 1.25, 1.32, 1.34, 1.36, 1.45]  # fmt: skip
_GAP_AVG = [0.00, 11.50, 23.32, 35.95, 49.02, 62.44, 76.28, 90.26, 104.86, 119.97, 134.86,
            150.54, 164.86, 179.57, 194.39, 209.38, 224.74, 238.96, 253.72, 270.79]  # fmt: skip
_GAP_SD = [0.00, 9.88, 14.34, 17.99, 21.10, 23.89, 26.55, 29.00, 31.11, 33.10, 35.02,
           36.03, 37.19, 38.82, 41.04, 43.35, 45.45, 48.41, 50.87, 52.27]  # fmt: skip


def _z_score(fragment, length, similarity, gaps) -> float:
    if fragment != 8 or length < 1:
        return 0.0
    if length <= 20:
        s_avg, s_sd = _SIM_AVG[length - 1], _SIM_SD[length - 1]
        g_avg, g_sd = _GAP_AVG[length - 1], _GAP_SD[length - 1]
    else:
        s_avg, s_sd = 0.209874 * length + 2.944714, 0.039487 * length + 0.675735
        g_avg, g_sd = 14.949173 * length - 14.581193, 2.045067 * length + 13.191095
    z1 = 0.0 if similarity > s_avg else (s_avg - similarity) / s_sd
    z2 = 0.0 if gaps > g_avg else (g_avg - gaps) / g_sd
    p = _Z_TO_P[min(max(int(z1 / 0.1), 0), 149)] * _Z_TO_P[min(max(int(z2 / 0.1), 0), 149)]
    return float(_P_TO_Z[min(max(int(-np.log10(p) * 3.0), 0), 149)])


@njit(cache=True)
def _distances(x):
    n = x.shape[0]
    d = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            v = np.sqrt((x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2
                        + (x[i, 2] - x[j, 2]) ** 2)  # fmt: skip
            d[i, j] = v
            d[j, i] = v
    return d


@njit(cache=True)
def _sim_between(dA, dB, iA, iB, jA, jB, m):
    """Similarity of two aligned fragment pairs (CE distance measure i)."""
    s = abs(dA[iA, jA] - dB[iB, jB]) + abs(dA[iA + m - 1, jA + m - 1] - dB[iB + m - 1, jB + m - 1])
    for k in range(1, m - 1):
        s += abs(dA[iA + k, jA + m - 1 - k] - abs(dB[iB + k, jB + m - 1 - k]))
    return -s / m


@njit(cache=True)
def _paths(A, B, m, gap_max):
    """The best CE paths: (count, lengths in fragments, similarities, paths[k, i] = (iA, iB))."""
    lenA, lenB = A.shape[0], B.shape[0]
    dA, dB = _distances(A), _distances(B)
    rows, cols = lenA - m + 1, lenB - m + 1
    terms = (m - 1) * (m - 2) // 2
    S = np.empty((rows, cols))
    for iA in range(rows):
        for iB in range(cols):
            s = 0.0
            for k in range(m - 2):
                for q in range(k + 2, m):
                    s += abs(dA[iA + k, iA + q] - dB[iB + k, iB + q])
            S[iA, iB] = -s / terms
    smaller = min(lenA, lenB)
    buf_len = np.zeros(_MAX_PATHS, np.int64)
    buf_sim = np.full(_MAX_PATHS, -1e6)
    buf_path = np.full((_MAX_PATHS, smaller, 2), -1, np.int64)
    nbuf = 0
    cur = np.full((smaller, 2), -1, np.int64)
    tmp = np.empty((smaller, 2), np.int64)
    for iA in range(rows):
        if nbuf > 0 and iA > lenA - m * (buf_len[nbuf - 1] - 1):
            break
        for iB in range(cols):
            if S[iA, iB] <= _D0:
                continue
            if nbuf > 0 and iB > lenB - m * (buf_len[nbuf - 1] - 1):
                break
            cur[:, :] = -1
            cur[0, 0], cur[0, 1] = iA, iB
            length = 1
            sim = S[iA, iB]
            while True:
                gap_best, gap_index = -1e6, -1
                for g in range(2 * gap_max + 1):
                    jA = cur[length - 1, 0] + m
                    jB = cur[length - 1, 1] + m
                    if (g + 1) % 2 == 0:
                        jA += (g + 1) // 2
                    else:
                        jB += (g + 1) // 2
                    if jA > lenA - m or jB > lenB - m:
                        continue
                    if S[jA, jB] <= _D0:
                        continue
                    cs = 0.0
                    for s in range(length):
                        cs += _sim_between(dA, dB, cur[s, 0], cur[s, 1], jA, jB, m)
                    cs /= length
                    if cs > _D1 and cs > gap_best:
                        cur[length, 0], cur[length, 1] = jA, jB
                        gap_best, gap_index = cs, g
                if gap_index == -1:
                    break
                jA, jB = cur[length, 0], cur[length, 1]
                n = float(length)
                cur_terms = n + n * (n - 1) / 2
                new_terms = n + 1 + n * (n + 1) / 2
                new_sim = (cur_terms * sim + n * gap_best + S[jA, jB]) / new_terms
                if new_sim > _D1:
                    sim = new_sim
                    length += 1
                else:
                    break
            # insert into the buffer, displacing worse paths down the list
            for i in range(nbuf):
                if length > buf_len[i] or (length == buf_len[i] and sim > buf_sim[i]):
                    tl, ts = buf_len[i], buf_sim[i]
                    tmp[:, :] = buf_path[i]
                    buf_len[i], buf_sim[i] = length, sim
                    buf_path[i] = cur
                    length, sim = tl, ts
                    cur[:, :] = tmp
            if nbuf < _MAX_PATHS:
                buf_len[nbuf], buf_sim[nbuf] = length, sim
                buf_path[nbuf] = cur
                nbuf += 1
    return nbuf, buf_len, buf_sim, buf_path


def _fit(A, B):
    """(rms, rotation, translation) superposing B onto A."""
    from .align import kabsch

    R, t = kabsch(B, A)
    diff = B @ R.T + t - A
    return float(np.sqrt((diff * diff).sum() / len(A))), R, t


def ce_align(reference, mobile, window: int = 8, max_gap: int = 30,
             final_optimization: bool = True):  # fmt: skip
    """Align ``mobile`` onto ``reference`` (C-alpha coordinates, (n, 3) arrays) by CE.

    Returns (reference indices, mobile indices, RMSD, rotation, translation,
    z-score); ``mobile @ rotation.T + translation`` superposes the mobile
    coordinates.  As in Biopython, the longest paths are ranked by RMSD and
    a significant alignment (z >= 3.5) is refined by shifting each aligned
    position by up to half a window.
    """
    A = np.ascontiguousarray(reference, dtype=np.float64).reshape(-1, 3)
    B = np.ascontiguousarray(mobile, dtype=np.float64).reshape(-1, 3)
    if len(A) < 2 * window or len(B) < 2 * window:
        raise ValueError(f"CE needs at least {2 * window} C-alpha atoms in each structure")
    nbuf, lengths, sims, paths = _paths(A, B, int(window), int(max_gap))
    if nbuf == 0:
        raise RuntimeError("CE found no alignment")
    offsets = np.arange(window)
    candidates = []
    for k in range(nbuf):
        L = int(lengths[k])
        p = paths[k, :L]
        gaps = int((np.diff(p[:, 0]) - 1).sum() + (np.diff(p[:, 1]) - 1).sum()) if L > 1 else 0
        z = _z_score(window, L, float(sims[k]), gaps)
        idx = [(p[:, side, None] + offsets).ravel().tolist() for side in (0, 1)]
        candidates.append((L * window, idx, z))
    best_len = candidates[0][0]
    best, best_rms, best_z, motion = None, np.inf, 0.0, None
    for length, idx, z in candidates:
        if length != best_len:
            continue
        rms, R, t = _fit(A[idx[0]], B[idx[1]])
        if rms < best_rms:
            best, best_rms, best_z, motion = idx, rms, z, (R, t)
    if final_optimization and best_z >= 3.5:
        half = window // 2
        for side in (0, 1):
            path = best[side]
            for i in range(1, len(path) - 1):
                left, center, right = path[i - 1], path[i], path[i + 1]
                best_shift = 0
                for shift in range(max(-half, left - center + 1), min(half + 1, right - center)):
                    path[i] = center + shift
                    rms, R, t = _fit(A[best[0]], B[best[1]])
                    path[i] = center
                    if rms < best_rms:
                        best_shift, best_rms, motion = shift, rms, (R, t)
                path[i] = center + best_shift
    return (np.array(best[0], np.int64), np.array(best[1], np.int64), best_rms,
            motion[0], motion[1], best_z)  # fmt: skip


@dataclass
class CEResult:
    """Outcome of ``cealign``: transform, fit and the paired guide atoms."""

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    z_score: float
    mobile_atoms: np.ndarray
    reference_atoms: np.ndarray

    def apply(self, positions) -> np.ndarray:
        return np.asarray(positions, dtype=np.float64) @ self.rotation.T + self.translation


def _guide_atoms(system, sel):
    """CA of amino acids and C4' of nucleotides, in residue order (Biopython's guide atoms)."""
    from .align import THREE2ONE
    from .matchmaker import NUCLEIC

    ids = np.arange(system.natoms) if sel is None else system.select(sel).ids
    names = system.atoms["name"][ids]
    resnames = np.strings.upper(system.residues["name"][system.atoms["residue"][ids]])
    amino = np.isin(resnames, list(THREE2ONE)) & (names == "CA")
    nucleic = np.isin(resnames, list(NUCLEIC)) & (names == "C4'")
    return ids[amino | nucleic]


def cealign(mobile, reference, sel=None, ref_sel=None, window: int = 8, max_gap: int = 30,
            apply: bool = True) -> CEResult:  # fmt: skip
    """Superpose ``mobile`` onto ``reference`` (Systems) by CE structural alignment.

    Like PyMOL's cealign, no sequence similarity is needed.  ``sel`` and
    ``ref_sel`` restrict the atoms considered (e.g. "chain A"); guide atoms
    are CA (and C4' for nucleotides).  ``apply`` moves ``mobile``.
    """
    mids = _guide_atoms(mobile, sel)
    rids = _guide_atoms(reference, ref_sel if ref_sel is not None else sel)
    ia, ib, rms, R, t, z = ce_align(reference.positions[rids], mobile.positions[mids],
                                    window, max_gap)  # fmt: skip
    result = CEResult(R, t, rms, z, mids[ib], rids[ia])
    if apply:
        mobile.positions = result.apply(mobile.positions)
        if mobile.cell.any():
            mobile.cell = mobile.cell @ R.T
    return result
