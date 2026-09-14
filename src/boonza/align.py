"""Rigid-body superposition: Kabsch fits, RMSD, and matchmaker-style alignment.

``superpose`` pairs atoms by index order or, like ChimeraX matchmaker and
viswizard, by aligning the two residue sequences (Needleman-Wunsch), so
residue numbering does not have to match.  Fits are then repeated while
dropping pairs farther apart than ``cutoff``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "HID": "H",
    "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H", "CYX": "C", "CYM": "C",
    "LYN": "K", "ASH": "D", "GLH": "E", "MSE": "M", "SEC": "U", "PYL": "O", "LYSH": "K",
    "HISB": "H", "NLE": "L", "ASPH": "D", "GLUH": "E",
}  # fmt: skip


def kabsch(mobile, target, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimizing |mobile @ R.T + t - target| (weighted)."""
    P = np.asarray(mobile, dtype=np.float64).reshape(-1, 3)
    Q = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if len(P) != len(Q) or len(P) == 0:
        raise ValueError("kabsch needs two equally long, non-empty sets of positions")
    w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    pc, qc = w @ P, w @ Q
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def rmsd(a, b, weights=None, superpose: bool = False) -> float:
    """RMSD between two sets of positions, optionally after the best rigid fit."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if superpose:
        R, t = kabsch(a, b, weights)
        a = a @ R.T + t
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(np.sqrt((w * ((a - b) ** 2).sum(1)).sum() / w.sum()))


@njit(cache=True)
def _needleman_wunsch(a, b, match, mismatch, gap):
    """Global alignment of two code arrays; returns index pairs of identical aligned codes."""
    n, m = a.shape[0], b.shape[0]
    S = np.zeros((n + 1, m + 1))
    P = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        S[i, 0] = gap * i
        P[i, 0] = 1
    for j in range(1, m + 1):
        S[0, j] = gap * j
        P[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            u = S[i - 1, j] + gap
            left = S[i, j - 1] + gap
            if d >= u and d >= left:
                S[i, j], P[i, j] = d, 0
            elif u >= left:
                S[i, j], P[i, j] = u, 1
            else:
                S[i, j], P[i, j] = left, 2
    pi = np.empty(min(n, m), np.int64)
    pj = np.empty(min(n, m), np.int64)
    k = 0
    i, j = n, m
    while i > 0 or j > 0:
        p = P[i, j]
        if p == 0 and i > 0 and j > 0:
            i -= 1
            j -= 1
            if a[i] == b[j]:
                pi[k], pj[k] = i, j
                k += 1
        elif p == 1 and i > 0:
            i -= 1
        else:
            j -= 1
    return pi[:k][::-1].copy(), pj[:k][::-1].copy()


def _codes(system, ids) -> np.ndarray:
    names = system.residues["name"][system.atoms["residue"][ids]].tolist()
    return np.array([ord(THREE2ONE.get(nm.upper(), "X")) for nm in names], np.int64)


@dataclass
class Superposition:
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    n_matched: int
    n_used: int
    mobile_atoms: np.ndarray
    reference_atoms: np.ndarray

    def apply(self, positions) -> np.ndarray:
        return np.asarray(positions, dtype=np.float64) @ self.rotation.T + self.translation


def superpose(mobile, reference, sel: str = "protein and name CA", ref_sel: str | None = None,
              match: str = "sequence", cutoff: float | None = 2.0, iterations: int = 50,
              apply: bool = True) -> Superposition:  # fmt: skip
    """Superpose ``mobile`` onto ``reference`` (Systems) using the selected atoms.

    ``match="sequence"`` pairs one atom per residue by aligning residue
    sequences; ``match="order"`` pairs the selections in order.  With
    ``cutoff``, the pairs farthest apart after a fit are pruned, at most 10%
    per cycle and only those beyond ``cutoff`` (like ChimeraX matchmaker),
    and the fit is repeated until none are beyond it.  ``apply`` moves
    ``mobile`` (positions and cell).
    """
    mids = mobile.select(sel).ids
    rids = reference.select(ref_sel or sel).ids
    if match == "sequence":
        i, j = _needleman_wunsch(_codes(mobile, mids), _codes(reference, rids), 2.0, -1.0, -2.0)
        mids, rids = mids[i], rids[j]
    elif match == "order":
        if len(mids) != len(rids):
            raise ValueError(f"selections have {len(mids)} and {len(rids)} atoms")
    else:
        raise ValueError("match must be 'sequence' or 'order'")
    if len(mids) < 3:
        raise ValueError(f"only {len(mids)} matched atoms; need at least 3")
    P, Q = mobile.positions[mids], reference.positions[rids]
    keep = np.arange(len(P))
    for _ in range(iterations if cutoff is not None else 0):
        R, t = kabsch(P[keep], Q[keep])
        d = np.linalg.norm(P[keep] @ R.T + t - Q[keep], axis=1)
        far = np.flatnonzero(d > cutoff)
        if len(far) == 0:
            break
        worst = far[np.argsort(d[far])[::-1][: max(1, len(keep) // 10)]]
        if len(keep) - len(worst) < 3:
            break
        keep = np.delete(keep, worst)
    R, t = kabsch(P[keep], Q[keep])
    fit = rmsd(P[keep] @ R.T + t, Q[keep])
    result = Superposition(R, t, fit, len(mids), len(keep), mids[keep], rids[keep])
    if apply:
        mobile.positions = result.apply(mobile.positions)
        if mobile.cell.any():
            mobile.cell = mobile.cell @ R.T
    return result
