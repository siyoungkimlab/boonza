# SPDX-License-Identifier: LGPL-2.1-only
#
# Not under boonza's MIT license: this file ports LGPL code, so it is
# distributed under the GNU Lesser General Public License version 2.1
# (LICENSES/LGPL-2.1.txt; see NOTICE).
#
# dssp and backbone_hbonds port MDTraj's DSSP (mdtraj/geometry/src/dssp.cpp):
#
#   DSSP secondary structure assignment
#   Copyright [2014] Stanford University and the Authors
#   Authors: Robert T. McGibbon, Maarten L. Hekkelman
#
#   This code is adapted from DSSP-2.2.0, written by Maarten L. Hekkelman,
#   and ported to MDTraj by Robert T. McGibbon. DSSP-2.2.0 is distributed
#   under the Boost Software License, Version 1.0. This code, as part of
#   MDTraj, is distributed under the GNU LGPL.
#
# chimerax_ss ports UCSF ChimeraX's Kabsch-Sander assignment (atomstruct CompSS.cpp):
#
#   === UCSF ChimeraX Copyright ===
#   Copyright 2022 Regents of the University of California. All rights reserved.
#   The ChimeraX application is provided pursuant to the ChimeraX license
#   agreement, which covers academic and commercial uses. For more details, see
#   <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
#   This particular file is part of the ChimeraX library. You can also
#   redistribute and/or modify it under the terms of the GNU Lesser General
#   Public License version 2.1 as published by the Free Software Foundation.
#   For more details, see
#   <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
#   THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
#   EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
#   LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
#   VERSION 2.1
#
#   This notice must be embedded in or attached to all copies, including partial
#   copies, of the software or any revisions or derivations thereof.
#   === UCSF ChimeraX Copyright ===
"""DSSP secondary structure (Kabsch & Sander), following DSSP 2.2 as ported by mdtraj.

    codes = boonza.dssp(system)                    # (1, nresidues) for the current positions
    codes = boonza.dssp(system, frames.positions)  # (nframes, nresidues)

Codes: H alpha helix, B isolated bridge, E strand, G 3-10 helix, I pi helix,
T turn, S bend, ' ' loop, and NA for residues without backbone N, CA, C and
O.  ``simplified=True`` maps them to H (H, G, I), E (E, B) and C (the rest).

Backbone hydrogens are rebuilt from the previous residue's C=O as in DSSP.
The arithmetic is float32 in nm, as in mdtraj, so borderline hydrogen bonds
resolve the same way.

Chain breaks (``breaks=True``, the default): as in the DSSP program, a chain
is also cut where the C(i-1)-N(i) distance exceeds 2.5 Å or a residue lacks
backbone atoms (missing residues in a crystal structure).  Helices, turns,
bends and bridges never span a break, and the residue after one gets its H on
N.  mdtraj's port uses the topology chains only; ``breaks=False`` reproduces
it exactly.

Periodic boxes: with ``box`` (one (3, 3) box or one per frame, Å), or with
Frames and trajectories that carry boxes, every distance uses the minimum
image, so proteins split across the box need no glueing.  A PDB's CRYST1
cell is not used unless passed: it would add contacts to symmetry mates.

    codes = boonza.dssp(system, traj)                  # boxes from the trajectory
    codes = boonza.dssp(system, frames, box=cell)
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from numba import njit

from .pbc import _mic

LOOP, HELIX, BRIDGE, STRAND, HELIX3, HELIX5, TURN, BEND = range(8)
CODES = np.array([" ", "H", "B", "E", "G", "I", "T", "S"])
_SIMPLE = {" ": "C", "H": "H", "B": "E", "E": "E", "G": "H", "I": "H", "T": "C", "S": "C"}
_NONE, _START, _END, _START_END, _MIDDLE = range(5)

_CUT_E = np.float32(-0.5)  # kcal/mol
_MIN_E = np.float32(-9.9)
_CA2 = np.float32(0.81)  # (0.9 nm)^2
_Q = np.float32(2.7888)  # 332 * 0.42 * 0.20 kcal nm/mol / 10
_NH = np.float32(0.1)  # N-H length, nm


@njit(cache=True)
def _dv(a, b, box, inv, periodic, ortho):
    """a - b (float32 3-vectors), as the minimum image when periodic."""
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    if periodic:
        x, y, z = _mic(np.float64(dx), np.float64(dy), np.float64(dz), box, inv, ortho)
        dx, dy, dz = np.float32(x), np.float32(y), np.float32(z)
    return dx, dy, dz


@njit(cache=True)
def _sq(dx, dy, dz):
    return dx * dx + dy * dy + dz * dz


@njit(cache=True)
def _energy(xyz, h, nco, donor, acceptor, box, inv, periodic, ortho):
    n, c, o = nco[donor, 0], nco[acceptor, 1], nco[acceptor, 2]
    d0 = _sq(*_dv(h[donor], xyz[o], box, inv, periodic, ortho))
    d1 = _sq(*_dv(xyz[n], xyz[c], box, inv, periodic, ortho))
    d2 = _sq(*_dv(h[donor], xyz[c], box, inv, periodic, ortho))
    d3 = _sq(*_dv(xyz[n], xyz[o], box, inv, periodic, ortho))
    one = np.float32(1.0)
    # same operations and summation order as mdtraj's vectorized dot4
    e = -_Q * (one / np.sqrt(d0)) + -_Q * (one / np.sqrt(d1))
    e = e + _Q * (one / np.sqrt(d2))
    e = e + _Q * (one / np.sqrt(d3))
    return _MIN_E if e < _MIN_E else e


@njit(cache=True)
def _store(hb, he, donor, acceptor, e):
    if e < he[donor, 0]:
        hb[donor, 1] = hb[donor, 0]
        he[donor, 1] = he[donor, 0]
        hb[donor, 0] = acceptor
        he[donor, 0] = e
    elif e < he[donor, 1]:
        hb[donor, 1] = acceptor
        he[donor, 1] = e


@njit(cache=True)
def _hbonds(xyz, nco, ca, is_pro, skip, linked, box, inv, periodic, ortho):
    """The two strongest backbone H-bonds (acceptor residue, energy) donated by each residue.

    ``linked[i]``: residue i follows residue i-1 in the chain (its H is built
    from that C=O; otherwise it sits on N)."""
    n = ca.shape[0]
    h = np.zeros((n, 3), np.float32)
    for ri in range(n):
        if skip[ri]:
            continue
        nn = nco[ri, 0]
        h[ri] = xyz[nn]
        if linked[ri] and nco[ri - 1, 1] >= 0 and nco[ri - 1, 2] >= 0:
            cx, cy, cz = _dv(xyz[nco[ri - 1, 1]], xyz[nco[ri - 1, 2]], box, inv, periodic, ortho)
            norm = np.sqrt(_sq(cx, cy, cz))
            h[ri, 0] = xyz[nn, 0] + cx / norm * _NH
            h[ri, 1] = xyz[nn, 1] + cy / norm * _NH
            h[ri, 2] = xyz[nn, 2] + cz / norm * _NH
    hb = np.full((n, 2), -1, np.int64)
    he = np.zeros((n, 2), np.float32)
    for ri in range(n):
        if skip[ri]:
            continue
        for rj in range(ri + 1, n):
            if skip[rj]:
                continue
            if _sq(*_dv(xyz[ca[ri]], xyz[ca[rj]], box, inv, periodic, ortho)) < _CA2:
                e = _energy(xyz, h, nco, ri, rj, box, inv, periodic, ortho)
                if e < _CUT_E and not is_pro[ri]:
                    _store(hb, he, ri, rj, e)
                if rj != ri + 1:
                    e = _energy(xyz, h, nco, rj, ri, box, inv, periodic, ortho)
                    if e < _CUT_E and not is_pro[rj]:
                        _store(hb, he, rj, ri, e)
    return hb, he


@njit(cache=True)
def _bond(hb, donor, acceptor):
    return hb[donor, 0] == acceptor or hb[donor, 1] == acceptor


@njit(cache=True)
def _bridge(i, j, n, chain, hb):
    a, b, c, d, e, f = i - 1, i, i + 1, j - 1, j, j + 1
    if a >= 0 and c < n and chain[a] == chain[c] and d >= 0 and f < n and chain[d] == chain[f]:
        if (_bond(hb, c, e) and _bond(hb, e, a)) or (_bond(hb, f, b) and _bond(hb, b, d)):
            return 1  # parallel
        if (_bond(hb, c, d) and _bond(hb, f, a)) or (_bond(hb, e, b) and _bond(hb, b, e)):
            return 2  # antiparallel
    return 0


@njit(cache=True)
def _sheets(chain, hb, skip, ss):
    n = chain.shape[0]
    # each ladder: type, chain_i, i_front, i_back, j_front, j_back, size (number of i's)
    lad = np.zeros((16, 7), np.int64)
    m = 0
    for i in range(1, n - 4):
        for j in range(i + 3, n - 1):
            kind = _bridge(j, i, n, chain, hb)
            if kind == 0 or skip[i] or skip[j]:
                continue
            found = False
            for k in range(m):
                if kind != lad[k, 0] or i != lad[k, 3] + 1:
                    continue
                if kind == 1 and lad[k, 5] + 1 == j:
                    lad[k, 3], lad[k, 5], lad[k, 6] = i, j, lad[k, 6] + 1
                    found = True
                    break
                if kind == 2 and lad[k, 4] - 1 == j:
                    lad[k, 3], lad[k, 4], lad[k, 6] = i, j, lad[k, 6] + 1
                    found = True
                    break
            if not found:
                if m == lad.shape[0]:
                    grown = np.zeros((2 * m, 7), np.int64)
                    grown[:m] = lad
                    lad = grown
                lad[m] = (kind, chain[i], i, i, j, j, 1)
                m += 1
    lad = lad[:m]
    if m:  # stable sort by (chain, first i)
        lad = lad[np.argsort(lad[:, 1] * (n + 1) + lad[:, 2], kind="mergesort")]
    i = 0
    while i < m:
        j = i + 1
        while j < m:
            ibi, iei, jbi, jei = lad[i, 2], lad[i, 3], lad[i, 4], lad[i, 5]
            ibj, iej, jbj, jej = lad[j, 2], lad[j, 3], lad[j, 4], lad[j, 5]
            if (
                lad[i, 0] != lad[j, 0]
                or chain[min(ibi, ibj)] != chain[max(iei, iej)]
                or chain[min(jbi, jbj)] != chain[max(jei, jej)]
                or ibj - iei >= 6
                or (iei >= ibj and ibi <= iej)
            ):
                j += 1
                continue
            if lad[i, 0] == 1:
                bulge = jbj > jbi and ((jbj - jei < 6 and ibj - iei < 3) or jbj - jei < 3)
            else:
                bulge = jbj < jbi and ((jbi - jej < 6 and ibj - iei < 3) or jbi - jej < 3)
            if bulge:
                lad[i, 3] = iej
                if lad[i, 0] == 1:
                    lad[i, 5] = jej
                else:
                    lad[i, 4] = jbj
                lad[i, 6] += lad[j, 6]
                lad[j : m - 1] = lad[j + 1 : m].copy()
                m -= 1
            else:
                j += 1
        i += 1
    for k in range(m):
        code = BRIDGE if lad[k, 6] == 1 else STRAND
        for r in range(lad[k, 2], lad[k, 3] + 1):
            if ss[r] != STRAND:
                ss[r] = code
        for r in range(lad[k, 4], lad[k, 5] + 1):
            if ss[r] != STRAND:
                ss[r] = code


@njit(cache=True)
def _starts(flag):
    return flag == _START or flag == _START_END


@njit(cache=True)
def _helices(xyz, ca, chain, hb, skip, ss, box, inv, periodic, ortho):
    n = chain.shape[0]
    flags = np.zeros((n, 6), np.int8)
    for stride in range(3, 6):
        for i in range(n):
            if i + stride < n and _bond(hb, i + stride, i) and chain[i] == chain[i + stride]:
                flags[i + stride, stride] = _END
                for j in range(i + 1, i + stride):
                    if flags[j, stride] == _NONE:
                        flags[j, stride] = _MIDDLE
                flags[i, stride] = _START_END if flags[i, stride] == _END else _START
    for i in range(1, n - 4):
        if _starts(flags[i, 4]) and _starts(flags[i - 1, 4]):
            for j in range(i, i + 4):
                ss[j] = HELIX
    for i in range(1, n - 3):
        if _starts(flags[i, 3]) and _starts(flags[i - 1, 3]):
            empty = True
            for j in range(i, i + 3):
                empty = empty and (ss[j] == LOOP or ss[j] == HELIX3)
            if empty:
                for j in range(i, i + 3):
                    ss[j] = HELIX3
    for i in range(1, n - 5):
        if _starts(flags[i, 5]) and _starts(flags[i - 1, 5]):
            empty = True
            for j in range(i, i + 5):
                empty = empty and (ss[j] == LOOP or ss[j] == HELIX5 or ss[j] == HELIX)
            if empty:
                for j in range(i, i + 5):
                    ss[j] = HELIX5
    bend = np.zeros(n, np.bool_)
    limit = 70.0 * np.pi / 180.0
    for i in range(2, n - 2):
        if chain[i - 2] == chain[i + 2] and not (skip[i - 2] or skip[i] or skip[i + 2]):
            ux, uy, uz = _dv(xyz[ca[i - 2]], xyz[ca[i]], box, inv, periodic, ortho)
            vx, vy, vz = _dv(xyz[ca[i]], xyz[ca[i + 2]], box, inv, periodic, ortho)
            cos = (ux * vx + uy * vy + uz * vz) / np.sqrt(_sq(ux, uy, uz) * _sq(vx, vy, vz))
            cos = min(max(cos, np.float32(-1.0)), np.float32(1.0))
            bend[i] = np.arccos(cos) > limit
    for i in range(1, n - 1):
        if ss[i] == LOOP and not skip[i]:
            turn = False
            for stride in range(3, 6):
                for k in range(1, stride):
                    if i >= k and _starts(flags[i - k, stride]):
                        turn = True
            if turn:
                ss[i] = TURN
            elif bend[i]:
                ss[i] = BEND


@njit(cache=True)
def _frame(xyz, nco, ca, is_pro, chain, skip, linked, box, inv, periodic, ortho):
    hb, _ = _hbonds(xyz, nco, ca, is_pro, skip, linked, box, inv, periodic, ortho)
    ss = np.zeros(ca.shape[0], np.int8)
    _sheets(chain, hb, skip, ss)
    _helices(xyz, ca, chain, hb, skip, ss, box, inv, periodic, ortho)
    return ss


def _backbone(system):
    """Per residue: indices of N, C, O (first atom of each name) and CA; -1 when missing."""
    names = system.atoms["name"]
    res = system.atoms["residue"]
    nres = system.nresidues
    out = []
    for name in ("N", "C", "O", "CA"):
        idx = np.full(nres, -1, np.int64)
        ids = np.flatnonzero(names == name)
        r, first = np.unique(res[ids], return_index=True)
        idx[r] = ids[first]
        out.append(idx)
    # C-terminal carbonyl oxygens often carry other names (mdtraj renames them to O)
    for alt in ("OT1", "OC1", "O1"):
        missing = out[2] < 0
        if not missing.any():
            break
        ids = np.flatnonzero(names == alt)
        r, first = np.unique(res[ids], return_index=True)
        take = missing[r]
        out[2][r[take]] = ids[first][take]
    nco = np.column_stack(out[:3])
    ca = out[3]
    skip = (nco < 0).any(axis=1) | (ca < 0)
    chain = system.residues["chain"].astype(np.int64)
    is_pro = system.residues["name"] == "PRO"
    return nco, ca, skip, chain, is_pro


def _frames(system, positions) -> np.ndarray:
    if positions is None:
        positions = system.positions
    positions = getattr(positions, "positions", positions)
    xyz = np.asarray(positions)
    if xyz.ndim == 2:
        xyz = xyz[None]
    if xyz.shape[1:] != (system.natoms, 3):
        raise ValueError(f"positions must be (nframes, {system.natoms}, 3)")
    return xyz


def _dssp_blocks(system, positions, box, atoms):
    """Blocks of (positions of ``atoms`` (nframes, n, 3) in Å, boxes (nframes, 3, 3) or None)."""
    boxes = None
    if positions is None:
        xyz = system.positions[atoms][None]
        blocks = [(xyz, None)]
    elif hasattr(positions, "chunks") and not hasattr(positions, "positions"):  # a Trajectory
        blocks = ((b.positions, b.boxes) for b in positions.chunks(256, atoms=atoms))
    else:
        xyz = np.asarray(getattr(positions, "positions", positions))
        boxes = getattr(positions, "boxes", None)
        if xyz.ndim == 2:
            xyz = xyz[None]
            boxes = None if boxes is None else np.asarray(boxes)[None]
        if xyz.shape[1:] != (system.natoms, 3):
            raise ValueError(f"positions must be (nframes, {system.natoms}, 3)")
        blocks = [(xyz[:, atoms], boxes)]
    for xyz, own in blocks:
        if box is False:
            yield xyz, None
            continue
        use = own if box is None else np.asarray(box, np.float64)
        if use is not None and use.ndim < 3:  # one box, (3, 3) or lengths and angles
            use = np.broadcast_to(use, (len(xyz),) + use.shape)
        yield xyz, use


def _frame_inputs(X, bx, nco_l, skip, chain, breaks):
    """Per frame: nm float32 coordinates, chain segments, linked flags and box arguments."""
    from .pbc import _prepare, minimum_image

    box, inv, periodic, ortho = _prepare(bx)
    n = len(chain)
    if breaks:
        same = chain[1:] == chain[:-1]
        if n > 1:
            gap = minimum_image(X[nco_l[1:, 0]] - X[nco_l[:-1, 1]], box if periodic else None)
            same &= ~(skip[1:] | skip[:-1] | ((gap * gap).sum(1) > 2.5 * 2.5))
        seg = np.concatenate([[0], np.cumsum(~same)]).astype(np.int64)
        linked = np.concatenate([[False], same])
    else:
        seg = chain
        linked = np.arange(n) > 0
    nm = (np.asarray(X, dtype=np.float64) * 0.1).astype(np.float32)
    return nm, seg, linked, box * 0.1, inv * 10.0, periodic, ortho


def _local_backbone(system):
    nco, ca, skip, chain, is_pro = _backbone(system)
    atoms = np.unique(np.concatenate([nco[nco >= 0], ca[ca >= 0]]))
    if len(atoms) == 0:
        atoms = np.zeros(1, np.int64)
    nco_l = np.where(nco < 0, 0, np.searchsorted(atoms, nco))
    ca_l = np.where(ca < 0, 0, np.searchsorted(atoms, ca))
    return atoms, nco_l, ca_l, skip, chain, is_pro


def dssp(system, positions=None, simplified: bool = False, box=None,
         breaks: bool = True) -> np.ndarray:  # fmt: skip
    """DSSP codes, shape (nframes, nresidues).

    ``positions``: one frame, a (nframes, natoms, 3) array, a Frames block or
    a Trajectory (read chunk by chunk, backbone atoms only).  ``box``: a (3, 3)
    box or one per frame (Å) for minimum-image distances; Frames and
    trajectories use their own boxes unless ``box`` is given (``box=False``
    ignores them).  ``breaks``: cut chains at C-N gaps over 2.5 Å as the
    DSSP program does (False: topology chains only, as mdtraj).
    """
    atoms, nco_l, ca_l, skip, chain, is_pro = _local_backbone(system)
    table = CODES if not simplified else np.array([_SIMPLE[c] for c in CODES])
    rows = []
    for xyz, boxes in _dssp_blocks(system, positions, box, atoms):
        for f in range(len(xyz)):
            nm, seg, linked, b, inv, periodic, ortho = _frame_inputs(
                xyz[f], None if boxes is None else boxes[f], nco_l, skip, chain, breaks
            )
            rows.append(table[_frame(nm, nco_l, ca_l, is_pro, seg, skip, linked, b, inv,
                                     periodic, ortho)])  # fmt: skip
    out = np.array(rows, dtype="<U2").reshape(len(rows), system.nresidues)
    out[:, skip] = "NA"
    return out


def backbone_hbonds(system, positions=None, box=None, breaks: bool = True):
    """Kabsch-Sander backbone H-bonds of one frame: (donor residue, acceptor residue, kcal/mol).

    ``box`` and ``breaks`` as in ``dssp``."""
    atoms, nco_l, ca_l, skip, chain, is_pro = _local_backbone(system)
    xyz, boxes = next(_dssp_blocks(system, positions, box, atoms))
    nm, _, linked, b, inv, periodic, ortho = _frame_inputs(
        xyz[0], None if boxes is None else boxes[0], nco_l, skip, chain, breaks
    )
    hb, he = _hbonds(nm, nco_l, ca_l, is_pro, skip, linked, b, inv, periodic, ortho)
    donor, slot = np.nonzero(hb >= 0)
    return donor, hb[donor, slot], he[donor, slot].astype(np.float64)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def chimerax_ss(system, positions=None, energy_cutoff: float = -0.5, min_helix_len: int = 3,
                min_strand_len: int = 3) -> np.ndarray:  # fmt: skip
    """Per-residue secondary structure 'H' (helix), 'S' (strand) or 'O', as ChimeraX assigns it.

    A port of ChimeraX's own Kabsch-Sander implementation (atomstruct
    CompSS.cpp, Copyright Regents of the University of California, LGPL 2.1),
    which matchmaker uses to guide sequence alignment.  It differs from
    ``dssp`` (the mdtraj DSSP 2.2 port): hydrogens named H are used when
    present, residues are consecutive in residue order (bonded neighbors get
    rebuilt imide hydrogens), helices of any kind (3-10, alpha, pi) count,
    beta bulges merge ladders, ladders shorter than ``min_strand_len`` are
    dropped, and a helix overrides a strand.
    """
    from .spatial import pairs_within

    s = system
    xyz = np.asarray(s.positions if positions is None else positions, dtype=np.float64)
    names, res = s.atoms["name"], s.atoms["residue"]
    nres = s.nresidues
    first = {}
    for nm in ("C", "N", "CA", "O", "H", "CD"):
        a = np.full(nres, -1, np.int64)
        ids = np.flatnonzero(names == nm)
        r, k = np.unique(res[ids], return_index=True)
        a[r] = ids[k]
        first[nm] = a
    out = np.full(nres, "O", dtype="<U1")
    rlist = np.flatnonzero((first["C"] >= 0) & (first["N"] >= 0) & (first["CA"] >= 0)
                           & (first["O"] >= 0))  # fmt: skip
    L = len(rlist)
    if L == 0:
        return out
    c, n, o, ca = (xyz[first[k][rlist]] for k in ("C", "N", "O", "CA"))
    h = np.full((L, 3), np.nan)
    has_h = first["H"][rlist] >= 0
    h[has_h] = xyz[first["H"][rlist][has_h]]
    bi, bj = s.bonds["i"], s.bonds["j"]
    ri, rj = res[bi], res[bj]
    linked = set(zip(np.minimum(ri, rj).tolist(), np.maximum(ri, rj).tolist(), strict=True))
    for k in range(1, L):
        pair = (int(rlist[k - 1]), int(rlist[k]))
        if has_h[k] or pair not in linked:
            continue
        n2ca, n2c, c2o = _unit(ca[k] - n[k]), _unit(c[k - 1] - n[k]), _unit(o[k - 1] - c[k - 1])
        opp = _unit(_unit(n2ca + n2c) + c2o)
        h[k] = n[k] - opp * 1.01
    cd = first["CD"][rlist]
    is_pro = np.array([cd[k] >= 0 and s.find_bond(first["N"][rlist[k]], cd[k]) is not None
                       for k in range(L)])  # fmt: skip

    # hydrogen bonds O(a)...H-N(d) for residues a, d with N atoms within 10 A, |a - d| >= 2
    pi, pj, _ = pairs_within(n, 10.0)
    far = pj > pi + 1
    pi, pj = pi[far], pj[far]
    acc = np.concatenate([pi, pj])
    don = np.concatenate([pj, pi])
    ok = ~is_pro[don] & ~np.isnan(h[don, 0])
    rcn2 = ((c[acc] - n[don]) ** 2).sum(1)
    ok &= rcn2 <= 49.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r_on = np.linalg.norm(o[acc] - n[don], axis=1)
        r_ch = np.linalg.norm(c[acc] - h[don], axis=1)
        r_oh = np.linalg.norm(o[acc] - h[don], axis=1)
        energy = 0.42 * 0.20 * 332 * (1 / r_on + 1 / r_ch - 1 / r_oh - 1 / np.sqrt(rcn2))
    ok &= energy < np.float32(energy_cutoff)
    hb = set(zip(acc[ok].tolist(), don[ok].tolist(), strict=True))

    DONOR, ACCEPTOR, GAP, HELIX = {}, {}, {}, {}
    for t, bit in zip((3, 4, 5), (0, 4, 8), strict=True):
        DONOR[t], ACCEPTOR[t], GAP[t], HELIX[t] = 1 << bit, 2 << bit, 4 << bit, 8 << bit
    flags = [0] * L
    for t in (3, 4, 5):
        for i in range(L - t):
            if (i, i + t) in hb:
                flags[i] |= ACCEPTOR[t]
                for j in range(1, t):
                    flags[i + j] |= GAP[t]
                flags[i + t] |= DONOR[t]
        for i in range(1, L - t):
            if flags[i - 1] & ACCEPTOR[t] and flags[i] & ACCEPTOR[t]:
                for j in range(t):
                    flags[i + j] |= HELIX[t]

    helices = []
    first_i, cur_type, acc_run, initial_acc = -1, 0, 0, False
    for i in range(L):
        f = flags[i]
        htype, hflags, acc_only = 0, 0, False
        if f & HELIX[3]:
            htype, hflags = 3, ACCEPTOR[3] | DONOR[3] | GAP[3]
            acc_only = not (f & DONOR[3])
        elif f & (HELIX[4] | HELIX[5]):
            htype = 4
            hflags = ACCEPTOR[4] | DONOR[4] | GAP[4] | ACCEPTOR[5] | DONOR[5] | GAP[5]
            acc_only = bool(f & ACCEPTOR[4]) and not (f & DONOR[4])
        if htype and (f & hflags):
            if first_i < 0:
                first_i, cur_type, initial_acc = i, htype, True
            elif htype != cur_type:
                if i - first_i >= min_helix_len:
                    helices.append((first_i, i - 1))
                first_i, cur_type, acc_run = i, htype, 0
            else:
                initial_acc = initial_acc and acc_only
            if initial_acc:
                initial_acc = acc_only or i == first_i
            elif acc_only:
                if acc_run > 0:
                    if i - 1 - first_i >= min_helix_len:
                        helices.append((first_i, i - 2))
                    first_i, cur_type, acc_run, initial_acc = i - 1, htype, 0, True
                else:
                    acc_run += 1
            else:
                acc_run = 0
        elif first_i >= 0:
            if i - first_i >= min_helix_len:
                helices.append((first_i, i - 1))
            first_i, acc_run = -1, 0
    if first_i >= 0 and L - first_i >= min_helix_len:
        helices.append((first_i, L - 1))

    # bridges between residues with N atoms within 20 A
    bridge = {}
    bi2, bj2, _ = pairs_within(n, 20.0)
    for i, j in zip(bi2.tolist(), bj2.tolist(), strict=True):
        if i >= L - 1:
            continue
        if (i > 0 and (i - 1, j) in hb and (j, i + 1) in hb) or \
                (j < L - 1 and (j - 1, i) in hb and (i, j + 1) in hb):  # fmt: skip
            bridge[(i, j)] = "P"
        elif ((i, j) in hb and (j, i) in hb) or (
            i > 0 and j < L - 1 and (i - 1, j + 1) in hb and (j - 1, i + 1) in hb
        ):
            bridge[(i, j)] = "A"
    ladders = []  # [type, start0, end0, start1, end1, is_bulge]

    def ladder(kind, s1, e1, s2, e2, bulge=False):
        return [kind, min(s1, e1), max(s1, e1), min(s2, e2), max(s2, e2), bulge]

    for i, j in sorted(bridge):
        kind = bridge[(i, j)]
        k = 0
        if kind == "P":
            while i + k < L and j + k < L and bridge.get((i + k, j + k)) == "P":
                bridge[(i + k, j + k)] = "p"
                k += 1
            ladders.append(ladder(1, i, i + k - 1, j, j + k - 1))
        elif kind == "A":
            while i + k < L and j - k >= 0 and bridge.get((i + k, j - k)) == "A":
                bridge[(i + k, j - k)] = "a"
                k += 1
            ladders.append(ladder(2, i, i + k - 1, j - k + 1, j))

    def merge(l1, l2):
        if l1[0] != l2[0]:
            return None
        if l1[1] > l2[1]:
            l1, l2 = l2, l1
        d0 = l2[1] - l1[2]
        if d0 < 0 or d0 > 4:
            return None
        d1 = l2[3] - l1[4] if l1[0] == 1 else l1[3] - l2[4]
        if d1 < 0 or d1 > 4 or (d0 > 1 and d1 > 1):
            return None
        if l1[0] == 1:
            return ladder(1, l1[1], l2[2], l1[3], l2[4], True)
        return ladder(2, l1[1], l2[2], l2[3], l1[4], True)

    merged = True
    while merged:
        merged = False
        for a in range(len(ladders)):
            if ladders[a][5]:
                continue
            for b in range(a + 1, len(ladders)):
                if ladders[b][5]:
                    continue
                m = merge(ladders[a], ladders[b])
                if m is not None:
                    ladders = [x for k, x in enumerate(ladders) if k not in (a, b)] + [m]
                    merged = True
                    break
            if merged:
                break
    short = min_strand_len - 1
    ladders = [x for x in ladders if x[2] - x[1] >= short and x[4] - x[3] >= short]
    ss = np.full(L, "O", dtype="<U1")
    for x in ladders:
        ss[x[1] : x[2] + 1] = "S"
        ss[x[3] : x[4] + 1] = "S"
    for a, b in helices:
        ss[a : b + 1] = "H"
    out[rlist] = ss
    return out


BackboneDihedrals = namedtuple("BackboneDihedrals", "phi psi omega")


def backbone_dihedrals(system, positions=None, box=None) -> BackboneDihedrals:
    """Backbone phi, psi and omega per residue in degrees, each (nframes, nresidues).

    phi(i) = C(i-1)-N(i)-CA(i)-C(i), psi(i) = N(i)-CA(i)-C(i)-N(i+1) and
    omega(i) = CA(i-1)-C(i-1)-N(i)-CA(i), the peptide bond before residue i,
    as mdtraj and the PDB define them.  Residues are linked when consecutive
    in one chain with C-N under 2.5 Å in the first frame (the DSSP break
    rule); angles lacking a linked neighbor or a backbone atom are NaN.
    ``box`` (3x3, Å) applies the minimum image to each bond vector.
    """
    from .pbc import dihedrals, minimum_image

    xyz = _frames(system, positions)
    nco, ca, _, chain, _ = _backbone(system)
    n_at, c_at = nco[:, 0], nco[:, 1]
    nres = system.nresidues
    have = (n_at >= 0) & (c_at >= 0) & (ca >= 0)
    sn, sc, sca = (np.where(a < 0, 0, a) for a in (n_at, c_at, ca))
    linked = np.zeros(nres, bool)  # residue i joined to residue i - 1
    if nres > 1:
        cand = np.flatnonzero(have[1:] & have[:-1] & (chain[1:] == chain[:-1])) + 1
        first = np.asarray(xyz[0], dtype=np.float64)
        gap = minimum_image(first[sn[cand]] - first[sc[cand - 1]], box)
        linked[cand[np.linalg.norm(gap, axis=1) < 2.5]] = True
    after = np.zeros(nres, bool)
    after[:-1] = linked[1:]
    out = [np.full((len(xyz), nres), np.nan) for _ in range(3)]
    r, q = np.flatnonzero(linked), np.flatnonzero(after)
    for f, frame in enumerate(xyz):
        p = np.asarray(frame, dtype=np.float64)
        out[0][f, r] = np.degrees(dihedrals(p[sc[r - 1]], p[sn[r]], p[sca[r]], p[sc[r]], box))
        out[1][f, q] = np.degrees(dihedrals(p[sn[q]], p[sca[q]], p[sc[q]], p[sn[q + 1]], box))
        out[2][f, r] = np.degrees(dihedrals(p[sca[r - 1]], p[sc[r - 1]], p[sn[r]], p[sca[r]], box))
    return BackboneDihedrals(*out)
