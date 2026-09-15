# SPDX-License-Identifier: LGPL-2.1-only
#
# Not under boonza's MIT license: this file ports LGPL code from MDTraj, so it
# is distributed under the GNU Lesser General Public License version 2.1
# (LICENSES/LGPL-2.1.txt; see NOTICE).  The Shrake-Rupley frame loop, sphere
# points and radii follow MDTraj's mdtraj/geometry/src/sasa.cpp and
# mdtraj/geometry/sasa.py, which carry this notice:
#
#   MDTraj: A Python Library for Loading, Saving, and Manipulating
#           Molecular Dynamics Trajectories.
#   Copyright 2012-2016 Stanford University and the Authors
#   Authors: Robert McGibbon, Peter Eastman
#
#   MDTraj is free software: you can redistribute it and/or modify it under
#   the terms of the GNU Lesser General Public License as published by the
#   Free Software Foundation, either version 2.1 of the License, or (at your
#   option) any later version.
"""Solvent accessible surface area (Shrake-Rupley), as mdtraj's ``shrake_rupley``.

    areas = boonza.sasa(system)                         # (nframes, natoms), Å^2
    areas = boonza.sasa(system, frames, mode="residue")

The same float32 sphere points, radii and arithmetic as mdtraj, so the
areas agree with it.  Kept apart from :mod:`boonza.analysis` because it is
LGPL-2.1 code (see the header of this file).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .hbonds import _frames

# mdtraj's radii (nm) for the elements proteins and common ligands contain
_SASA_RADII = {
    "H": 0.120, "He": 0.140, "Li": 0.076, "Be": 0.059, "B": 0.192, "C": 0.170, "N": 0.155,
    "O": 0.152, "F": 0.147, "Ne": 0.154, "Na": 0.102, "Mg": 0.086, "Al": 0.184, "Si": 0.210,
    "P": 0.180, "S": 0.180, "Cl": 0.181, "Ar": 0.188, "K": 0.138, "Ca": 0.114, "Sc": 0.211,
    "Ti": 0.200, "V": 0.200, "Cr": 0.200, "Mn": 0.200, "Fe": 0.200, "Co": 0.200, "Ni": 0.163,
    "Cu": 0.140, "Zn": 0.139, "Ga": 0.187, "Ge": 0.211, "As": 0.185, "Se": 0.190,
    "Br": 0.185, "Kr": 0.202, "Rb": 0.303, "Sr": 0.249, "Y": 0.200, "Zr": 0.200,
}  # fmt: skip


def _sphere_points(n: int) -> np.ndarray:
    """mdtraj's golden-section spiral, with its float/double promotions."""
    inc = np.float32(math.pi * (3.0 - math.sqrt(5.0)))
    offset = np.float32(2.0 / n)
    pts = np.empty((n, 3), np.float32)
    for i in range(n):
        y = np.float32(float(np.float32(i * offset)) - 1.0 + float(offset) / 2.0)
        r = np.float32(math.sqrt(1.0 - float(np.float32(y * y))))
        phi = np.float32(np.float32(i) * inc)
        pts[i] = (np.float32(math.cos(float(phi)) * float(r)), y,
                  np.float32(math.sin(float(phi)) * float(r)))  # fmt: skip
    return pts


@njit(parallel=True, cache=True)
def _sasa_frame(xyz, radii, points, noff, nbr, selected):
    n = xyz.shape[0]
    npts = points.shape[0]
    constant = np.float32(4.0 * math.pi / npts)
    areas = np.zeros(n, np.float32)
    for i in prange(n):
        if not selected[i]:
            continue
        ri = radii[i]
        count = 0
        k0 = 0
        m = noff[i + 1] - noff[i]
        for p in range(npts):
            px = xyz[i, 0] + ri * points[p, 0]
            py = xyz[i, 1] + ri * points[p, 1]
            pz = xyz[i, 2] + ri * points[p, 2]
            accessible = True
            for kk in range(k0, k0 + m):
                j = nbr[noff[i] + kk % m]
                dx, dy, dz = px - xyz[j, 0], py - xyz[j, 1], pz - xyz[j, 2]
                rj = radii[j]
                if dx * dx + dy * dy + dz * dz < rj * rj:
                    k0 = kk
                    accessible = False
                    break
            if accessible:
                count += 1
        areas[i] = np.float32(count) * constant * ri * ri
    return areas


def sasa(system, positions=None, probe_radius: float = 1.4, n_sphere_points: int = 960,
         mode: str = "atom", atoms=None, radii=None) -> np.ndarray:  # fmt: skip
    """Solvent accessible surface area (Å^2) per atom or residue, per frame (mdtraj).

    ``radii``: {element symbol: radius in Å} overriding mdtraj's defaults.
    ``atoms``: compute only these atoms (others report -1, as mdtraj);
    they are still occluded by every atom.  Frames are not made whole.
    """
    from .analysis import _ids
    from .elements import msys_symbol

    if mode not in ("atom", "residue"):
        raise ValueError('mode must be one of "residue", "atom"')
    xyz, _ = _frames(system, positions, None)
    table = {k: v * 10 for k, v in _SASA_RADII.items()}
    if radii:
        table.update(radii)
    symbols = [msys_symbol(int(z)) for z in system.atoms["anum"].tolist()]
    try:
        vdw = np.array([table[sym] for sym in symbols], np.float64)
    except KeyError as e:
        raise KeyError(f"no SASA radius for element {e.args[0]}; pass radii=") from None
    # mdtraj works in float32 nm; keep its arithmetic so sphere points agree
    r_nm = (np.float32(vdw / 10) + np.float32(probe_radius / 10)).astype(np.float32)
    points = _sphere_points(int(n_sphere_points))
    selected = np.ones(system.natoms, bool) if atoms is None else np.zeros(system.natoms, bool)
    if atoms is not None:
        selected[_ids(system, atoms)] = True
    groups = np.arange(system.natoms) if mode == "atom" else system.atoms["residue"]
    ngroups = system.natoms if mode == "atom" else system.nresidues
    out = np.zeros((len(xyz), ngroups), np.float64)
    if atoms is not None:
        out[:] = -1
        out[:, groups[selected]] = 0
    for f, frame in enumerate(xyz):
        nm = (frame * 0.1).astype(np.float32)
        pi, pj, d2 = _neighbors(nm, r_nm)
        order = np.argsort(pi, kind="stable")
        noff = np.zeros(system.natoms + 1, np.int64)
        np.cumsum(np.bincount(pi, minlength=system.natoms), out=noff[1:])
        areas = _sasa_frame(nm, r_nm, points, noff, pj[order], selected)
        np.add.at(out[f], groups[selected], areas[selected].astype(np.float64) * 100.0)
    return out


def _neighbors(nm, r_nm):
    """Directed pairs (i, j), i != j, with |ri - rj|^2 < (radius_i + radius_j)^2 in float32."""
    from .spatial import pairs_within

    i, j, _ = pairs_within(nm.astype(np.float64), 2 * float(r_nm.max()) + 1e-4)
    diff = nm[i] - nm[j]
    r2 = (diff * diff).sum(1, dtype=np.float32)
    cut = r_nm[i] + r_nm[j]
    keep = r2 < cut * cut
    i, j = i[keep], j[keep]
    return np.concatenate([i, j]), np.concatenate([j, i]), None
