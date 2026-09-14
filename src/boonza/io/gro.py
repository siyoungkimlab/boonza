"""GROMACS .gro files (first frame), read the way MDAnalysis does.

Positions and box are converted from nm to Å, velocities from nm/ps to Å/ps.
The coordinate precision is detected from the first atom line.  A new
residue starts whenever the residue number or name changes from one atom to
the next; residue numbers that wrap past 99999 are unwrapped.  Elements are
guessed from atom names and bonds from geometry (``guess_bonds=False`` to
skip).  The title becomes the ct name.
"""

from __future__ import annotations

import bz2
import gzip
import os
import re
import warnings

import numpy as np

from .._columns import STR
from ..system import System
from ._maeparse import read_text
from .dms import _factorize
from .pdb import _element

NM = 10.0  # Å per nm


class GroError(ValueError):
    """Malformed GRO content."""


def load_gro(path, guess_bonds: bool = True) -> System:
    path = os.fspath(path)
    lines = read_text(path).split("\n")
    if len(lines) < 3:
        raise GroError("a gro file needs a title, an atom count and a box line")
    title = lines[0].strip()
    try:
        n = int(lines[1])
    except ValueError:
        raise GroError(f"bad atom count line: {lines[1]!r}") from None
    if len(lines) < n + 3:
        raise GroError(f"expected {n} atom lines and a box line")
    atoms = [line.rstrip("\r") for line in lines[2 : 2 + n]]
    box = lines[2 + n]

    s = System(path)
    ct = s.add_ct(title)
    chain = s.add_chain(ct)
    if n:
        _read_atoms(s, chain, atoms)
    s.cell = _read_box(box)
    if guess_bonds and n:
        s.guess_bonds()
    s.name = path
    return s


def _column(mat: np.ndarray, a: int, b: int) -> np.ndarray:
    n = len(mat)
    return np.char.strip(np.ascontiguousarray(mat[:, a:b]).view(f"S{b - a}").reshape(n))


def _numbers(field: np.ndarray, what: str):
    blank = field == b""
    try:
        return np.where(blank, b"0", field).astype(np.float64), blank
    except ValueError:
        raise GroError(f"bad {what} field in atom lines") from None


def _read_atoms(s: System, chain, lines: list[str]) -> None:
    n = len(lines)
    cs = lines[0][25:].find(".") + 1  # field width, from the precision of the file
    if cs <= 0:
        raise GroError("cannot find coordinates in the first atom line")
    width = 20 + 6 * cs
    raw = "".join(line[:width].ljust(width) for line in lines).encode("latin-1", "replace")
    mat = np.frombuffer(raw, dtype="S1").reshape(n, width)

    try:
        resid = _column(mat, 0, 5).astype(np.int64)
    except ValueError:
        raise GroError("bad residue number in atom lines") from None
    resname = _column(mat, 5, 10).astype(STR)
    name = _column(mat, 10, 15).astype(STR)
    pos = np.empty((n, 3))
    vel = np.zeros((n, 3))
    missing_vel = np.zeros(n, bool)
    for k in range(3):
        values, blank = _numbers(_column(mat, 20 + cs * k, 20 + cs * (k + 1)), "coordinate")
        if blank.any():
            raise GroError("missing coordinates in atom lines")
        pos[:, k] = values * NM
        values, blank = _numbers(_column(mat, 20 + cs * (k + 3), 20 + cs * (k + 4)), "velocity")
        vel[:, k] = values * NM
        missing_vel |= blank
    if vel.any() and missing_vel.any():
        warnings.warn("Not all velocities were present; unset velocities set to zero",
                      stacklevel=3)  # fmt: skip

    # residue numbers are 5 digits wide: unwrap each return to 0 after the start
    wraps = np.flatnonzero(resid == 0)
    if len(wraps):
        starts = np.concatenate([wraps[:1], wraps[1:][np.diff(wraps) != 1]])
        for start in starts[starts > 0].tolist():
            resid[start:] += 100000

    new = np.ones(n, bool)
    new[1:] = (resid[1:] != resid[:-1]) | (resname[1:] != resname[:-1])
    first = np.flatnonzero(new)
    s._residues.append(len(first), {"chain": chain.id, "resid": resid[first],
                                    "name": resname[first]})  # fmt: skip
    combo, reps = _factorize(name, resname)
    guesses = [_element("", str(name[k]), str(resname[k])) for k in reps.tolist()]
    anum = np.array(guesses, np.int64)[combo]
    from ..elements import guessed_masses

    s._atoms.append(n, {"residue": np.cumsum(new) - 1, "name": name, "anum": anum, "pos": pos,
                        "vel": vel, "mass": guessed_masses(anum)})  # fmt: skip
    s._cache.clear()


def _read_box(line: str) -> np.ndarray:
    try:
        v = [float(x) for x in line.split()]
    except ValueError:  # fields may run together when the box is wide
        v = [float(x) for x in re.findall(r"(\d+\.\d{5})", line)]
    cell = np.zeros((3, 3))
    if len(v) == 3:
        np.fill_diagonal(cell, v)
    elif len(v) == 9:
        cell[0] = (v[0], v[3], v[4])
        cell[1] = (v[5], v[1], v[6])
        cell[2] = (v[7], v[8], v[2])
    else:
        raise GroError("GRO box line has neither 3 nor 9 entries")
    return cell * NM


def save_gro(system: System, path, precision: int = 3) -> None:
    """Write the system as .gro (nm); velocities are written when any is nonzero."""
    path = os.fspath(path)
    A, R = system._atoms, system._residues
    n = system.natoms
    res = A.column("residue")
    w = precision + 5
    pos = (A.column("pos") / NM).tolist()
    vel = A.column("vel") / NM
    has_vel = bool(vel.any())
    vel = vel.tolist()
    resids = (R.column("resid")[res] % 100000).tolist()
    resnames = [r[:5] for r in R.column("name")[res].tolist()]
    names = [a[:5] for a in A.column("name").tolist()]
    title = system._ct_names[0] if system.ncts and system._ct_names[0] else "written by boonza"
    lines = [title, f"{n:5d}"]
    for k in range(n):
        x, y, z = pos[k]
        line = (f"{resids[k]:5d}{resnames[k]:<5s}{names[k]:>5s}{(k + 1) % 100000:5d}"
                f"{x:{w}.{precision}f}{y:{w}.{precision}f}{z:{w}.{precision}f}")  # fmt: skip
        if has_vel:
            vx, vy, vz = vel[k]
            p = precision + 1
            line += f"{vx:{w}.{p}f}{vy:{w}.{p}f}{vz:{w}.{p}f}"
        lines.append(line)
    c = system.cell / NM
    if np.count_nonzero(c - np.diag(np.diag(c))) == 0:
        lines.append("".join(f"{c[k, k]:10.5f}" for k in range(3)))
    else:
        order = [c[0, 0], c[1, 1], c[2, 2], c[0, 1], c[0, 2], c[1, 0], c[1, 2], c[2, 0], c[2, 1]]
        lines.append("".join(f"{v:10.5f}" for v in order))
    opener = gzip.open if path.endswith(".gz") else bz2.open if path.endswith(".bz2") else open
    with opener(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
