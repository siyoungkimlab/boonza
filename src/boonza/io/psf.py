"""CHARMM, NAMD and X-PLOR PSF topologies.

    s = boonza.load("step5_input.psf", coordinates="step5_input.pdb")

A PSF lists atoms (segment, residue, name, CHARMM type, charge, mass) and
bonds.  The parameters live in separate CHARMM files, so no force-field
tables are made here; for a full force field go through OpenMM
(``boonza.from_openmm(psf.topology, psf.createSystem(params))``).

Standard and EXT (extended) layouts are read by their fixed columns, with
or without X-PLOR type names and CHEQ columns; files flagged NAMD are split
on whitespace.  Each segment becomes a chain named after its segid.  A
residue number with a letter (``1H``) gives the insertion code.  Elements
are guessed from masses, so lone pairs and Drude particles (mass below 1)
are pseudo particles.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .._columns import STR
from ..elements import guess_atomic_number
from ..system import System
from .amber import _read_text


class PSFError(ValueError):
    """Unreadable PSF content."""


_RESID = re.compile(r"\s*(-?\d+)(\S*)")


def _section(lines: list[str], tag: str):
    """(line index of the ``!TAG`` header, its count) or (None, 0)."""
    for k, line in enumerate(lines):
        if f"!{tag}" in line:
            return k, int(line.split()[0])
    return None, 0


def _ints_after(lines: list[str], start: int, count: int) -> np.ndarray:
    vals: list[str] = []
    k = start + 1
    while len(vals) < count and k < len(lines):
        vals.extend(lines[k].split())
        k += 1
    if len(vals) < count:
        raise PSFError(f"expected {count} numbers after line {start + 1}")
    return np.array(vals[:count], dtype=np.int64)


def _atom_fields(line: str, layout: str):
    if layout == "NAMD":
        f = line.split()
        return f[1], f[2], f[3], f[4], f[5], f[6], f[7]
    if layout == "EXT":
        seg, res, resname, name = line[11:19], line[20:28], line[29:37], line[38:46]
        rest = line[47:].split()
    else:
        seg, res, resname, name = line[9:13], line[14:18], line[19:23], line[24:28]
        rest = line[29:].split()
    if len(rest) < 3:
        raise PSFError(f"cannot read atom line {line!r}")
    return seg.strip(), res.strip(), resname.strip(), name.strip(), rest[0], rest[1], rest[2]


def load_psf(path, coordinates=None) -> System:
    """Read a PSF file; ``coordinates``: a file boonza can load with the same
    atoms (PDB, CRD-like formats, ...) or an (natoms, 3) array."""
    lines = _read_text(path).splitlines()
    header = lines[0].split() if lines else []
    if not header or header[0] != "PSF":
        raise PSFError(f"{path}: not a PSF file")
    layout = "NAMD" if "NAMD" in header else "EXT" if "EXT" in header else "STANDARD"
    start, natoms = _section(lines, "NATOM")
    if start is None:
        raise PSFError(f"{path}: no !NATOM section")
    rows = [_atom_fields(line, layout) for line in lines[start + 1 : start + 1 + natoms]]
    if len(rows) != natoms:
        raise PSFError(f"{path}: expected {natoms} atoms")
    seg, res, resname, name, types, charge, mass = (list(c) for c in zip(*rows, strict=True)) \
        if rows else ([], [], [], [], [], [], [])  # fmt: skip
    resid = np.empty(natoms, np.int64)
    insertion = []
    for k, text in enumerate(res):
        m = _RESID.match(text)
        if m is None:
            raise PSFError(f"{path}: bad residue number {text!r}")
        resid[k] = int(m.group(1))
        insertion.append(m.group(2))
    charge = np.array(charge, dtype=np.float64)
    mass = np.array(mass, dtype=np.float64)

    # residues and chains start wherever their identifying fields change
    seg_a = np.array(seg, dtype=STR)
    key = list(zip(seg, resid.tolist(), insertion, resname, strict=True))
    new_res = np.array([k == 0 or key[k] != key[k - 1] for k in range(natoms)], bool)
    new_chain = np.array([k == 0 or seg[k] != seg[k - 1] for k in range(natoms)], bool)
    residue = np.cumsum(new_res) - 1
    chain_of_res = (np.cumsum(new_chain) - 1)[new_res]
    first_res = np.flatnonzero(new_res)
    first_chain = np.flatnonzero(new_chain)

    s = System(os.fspath(path))
    ct = s.add_ct()
    s._chains.append(len(first_chain), {"ct": ct.id, "name": seg_a[first_chain],
                                        "segid": seg_a[first_chain]})  # fmt: skip
    s._residues.append(len(first_res), {
        "chain": chain_of_res, "resid": resid[first_res],
        "name": np.array(resname, dtype=STR)[first_res],
        "insertion": np.array(insertion, dtype=STR)[first_res]})  # fmt: skip
    s._atoms.append(natoms, {"residue": residue, "name": np.array(name, dtype=STR),
                             "anum": guess_atomic_number(mass), "charge": charge,
                             "mass": mass})  # fmt: skip
    s._cache.clear()
    s.atoms["type"] = np.array(types, dtype=STR)

    bstart, nbonds = _section(lines, "NBOND")
    if bstart is not None and nbonds:
        s.add_bonds(_ints_after(lines, bstart, 2 * nbonds).reshape(-1, 2) - 1)
    if coordinates is not None:
        if isinstance(coordinates, str | os.PathLike):
            from . import load

            other = load(coordinates)
            if other.natoms != natoms:
                raise PSFError(f"{coordinates} has {other.natoms} atoms, the PSF {natoms}")
            s.positions = other.positions
            if other.cell.any():
                s.cell = other.cell
        else:
            s.positions = np.asarray(coordinates, dtype=np.float64).reshape(natoms, 3)
    return s
