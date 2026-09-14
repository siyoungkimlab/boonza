"""Amber prmtop/parm7 topologies and inpcrd/rst7 coordinates, read as msys reads them.

    s = boonza.load("complex.prmtop", coordinates="complex.rst7")

The force field becomes the tables msys's own converter (``msys.LoadPrmTop``)
produces:

- ``stretch_harm``, ``angle_harm`` (theta0 in degrees);
- ``dihedral_trig``, with the zero-phase terms of each atom quartet merged
  into one Fourier series and phases of +-180 degrees turned into a sign;
- ``pair_12_6_es`` for the 1-4 pairs, scaled by SCEE/SCNB (1.2 and 2.0 when
  the file does not give them);
- ``vdw_12_6`` with arithmetic/geometric (Lorentz-Berthelot) combining,
  sigma and epsilon from the A/B coefficients of each atom's type, and the
  Amber atom type in a ``type`` column;
- ``exclusion``, and CMAP grids (``torsiontorsion_cmap`` with ``cmap1``,
  ``cmap2`` ... auxiliary tables).

Charges are divided by 18.2223 and elements are guessed from masses, as msys
does (so a hydrogen-mass-repartitioned H of 3 u reads as helium).  All
residues go into one chain.  H-H bonds are not made: that is how Amber writes
rigid water.
"""

from __future__ import annotations

import bz2
import gzip
import math
import os
import re

import numpy as np

from .._columns import STR
from ..elements import guess_atomic_number
from ..system import System
from ..terms import ParamTable

CHARGE_SCALE = 18.2223
VELOCITY_SCALE = 20.455  # Amber velocity units to Å/ps
_POINTERS = ("Natom Ntypes Nbonh Nbona Ntheth Ntheta Nphih Nphia Jparm Nparm Nnb Nres Mbona "
             "Mtheta Mphia Numbnd Numang Nptra Natyp Nphb Ifpert Nbper Ngper Ndper Mbper Mgper "
             "Mdper IfBox Nmxrs IfCap").split()  # fmt: skip


class PrmtopError(ValueError):
    """Unreadable or unsupported prmtop content."""


def _read_text(path) -> str:
    path = os.fspath(path)
    opener = gzip.open if path.endswith(".gz") else bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt") as fh:
        return fh.read()


def _sections(text: str) -> dict:
    """{flag: (width, data)}: each section's field width and its lines joined."""
    lines = text.split("\n")
    out = {}
    k, n = 1, len(lines)  # the first line is the version
    while k < n and not (len(lines[k]) > 6 and lines[k].startswith("%FLAG")):
        k += 1
    while k < n:
        flag = lines[k][5:].strip()
        k += 1
        while k < n and (not lines[k] or lines[k].startswith("%COMMENT")):
            k += 1
        m = re.search(r"\(\s*(\d+)\s*([A-Za-z])\s*(\d+)", lines[k] if k < n else "")
        if m is None:
            raise PrmtopError(f"section {flag}: expected %FORMAT(...)")
        width = int(m.group(3))
        k += 1
        data = []
        while k < n and not lines[k].startswith("%FLAG"):
            if lines[k]:
                data.append(lines[k])
            k += 1
        out[flag] = (width, "".join(data))
    return out


class _Sections:
    def __init__(self, text: str):
        self.map = _sections(text)

    def __contains__(self, name) -> bool:
        return name in self.map

    def raw(self, name: str, count: int) -> np.ndarray:
        if name not in self.map:
            raise PrmtopError(f"missing section {name}")
        width, data = self.map[name]
        if count == 0:
            return np.empty(0, f"S{width}")
        if len(data) < count * width:
            raise PrmtopError(f"section {name}: expected {count} values")
        return np.frombuffer(data[: count * width].encode("latin-1"), dtype=f"S{width}")

    def ints(self, name: str, count: int) -> np.ndarray:
        return self.raw(name, count).astype(np.int64)

    def floats(self, name: str, count: int) -> np.ndarray:
        return np.char.replace(self.raw(name, count), b"D", b"E").astype(np.float64)

    def entries(self, *sections, width: int) -> np.ndarray:
        """Rows of ``width`` ints from (section, row count) pairs, concatenated."""
        blocks = [self.ints(name, width * count).reshape(-1, width) for name, count in sections]
        return np.concatenate(blocks) if blocks else np.empty((0, width), np.int64)

    def strs(self, name: str, count: int) -> np.ndarray:
        return np.char.strip(self.raw(name, count).astype(str)).astype(STR)


def load_prmtop(path, coordinates=None, structure_only: bool = False,
                without_tables: bool = False) -> System:  # fmt: skip
    """Read an Amber prmtop (or parm7) file, optionally with coordinates.

    ``coordinates``: an inpcrd/rst7 file (ASCII) or an Amber NetCDF restart
    (positions, velocities and the cell).  ``structure_only`` or
    ``without_tables`` skip the force-field tables (bonds are still made).
    """
    S = _Sections(_read_text(path))
    ptr = dict(zip(_POINTERS, S.ints("POINTERS", len(_POINTERS)).tolist(), strict=True))
    if ptr["Nphb"] > 0:
        coefs = [S.floats(name, ptr["Nphb"]) for name in ("HBOND_ACOEF", "HBOND_BCOEF", "HBCUT")]
        if any(c.any() for c in coefs):
            raise PrmtopError("NPHB > 0: 10-12 hydrogen-bond terms with nonzero coefficients")
    if ptr["Ifpert"] > 0:
        raise PrmtopError("IFPERT > 0: perturbation information is not supported")
    natoms, nres = ptr["Natom"], ptr["Nres"]
    resptr = S.ints("RESIDUE_POINTER", nres)
    charge = S.floats("CHARGE", natoms) / CHARGE_SCALE
    mass = S.floats("MASS", natoms)
    anum = guess_atomic_number(mass)

    s = System(os.fspath(path))
    ct = s.add_ct()
    s._chains.append(1, {"ct": ct.id, "name": np.array([""], STR), "segid": np.array([""], STR)})
    s._residues.append(nres, {"chain": np.zeros(nres, np.int64),
                              "resid": np.arange(1, nres + 1, dtype=np.int64),
                              "name": S.strs("RESIDUE_LABEL", nres),
                              "insertion": np.full(nres, "", dtype=STR)})  # fmt: skip
    residue = np.searchsorted(resptr - 1, np.arange(natoms), side="right") - 1
    s._atoms.append(natoms, {"residue": residue, "name": S.strs("ATOM_NAME", natoms),
                             "anum": anum, "charge": charge, "mass": mass})  # fmt: skip
    s._cache.clear()

    bonds = S.entries(
        ("BONDS_INC_HYDROGEN", ptr["Nbonh"]), ("BONDS_WITHOUT_HYDROGEN", ptr["Nbona"]), width=3
    )
    pairs = bonds[:, :2] // 3
    real = ~((anum[pairs[:, 0]] == 1) & (anum[pairs[:, 1]] == 1))  # H-H: Amber's rigid water
    s.add_bonds(pairs[real])
    if not (structure_only or without_tables):
        _force_field(s, S, ptr, pairs[real], bonds[real, 2] - 1, anum, charge)
    if coordinates is not None:
        pos, vel, cell = read_amber_coordinates(coordinates, natoms)
        s.positions = pos
        if vel is not None:
            s.atoms["vel"] = vel
        if cell is not None:
            s.cell = cell
    return s


def _force_field(s: System, S: _Sections, ptr: dict, bond_atoms, bond_params, anum, charge):
    natoms = ptr["Natom"]
    st = s.add_table_from_schema("stretch_harm")
    st.params.add_params(ptr["Numbnd"], r0=S.floats("BOND_EQUIL_VALUE", ptr["Numbnd"]),
                         fc=S.floats("BOND_FORCE_CONSTANT", ptr["Numbnd"]))  # fmt: skip
    st.add_terms(bond_atoms, params=bond_params)

    at = s.add_table_from_schema("angle_harm")
    at.params.add_params(ptr["Numang"],
                         theta0=S.floats("ANGLE_EQUIL_VALUE", ptr["Numang"]) * 180 / math.pi,
                         fc=S.floats("ANGLE_FORCE_CONSTANT", ptr["Numang"]))  # fmt: skip
    angles = S.entries(
        ("ANGLES_INC_HYDROGEN", ptr["Ntheth"]), ("ANGLES_WITHOUT_HYDROGEN", ptr["Ntheta"]), width=4
    )
    at.add_terms(angles[:, :3] // 3, params=angles[:, 3] - 1)

    pairs = _torsions(s, S, ptr)
    _nonbonded(s, S, ptr, pairs, charge)

    if ptr["Nnb"] > 0:
        count = S.ints("NUMBER_EXCLUDED_ATOMS", natoms)
        listed = S.ints("EXCLUDED_ATOMS_LIST", ptr["Nnb"])
        ai = np.repeat(np.arange(natoms), count)
        aj = listed[: len(ai)]
        keep = aj != 0
        s.add_table_from_schema("exclusion").add_terms(np.column_stack([ai[keep], aj[keep] - 1]))
    _cmap(s, S)


def _torsions(s: System, S: _Sections, ptr: dict) -> list:
    n = ptr["Nptra"]
    phase = S.floats("DIHEDRAL_PHASE", n)
    fc = S.floats("DIHEDRAL_FORCE_CONSTANT", n)
    period = S.floats("DIHEDRAL_PERIODICITY", n)
    scee = S.floats("SCEE_SCALE_FACTOR", n) if "SCEE_SCALE_FACTOR" in S else np.full(n, 1.2)
    scnb = S.floats("SCNB_SCALE_FACTOR", n) if "SCNB_SCALE_FACTOR" in S else np.full(n, 2.0)
    entries = S.entries(
        ("DIHEDRALS_INC_HYDROGEN", ptr["Nphih"]),
        ("DIHEDRALS_WITHOUT_HYDROGEN", ptr["Nphia"]),
        width=5,
    )
    rows: list[list[float]] = []  # phi0, fc0 ... fc6
    terms, merged, pairs = [], {}, []
    for ai, aj, ak, al, ind in entries.tolist():
        ai, aj, ak, al, ind = ai // 3, aj // 3, ak // 3, al // 3, ind - 1
        needs_pair = ak >= 0  # a negative third atom: no 1-4 pair (ring or multi-term)
        ak, al = abs(ak), abs(al)  # a negative fourth atom marks an improper
        if needs_pair:
            pairs.append((min(ai, al), max(ai, al), scee[ind], scnb[ind]))
        fc_orig = float(fc[ind])
        fc_phased = fc_orig
        degrees = float(phase[ind]) * 180 / math.pi
        if 179.9 < abs(degrees) < 180.1:  # Amber files approximate pi by 3.141594
            degrees, fc_phased = 0.0, -fc_orig
        key = (ai, aj, ak, al)
        if degrees == 0:
            p = merged.get(key)
            if p is None:
                p = merged[key] = len(rows)
                rows.append([0.0] * 8)
                terms.append((key, p))
        else:
            p = len(rows)
            rows.append([degrees] + [0.0] * 7)
            terms.append((key, p))
        col = 1 + int(period[ind])
        if not 1 <= col <= 7:
            raise PrmtopError(f"dihedral periodicity {period[ind]} is not in 0..6")
        old = rows[p][col]
        if old == 0:
            rows[p][col] = fc_phased
        elif old != fc_phased:
            raise PrmtopError(f"conflicting force constants for period {period[ind]} of {key}")
        rows[p][1] += fc_orig
    dt = s.add_table_from_schema("dihedral_trig")
    values = np.array(rows, dtype=np.float64).reshape(-1, 8)
    names = ["phi0"] + [f"fc{k}" for k in range(7)]
    pids = dt.params.add_params(len(values), **{nm: values[:, c] for c, nm in enumerate(names)})
    if terms:
        dt.add_terms(np.array([t for t, _ in terms], np.int64), params=pids[[p for _, p in terms]])
    return pairs


def _lj(S: _Sections, ntypes: int, types_i, types_j):
    inds = S.ints("NONBONDED_PARM_INDEX", ntypes * ntypes)
    npair = ntypes * (ntypes + 1) // 2
    acoef = S.floats("LENNARD_JONES_ACOEF", npair)
    bcoef = S.floats("LENNARD_JONES_BCOEF", npair)
    ico = inds[ntypes * (types_i - 1) + types_j - 1]
    return acoef[ico - 1], bcoef[ico - 1]


def _nonbonded(s: System, S: _Sections, ptr: dict, pairs: list, charge) -> None:
    natoms, ntypes = ptr["Natom"], ptr["Ntypes"]
    types = S.ints("ATOM_TYPE_INDEX", natoms)
    c12, c6 = _lj(S, ntypes, types, types)
    ok = (c12 != 0) & (c6 != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(ok, (c12 / c6) ** (1.0 / 6.0), 0.0)
        epsilon = np.where(ok, c6 * c6 / (4 * c12), 0.0)
    nb = s.add_nonbonded_from_schema("vdw_12_6", "arithmetic/geometric")
    nb.params.add_prop("type", str)
    pids = nb.params.add_params(natoms, sigma=sigma, epsilon=epsilon,
                                type=S.strs("AMBER_ATOM_TYPE", natoms))  # fmt: skip
    nb.add_terms(np.arange(natoms)[:, None], params=pids)

    pt = s.add_table_from_schema("pair_12_6_es")
    if pairs:
        arr = np.array(pairs, dtype=np.float64)
        pi, pj = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
        es, lj = 1.0 / arr[:, 2], 1.0 / arr[:, 3]
        a12, b6 = _lj(S, ntypes, types[pi], types[pj])
        pids = pt.params.add_params(len(arr), aij=lj * a12, bij=lj * b6,
                                    qij=es * charge[pi] * charge[pj])  # fmt: skip
        pt.add_terms(np.column_stack([pi, pj]), params=pids)


def _cmap(s: System, S: _Sections) -> None:
    prefix = "" if "CMAP_COUNT" in S else "CHARMM_" if "CHARMM_CMAP_COUNT" in S else None
    if prefix is None:
        return
    nterms, ntables = S.ints(prefix + "CMAP_COUNT", 2).tolist()
    resolution = S.ints(prefix + "CMAP_RESOLUTION", ntables)
    for i, n in enumerate(resolution.tolist()):
        grid = S.floats(f"{prefix}CMAP_PARAMETER_{i + 1:02d}", n * n)
        axis = -180.0 + np.arange(n) * (360.0 / n)
        aux = ParamTable()
        for prop in ("phi", "psi", "energy"):
            aux.add_prop(prop)
        aux.add_params(n * n, phi=np.repeat(axis, n), psi=np.tile(axis, n), energy=grid)
        s.aux_tables[f"cmap{i + 1}"] = aux
    idx = S.ints(prefix + "CMAP_INDEX", 6 * nterms).reshape(-1, 6) - 1
    atoms = idx[:, [0, 1, 2, 3, 1, 2, 3, 4]]
    table = s.add_table_from_schema("torsiontorsion_cmap")
    pids = table.params.add_params(nterms, cmapid=np.array([f"cmap{k + 1}" for k in idx[:, 5]],
                                                           dtype=STR))  # fmt: skip
    table.add_terms(atoms, params=pids)


def read_amber_coordinates(path, natoms: int | None = None):
    """(positions, velocities or None, cell or None) from an inpcrd/rst7 file or an
    Amber NetCDF restart.  Velocities are converted to Å/ps."""
    path = os.fspath(path)
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head[:3] == b"CDF" or head == b"\x89HDF":
        from .ncdf import NCDFTrajectory

        frame = NCDFTrajectory(path)[0]
        if natoms is not None and len(frame.positions) != natoms:
            raise PrmtopError(f"{path} has {len(frame.positions)} atoms, want {natoms}")
        return (frame.positions.astype(np.float64), None,
                frame.box if frame.box.any() else None)  # fmt: skip
    lines = _read_text(path).split("\n")
    m = re.match(r"\s*(\d+)", lines[1] if len(lines) > 1 else "")
    if m is None:
        raise PrmtopError(f"{path}: no atom count on the second line")
    n = int(m.group(1))
    if natoms is not None and n != natoms:
        raise PrmtopError(f"{path} has {n} atoms, want {natoms}")
    per = (n + 1) // 2  # lines holding n triples, two per line

    def triples(block):
        values = [float(line[12 * k : 12 * k + 12]) for line in block for k in range(6)
                  if line[12 * k : 12 * k + 12].strip()]  # fmt: skip
        if len(values) < 3 * n:
            raise PrmtopError(f"{path}: expected {3 * n} values")
        return np.array(values[: 3 * n]).reshape(n, 3)

    pos = triples(lines[2 : 2 + per])
    rest = [line for line in lines[2 + per :] if line.strip()]
    vel = None
    if len(rest) >= per and per > 0 and len(rest) != 1:
        vel = triples(rest[:per]) * VELOCITY_SCALE
        rest = rest[per:]
    cell = None
    if rest:
        from .pdb import cell_from_lengths_angles

        box = [float(rest[0][12 * k : 12 * k + 12]) for k in range(6)
               if rest[0][12 * k : 12 * k + 12].strip()]  # fmt: skip
        if len(box) >= 3:
            box += [90.0] * (6 - len(box))
            cell = cell_from_lengths_angles(*box[:6])
    return pos, vel, cell
