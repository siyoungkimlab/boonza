"""GROMACS topologies (.top), read natively.

    s = boonza.load("topol.top", coordinates="conf.gro")
    s = boonza.load("topol.top", defines={"FLEXIBLE": ""}, include_dirs=["/usr/share/gromacs/top"])

The preprocessor directives GROMACS uses are followed: ``#include``
(searched next to the including file, then in ``include_dirs``, $GMXLIB
and $GMXDATA/top), ``#define`` (including parameter macros such as
GROMOS's ``gb_1``), ``#ifdef``/``#ifndef``/``#else``/``#endif`` and
``#undef``.  Each molecule type is read once and repeated as
``[ molecules ]`` says.  Molecules of several residues get one chain each;
single-residue molecules (water, ions) get one chain per ``[ molecules ]``
line.  The segid is the molecule type's name.  Residue numbers continue
over molecule copies, as in .gro files.

The force field becomes msys tables, in kcal/mol and Å:

- bonds of type 1 as ``stretch_harm``; angles of type 1 as ``angle_harm``,
  and Urey-Bradley angles (5) as an angle plus a 1-3 stretch;
- periodic dihedrals (types 1, 4, 9) and Ryckaert-Bellemans dihedrals (3,
  rewritten exactly as a cosine series) as ``dihedral_trig``; harmonic
  impropers (2) as ``improper_harm``;
- Lennard-Jones with combination rule 1 (C6/C12), 2 or 3, and
  ``[ nonbond_params ]`` as pair overrides;
- ``[ pairs ]`` as ``pair_12_6_es``: parameters from the line, from
  ``[ pairtypes ]``, or generated from the atom types and scaled by fudgeLJ;
  charges scaled by fudgeQQ;
- exclusions from ``nrexcl`` and ``[ exclusions ]``; ``[ settles ]`` as rigid
  water (``constraint_hoh``); ``[ position_restraints ]`` as ``posre_harm``.

Other function types (GROMOS quartic bonds and cosine angles, Morse bonds,
tabulated terms, CMAP, virtual sites) raise GromacsError; pass
``structure_only=True`` to read atoms, residues and bonds only.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .._columns import STR
from ..elements import guess_atomic_number
from ..system import System

KJ = 4.184  # kJ per kcal
_PTYPES = {"A", "S", "V", "D"}


class GromacsError(ValueError):
    """Unreadable or unsupported GROMACS topology content."""


def _is_int(text: str) -> bool:
    return text.lstrip("+-").isdigit()


# ---------------------------------------------------------------------------
# preprocessing


class _Preprocessor:
    """Records (section, tokens) of a topology after #include/#define/#ifdef handling."""

    def __init__(self, defines, include_dirs):
        self.defines = {k: "" if v is None else str(v) for k, v in (defines or {}).items()}
        self.dirs = [os.fspath(d) for d in include_dirs or []]
        if os.environ.get("GMXLIB"):
            self.dirs += os.environ["GMXLIB"].split(os.pathsep)
        if os.environ.get("GMXDATA"):
            self.dirs.append(os.path.join(os.environ["GMXDATA"], "top"))
        self.records: list[tuple[str | None, list[str]]] = []
        self.section: str | None = None

    def _find(self, name: str, here: str) -> str:
        for d in (here, *self.dirs):
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
        raise GromacsError(f"cannot find #include file {name!r}; pass include_dirs=[...]")

    def read(self, path: str, depth: int = 0) -> None:
        if depth > 32:
            raise GromacsError("#include nesting too deep")
        here = os.path.dirname(os.path.abspath(path))
        with open(path) as fh:
            lines = fh.read().splitlines()
        stack: list[bool] = []
        pending = ""
        for raw in lines:
            line = raw.split(";", 1)[0].rstrip()
            if line.endswith("\\"):
                pending += line[:-1] + " "
                continue
            line, pending = (pending + line).strip(), ""
            if not line:
                continue
            active = all(stack)
            if line.startswith("#"):
                parts = line[1:].split(None, 1)
                word = parts[0] if parts else ""
                arg = parts[1].strip() if len(parts) > 1 else ""
                if word == "ifdef":
                    stack.append(arg.split()[0] in self.defines)
                elif word == "ifndef":
                    stack.append(arg.split()[0] not in self.defines)
                elif word == "else":
                    if not stack:
                        raise GromacsError(f"{path}: #else without #ifdef")
                    stack[-1] = not stack[-1]
                elif word == "endif":
                    if not stack:
                        raise GromacsError(f"{path}: #endif without #ifdef")
                    stack.pop()
                elif not active:
                    continue
                elif word == "define":
                    name, _, value = arg.partition(" ")
                    self.defines[name] = value.strip()
                elif word == "undef":
                    self.defines.pop(arg.split()[0], None)
                elif word == "include":
                    self.read(self._find(arg.strip().strip('"<>'), here), depth + 1)
                else:
                    raise GromacsError(f"{path}: unsupported directive #{word}")
                continue
            if not active:
                continue
            if line.startswith("["):
                self.section = line.strip("[] \t").lower()
                continue
            tokens: list[str] = []
            for token in line.split():
                value = self.defines.get(token)
                tokens.extend(value.split() if value else [token])
            self.records.append((self.section, tokens))
        if stack:
            raise GromacsError(f"{path}: #ifdef without #endif")


# ---------------------------------------------------------------------------
# parsing


@dataclass
class _Molecule:
    name: str
    nrexcl: int
    sections: dict = field(default_factory=dict)

    def lines(self, name: str) -> list[list[str]]:
        return self.sections.get(name, [])


_MOLECULE_SECTIONS = {
    "atoms", "bonds", "pairs", "angles", "dihedrals", "settles", "exclusions", "constraints",
    "position_restraints", "virtual_sites2", "virtual_sites3", "virtual_sites4",
    "virtual_sitesn", "cmap", "pairs_nb", "dihedral_restraints", "distance_restraints",
    "angle_restraints", "orientation_restraints",
}  # fmt: skip


def _atomtype(t: list[str]) -> dict:
    p = len(t) - 3
    if p < 1 or t[p] not in _PTYPES:
        raise GromacsError(f"cannot read atom type line: {' '.join(t)}")
    head = t[1:p]
    bond_type, anum, mass, charge = t[0], 0, 0.0, 0.0
    if len(head) == 4:
        bond_type, anum, mass, charge = head[0], int(head[1]), float(head[2]), float(head[3])
    elif len(head) == 3 and _is_int(head[0]):
        anum, mass, charge = int(head[0]), float(head[1]), float(head[2])
    elif len(head) == 3:
        bond_type, mass, charge = head[0], float(head[1]), float(head[2])
    elif len(head) == 2:
        mass, charge = float(head[0]), float(head[1])
    else:
        raise GromacsError(f"cannot read atom type line: {' '.join(t)}")
    return {"bond_type": bond_type, "anum": anum, "mass": mass, "charge": charge,
            "v": float(t[p + 1]), "w": float(t[p + 2])}  # fmt: skip


class _Topology:
    def __init__(self, records):
        self.defaults = [1, 1, False, 1.0, 1.0]
        self.atomtypes: dict[str, dict] = {}
        self.bondtypes: dict = {}
        self.angletypes: dict = {}
        self.dihedraltypes: dict = {}
        self.pairtypes: dict = {}
        self.nonbond_params: dict = {}
        self.molecules: dict[str, _Molecule] = {}
        self.system: list[tuple[str, int]] = []
        mol = None
        for section, t in records:
            if section == "defaults":
                self.defaults = [int(t[0]), int(t[1]) if len(t) > 1 else 1,
                                 len(t) > 2 and t[2].lower() == "yes",
                                 float(t[3]) if len(t) > 3 else 1.0,
                                 float(t[4]) if len(t) > 4 else 1.0]  # fmt: skip
            elif section == "atomtypes":
                self.atomtypes[t[0]] = _atomtype(t)
            elif section == "bondtypes":
                self._both(self.bondtypes, t[:2], int(t[2]), t[3:])
            elif section == "constrainttypes":
                continue
            elif section == "angletypes":
                self._both(self.angletypes, t[:3], int(t[3]), t[4:])
            elif section == "dihedraltypes":
                n = 2 if _is_int(t[2]) else 4
                key = (tuple(t[:n]), int(t[n]))
                self.dihedraltypes.setdefault(key, []).append(t[n + 1 :])
            elif section == "pairtypes":
                self._both(self.pairtypes, t[:2], int(t[2]), t[3:])
            elif section == "nonbond_params":
                self._both(self.nonbond_params, t[:2], int(t[2]), t[3:])
            elif section == "moleculetype":
                mol = self.molecules[t[0]] = _Molecule(t[0], int(t[1]) if len(t) > 1 else 3)
            elif section in _MOLECULE_SECTIONS:
                if mol is None:
                    raise GromacsError(f"[ {section} ] outside a [ moleculetype ]")
                mol.sections.setdefault(section, []).append(t)
            elif section == "molecules":
                self.system.append((t[0], int(t[1])))

    @staticmethod
    def _both(table: dict, types: list[str], funct: int, params: list[str]) -> None:
        table[(*types, funct)] = params
        table.setdefault((*types[::-1], funct), params)

    def bond_type(self, atype: str) -> str:
        entry = self.atomtypes.get(atype)
        if entry is None:
            raise GromacsError(f"unknown atom type {atype!r}")
        return entry["bond_type"]

    def dihedral_params(self, types: tuple[str, ...], funct: int) -> list[list[str]]:
        """Parameter lines for a dihedral from [ dihedraltypes ]: the match with the fewest
        wildcards (X), in either direction; two-type entries match the central atoms
        (propers) or the outer atoms (impropers)."""
        best, found = -1, None
        for (key, f), lines in self.dihedraltypes.items():
            if f != funct:
                continue
            if len(key) == 2:
                probe = (types[0], types[3]) if funct in (2, 4) else (types[1], types[2])
            else:
                probe = types
            for cand in (probe, probe[::-1]):
                if all(k == "X" or k == c for k, c in zip(key, cand, strict=True)):
                    score = sum(k != "X" for k in key) + (10 if len(key) == 4 else 0)
                    if score > best:
                        best, found = score, lines
        if found is None:
            raise GromacsError(f"no [ dihedraltypes ] entry for {'-'.join(types)} (funct {funct})")
        return found


# ---------------------------------------------------------------------------
# force-field terms of one molecule type (local atom indices, boonza units)


def _rb_fourier(c: list[float]) -> list[float]:
    """Ryckaert-Bellemans C0..C5 (in psi = phi - 180) as fc0, fc1 ... fc5 of
    fc0 + sum_n fcn cos(n phi)."""
    a = [c[n] * (-1) ** n for n in range(6)]
    return [a[0] + a[2] / 2 + 3 * a[4] / 8, a[1] + 3 * a[3] / 4 + 10 * a[5] / 16,
            a[2] / 2 + a[4] / 2, a[3] / 4 + 5 * a[5] / 16, a[4] / 8, a[5] / 16]  # fmt: skip


def _unsupported(kind: str, funct: int, what: str) -> GromacsError:
    return GromacsError(f"{kind} function type {funct} ({what}) has no msys table; "
                        "load with structure_only=True")  # fmt: skip


class _Terms:
    def __init__(self):
        self.stretch, self.angle, self.dihedral, self.improper = [], [], [], []
        self.pair, self.hoh, self.posre = [], [], []


def _molecule_terms(top: _Topology, mol: _Molecule, types, charge) -> _Terms:
    out = _Terms()
    comb, fudge_lj, fudge_qq = top.defaults[1], top.defaults[3], top.defaults[4]
    bt = [top.bond_type(t) for t in types]

    def local(tokens, n):
        return [int(x) - 1 for x in tokens[:n]]

    for t in mol.lines("bonds"):
        i, j = local(t, 2)
        funct = int(t[2])
        params = t[3:] or top.bondtypes.get((bt[i], bt[j], funct))
        if funct == 5:
            continue
        if funct not in (1, 6):
            names = {2: "GROMOS quartic", 3: "Morse", 4: "cubic", 7: "FENE"}
            raise _unsupported("bond", funct, names.get(funct, "not harmonic"))
        if params is None:
            raise GromacsError(f"no [ bondtypes ] entry for {bt[i]}-{bt[j]}")
        b0, kb = float(params[0]), float(params[1])
        out.stretch.append((i, j, b0 * 10, kb / 2 / KJ / 100))
    for t in mol.lines("angles"):
        i, j, k = local(t, 3)
        funct = int(t[3])
        params = t[4:] or top.angletypes.get((bt[i], bt[j], bt[k], funct))
        if funct not in (1, 5):
            names = {2: "GROMOS cosine", 3: "cross bond-bond", 4: "cross bond-angle",
                     6: "quartic", 8: "tabulated", 10: "restricted bending"}  # fmt: skip
            raise _unsupported("angle", funct, names.get(funct, "not harmonic"))
        if params is None:
            raise GromacsError(f"no [ angletypes ] entry for {bt[i]}-{bt[j]}-{bt[k]}")
        out.angle.append((i, j, k, float(params[0]), float(params[1]) / 2 / KJ))
        if funct == 5 and float(params[3]) != 0:  # Urey-Bradley 1-3 spring
            out.stretch.append((i, k, float(params[2]) * 10, float(params[3]) / 2 / KJ / 100))
    for t in mol.lines("dihedrals"):
        idx = local(t, 4)
        funct = int(t[4])
        key = tuple(bt[a] for a in idx)
        if funct in (1, 4, 9):
            lines = [t[5:]] if t[5:] else top.dihedral_params(key, funct)
            for p in lines:
                phase, k, mult = float(p[0]), float(p[1]) / KJ, int(float(p[2]))
                if not 0 <= mult <= 6:
                    raise GromacsError(f"dihedral multiplicity {mult} is beyond 6")
                fcs = [0.0] * 7
                fcs[0] += k
                fcs[mult] += k
                out.dihedral.append((*idx, phase, *fcs))
        elif funct == 2:
            p = t[5:] or top.dihedral_params(key, funct)[0]
            out.improper.append((*idx, float(p[0]), float(p[1]) / 2 / KJ))
        elif funct == 3:
            p = t[5:] or top.dihedral_params(key, funct)[0]
            fcs = [x / KJ for x in _rb_fourier([float(x) for x in p[:6]])]
            out.dihedral.append((*idx, 0.0, *fcs, 0.0))
        else:
            raise _unsupported("dihedral", funct, "not periodic, harmonic or RB")
    for t in mol.lines("pairs"):
        i, j = local(t, 2)
        funct = int(t[2])
        if funct != 1:
            raise _unsupported("pair", funct, "not LJ-14")
        explicit = t[3:5] or top.pairtypes.get((bt[i], bt[j], 1)) or \
            top.pairtypes.get((types[i], types[j], 1))  # fmt: skip
        if explicit:
            aij, bij = _lj_ab(comb, float(explicit[0]), float(explicit[1]))
        elif top.defaults[2]:
            aij, bij = _lj_ab(comb, *_combine(comb, top.atomtypes[types[i]],
                                              top.atomtypes[types[j]]))  # fmt: skip
            aij, bij = aij * fudge_lj, bij * fudge_lj
        else:
            raise GromacsError(f"no pair parameters for {types[i]}-{types[j]} (gen-pairs no)")
        out.pair.append((i, j, aij, bij, fudge_qq * charge[i] * charge[j]))
    for t in mol.lines("settles"):
        o = int(t[0]) - 1
        doh, dhh = float(t[2]) * 10, float(t[3]) * 10
        out.hoh.append((o, o + 1, o + 2, math.degrees(2 * math.asin(dhh / (2 * doh))), doh, doh))
    for t in mol.lines("position_restraints"):
        (i,) = local(t, 1)
        if int(t[1]) != 1:
            raise _unsupported("position restraint", int(t[1]), "not harmonic")
        out.posre.append((i, *(float(x) / KJ / 100 for x in t[2:5])))
    for section in ("virtual_sites2", "virtual_sites3", "virtual_sites4", "virtual_sitesn",
                    "cmap"):  # fmt: skip
        if mol.lines(section):
            raise GromacsError(f"[ {section} ] is not supported; load with structure_only=True")
    return out


def _combine(comb: int, ti: dict, tj: dict) -> tuple[float, float]:
    """Pair V, W from two atom types by the combination rule (GROMACS units)."""
    if comb == 2:
        return (ti["v"] + tj["v"]) / 2, math.sqrt(ti["w"] * tj["w"])
    return math.sqrt(ti["v"] * tj["v"]), math.sqrt(ti["w"] * tj["w"])


def _lj_ab(comb: int, v: float, w: float) -> tuple[float, float]:
    """(A, B) of A/r^12 - B/r^6 in kcal/mol, Å from GROMACS V, W."""
    if comb == 1:  # V = C6, W = C12
        return w / KJ * 1e12, v / KJ * 1e6
    sigma, eps = v * 10, w / KJ
    return 4 * eps * sigma**12, 4 * eps * sigma**6


def _sigma_eps(comb: int, v: float, w: float) -> tuple[float, float]:
    """(sigma Å, epsilon kcal/mol) from GROMACS V, W."""
    if comb == 1:
        if v == 0 or w == 0:
            return 0.0, 0.0
        return (w / v) ** (1 / 6) * 10, v * v / (4 * w) / KJ
    return v * 10, w / KJ


def _exclusions(nat: int, bonds: list[tuple[int, int]], nrexcl: int, extra) -> set:
    """Pairs within ``nrexcl`` bonds, plus [ exclusions ] lines."""
    adj: list[list[int]] = [[] for _ in range(nat)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    out = set()
    for a in range(nat):
        seen, frontier = {a}, [a]
        for _ in range(nrexcl):
            frontier = [y for x in frontier for y in adj[x] if y not in seen]
            seen.update(frontier)
        out.update((min(a, b), max(a, b)) for b in seen if b != a)
    for t in extra:
        a = int(t[0]) - 1
        out.update((min(a, int(b) - 1), max(a, int(b) - 1)) for b in t[1:] if int(b) - 1 != a)
    return out


# ---------------------------------------------------------------------------
# building the system


def load_top(path, coordinates=None, defines=None, include_dirs=None,
             structure_only: bool = False) -> System:  # fmt: skip
    """Read a GROMACS topology; ``coordinates``: a .gro/.pdb (anything boonza loads)."""
    pre = _Preprocessor(defines, include_dirs)
    pre.read(os.fspath(path))
    top = _Topology(pre.records)
    if not top.system:
        raise GromacsError(f"{path}: no [ molecules ]")
    if not structure_only and top.defaults[0] != 1:
        raise GromacsError("nonbonded function type 2 (Buckingham) has no msys table")

    s = System(os.fspath(path))
    ct = s.add_ct()
    names, types, charge, mass, anum = [], [], [], [], []
    residue, res_chain, res_name, res_id = [], [], [], []
    chain_name, chain_segid = [], []
    bonds = []
    per_mol: dict[str, tuple] = {}
    offset = 0
    for molname, count in top.system:
        mol = top.molecules.get(molname)
        if mol is None:
            raise GromacsError(f"[ molecules ] names unknown molecule type {molname!r}")
        info = per_mol.get(molname)
        if info is None:
            info = per_mol[molname] = _prepare(top, mol, structure_only)
        m = info["n"]
        nres = info["nres"]
        one_chain = nres == 1  # waters and ions: one chain per [ molecules ] line
        if one_chain and count:
            chain_name.append("")
            chain_segid.append(molname)
        for _ in range(count):
            if not one_chain:
                chain_name.append("")
                chain_segid.append(molname)
            base_res = len(res_name)
            residue.extend((info["res_local"] + base_res).tolist())
            res_chain.extend([len(chain_name) - 1] * nres)
            res_name.extend(info["res_names"])
            # residue numbers continue over molecule copies (as MDAnalysis numbers
            # them): each molecule's are shifted by the last number given before it
            shift = res_id[-1] if res_id else 0
            res_id.extend(r + shift for r in info["res_ids"])
        n_all = m * count
        names.extend(info["names"] * count)
        types.extend(info["types"] * count)
        charge.append(np.tile(info["charge"], count))
        mass.append(np.tile(info["mass"], count))
        anum.append(np.tile(info["anum"], count))
        shifts = offset + m * np.arange(count)
        if len(info["bonds"]):
            bonds.append((info["bonds"][None, :, :] + shifts[:, None, None]).reshape(-1, 2))
        info.setdefault("copies", []).append(shifts)
        offset += n_all

    natoms = offset
    s._chains.append(len(chain_name), {"ct": ct.id, "name": np.array(chain_name, STR),
                                       "segid": np.array(chain_segid, STR)})  # fmt: skip
    s._residues.append(len(res_name), {"chain": np.array(res_chain, np.int64),
                                       "resid": np.array(res_id, np.int64),
                                       "name": np.array(res_name, STR),
                                       "insertion": np.full(len(res_name), "", STR)})  # fmt: skip
    s._atoms.append(natoms, {"residue": np.array(residue, np.int64),
                             "name": np.array(names, STR),
                             "anum": np.concatenate(anum).astype(np.int64),
                             "charge": np.concatenate(charge),
                             "mass": np.concatenate(mass)})  # fmt: skip
    s._cache.clear()
    s.atoms["type"] = np.array(types, STR)
    if bonds:
        s.add_bonds(np.concatenate(bonds))
    if not structure_only:
        _force_field(s, top, per_mol, np.array(types))
    if coordinates is not None:
        from . import load

        other = load(coordinates)
        if other.natoms != natoms:
            raise GromacsError(f"{coordinates} has {other.natoms} atoms, the topology {natoms}")
        s.positions = other.positions
        if other.cell.any():
            s.cell = other.cell
        if "posre_harm" in s.tables:
            t = s.table("posre_harm")
            xyz = s.positions[t.atoms[:, 0]]
            for c, key in enumerate(("x0", "y0", "z0")):
                t._t.set(key, xyz[:, c], np.arange(len(t)))
    return s


def _prepare(top: _Topology, mol: _Molecule, structure_only: bool) -> dict:
    atoms = mol.lines("atoms")
    n = len(atoms)
    if [int(t[0]) for t in atoms] != list(range(1, n + 1)):
        raise GromacsError(f"molecule {mol.name}: atoms must be numbered 1..{n}")
    types = [t[1] for t in atoms]
    at = [top.atomtypes.get(t) for t in types]
    if not structure_only and any(a is None for a in at):
        missing = sorted({t for t, a in zip(types, at, strict=True) if a is None})
        raise GromacsError(f"unknown atom types {missing}")
    charge = np.array([float(t[6]) if len(t) > 6 else (a or {}).get("charge", 0.0)
                       for t, a in zip(atoms, at, strict=True)])  # fmt: skip
    mass = np.array([float(t[7]) if len(t) > 7 else (a or {}).get("mass", 0.0)
                     for t, a in zip(atoms, at, strict=True)])  # fmt: skip
    anum = np.array([(a or {}).get("anum", 0) for a in at], np.int64)
    guess = anum <= 0
    anum[guess] = guess_atomic_number(mass[guess]) if guess.any() else anum[guess]
    rkey = [(t[2], t[3]) for t in atoms]
    starts = [k for k in range(n) if k == 0 or rkey[k] != rkey[k - 1]]
    res_local = np.cumsum([k in set(starts) for k in range(n)]) - 1
    chem = []
    for t in mol.lines("bonds"):
        if int(t[2]) != 6:  # type 6: a harmonic spring, not a chemical bond
            chem.append((int(t[0]) - 1, int(t[1]) - 1))
    for t in mol.lines("constraints"):
        if int(t[2]) == 1:
            chem.append((int(t[0]) - 1, int(t[1]) - 1))
    for t in mol.lines("settles"):
        o = int(t[0]) - 1
        chem += [(o, o + 1), (o, o + 2)]
    info = {"n": n, "names": [t[4] for t in atoms], "types": types, "charge": charge,
            "mass": mass, "anum": anum, "res_local": res_local, "nres": len(starts),
            "res_names": [atoms[k][3] for k in starts],
            "res_ids": [int("".join(ch for ch in atoms[k][2] if ch.isdigit() or ch == "-") or 0)
                        for k in starts],
            "bonds": np.array(chem, np.int64).reshape(-1, 2)}  # fmt: skip
    if not structure_only:
        info["terms"] = _molecule_terms(top, mol, types, charge)
        info["exclusions"] = np.array(sorted(_exclusions(n, chem, mol.nrexcl,
                                                         mol.lines("exclusions"))),
                                      np.int64).reshape(-1, 2)  # fmt: skip
    return info


def _replicate(info: dict, rows: list, natoms_per: int, width: int):
    """Local-index term rows repeated for every copy: (atoms (m, width), values (m, k))."""
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float64)
    local, values = arr[:, :width].astype(np.int64), arr[:, width:]
    shifts = np.concatenate(info["copies"])
    atoms = (local[None, :, :] + shifts[:, None, None]).reshape(-1, width)
    return atoms, np.tile(values, (len(shifts), 1))


def _force_field(s: System, top: _Topology, per_mol: dict, types: np.ndarray) -> None:
    comb = top.defaults[1]
    plan = [("stretch", "stretch_harm", 2, ["r0", "fc"]),
            ("angle", "angle_harm", 3, ["theta0", "fc"]),
            ("dihedral", "dihedral_trig", 4, ["phi0"] + [f"fc{k}" for k in range(7)]),
            ("improper", "improper_harm", 4, ["phi0", "fc"]),
            ("pair", "pair_12_6_es", 2, ["aij", "bij", "qij"]),
            ("hoh", "constraint_hoh", 3, ["theta", "r1", "r2"])]  # fmt: skip
    for attr, table_name, width, cols in plan:
        blocks = [_replicate(info, getattr(info["terms"], attr), info["n"], width)
                  for info in per_mol.values()]  # fmt: skip
        blocks = [b for b in blocks if b is not None]
        if not blocks:
            continue
        atoms = np.concatenate([b[0] for b in blocks])
        values = np.concatenate([b[1] for b in blocks])
        table = s.add_table_from_schema(table_name)
        pids = table.params.add_params(len(values),
                                       **{c: values[:, k] for k, c in enumerate(cols)})  # fmt: skip
        table.add_terms(atoms, params=pids)
    posre = [b for b in (_replicate(info, info["terms"].posre, info["n"], 1)
                         for info in per_mol.values()) if b is not None]  # fmt: skip
    if posre:
        atoms = np.concatenate([b[0] for b in posre])
        values = np.concatenate([b[1] for b in posre])
        table = s.add_table_from_schema("posre_harm")
        pids = table.params.add_params(len(values), fcx=values[:, 0], fcy=values[:, 1],
                                       fcz=values[:, 2])  # fmt: skip
        table.add_terms(atoms, params=pids)
    excl = [(info["exclusions"][None, :, :] + np.concatenate(info["copies"])[:, None, None])
            .reshape(-1, 2) for info in per_mol.values() if len(info["exclusions"])]  # fmt: skip
    if excl:
        s.add_table_from_schema("exclusion").add_terms(np.concatenate(excl))

    rule = {1: "geometric", 2: "arithmetic/geometric", 3: "geometric"}.get(comb)
    if rule is None:
        raise GromacsError(f"combination rule {comb} is not supported")
    nb = s.add_nonbonded_from_schema("vdw_12_6", rule)
    nb.params.add_prop("type", str)
    used, pid_of = np.unique(types, return_inverse=True)
    sig_eps = [_sigma_eps(comb, top.atomtypes[t]["v"], top.atomtypes[t]["w"]) for t in used]
    pids = nb.params.add_params(len(used), sigma=[x[0] for x in sig_eps],
                                epsilon=[x[1] for x in sig_eps], type=used.astype(str))  # fmt: skip
    nb.add_terms(np.arange(s.natoms)[:, None], params=pids[pid_of.reshape(-1)])
    index = {t: k for k, t in enumerate(used.tolist())}
    for (a, b, funct), params in top.nonbond_params.items():
        if funct != 1:
            raise GromacsError("[ nonbond_params ] other than Lennard-Jones are not supported")
        if a in index and b in index and index[a] <= index[b]:
            sigma, eps = _sigma_eps(comb, float(params[0]), float(params[1]))
            nb.overrides.set(int(pids[index[a]]), int(pids[index[b]]), sigma=sigma, epsilon=eps)
