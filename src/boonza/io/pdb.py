"""PDB files, read and written the way msys does.

Reading: ATOM/HETATM records by fixed columns; ``TER`` starts a new chain
even when the chain id repeats; every ``END``/``ENDMDL`` closes a model, and
each model becomes a ct.  Elements come from columns 77-78, or are guessed
from the atom name.  ``occupancy``, ``bfactor`` and ``altloc`` become atom
properties; the CRYST1 space group and Z become ct properties.

Bonds: guessed from geometry (msys rules) unless ``guess_bonds=False``, then
the file's own records are applied (msys ignores them):

- ``SSBOND``: the SG atoms of the two cysteines are bonded, whatever their
  distance (bonds to a symmetry copy, operator other than 1555, are skipped).
- ``CONECT``: the listed bonds are added; an entry repeated two or three
  times sets the bond order (the PyMOL/Open Babel convention).  For atoms
  that have their own CONECT record, the records are authoritative: a
  guessed bond between two such atoms that CONECT does not list is removed.
  Bonds from them to atoms without records (a ligand's link to the protein
  when only the ligand is listed) stay as guessed.

Writing: one model; several MODELs only for an ensemble (several cts with
the same atoms, as read from an NMR file) or with ``models=True`` (msys: one
MODEL per ct).  TER records where a chain id is reused, and hexadecimal
serials/resids past 99999/9999 (as msys and VMD do).  ``conect="auto"``
writes CONECT records only for atoms whose bonds re-guessing would get
wrong, so bonds survive a round trip; ``True`` writes them for every bonded
atom, ``False`` never.
"""

from __future__ import annotations

import bz2
import gzip
import math
import os

import numpy as np

from .._columns import STR
from ..elements import element_for_abbreviation, msys_symbol
from ..system import System
from ._maeparse import _atof, _atoi, read_text
from .dms import _factorize

SPACE_GROUP = "pdb_space_group"
Z_VALUE = "pdb_z_value"


def load_pdb(path, guess_bonds: bool = True, conect: bool = True, ssbond: bool = True) -> System:
    """Read a PDB file (optionally gzip/bzip2 compressed).

    ``guess_bonds``: bond atoms by distance (msys rules).  ``conect`` and
    ``ssbond``: apply the file's CONECT and SSBOND records on top (see the
    module notes); False for msys behavior.
    """
    path = os.fspath(path)
    out = System(path)
    lines = read_text(path).split("\n")
    records = _bond_records(lines, conect, ssbond)
    for atoms, ters, cryst in _models(lines):
        out.append(_model(atoms, ters, cryst, guess_bonds, records))
    out.name = path
    return out


# ---------------------------------------------------------------------------
# reading


def _models(lines):
    """Yield (atom lines, TER positions, CRYST1 line) per model; stop at an empty model."""
    k, n = 0, len(lines)
    while True:
        atoms, ters, cryst = [], [], None
        while k < n:
            line = lines[k][:81].rstrip("\r")
            k += 1
            if line.startswith(("ATOM ", "HETATM")):
                atoms.append(line)
            elif line.startswith(("CONECT", "REMARK", "HEADER")):
                continue
            elif line.startswith("CRYST1"):
                cryst = line
            elif line.startswith("TER"):
                ters.append(len(atoms))
            elif line.startswith("END"):
                break
        if not atoms:
            return
        yield atoms, ters, cryst
        if k >= n:
            return


class _Columns:
    """Fixed-width fields of many records at once."""

    def __init__(self, lines):
        self.n = len(lines)
        raw = "".join(line[:80].ljust(80) for line in lines).encode("latin-1", "replace")
        self.mat = np.frombuffer(raw, dtype="S1").reshape(self.n, 80)

    def raw(self, a: int, b: int) -> np.ndarray:
        return np.ascontiguousarray(self.mat[:, a:b]).view(f"S{b - a}").reshape(self.n)

    def mapped(self, a: int, b: int, func, dtype):
        """Apply ``func`` to each distinct field value (fields repeat a lot)."""
        uniq, inv = np.unique(self.raw(a, b), return_inverse=True)
        vals = np.array([func(u.decode("latin-1")) for u in uniq.tolist()], dtype=dtype)
        return vals[inv.reshape(-1)]

    def token(self, a: int, b: int) -> np.ndarray:
        return self.mapped(a, b, lambda s: s.split()[0] if s.split() else "", STR)

    def floats(self, a: int, b: int) -> np.ndarray:
        field = self.raw(a, b)
        try:
            return np.char.strip(field).astype(np.float64)
        except ValueError:
            return np.array([_atof(f.decode("latin-1")) for f in field.tolist()])


def _alpha_run(s: str) -> str:
    """The first run of letters in ``s``."""
    i = 0
    while i < len(s) and not (s[i].isascii() and s[i].isalpha()):
        i += 1
    j = i
    while j < len(s) and s[j].isascii() and s[j].isalpha():
        j += 1
    return s[i:j]


def _element(element: str, name: str, resname: str) -> int:
    if element:
        anum = element_for_abbreviation(element)
        return 1 if anum == 0 and element == "D" else anum
    if not name:
        return 0
    # guess from the name: the whole name when it matches the residue name
    # (ions such as NA or CL), else its first letter, else the whole name
    nm = _alpha_run(name)
    anum = element_for_abbreviation(nm) if nm == resname else 0
    if anum == 0:
        anum = element_for_abbreviation(nm[:1])
    if anum == 0:
        anum = element_for_abbreviation(nm)
    return anum


def _serial_value(field: str) -> int:
    """An atom serial: decimal, or hexadecimal as msys and VMD write past 99999."""
    s = field.strip()
    if not s:
        return -1
    try:
        return int(s)
    except ValueError:
        try:
            return int(s, 16)
        except ValueError:
            return -1


def _bond_records(lines, conect: bool, ssbond: bool):
    """CONECT pairs {(serial, serial): max multiplicity}, base serials, SSBOND residue pairs."""
    pairs: dict[tuple[int, int], int] = {}
    bases: set[int] = set()
    ss = []
    for line in lines:
        if conect and line.startswith("CONECT"):
            rec = line.rstrip("\r").ljust(31)
            base = _serial_value(rec[6:11])
            if base < 0:
                continue
            bases.add(base)
            count: dict[int, int] = {}
            for k in (11, 16, 21, 26):
                other = _serial_value(rec[k : k + 5])
                if other >= 0 and other != base:
                    count[other] = count.get(other, 0) + 1
            for other, c in count.items():
                key = (min(base, other), max(base, other))
                pairs[key] = max(pairs.get(key, 0), c)
        elif ssbond and line.startswith("SSBOND"):
            rec = line.rstrip("\r").ljust(80)
            sym1, sym2 = rec[59:65].strip(), rec[66:72].strip()
            if sym1 and sym2 and sym1 != sym2:
                continue  # bonded to a crystal symmetry copy
            first = (rec[15].strip(), _atoi(rec[17:21]), rec[21].strip())
            second = (rec[29].strip(), _atoi(rec[31:35]), rec[35].strip())
            ss.append((first, second))
    if not pairs and not ss:
        return None
    return pairs, bases, ss


def _apply_bond_records(s: System, serials: np.ndarray, chain, resid, insertion, name,
                        records) -> None:  # fmt: skip
    pairs, bases, ss = records
    if pairs:
        uniq, counts = np.unique(serials[serials >= 0], return_counts=True)
        ok = set(uniq[counts == 1].tolist())  # ambiguous serials are ignored
        index = {int(v): k for k, v in enumerate(serials.tolist()) if v in ok}
        found = [(index[a], index[b], c) for (a, b), c in pairs.items()
                 if a in index and b in index]  # fmt: skip
        covered = np.zeros(s.natoms, bool)
        covered[[index[a] for a in bases if a in index]] = True
        listed = {(min(i, j), max(i, j)) for i, j, _ in found}
        bi, bj = s._bonds.column("i"), s._bonds.column("j")
        drop = np.flatnonzero(covered[bi] & covered[bj])
        drop = [b for b in drop.tolist() if (int(bi[b]), int(bj[b])) not in listed]
        if drop:
            s.delete_bonds(drop)
        if found:
            arr = np.array(found, np.int64)
            ids = s.add_bonds(arr[:, :2])
            multi = arr[:, 2] > 1
            if multi.any():
                orders = s._bonds.column("order").copy()
                orders[ids[multi]] = np.minimum(arr[multi, 2], 3)
                s.bonds["order"] = orders
    if ss:
        sg = np.flatnonzero(name == "SG")
        where: dict[tuple, int] = {}
        for k in sg.tolist():
            where.setdefault((str(chain[k]), int(resid[k]), str(insertion[k])), k)
        bonds = [(where[a], where[b]) for a, b in ss if a in where and b in where
                 and where[a] != where[b]]  # fmt: skip
        if bonds:
            s.add_bonds(bonds)


def _model(lines, ters, cryst, guess_bonds: bool, records=None) -> System:
    f = _Columns(lines)
    n = f.n
    name, resname = f.token(12, 16), f.token(17, 21)
    chain, segid = f.token(21, 22), f.token(72, 76)
    altloc, insertion = f.token(16, 17), f.token(26, 27)
    element = f.token(76, 78)
    resid = f.mapped(22, 26, _atoi, np.int64)
    combo, first = _factorize(element, name, resname)
    guesses = [_element(str(element[k]), str(name[k]), str(resname[k])) for k in first.tolist()]
    anum = np.array(guesses, np.int64)[combo]

    # a TER ends the chain of the atom before it: later atoms with the same
    # chain id and segid start a new chain
    key, _ = _factorize(chain, segid)
    epoch = np.zeros(n, np.int64)
    p = np.unique(np.asarray(ters, dtype=np.int64))
    p = p[p > 0]
    if len(p):
        codes = np.sort(key[p - 1] * (n + 1) + p)
        base = key * (n + 1)
        epoch = np.searchsorted(codes, base + np.arange(n), "right") - np.searchsorted(
            codes, base, "left"
        )
    chain_code, chain_first = _factorize(key, epoch)
    res_code, res_first = _factorize(chain_code, resid, resname, insertion)

    s = System()
    ct = s.add_ct()
    s._chains.append(
        len(chain_first), {"ct": ct.id, "name": chain[chain_first], "segid": segid[chain_first]}
    )
    s._residues.append(
        len(res_first),
        {
            "chain": chain_code[res_first],
            "resid": resid[res_first],
            "name": resname[res_first],
            "insertion": insertion[res_first],
        },
    )
    pos = np.column_stack([f.floats(30, 38), f.floats(38, 46), f.floats(46, 54)])
    s._atoms.append(
        n,
        {
            "residue": res_code,
            "name": name,
            "anum": anum,
            "pos": pos,
            "formal_charge": f.mapped(78, 80, _atoi, np.int64),
        },
    )
    s.atoms["occupancy"] = f.floats(54, 60)
    s.atoms["bfactor"] = f.floats(60, 66)
    if (altloc != "").any():
        s.atoms["altloc"] = altloc
    if cryst is not None:
        _read_cryst1(s, cryst)
    if guess_bonds:
        s.guess_bonds()
    if records is not None:
        serials = f.mapped(6, 11, _serial_value, np.int64)
        _apply_bond_records(s, serials, chain, resid, insertion, name, records)
    return s


def _read_cryst1(s: System, line: str) -> None:
    rec = line.ljust(80)
    a, b, c = (_atof(rec[k : k + 9]) for k in (6, 15, 24))
    alpha, beta, gamma = (_atof(rec[k : k + 7]) for k in (33, 40, 47))
    space = rec[55:66].strip()
    z = _atoi(rec[66:80])
    if space:
        s._ct_props[0][SPACE_GROUP] = space
    if z:
        s._ct_props[0][Z_VALUE] = z
    s.cell = cell_from_lengths_angles(a, b, c, alpha, beta, gamma)


def cell_from_lengths_angles(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Box vectors (rows) from lengths and angles in degrees, a along x, b in the xy plane."""
    cos_bc = math.sin(((90 - alpha) / 180) * math.pi)
    cos_ac = math.sin(((90 - beta) / 180) * math.pi)
    cos_ab = math.sin(((90 - gamma) / 180) * math.pi)
    sin_ab = math.cos(((90 - gamma) / 180) * math.pi)
    cx = cy = cz = 0.0
    if sin_ab != 0:
        cx = cos_ac
        cy = (cos_bc - cos_ac * cos_ab) / sin_ab
        v = 1 - cx * cx - cy * cy
        cz = math.sqrt(v) if v >= 0 else math.nan
        cx, cy, cz = cx * c, cy * c, cz * c
    return np.array([[a, 0.0, 0.0], [b * cos_ab, b * sin_ab, 0.0], [cx, cy, cz]])


def lengths_angles_from_cell(cell) -> tuple[float, float, float, float, float, float]:
    A, B, C = (np.asarray(v, dtype=np.float64) for v in np.asarray(cell).reshape(3, 3))
    a, b, c = (math.sqrt(float(v @ v)) for v in (A, B, C))
    alpha = beta = gamma = 90.0
    if a and b and c:

        def angle(cos):
            return 90.0 - math.asin(min(1.0, max(-1.0, cos))) * 90.0 / (math.pi / 2)

        gamma = angle(float(A @ B) / (a * b))
        beta = angle(float(A @ C) / (a * c))
        alpha = angle(float(B @ C) / (b * c))
    return a, b, c, alpha, beta, gamma


# ---------------------------------------------------------------------------
# writing


def _serial(i: int) -> str:
    return f"{i:5d}" if i < 100000 else (f"{i:05x}" if i < 1048576 else "*****")


def _resid(r: int) -> str:
    return f"{r:4d}" if r < 10000 else (f"{r:04x}" if r < 65536 else "****")


def _name(nm: str) -> str:
    if len(nm) in (0, 4):
        return nm
    return " " + nm if len(nm) < 4 else nm[:3]


def _charge(q: int) -> str:
    if -10 < q < 0:
        return str(q)
    if 0 < q < 10:
        return f"+{q}"
    return ""


def _mapped(values, func) -> list:
    uniq, inv = np.unique(np.asarray(values), return_inverse=True)
    out = [func(u) for u in uniq.tolist()]
    return [out[k] for k in inv.reshape(-1).tolist()]


def save_pdb(system: System, path, append: bool = False, reorder: bool = False,
             models="auto", conect="auto") -> None:  # fmt: skip
    """Write a PDB file; ``reorder`` groups atoms by chain and residue first.

    ``models``: "auto" writes one model, or one MODEL per ct when the cts are
    an ensemble (same atoms in each); True always one MODEL per ct (msys);
    False always one model.  ``conect``: "auto" writes CONECT records for the
    atoms whose bonds re-guessing on reading would get wrong; True for every
    bonded atom; False never.
    """
    path = os.fspath(path)
    A, R, C = system._atoms, system._residues, system._chains
    n = system.natoms
    order = np.arange(n)
    res = A.column("residue")
    chn = R.column("chain")[res]
    if reorder:
        order = np.lexsort((order, res, chn))
    res, chn = res[order], chn[order]
    ct = C.column("ct")[chn]

    lines = ["ENDMDL"] if append else []
    props = system._ct_props[0] if system.ncts else {}
    if system.cell.any() or SPACE_GROUP in props:
        a, b, c, al, be, ga = lengths_angles_from_cell(system.cell)
        space = str(props.get(SPACE_GROUP, "P 1"))
        z = props.get(Z_VALUE, 1)
        z = z if isinstance(z, int) else 1
        lines.append(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {space:<11s}{z:4d}"
        )

    pos = A.column("pos")[order]
    occ = A.column("occupancy")[order] if "occupancy" in A else np.ones(n)
    bfac = A.column("bfactor")[order] if "bfactor" in A else np.zeros(n)
    alt = (
        _mapped(A.column("altloc")[order], lambda s: s[:1] or " ") if "altloc" in A else ([" "] * n)
    )
    names = _mapped(A.column("name")[order], _name)
    resnames = _mapped(R.column("name")[res], lambda s: s[:4])
    resids = _mapped(R.column("resid")[res], _resid)
    ins = _mapped(R.column("insertion")[res], lambda s: s[:1] or " ")
    chains = _mapped(C.column("name")[chn], lambda s: s[:1] or " ")
    segids = _mapped(C.column("segid")[chn], lambda s: s[:4])
    elems = _mapped(A.column("anum")[order], lambda z: msys_symbol(z)[:2])
    charges = _mapped(A.column("formal_charge")[order], _charge)
    if models == "auto":
        multi = _is_ensemble(ct, A.column("name")[order], R.column("name")[res])
    else:
        multi = bool(models) and system.ncts > 1 and len(np.unique(ct)) > 1
    model_of = ct if multi else np.zeros(n, np.int64)
    ter_after = _ter_positions(chn, model_of, C.column("name"), C.column("segid"))

    fmt = "%-6s%5s %-4s%s%-4s%s%4s%s   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%2s"
    serial = 0
    model = 0
    xyz, occ, bfac = pos.tolist(), occ.tolist(), bfac.tolist()
    for k in range(n):
        if multi and (k == 0 or ct[k] != ct[k - 1]):
            if k:
                lines.append("ENDMDL")
            model += 1
            lines.append(f"MODEL     {model:4d}")
        serial += 1
        x, y, z = xyz[k]
        lines.append(fmt % ("ATOM", _serial(serial), names[k], alt[k], resnames[k], chains[k],
                            resids[k], ins[k], x, y, z, occ[k], bfac[k], segids[k], elems[k],
                            charges[k]))  # fmt: skip
        if ter_after[k]:
            serial += 1
            lines.append(
                f"TER   {_serial(serial)}      {resnames[k]:<4s}{chains[k]}{resids[k]:>4s}"
            )
    if multi:
        lines.append("ENDMDL")
    if conect and system.nbonds:
        serials = np.arange(1, n + 1) + np.concatenate([[0], np.cumsum(ter_after)[:-1]])
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n)
        lines.extend(_conect_lines(system, rank, serials, np.round(pos, 3), model_of,
                                   everything=conect is True))  # fmt: skip
    text = "\n".join(lines) + "\n"
    mode = "at" if append else "wt"
    opener = gzip.open if path.endswith(".gz") else bz2.open if path.endswith(".bz2") else open
    with opener(path, mode) as fh:
        fh.write(text)


def _is_ensemble(ct, names, resnames) -> bool:
    """Several cts with the same atoms in the same order: models of one structure."""
    starts = np.flatnonzero(np.r_[True, ct[1:] != ct[:-1]])
    if len(starts) < 2 or len(np.unique(ct)) != len(starts):
        return False
    sizes = np.diff(np.r_[starts, len(ct)])
    if (sizes != sizes[0]).any():
        return False
    m = int(sizes[0])
    return bool((names.reshape(-1, m) == names[:m]).all()
                and (resnames.reshape(-1, m) == resnames[:m]).all())  # fmt: skip


def _conect_lines(system, rank, serials, pos, model_of, everything: bool) -> list[str]:
    """CONECT records (atoms in written order) for bonds the reader would not guess."""
    from ..bonds import guessed_pairs

    bi = rank[system._bonds.column("i")]
    bj = rank[system._bonds.column("j")]
    orders = system._bonds.column("order")
    n = len(rank)
    lo, hi = np.minimum(bi, bj), np.maximum(bi, bj)
    if everything:
        covered = np.zeros(n, bool)
        covered[lo] = covered[hi] = True
    else:
        anum = system._atoms.column("anum")[np.argsort(rank)]
        g = guessed_pairs(pos, anum, model_of if model_of.any() else None)
        gkeys = np.minimum(g[:, 0], g[:, 1]) * n + np.maximum(g[:, 0], g[:, 1])
        bkeys = lo * n + hi
        wrong = np.concatenate([np.setdiff1d(bkeys, gkeys), np.setdiff1d(gkeys, bkeys)])
        covered = np.zeros(n, bool)
        covered[wrong // n] = covered[wrong % n] = True
    if not covered.any():
        return []
    partners: dict[int, list[int]] = {}
    for a, b, o in zip(lo.tolist(), hi.tolist(), orders.tolist(), strict=True):
        o = min(max(int(o), 1), 3)
        if covered[a]:
            partners.setdefault(a, []).extend([b] * o)
        if covered[b]:
            partners.setdefault(b, []).extend([a] * o)
    out = []
    for a in sorted(partners):
        if serials[a] >= 1048576:
            continue
        refs = [_serial(int(serials[b])) for b in partners[a] if serials[b] < 1048576]
        for k in range(0, len(refs), 4):
            out.append("CONECT" + _serial(int(serials[a])) + "".join(refs[k : k + 4]))
    return out


def _ter_positions(chn, ct, chain_names, chain_segids) -> np.ndarray:
    """True after the last atom of a chain whose id and segid are reused by a later chain
    in the same model; without a TER that later chain would be merged into it on reading."""
    n = len(chn)
    ter = np.zeros(n, bool)
    if n == 0:
        return ter
    last: dict[int, int] = {}
    first: dict[int, int] = {}
    for k, c in enumerate(chn.tolist()):
        first.setdefault(c, k)
        last[c] = k
    latest_first: dict[tuple, int] = {}
    for c, k in first.items():
        key = (int(ct[k]), str(chain_names[c]), str(chain_segids[c]))
        latest_first[key] = max(latest_first.get(key, -1), k)
    for c, k in last.items():
        key = (int(ct[k]), str(chain_names[c]), str(chain_segids[c]))
        if latest_first[key] > k:
            ter[k] = True
    return ter
