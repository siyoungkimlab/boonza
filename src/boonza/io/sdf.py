"""SDF / MOL (V2000) files, read and written the way msys does.

Each entry becomes a ct holding one chain and one residue.  Atom names are
element symbols.  Formal charges come from the atom block, replaced by any
``M  CHG`` lines; ``M  ISO`` sets an ``isotope`` atom property.  Data fields
become ct properties, typed as int, float or string.  Atom stereo parity,
bond stereo and aromatic bonds (order 4) are kept as the ``stereo_parity``,
``stereo`` and ``aromatic`` properties when present.
"""

from __future__ import annotations

import bz2
import gzip
import math
import os
import re

import numpy as np

from ..elements import element_for_abbreviation, msys_symbol
from ..system import System
from ._maeparse import read_text

_BAD = 9999
_FLT_MAX = float(np.finfo(np.float32).max)
_ONE_LETTER = {"C": 6, "H": 1, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "K": 19, "B": 5,
               "V": 23, "Y": 39, "I": 53, "W": 74, "U": 92}  # fmt: skip
_INT = re.compile(r"[ \t\n\v\f\r]*[+-]?\d+")
_FLOAT = re.compile(r"[ \t\n\v\f\r]*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class SdfError(ValueError):
    """Malformed SDF content."""


def load_sdf(path) -> System:
    """Read every entry of an SDF file (optionally gzip/bzip2 compressed)."""
    path = os.fspath(path)
    reader = _Lines(read_text(path))
    out = System(path)
    while (entry := _entry(reader)) is not None:
        out.append(entry)
    out.name = path
    return out


class _Lines:
    """Line reader that reports end-of-file the way C ``fgets``/``feof`` do."""

    def __init__(self, text: str):
        self.lines = text.split("\n")
        self.partial_last = not text.endswith("\n") and text != ""
        if not self.partial_last:
            self.lines.pop()  # the empty string after the final newline
        self.k = 0
        self.buf = ""
        self.nl = True
        self.eof = False

    def getline(self) -> bool:
        if self.k >= len(self.lines):
            self.eof = True
            return False
        self.buf = self.lines[self.k].rstrip("\r")
        self.k += 1
        self.nl = not (self.k == len(self.lines) and self.partial_last)
        if not self.nl:
            self.eof = True
        return True

    def skip_to_end(self) -> str:
        where = f"line {self.k}: {self.buf}"
        while self.getline():
            if self.buf.startswith("$$$$"):
                break
        return where


def _count(s: str, k: int) -> int:
    """A right-justified 3-character integer field, as msys reads it."""
    field = (s[k : k + 3] + "\0\0\0")[:3]
    n = sign = 1
    n = 0
    pos = 0
    if field[pos].isspace():
        n, pos = n + 1, pos + 1
    if pos < 3 and field[pos].isspace():
        n, pos = n + 1, pos + 1
    if pos < 3 and field[pos] == "-":
        sign, n, pos = -1, n + 1, pos + 1
    result = 0
    while n < 3:
        c = field[pos]
        if not ("0" <= c <= "9"):
            return _BAD
        result += (100, 10, 1)[n] * int(c)
        n, pos = n + 1, pos + 1
    return sign * result


def _coord(s: str, k: int) -> float:
    """A %10.4f coordinate field, parsed exactly like msys (float32 precision)."""
    field = s[k : k + 10] + "\0" * 10
    n = pos = 0
    scale = np.float32(10000.0)
    while n < 4 and field[pos].isspace():
        n, pos = n + 1, pos + 1
    if field[pos] == "-":
        n, pos, scale = n + 1, pos + 1, -scale
    result = 0
    while n < 5:
        c = field[pos]
        if not ("0" <= c <= "9"):
            return _FLT_MAX
        result += (100000000, 10000000, 1000000, 100000, 10000)[n] * int(c)
        n, pos = n + 1, pos + 1
    if field[pos] != ".":
        return _FLT_MAX
    for w, c in zip((1000, 100, 10, 1), field[pos + 1 : pos + 5], strict=True):
        result += w * (ord(c) - 48)
    return float(np.float32(result & 0xFFFFFFFF) / scale)


def _element(s: str) -> int:
    field = s[31:34] + "\0\0\0"
    pos = 1 if field[0].isspace() else 0
    if field[pos].isspace():
        return _ONE_LETTER.get(field[pos + 1], 0)
    return element_for_abbreviation(field[pos : pos + 2].replace("\0", ""))


def _typed(val: str):
    """Data field value as msys types it: int, finite float, else string."""
    if val == "" or _INT.fullmatch(val):
        return int(val) if val else 0
    if _FLOAT.fullmatch(val):
        v = float(val)
        if math.isfinite(v):
            return v
    return val


def _add_field(props: dict, key: str, val: str) -> None:
    if not key:
        return
    val = "" if len(val) == 1 else val[:-2]  # drop the value's newline and the blank line's
    props[key] = _typed(val)


def _v2000(r: _Lines, counts: str):
    """V2000 atom and bond blocks plus the properties block, parsed exactly like msys."""
    natoms, nbonds = _count(counts, 0), _count(counts, 3)
    if natoms == _BAD or nbonds == _BAD:
        raise SdfError(f"Bad counts line: {r.skip_to_end()}")

    anum = np.zeros(natoms, np.int64)
    pos = np.zeros((natoms, 3))
    charge = np.zeros(natoms, np.int64)
    parity = np.zeros(natoms, np.int64)
    for a in range(natoms):
        if not r.getline():
            r.skip_to_end()
            raise SdfError("Missing expected atom record")
        line = r.buf
        size = len(line) + r.nl
        if size < 34:
            raise SdfError(f"Malformed atom line: {r.skip_to_end()}")
        pos[a] = (_coord(line, 0), _coord(line, 10), _coord(line, 20))
        anum[a] = _element(line)
        if size >= 39:
            q = _count(line, 36)
            if q == _BAD:
                raise SdfError(f"Bad charge: {r.skip_to_end()}")
            charge[a] = 0 if q == 0 else 4 - q
        if size >= 42:
            p = _count(line, 39)
            if p == _BAD:
                raise SdfError(f"Bad stereo: {r.skip_to_end()}")
            parity[a] = p

    pairs = np.zeros((nbonds, 2), np.int64)
    order = np.ones(nbonds, np.int64)
    stereo = np.zeros(nbonds, np.int64)
    aromatic = np.zeros(nbonds, np.int64)
    for b in range(nbonds):
        if not r.getline():
            r.skip_to_end()
            raise SdfError("Missing expected bond record")
        line = r.buf
        if len(line) + r.nl < 9:
            raise SdfError(f"Malformed bond line: {r.skip_to_end()}")
        ai, aj = _count(line, 0) - 1, _count(line, 3) - 1
        if not (0 <= ai < natoms and 0 <= aj < natoms) or ai == aj:
            raise SdfError(f"Bad atoms in bond record: {r.skip_to_end()}")
        bo, st = _count(line, 6), _count(line, 9)
        st = 0 if st == _BAD else st
        if st in (1, 6) and ai > aj:  # bonds are stored low-high; flip the wedge
            st = -st
        if bo == 4:
            bo, aromatic[b] = 1, 1
        elif bo < 0 or bo > 4:
            raise SdfError(f"Unsupported bond type in bond record: {r.skip_to_end()}")
        pairs[b] = (ai, aj)
        order[b], stereo[b] = bo, st

    isotope = None
    cleared = False
    while r.getline():
        line = r.buf
        if line.startswith("M  "):
            tag = line[3:6]
            if tag == "END":
                break
            if tag in ("CHG", "ISO"):
                if tag == "CHG" and not cleared:
                    charge[:] = 0
                    cleared = True
                if tag == "ISO" and isotope is None:
                    isotope = np.zeros(natoms, np.int64)
                count = _count(line, 6)
                if count == _BAD:
                    raise SdfError(f"Malformed {tag} line: {r.skip_to_end()}")
                for k in range(count):
                    aid, val = _count(line, 10 + 8 * k), _count(line, 14 + 8 * k)
                    if aid == _BAD or val == _BAD or not 1 <= aid <= natoms:
                        raise SdfError(f"Malformed {tag} line: {r.skip_to_end()}")
                    (charge if tag == "CHG" else isotope)[aid - 1] = val
        elif line.startswith(("A  ", "G  ")):
            r.getline()
        elif not line.startswith("V  "):
            raise SdfError(f"Malformed properties line: {r.skip_to_end()}")
    return anum, pos, charge, parity, isotope, pairs, order, stereo, aromatic


_V30_TOKEN = re.compile(r'[^\s=]+=\([^)]*\)|"(?:[^"]|"")*"|\S+')
_V30_WEDGE = {1: 1, 2: 4, 3: 6}  # V3000 bond CFG -> V2000 bond stereo


def _v30_records(r: _Lines):
    """Logical ``M  V30`` records up to ``M  END``, with continuation lines joined."""
    pending = ""
    while r.getline():
        line = r.buf
        if line.startswith("M  END"):
            return
        if not line.startswith("M  V30 "):
            raise SdfError(f"Malformed V3000 line: {r.skip_to_end()}")
        body = line[7:]
        if body.endswith("-"):
            pending += body[:-1]
            continue
        yield _V30_TOKEN.findall(pending + body)
        pending = ""


def _v3000(r: _Lines):
    """V3000 connection table (CTAB block); other blocks such as SGROUP are skipped."""
    atoms, bonds = [], []
    block = None
    skip_depth = 0
    for tok in _v30_records(r):
        if not tok:
            continue
        head = tok[0].upper()
        if skip_depth:
            if head == "BEGIN":
                skip_depth += 1
            elif head == "END":
                skip_depth -= 1
            continue
        if head == "BEGIN":
            what = tok[1].upper() if len(tok) > 1 else ""
            if what in ("ATOM", "BOND", "CTAB"):
                block = what
            else:
                skip_depth = 1
        elif head == "END":
            block = None
        elif head == "COUNTS" or block is None or block == "CTAB":
            continue
        elif block == "ATOM":
            atoms.append(tok)
        elif block == "BOND":
            bonds.append(tok)

    index = {}
    n = len(atoms)
    anum = np.zeros(n, np.int64)
    pos = np.zeros((n, 3))
    charge = np.zeros(n, np.int64)
    parity = np.zeros(n, np.int64)
    isotope = None
    for a, tok in enumerate(atoms):
        if len(tok) < 5:
            raise SdfError(f"Malformed V3000 atom record: {' '.join(tok)}")
        index[tok[0]] = a
        symbol = tok[1].strip('"')
        anum[a] = element_for_abbreviation(symbol) if symbol.isalpha() else 0
        try:
            pos[a] = [float(v) for v in tok[2:5]]
        except ValueError:
            raise SdfError(f"Bad V3000 coordinates: {' '.join(tok)}") from None
        for kv in tok[6:]:
            key, _, value = kv.partition("=")
            key = key.upper()
            if key == "CHG":
                charge[a] = int(value)
            elif key == "MASS":
                if isotope is None:
                    isotope = np.zeros(n, np.int64)
                isotope[a] = int(float(value))
            elif key == "CFG":
                parity[a] = int(value)

    m = len(bonds)
    pairs = np.zeros((m, 2), np.int64)
    order = np.ones(m, np.int64)
    stereo = np.zeros(m, np.int64)
    aromatic = np.zeros(m, np.int64)
    for b, tok in enumerate(bonds):
        if len(tok) < 4 or tok[2] not in index or tok[3] not in index:
            raise SdfError(f"Malformed V3000 bond record: {' '.join(tok)}")
        ai, aj = index[tok[2]], index[tok[3]]
        if ai == aj:
            raise SdfError(f"V3000 bond to self: {' '.join(tok)}")
        bo = int(tok[1])
        if bo == 4:
            bo, aromatic[b] = 1, 1
        elif bo in (9, 10):  # coordination and hydrogen bonds
            bo = 0
        elif not 1 <= bo <= 3:
            raise SdfError(f"Unsupported V3000 bond type {bo}")
        st = 0
        for kv in tok[4:]:
            key, _, value = kv.partition("=")
            if key.upper() == "CFG":
                st = _V30_WEDGE.get(int(value), 0)
        if st in (1, 6) and ai > aj:
            st = -st
        pairs[b] = (ai, aj)
        order[b], stereo[b] = bo, st
    return anum, pos, charge, parity, isotope, pairs, order, stereo, aromatic


def _entry(r: _Lines) -> System | None:
    if not r.getline():
        return None
    name = r.buf
    for _ in range(3):
        r.getline()
    if r.eof:
        return None
    counts = r.buf
    if "V3000" in counts[33:]:
        table = _v3000(r)
    else:
        table = _v2000(r, counts)
    anum, pos, charge, parity, isotope, pairs, order, stereo, aromatic = table

    props: dict = {}
    if r.getline():
        need = False
        while True:
            if need and not r.getline():
                raise SdfError("Unexpected end of file")
            if r.eof:
                break
            need = True
            line = r.buf
            if line.startswith("$$$$"):
                break
            if line.startswith("> "):
                lo, hi = line.find("<", 2), line.find(">", 3)
                key = line[lo + 1 : hi] if lo >= 0 and hi > lo else ""
                val = ""
                while True:
                    if not r.getline() or r.buf.startswith(("$$$$", "> ")):
                        _add_field(props, key, val)
                        need = False
                        break
                    val += r.buf + ("\n" if r.nl else "")
            elif line != "":
                raise SdfError(f"Malformed data field line: {r.skip_to_end()}")

    s = System(name)
    res = s.add_residue(s.add_chain(s.add_ct(name, **props)))
    s.add_atoms(len(anum), res, name=[msys_symbol(int(a)) for a in anum], anum=anum, pos=pos,
                formal_charge=charge)  # fmt: skip
    if parity.any():
        s.atoms["stereo_parity"] = parity
    if isotope is not None:
        s.atoms["isotope"] = isotope
    if len(pairs):
        ids = s.add_bonds(pairs)
        s._bonds.set("order", order, ids)  # a repeated bond keeps its last order
        if stereo.any():
            s.bonds.add_prop("stereo", int)
            s._bonds.set("stereo", stereo, ids)
        if aromatic.any():
            s.bonds.add_prop("aromatic", int)
            s._bonds.set("aromatic", aromatic, ids)
    return s


# ---------------------------------------------------------------------------
# writing


def _value(v) -> str:
    if isinstance(v, bool | int | np.integer):
        return str(int(v))
    if isinstance(v, float | np.floating):
        return f"{float(v):.16g}"
    return str(v)


def _format_v3000(s: System) -> list[str]:
    n, m = s.natoms, s.nbonds
    A, B = s._atoms, s._bonds
    lines = ["  0  0  0     0  0            999 V3000", "M  V30 BEGIN CTAB",
             f"M  V30 COUNTS {n} {m} 0 0 1", "M  V30 BEGIN ATOM"]  # fmt: skip
    parity = A.column("stereo_parity") if "stereo_parity" in A else np.zeros(n, np.int64)
    iso = A.column("isotope") if "isotope" in A else np.zeros(n, np.int64)
    charge = A.column("formal_charge")
    for a, (x, y, z) in enumerate(A.column("pos").tolist()):
        sym = msys_symbol(int(A.column("anum")[a])) or "X"
        fields = (("CHG", charge[a]), ("MASS", iso[a]), ("CFG", parity[a]))
        extra = "".join(f" {k}={int(v)}" for k, v in fields if v)
        lines.append(f"M  V30 {a + 1} {sym} {x:.4f} {y:.4f} {z:.4f} 0{extra}")
    lines += ["M  V30 END ATOM", "M  V30 BEGIN BOND"]
    stereo = B.column("stereo") if "stereo" in B else np.zeros(m, np.int64)
    arom = B.column("aromatic") if "aromatic" in B else np.zeros(m, np.int64)
    cfg = {1: 1, 4: 2, 6: 3}
    for b in range(m):
        i, j, st = int(B.column("i")[b]) + 1, int(B.column("j")[b]) + 1, int(stereo[b])
        if st < 0:
            i, j, st = j, i, -st
        order = 4 if arom[b] else int(B.column("order")[b])
        order = order if order else 9  # zero-order bond: write as coordination
        extra = f" CFG={cfg[st]}" if st in cfg else ""
        lines.append(f"M  V30 {b + 1} {order} {i} {j}{extra}")
    lines += ["M  V30 END BOND", "M  V30 END CTAB", "M  END"]
    return lines


def _format_ct(s: System, v3000: bool | None = None) -> str:
    n, m = s.natoms, s.nbonds
    if v3000 is None:
        v3000 = n > 999 or m > 999
    A, B = s._atoms, s._bonds
    lines = [s._ct_names[0] if s.ncts else "", "", ""]
    if v3000:
        lines += _format_v3000(s)
        for key, value in (s._ct_props[0] if s.ncts else {}).items():
            lines += [f">  <{key}>", _value(value), ""]
        lines.append("$$$$")
        return "\n".join(lines) + "\n"
    if n > 999 or m > 999:
        raise ValueError(f"too many atoms ({n}) or bonds ({m}) for V2000; use v3000=True")
    lines.append(f"{n:3d}{m:3d}  0  0  1  0            999 V2000")
    pos = A.column("pos").astype(np.float32).astype(np.float64)
    parity = A.column("stereo_parity") if "stereo_parity" in A else np.zeros(n, np.int64)
    for a, (x, y, z) in enumerate(pos.tolist()):
        sym = msys_symbol(int(A.column("anum")[a]))[:2]
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<2s}  0  0{int(parity[a]):3d}  0  0  0")
    stereo = B.column("stereo") if "stereo" in B else np.zeros(m, np.int64)
    arom = B.column("aromatic") if "aromatic" in B else np.zeros(m, np.int64)
    for b in range(m):
        i, j, st = int(B.column("i")[b]) + 1, int(B.column("j")[b]) + 1, int(stereo[b])
        if st < 0:
            i, j, st = j, i, -st
        order = 4 if arom[b] else int(B.column("order")[b])
        lines.append(f"{i:3d}{j:3d}{order:3d}{st:3d}  0  0")
    for a in np.flatnonzero(A.column("formal_charge")).tolist():
        lines.append(f"M  CHG  1 {a + 1:3d} {int(A.column('formal_charge')[a]):3d}")
    if "isotope" in A:
        for a in np.flatnonzero(A.column("isotope")).tolist():
            lines.append(f"M  ISO  1 {a + 1:3d} {int(A.column('isotope')[a]):3d}")
    lines.append("M  END")
    for key, value in (s._ct_props[0] if s.ncts else {}).items():
        lines += [f">  <{key}>", _value(value), ""]
    lines.append("$$$$")
    return "\n".join(lines) + "\n"


def save_sdf(system: System, path, append: bool = False, v3000: bool | None = None) -> None:
    """Write each ct as one SDF entry; V3000 when forced or above 999 atoms or bonds."""
    path = os.fspath(path)
    parts = []
    for ct in range(max(system.ncts, 1)):
        ids = system.ct_atoms(ct) if system.ncts else np.arange(system.natoms)
        if system.ncts > 1 and len(ids) == 0:
            continue
        sub = system if system.ncts <= 1 else system.clone(ids)
        parts.append(_format_ct(sub, v3000))
    mode = "at" if append else "wt"
    opener = gzip.open if path.endswith(".gz") else bz2.open if path.endswith(".bz2") else open
    with opener(path, mode) as fh:
        fh.write("".join(parts))
