"""PDBx/mmCIF coordinate files.

Reading uses the ``_atom_site`` loop: ``auth_*`` fields are preferred (they
match PDB files), the chain is ``auth_asym_id`` and the segid is
``label_asym_id``, so ligands and waters that share a chain letter with a
polymer stay separate chains.  Each ``pdbx_PDB_model_num`` becomes a ct.
``occupancy``, ``bfactor`` and ``altloc`` become atom properties, the space
group and Z become ct properties, and bonds are guessed from geometry as for
PDB files.  ``.`` and ``?`` read as empty (or 0 for numbers).

Loop bodies without quoted values, comments or text fields are split with
numpy directly on the bytes, which keeps large files fast.
"""

from __future__ import annotations

import bz2
import gzip
import os
import re

import numpy as np

from .._columns import STR
from ..analyze import RES_NUCLEIC, RES_PROTEIN
from ..elements import msys_symbol
from ..system import System
from ._maeparse import read_text
from ._sqlite_scan import _gather
from .dms import _factorize
from .pdb import (
    SPACE_GROUP,
    Z_VALUE,
    _element,
    cell_from_lengths_angles,
    lengths_angles_from_cell,
    named_bonds,
)

_RESERVED = ("loop_", "data_", "save_", "global_", "stop_")
_TOKEN = re.compile(
    r"""'((?:[^'\n]|'(?=\S))*)'(?=\s|$)|"((?:[^"\n]|"(?=\S))*)"(?=\s|$)|(\#.*)|(\S+)"""
)
_QUOTE_OR_COMMENT = re.compile(r"\s['\"#]")
# the first line after a loop body: a text field, a tag or a reserved word
_LOOP_END = re.compile(r"^(?:;|[ \t]*(?:_|(?i:loop_|data_|save_|global_|stop_)))", re.M)
_NULL = ("?", ".")


class CifError(ValueError):
    """Malformed CIF content."""


class CifBlock:
    """One ``data_`` block: lower-case tag -> numpy array of values (loops have many)."""

    def __init__(self, name: str):
        self.name = name
        self.items: dict[str, np.ndarray] = {}

    def category(self, name: str) -> dict[str, np.ndarray]:
        prefix = name.lower().rstrip(".") + "."
        return {k[len(prefix) :]: v for k, v in self.items.items() if k.startswith(prefix)}

    def value(self, tag: str, default: str = "") -> str:
        v = self.items.get(tag.lower())
        if v is None or len(v) == 0:
            return default
        text = _text(v[:1])[0]
        return default if text == "" else str(text)

    def __repr__(self) -> str:
        return f"<CifBlock {self.name!r}: {len(self.items)} items>"


# ---------------------------------------------------------------------------
# tokenizing


def _line_tokens(line: str) -> list[tuple[str, bool]]:
    """(text, quoted) tokens of one line; a comment ends the line."""
    out = []
    for m in _TOKEN.finditer(line):
        single, double, comment, bare = m.groups()
        if comment is not None:
            break
        if single is not None:
            out.append((single, True))
        elif double is not None:
            out.append((double, True))
        else:
            out.append((bare, False))
    return out


def _text_field(lines: list[str], i: int) -> tuple[str, int]:
    """A ``;``-delimited text field starting at line ``i``; returns (value, next line)."""
    content = [lines[i][1:]]
    i += 1
    while i < len(lines) and not lines[i].startswith(";"):
        content.append(lines[i])
        i += 1
    if i >= len(lines):
        raise CifError("unterminated text field")
    return "\n".join(content), i + 1


def _ends_loop(stripped: str) -> bool:
    return stripped.startswith("_") or stripped[:7].lower().startswith(_RESERVED)


def parse_cif(text: str) -> list[CifBlock]:
    """Parse CIF text into its data blocks."""
    lines = text.split("\n")
    lengths = np.fromiter(map(len, lines), np.int64, len(lines))
    offsets = np.concatenate([[0], np.cumsum(lengths + 1)])  # start of each line in text
    blocks: list[CifBlock] = []
    cur: CifBlock | None = None
    pending: str | None = None
    i, n = 0, len(lines)

    def block() -> CifBlock:
        nonlocal cur
        if cur is None:
            cur = CifBlock("")
            blocks.append(cur)
        return cur

    while i < n:
        line = lines[i].rstrip("\r")
        if line.startswith(";"):
            value, i = _text_field(lines, i)
            if pending is None:
                raise CifError(f"line {i}: text field without a tag")
            block().items[pending] = np.array([value], dtype=STR)
            pending = None
            continue
        i += 1
        stripped = line.strip()
        if not stripped or stripped[0] == "#":
            continue
        low = stripped.lower()
        if low.startswith("data_"):
            cur = CifBlock(stripped[5:])
            blocks.append(cur)
            pending = None
            continue
        if low.startswith(("save_", "global_", "stop_")):
            continue
        if low.startswith("loop_"):
            i = _loop(text, lines, offsets, i, block())
            continue
        for tok, quoted in _line_tokens(line):
            if pending is None:
                if quoted or not tok.startswith("_"):
                    raise CifError(f"line {i}: value {tok!r} without a tag")
                pending = tok.lower()
            else:
                block().items[pending] = np.array([tok], dtype=STR)
                pending = None
    return blocks


def _needs_tokenizer(body: str) -> bool:
    """True when a value starts with a quote or a comment follows whitespace."""
    if "'" not in body and '"' not in body and "#" not in body:
        return False
    return body[:1] in ("'", '"', "#") or _QUOTE_OR_COMMENT.search(body) is not None


def _loop(text: str, lines: list[str], offsets: np.ndarray, i: int, blk: CifBlock) -> int:
    n = len(lines)
    tags: list[str] = []
    while i < n:
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
        elif s.startswith("_"):
            tags.append(s.split()[0].lower())
            i += 1
        else:
            break
    if not tags:
        raise CifError(f"line {i}: loop_ without tags")
    start = i
    m = _LOOP_END.search(text, int(offsets[start])) if start < n else None
    end = n if m is None else int(np.searchsorted(offsets, m.start(), "right")) - 1
    has_text = m is not None and text.startswith(";", m.start())
    if has_text:  # text fields inside the loop: find its end line by line
        i = end
        while i < n:
            if lines[i].startswith(";"):
                _, i = _text_field(lines, i)
                continue
            stripped = lines[i].strip()
            if stripped and _ends_loop(stripped):
                break
            i += 1
        end = i
    stop = end  # leave out the blank and comment lines that usually close a loop
    while stop > start and (not lines[stop - 1].strip() or lines[stop - 1].lstrip()[:1] == "#"):
        stop -= 1
    body = text[offsets[start] : offsets[stop]] if stop > start else ""
    if has_text or _needs_tokenizer(body):
        columns = _split_slow(lines[start:end], len(tags))
    else:
        columns = _split_fast(body, len(tags))
    for tag, column in zip(tags, columns, strict=True):
        blk.items[tag] = column
    return end


def _split_fast(body: str, ncol: int) -> list[np.ndarray]:
    """Split whitespace-separated tokens into columns of fixed-width bytes."""
    b = np.frombuffer(body.encode("utf-8"), np.uint8)
    if len(b) == 0:
        return [np.array([], dtype="S1") for _ in range(ncol)]
    ws = b <= 32
    before = np.empty_like(ws)
    before[0] = True
    before[1:] = ws[:-1]
    after = np.empty_like(ws)
    after[-1] = True
    after[:-1] = ws[1:]
    starts = np.flatnonzero(~ws & before)
    ends = np.flatnonzero(~ws & after) + 1
    if len(starts) % ncol:
        raise CifError(f"loop has {len(starts)} values, not a multiple of its {ncol} tags")
    out = []
    for c in range(ncol):
        s, e = starts[c::ncol], ends[c::ncol]
        width = int((e - s).max()) if len(s) else 1
        mat = _gather(b, s.astype(np.int64), (e - s).astype(np.int64), width)
        out.append(mat.view(f"S{width}").reshape(len(s)))
    return out


def _split_slow(lines: list[str], ncol: int) -> list[np.ndarray]:
    tokens: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(";"):
            value, i = _text_field(lines, i)
            tokens.append(value)
            continue
        tokens += [t for t, _ in _line_tokens(lines[i].rstrip("\r"))]
        i += 1
    if len(tokens) % ncol:
        raise CifError(f"loop has {len(tokens)} values, not a multiple of its {ncol} tags")
    return [np.array(tokens[c::ncol], dtype=STR) for c in range(ncol)]


# ---------------------------------------------------------------------------
# typed columns


def _text(col) -> np.ndarray:
    """Column as strings, with ``.`` and ``?`` as empty."""
    if col is None:
        return None
    uniq, inv = np.unique(col, return_inverse=True)
    words = [u.decode("utf-8", "replace") if isinstance(u, bytes) else u for u in uniq.tolist()]
    words = ["" if w in _NULL else w for w in words]
    return np.array(words, dtype=STR)[inv.reshape(-1)]


def _number(token: str) -> float:
    token = token.split("(")[0]  # drop a standard uncertainty such as 1.234(5)
    try:
        return float(token)
    except ValueError:
        raise CifError(f"bad number {token!r}") from None


def _floats(col, n: int, default: float) -> np.ndarray:
    if col is None:
        return np.full(n, default)
    if col.dtype.kind == "S":
        null = (col == b"?") | (col == b".")
        try:
            out = np.where(null, b"0", col).astype(np.float64)
        except ValueError:
            out = np.array([_number(t.decode("utf-8", "replace")) if not z else 0.0
                            for t, z in zip(col.tolist(), null.tolist(), strict=True)])  # fmt: skip
    else:
        words = col.tolist()
        null = np.array([w in _NULL for w in words], bool)
        out = np.array([0.0 if w in _NULL else _number(w) for w in words])
    out[null] = default
    return out


def _ints(col, n: int, default: int) -> np.ndarray:
    return np.rint(_floats(col, n, default)).astype(np.int64)


# ---------------------------------------------------------------------------
# reading


def load_cif(path, guess_bonds: bool = True, struct_conn: bool = True) -> System:
    """Read the first data block with an ``_atom_site`` loop.

    Bonds are guessed from distances (``guess_bonds``), then the
    ``_struct_conn`` records (disulfides, covalent links, metal coordination;
    not hydrogen bonds or bonds to symmetry copies) are added with
    ``struct_conn``, as SSBOND/LINK are for PDB files.
    """
    path = os.fspath(path)
    blocks = parse_cif(read_text(path))
    blk = next((b for b in blocks if "_atom_site.cartn_x" in b.items), None)
    if blk is None:
        raise CifError("no _atom_site coordinates found")
    site = blk.category("_atom_site")
    n = len(site["cartn_x"])

    def pick(*names):
        return next((site[k] for k in names if k in site), None)

    chain = _text(pick("auth_asym_id", "label_asym_id"))
    segid = _text(site["label_asym_id"]) if "auth_asym_id" in site and "label_asym_id" in site \
        else np.full(n, "", dtype=STR)  # fmt: skip
    empty = np.full(n, "", dtype=STR)
    fields = {
        "name": _text(pick("auth_atom_id", "label_atom_id")),
        "resname": _text(pick("auth_comp_id", "label_comp_id")),
        "chain": empty if chain is None else chain,
        "segid": segid,
        "resid": _ints(pick("auth_seq_id", "label_seq_id"), n, 0),
        "insertion": _text(pick("pdbx_pdb_ins_code")),
        "altloc": _text(pick("label_alt_id")),
        "element": _text(pick("type_symbol")),
        "occupancy": _floats(pick("occupancy"), n, 1.0),
        "bfactor": _floats(pick("b_iso_or_equiv"), n, 0.0),
        "formal_charge": _ints(pick("pdbx_formal_charge"), n, 0),
        "pos": np.column_stack([_floats(site[f"cartn_{c}"], n, 0.0) for c in "xyz"]),
    }
    for key in ("name", "resname", "insertion", "altloc", "element"):
        if fields[key] is None:
            fields[key] = empty
    props, cell = _crystal(blk)
    model, _ = _factorize(_ints(pick("pdbx_pdb_model_num"), n, 1))

    links = _struct_conn(blk) if struct_conn else []
    out = System(path)
    for m in range(int(model.max()) + 1 if n else 0):
        rows = np.flatnonzero(model == m)
        out.append(_model({k: v[rows] for k, v in fields.items()}, blk.name, props, cell,
                          guess_bonds, links))  # fmt: skip
    out.name = path
    return out


def _crystal(blk: CifBlock):
    props: dict = {}
    space = blk.value("_symmetry.space_group_name_h-m") or blk.value("_space_group.name_h-m_alt")
    if space:
        props[SPACE_GROUP] = space
    z = blk.value("_cell.z_pdb")
    if z:
        props[Z_VALUE] = int(round(_number(z)))
    lengths = [blk.value(f"_cell.length_{k}") for k in "abc"]
    angles = [blk.value(f"_cell.angle_{k}") for k in ("alpha", "beta", "gamma")]
    if all(lengths):
        a, b, c = (_number(v) for v in lengths)
        al, be, ga = (_number(v) if v else 90.0 for v in angles)
        return props, cell_from_lengths_angles(a, b, c, al, be, ga)
    return props, np.zeros((3, 3))


def _struct_conn(blk: CifBlock) -> list:
    """Named atom pairs ((chain, resid, insertion, atom, altloc) x 2) from _struct_conn."""
    conn = blk.category("_struct_conn")
    if not conn or "conn_type_id" not in conn:
        return []
    m = len(conn["conn_type_id"])

    def col(*names):
        for name in names:
            if name in conn:
                vals = _text(conn[name])
                return np.array(["" if v in ("?", ".") else str(v) for v in vals.tolist()])
        return np.full(m, "")

    kind = np.char.lower(col("conn_type_id").astype(str))
    sym1, sym2 = col("ptnr1_symmetry"), col("ptnr2_symmetry")
    ends = []
    for p in ("1", "2"):
        ends.append((col(f"ptnr{p}_auth_asym_id", f"ptnr{p}_label_asym_id"),
                     col(f"ptnr{p}_auth_seq_id", f"ptnr{p}_label_seq_id"),
                     col(f"pdbx_ptnr{p}_pdb_ins_code"),
                     col(f"ptnr{p}_auth_atom_id", f"ptnr{p}_label_atom_id"),
                     col(f"pdbx_ptnr{p}_label_alt_id")))  # fmt: skip
    pairs = []
    for k in range(m):
        if kind[k] == "hydrog" or (sym1[k] and sym2[k] and sym1[k] != sym2[k]):
            continue
        keys = []
        for chain, seq, ins, atom, alt in ends:
            try:
                resid = int(seq[k])
            except ValueError:
                break
            keys.append((chain[k], resid, ins[k], atom[k], alt[k]))
        if len(keys) == 2:
            pairs.append(tuple(keys))
    return pairs


def _model(f: dict, title: str, props: dict, cell, guess_bonds: bool, links=()) -> System:
    n = len(f["name"])
    combo, reps = _factorize(f["element"], f["name"], f["resname"])
    guesses = [_element(str(f["element"][k]), str(f["name"][k]), str(f["resname"][k]))
               for k in reps.tolist()]  # fmt: skip
    anum = np.array(guesses, np.int64)[combo]
    key, chain_first = _factorize(f["chain"], f["segid"])
    res_code, res_first = _factorize(key, f["resid"], f["resname"], f["insertion"])

    s = System()
    ct = s.add_ct(title, **props)
    s._chains.append(len(chain_first), {"ct": ct.id, "name": f["chain"][chain_first],
                                        "segid": f["segid"][chain_first]})  # fmt: skip
    s._residues.append(len(res_first), {
        "chain": key[res_first], "resid": f["resid"][res_first],
        "name": f["resname"][res_first], "insertion": f["insertion"][res_first],
    })  # fmt: skip
    s._atoms.append(n, {"residue": res_code, "name": f["name"], "anum": anum, "pos": f["pos"],
                        "formal_charge": f["formal_charge"]})  # fmt: skip
    s.atoms["occupancy"] = f["occupancy"]
    s.atoms["bfactor"] = f["bfactor"]
    if (f["altloc"] != "").any():
        s.atoms["altloc"] = f["altloc"]
    s.cell = cell
    if guess_bonds and n:
        s.guess_bonds()
    if links:
        alt = f["altloc"] if (f["altloc"] != "").any() else None
        named_bonds(s, f["chain"], f["resid"], f["insertion"], f["name"], alt, links)
    return s


# ---------------------------------------------------------------------------
# writing


def _quote(s: str) -> str:
    if s == "":
        return "."
    if (
        s in _NULL
        or s[0] in "_#$'\";[]"
        or s.lower().startswith(_RESERVED)
        or any(c.isspace() for c in s)
    ):
        if "'" not in s:
            return f"'{s}'"
        if '"' not in s:
            return f'"{s}"'
        raise ValueError(f"cannot quote CIF value {s!r}")
    return s


def _quoted(values) -> list[str]:
    uniq, inv = np.unique(np.asarray(values), return_inverse=True)
    words = [_quote(str(u)) for u in uniq.tolist()]
    return [words[k] for k in inv.reshape(-1).tolist()]


def save_cif(system: System, path) -> None:
    """Write an mmCIF file with ``_cell``, ``_symmetry`` and an ``_atom_site`` loop."""
    path = os.fspath(path)
    A, R, C = system._atoms, system._residues, system._chains
    n = system.natoms
    res = A.column("residue")
    chn = R.column("chain")[res]
    name = system._ct_names[0] if system.ncts and system._ct_names[0] else "boonza"
    lines = [f"data_{''.join(c if not c.isspace() else '_' for c in name)}", "#"]
    props = system._ct_props[0] if system.ncts else {}
    if system.cell.any():
        a, b, c, al, be, ga = lengths_angles_from_cell(system.cell)
        for tag, v in (("length_a", a), ("length_b", b), ("length_c", c),
                       ("angle_alpha", al), ("angle_beta", be), ("angle_gamma", ga)):  # fmt: skip
            lines.append(f"_cell.{tag:<12s} {v:.3f}")
        if Z_VALUE in props:
            lines.append(f"_cell.Z_PDB        {props[Z_VALUE]}")
        lines.append("#")
    if SPACE_GROUP in props:
        lines += [f"_symmetry.space_group_name_H-M {_quote(str(props[SPACE_GROUP]))}", "#"]

    from ..atomsel import _Ctx  # residue typing, cached on the system

    restype, _ = _Ctx(system).types()
    polymer = np.isin(restype[res], [RES_PROTEIN, RES_NUCLEIC])
    group = np.where(polymer, "ATOM", "HETATM").tolist()
    chains = C.column("name")[chn]
    segids = C.column("segid")[chn]
    auth_asym = _quoted(chains)
    label_asym = _quoted(np.where(segids != "", segids, chains))
    names = _quoted(A.column("name"))
    resnames = _quoted(R.column("name")[res])
    resids = R.column("resid")[res].tolist()
    ins = _quoted(R.column("insertion")[res])
    elems = _quoted([msys_symbol(int(z)) for z in A.column("anum").tolist()])
    alt = _quoted(A.column("altloc")) if "altloc" in A else ["."] * n
    occ = (A.column("occupancy") if "occupancy" in A else np.ones(n)).tolist()
    bfac = (A.column("bfactor") if "bfactor" in A else np.zeros(n)).tolist()
    charge = A.column("formal_charge").tolist()
    model = (C.column("ct")[chn] + 1).tolist()
    xyz = A.column("pos").tolist()
    fields = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
              "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
              "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
              "B_iso_or_equiv", "pdbx_formal_charge", "auth_seq_id", "auth_comp_id",
              "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"]  # fmt: skip
    lines.append("loop_")
    lines += [f"_atom_site.{f}" for f in fields]
    pol = polymer.tolist()
    for k in range(n):
        x, y, z = xyz[k]
        seq = str(resids[k]) if pol[k] else "."
        lines.append(
            f"{group[k]} {k + 1} {elems[k]} {names[k]} {alt[k]} {resnames[k]} {label_asym[k]} "
            f"? {seq} {ins[k]} {x!r} {y!r} {z!r} {occ[k]!r} {bfac[k]!r} {charge[k]} "
            f"{resids[k]} {resnames[k]} {auth_asym[k]} {names[k]} {model[k]}"
        )
    lines.append("#")
    opener = gzip.open if path.endswith(".gz") else bz2.open if path.endswith(".bz2") else open
    with opener(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
