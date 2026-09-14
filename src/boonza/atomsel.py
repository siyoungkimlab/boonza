"""msys/VMD-style atom selection language.

    s.select("protein and name CA")
    s.select("water and same residue as within 3 of resname LIG")
    s.select("pbwithin 5 of protein", pos=xyz, cell=box)

Grammar, keywords, macros and evaluation order follow msys.  Clauses narrow
the atoms chosen so far from left to right, which matters for ``nearest``:
``water and nearest 5 to protein`` picks the five waters closest to protein.
``within``, ``nearest``, ``same ... as`` and friends take everything to their
right as their argument, so ``within 3 of A and B`` means ``within 3 of (A and B)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import analyze, spatial
from ._columns import STR
from .elements import SYMBOLS


class SelectionError(ValueError):
    """The selection text could not be parsed or evaluated."""


_SYNTAX = frozenset(" '\"()<!=>+-%*/^\t\n")
_RESERVED = {
    "of", "same", "as", "to", "and", "or", "not", "within", "exwithin", "pbwithin",
    "withinbonds", "nearest", "pbnearest",
}  # fmt: skip
_FUNCS = {"sqr": np.square, "sqrt": np.sqrt, "abs": np.abs}
_INT = re.compile(r"[+-]?\d+")

MACROS = {
    "at": "resname ADE A THY T",
    "acidic": "resname ASP GLU",
    "cyclic": "resname HIS PHE PRO TRP TYR",
    "acyclic": "protein and not cyclic",
    "aliphatic": "resname ALA GLY ILE LEU VAL",
    "alpha": "protein and name CA",
    "amino": "protein",
    "aromatic": "resname HIS PHE TRP TYR",
    "basic": "resname ARG HIS LYS HSP",
    "bonded": "degree > 0",
    "buried": " resname ALA LEU VAL ILE PHE CYS MET TRP",
    "cg": "resname CYT C GUA G",
    "charged": "basic or acidic",
    "hetero": "not (protein or nucleic)",
    "hydrophobic": "resname ALA LEU VAL ILE PRO PHE MET TRP",
    "small": "resname ALA GLY SER",
    "medium": "resname VAL THR ASP ASN PRO CYS ASX PCA HYP",
    "large": "protein and not (small or medium)",
    "neutral": "resname VAL PHE GLN TYR HIS CYS MET TRP ASX GLX PCA HYP",
    "polar": "protein and not hydrophobic",
    "purine": "resname ADE A GUA G",
    "pyrimidine": "resname CYT C THY T URA U",
    "surface": "protein and not buried",
    "lipids": "lipid",
    "legacy_ion": "resname AL BA CA Ca CAL CD CES CLA CL Cl 'Cl-' CO CS CU Cu CU1 CUA HG IN "
    "IOD K 'K+' MG MN3 MO3 MO4 MO5 MO6 NA Na NAW OC7 PB POT PT RB SOD TB TL WO4 YB ZN ZN1 ZN2",
    "ion": "degree 0 and not atomicnumber 0 1 2 5 6 7 8 10 18 36 54 86",
    "ions": "ion",
    "sugar": "resname AGLC",
    "solvent": "not (protein or sugar or nucleic or lipid)",
    "carbon": "atomicnumber 6",
    "nitrogen": "atomicnumber 7",
    "sulfur": "atomicnumber 16",
    "heme": "resname HEM HEME",
}

_INT_KEYS = {
    "atomicnumber", "ctnumber", "ct", "fragment", "fragid", "index", "numbonds", "degree",
    "resid", "residue",
}  # fmt: skip
_FLOAT_KEYS = {"mass", "charge", "x", "y", "z", "vx", "vy", "vz"}
_STR_KEYS = {"chain", "element", "name", "resname", "insertion", "segid", "segname"}
_STR_FUNCS = {"smarts", "paramtype", "sequence"}
_CMP = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
        "==": np.equal, "!=": np.not_equal}  # fmt: skip
_BINPREC = {"ADD": 1, "SUB": 1, "MUL": 2, "DIV": 2, "MOD": 2, "EXP": 3}


@dataclass
class _Tok:
    kind: str
    text: str
    start: int
    end: int
    value: object = None


# ---------------------------------------------------------------------------
# tokenizer


def _tokenize(s: str, is_key) -> list[_Tok]:
    toks: list[_Tok] = []
    n = len(s)
    loc = 0

    def syntax(i):  # the end of the string counts as syntax, as in msys
        return i >= n or s[i] in _SYNTAX

    while True:
        while loc < n and s[loc].isspace():
            loc += 1
        if loc >= n:
            break
        start = loc
        end = loc
        while end < n and s[end] not in _SYNTAX:
            end += 1
        if end > loc:
            text = s[loc:end]
            loc = end
            if loc < n and s[loc] == "'":  # primes inside names: C4', 3'A
                end = loc + 1
                while end < n and s[end] not in _SYNTAX:
                    end += 1
                loc = end
                toks.append(_Tok("VAL", s[start:end], start, end))
            elif text in _RESERVED:
                toks.append(_Tok(text.upper(), text, start, end))
            elif text in _FUNCS and loc < n and s[loc] == "(":
                toks.append(_Tok("FUNC", text, start, end, _FUNCS[text]))
            elif text in MACROS:
                toks.append(_Tok("MACRO", text, start, end, MACROS[text]))
            elif _INT.fullmatch(text):
                toks.append(_Tok("INT", text, start, end, int(text)))
            elif (f := _float(text)) is not None:
                toks.append(_Tok("FLT", text, start, end, f))
            elif is_key(text):
                toks.append(_Tok("KEY", text, start, end))
            else:
                toks.append(_Tok("VAL", text, start, end))
            continue

        c = s[loc]
        nxt = s[loc + 1] if loc + 1 < n else ""
        if c in "\"'":  # "regex" or 'literal'
            end = loc + 1
            while end < n and not (s[end] == c and s[end - 1] != "\\"):
                end += 1
            if end >= n:
                raise _error(s, start, n, "unterminated quote")
            kind = "REGEX" if c == '"' else "VAL"
            toks.append(_Tok(kind, s[loc + 1 : end], start, end + 1))
            loc = end + 1
            continue
        if c in "<>":
            op = c + "=" if nxt == "=" else c
            toks.append(_Tok("CMP", op, start, start + len(op)))
            loc += len(op)
            continue
        if c in "=!":
            if nxt != "=":
                raise _error(s, start, start + 1, "unexpected character")
            toks.append(_Tok("CMP", c + "=", start, start + 2))
            loc += 2
            continue
        kind = None
        if c == "(":
            kind = "LP"
        elif c == ")":
            kind = "RP"
        elif c == "-":
            unary = loc == 0 or (s[loc - 1] in _SYNTAX and (nxt == "(" or not syntax(loc + 1)))
            kind = "NEG" if unary else "SUB"
        elif c == "+":
            unary = loc == 0 or (s[loc - 1] in _SYNTAX and (nxt == "." or nxt.isdigit()))
            kind = "PLUS" if unary else "ADD"
        elif c == "*":
            if nxt == "*":
                toks.append(_Tok("EXP", "**", start, start + 2))
                loc += 2
                continue
            kind = "MUL"
        elif c == "/":
            kind = "DIV"
        elif c == "^":
            kind = "EXP"
        elif c == "%":
            kind = "MOD"
        if kind is None:
            raise _error(s, start, start + 1, "unexpected character")
        toks.append(_Tok(kind, c, start, start + 1))
        loc += 1
    toks.append(_Tok("EOF", "", n, n))
    return toks


def _float(text: str) -> float | None:
    if "_" in text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _error(text: str, start: int, stop: int, why: str = "") -> SelectionError:
    marks = " " * start + "^" + "-" * max(stop - start - 2, 0) + ("^" if stop - start > 1 else "")
    return SelectionError(f"Parse failed{': ' + why if why else ''}:\n{text}\n{marks}")


# ---------------------------------------------------------------------------
# parser


@dataclass
class _Values:
    ints: list = field(default_factory=list)
    floats: list = field(default_factory=list)
    strs: list = field(default_factory=list)
    irng: list = field(default_factory=list)
    frng: list = field(default_factory=list)
    regex: list = field(default_factory=list)


class _Parser:
    def __init__(self, text: str, system):
        self.text = text
        self.system = system
        self.toks = _tokenize(text, lambda w: is_keyword(w, system))
        self.i = 0

    def peek(self, k: int = 0) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self, kind: str | None = None) -> _Tok:
        tok = self.peek()
        if kind is not None and tok.kind != kind:
            raise _error(self.text, tok.start, tok.end, f"expected {kind.lower()}")
        self.i += 1
        return tok

    def parse(self):
        pred = self.or_()
        tok = self.peek()
        if tok.kind != "EOF":
            raise _error(self.text, tok.start, tok.end)
        return pred

    def or_(self):
        left = self.and_()
        while self.peek().kind == "OR":
            self.take()
            left = _Or(left, self.and_())
        return left

    def and_(self):
        left = self.unary()
        while self.peek().kind == "AND":
            self.take()
            left = _And(left, self.unary())
        return left

    def unary(self):
        tok = self.peek()
        k = tok.kind
        if k == "NOT":
            self.take()
            return _Not(self.unary())
        if k in ("WITHIN", "EXWITHIN", "PBWITHIN"):
            self.take()
            r = self.number()
            self.take("OF")
            return _Within(r, k == "EXWITHIN", k == "PBWITHIN", self.or_())
        if k in ("NEAREST", "PBNEAREST"):
            self.take()
            count = self.take("INT").value
            self.take("TO")
            return _Nearest(count, k == "PBNEAREST", self.or_())
        if k == "WITHINBONDS":
            self.take()
            count = self.take("INT").value
            self.take("OF")
            return _WithinBonds(count, self.or_())
        if k == "SAME":
            self.take()
            key = self.take("KEY").text
            self.take("AS")
            return _Same(key, self.or_())
        if k == "MACRO":
            self.take()
            return _Parser(tok.value, self.system).parse()
        if k == "VAL":
            self.take()
            return _Bool(tok.text)
        if k == "LP":
            save = self.i
            try:
                self.take()
                pred = self.or_()
                self.take("RP")
                return pred
            except SelectionError:
                self.i = save  # maybe a parenthesized arithmetic expression
        if k == "KEY" and self._starts_value(1):
            self.take()
            return _Key(tok.text, self.values())
        return self.comparison()

    def _starts_value(self, k: int) -> bool:
        kind = self.peek(k).kind
        if kind in ("INT", "FLT", "VAL", "REGEX"):
            return True
        return kind == "NEG" and self.peek(k + 1).kind in ("INT", "FLT")

    def number(self) -> float:
        tok = self.take()
        sign = 1
        if tok.kind in ("NEG", "PLUS"):
            sign = -1 if tok.kind == "NEG" else 1
            tok = self.take()
        if tok.kind not in ("INT", "FLT"):
            raise _error(self.text, tok.start, tok.end, "expected a number")
        return sign * tok.value

    def _list_number(self):
        neg = self.peek().kind == "NEG"
        if neg:
            self.take()
        tok = self.take()
        return (-tok.value if neg else tok.value), tok.kind

    def values(self) -> _Values:
        v = _Values()
        while self._starts_value(0):
            tok = self.peek()
            if tok.kind == "VAL":
                v.strs.append(self.take().text)
            elif tok.kind == "REGEX":
                v.regex.append(self.take().text)
            else:
                lo, kind = self._list_number()
                if self.peek().kind == "TO":
                    here = self.take()
                    if not self._starts_value(0) or self.peek().kind in ("VAL", "REGEX"):
                        raise _error(self.text, here.start, here.end, "range needs two numbers")
                    hi, kind2 = self._list_number()
                    if kind2 != kind:
                        raise _error(self.text, here.start, here.end, "mixed int/float range")
                    (v.irng if kind == "INT" else v.frng).append((lo, hi))
                else:
                    (v.ints if kind == "INT" else v.floats).append(lo)
        return v

    def comparison(self):
        lhs = self.expr()
        op = self.take("CMP").text
        return _Cmp(op, lhs, self.expr())

    def expr(self, minprec: int = 1):
        left = self.primary()
        while self.peek().kind in _BINPREC and _BINPREC[self.peek().kind] >= minprec:
            op = self.take().kind
            left = _Bin(op, left, self.expr(_BINPREC[op] + 1))
        return left

    def primary(self):
        tok = self.peek()
        if tok.kind in ("INT", "FLT"):
            self.take()
            return _Lit(float(tok.value))
        if tok.kind in ("NEG", "PLUS") and self.peek(1).kind in ("INT", "FLT"):
            return _Lit(float(self.number()))
        if tok.kind == "NEG":
            self.take()
            return _Neg(self.nexpr())
        return self.nexpr()

    def nexpr(self):
        tok = self.take()
        if tok.kind == "KEY":
            return _KeyExpr(tok.text)
        if tok.kind == "LP":
            e = self.expr()
            self.take("RP")
            return e
        if tok.kind == "FUNC":
            self.take("LP")
            e = self.expr()
            self.take("RP")
            return _Func(tok.value, e)
        raise _error(self.text, tok.start, tok.end)


# ---------------------------------------------------------------------------
# evaluation


class _Ctx:
    def __init__(self, system, pos=None, cell=None):
        self.s = system
        self.n = system.natoms
        self.pos = None if pos is None else np.asarray(pos, np.float32).reshape(self.n, 3)
        self.cell = cell
        self._keys: dict = {}

    def full(self) -> np.ndarray:
        return np.ones(self.n, bool)

    def xyz32(self) -> np.ndarray:
        if self.pos is not None:
            return self.pos
        return self.s._atoms.column("pos").astype(np.float32)

    def periodic_cell(self):
        return self.s.cell if self.cell is None else self.cell

    def types(self):
        c = self.s._cache.get("analyze")
        if c is None:
            c = self.s._cache["analyze"] = analyze.classify(self.s)
        return c

    def key(self, name: str):
        hit = self._keys.get(name)
        if hit is None:
            hit = self._keys[name] = self._compute_key(name)
        return hit

    def _compute_key(self, name: str):
        s = self.s
        A, R, C = s._atoms, s._residues, s._chains
        res = A.column("residue")
        chn = R.column("chain")[res]
        anum = A.column("anum")
        if name == "atomicnumber":
            return anum, "int"
        if name in ("mass", "charge"):
            return A.column(name), "float"
        if name in ("x", "y", "z"):
            d = "xyz".index(name)
            src = self.pos if self.pos is not None else A.column("pos")
            return src[:, d].astype(np.float64), "float"
        if name in ("vx", "vy", "vz"):
            return A.column("vel")[:, "xyz".index(name[1])], "float"
        if name == "chain":
            return C.column("name")[chn], "str"
        if name in ("segid", "segname"):
            return C.column("segid")[chn], "str"
        if name in ("ct", "ctnumber"):
            return C.column("ct")[chn] + (name == "ctnumber"), "int"
        if name == "element":
            table = np.array(list(SYMBOLS), dtype=STR)
            return table[np.clip(anum, 0, len(SYMBOLS) - 1)], "str"
        if name in ("fragment", "fragid"):
            return s.fragids, "int"
        if name == "index":
            return np.arange(self.n), "int"
        if name == "name":
            return A.column("name"), "str"
        if name in ("numbonds", "degree"):
            bi, bj = s._bonds.column("i"), s._bonds.column("j")
            if name == "numbonds":
                return np.bincount(np.concatenate([bi, bj]), minlength=self.n), "int"
            deg = np.bincount(bi[anum[bj] != 0], minlength=self.n) + np.bincount(
                bj[anum[bi] != 0], minlength=self.n
            )
            return np.where(anum == 0, 0, deg), "int"
        if name == "resid":
            return R.column("resid")[res], "int"
        if name == "residue":
            return res, "int"
        if name == "resname":
            return R.column("name")[res], "str"
        if name == "insertion":
            return R.column("insertion")[res], "str"
        if name in A.props:
            kind, shape, _ = A.spec(name)
            if shape:
                raise SelectionError(f"atom property {name!r} is not a scalar")
            return A.column(name), kind
        raise SelectionError(f"Unknown atom selection keyword '{name}'")


def is_keyword(name: str, system) -> bool:
    return (
        name in _INT_KEYS
        or name in _FLOAT_KEYS
        or name in _STR_KEYS
        or name in _STR_FUNCS
        or name in system._atoms.props
    )


def _str_isin(arr: np.ndarray, values) -> np.ndarray:
    values = list(dict.fromkeys(values))
    if len(values) <= 16:
        hit = np.zeros(len(arr), bool)
        for v in values:
            hit |= arr == v
        return hit
    uniq, inv = np.unique(arr, return_inverse=True)
    wanted = set(values)
    return np.array([u in wanted for u in uniq.tolist()], bool)[inv.reshape(-1)]


def _regex_match(arr: np.ndarray, patterns) -> np.ndarray:
    try:
        regs = [re.compile(p) for p in patterns]
    except re.error as e:
        raise SelectionError(f"bad regular expression: {e}") from None
    uniq, inv = np.unique(arr, return_inverse=True)
    ok = np.array([any(r.fullmatch(u) for r in regs) for u in uniq.tolist()], bool)
    return ok[inv.reshape(-1)]


class _Bool:
    def __init__(self, name):
        self.name = name

    def eval(self, ctx, m):
        name = self.name
        if name == "all":
            return m.copy()
        if name == "none":
            return np.zeros_like(m)
        anum = ctx.s._atoms.column("anum")
        if name == "hydrogen":
            return m & (anum == 1)
        if name == "oxygen":
            return m & (anum == 8)
        if name == "noh":
            return m & (anum != 1)
        restype, atomtype = ctx.types()
        if name == "backbone":
            return m & ((atomtype == analyze.ATOM_PROBACK) | (atomtype == analyze.ATOM_NUCBACK))
        if name == "sidechain":
            return m & (atomtype == analyze.ATOM_PROSIDE)
        kinds = {
            "protein": analyze.RES_PROTEIN,
            "nucleic": analyze.RES_NUCLEIC,
            "water": analyze.RES_WATER,
            "lipid": analyze.RES_LIPID,
        }
        if name in kinds:
            return m & (restype[ctx.s._atoms.column("residue")] == kinds[name])
        if name == "polymer":  # boonza extension: protein or nucleic acid
            rt = restype[ctx.s._atoms.column("residue")]
            return m & ((rt == analyze.RES_PROTEIN) | (rt == analyze.RES_NUCLEIC))
        raise SelectionError(f"Unrecognized boolean '{name}'")


class _Key:
    def __init__(self, name, values: _Values):
        self.name = name
        self.v = values

    def eval(self, ctx, m):
        if self.name in _STR_FUNCS:
            return m & _STRFUNC[self.name](ctx, self.v.strs + self.v.regex, m)
        arr, kind = ctx.key(self.name)
        v = self.v
        if kind == "int":
            if v.floats or v.strs or v.frng or v.regex:
                raise SelectionError(f"Selection keyword '{self.name}' expects integer values")
            hit = np.isin(arr, v.ints)
            for lo, hi in v.irng:
                hit |= (arr >= lo) & (arr <= hi)
        elif kind == "float":
            if v.strs or v.regex:
                raise SelectionError(f"Selection keyword '{self.name}' expects numeric values")
            with np.errstate(invalid="ignore"):
                trunc = arr.astype(np.int64)  # C-style truncation, as msys compares ints
            hit = np.isin(trunc, v.ints) | np.isin(arr, v.floats)
            for lo, hi in v.irng:
                hit |= (trunc >= lo) & (trunc <= hi)
            for lo, hi in v.frng:
                hit |= (arr >= lo) & (arr <= hi)
        else:
            if v.ints or v.floats or v.irng or v.frng:
                raise SelectionError(f"Selection keyword '{self.name}' expects string values")
            hit = _str_isin(arr, v.strs)
            if v.regex:
                hit |= _regex_match(arr, v.regex)
        return m & hit


class _And:
    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs

    def eval(self, ctx, m):
        return self.rhs.eval(ctx, self.lhs.eval(ctx, m))


class _Or:
    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs

    def eval(self, ctx, m):
        return self.lhs.eval(ctx, m) | self.rhs.eval(ctx, m)


class _Not:
    def __init__(self, sub):
        self.sub = sub

    def eval(self, ctx, m):
        return m & ~self.sub.eval(ctx, m)


class _Within:
    def __init__(self, r, exclude, periodic, sub):
        self.r, self.exclude, self.periodic, self.sub = r, exclude, periodic, sub

    def eval(self, ctx, m):
        sub = self.sub.eval(ctx, ctx.full())
        if self.exclude:
            m = m & ~sub
        if self.r <= 0:
            return m & sub
        cand, target = np.flatnonzero(m), np.flatnonzero(sub)
        xyz = ctx.xyz32()
        cell = ctx.periodic_cell() if self.periodic else None
        out = np.zeros_like(m)
        out[cand[spatial.within(xyz[cand], xyz[target], self.r, cell)]] = True
        return out


class _Nearest:
    def __init__(self, count, periodic, sub):
        self.count, self.periodic, self.sub = count, periodic, sub

    def eval(self, ctx, m):
        sub = self.sub.eval(ctx, ctx.full())
        m = m & ~sub
        cand, target = np.flatnonzero(m), np.flatnonzero(sub)
        xyz = ctx.xyz32()
        cell = ctx.periodic_cell() if self.periodic else None
        try:
            ids = spatial.nearest(xyz[cand], cand, xyz[target], self.count, cell)
        except ValueError as e:
            raise SelectionError(str(e)) from None
        out = np.zeros_like(m)
        out[ids] = True
        return out


class _WithinBonds:
    def __init__(self, count, sub):
        self.count, self.sub = count, sub

    def eval(self, ctx, m):
        cur = self.sub.eval(ctx, ctx.full())
        bi, bj = ctx.s._bonds.column("i"), ctx.s._bonds.column("j")
        for _ in range(self.count):
            add = np.zeros_like(cur)
            add[bj[cur[bi]]] = True
            add[bi[cur[bj]]] = True
            cur = cur | add
        return m & cur


class _Same:
    def __init__(self, key, sub):
        self.key, self.sub = key, sub

    def eval(self, ctx, m):
        sub = self.sub.eval(ctx, ctx.full())
        arr, kind = ctx.key(self.key)
        if kind == "str":
            return m & _str_isin(arr, np.unique(arr[sub]).tolist())
        return m & np.isin(arr, arr[sub])


class _Lit:
    def __init__(self, value):
        self.value = value

    def eval(self, ctx):
        return np.full(ctx.n, self.value)


class _KeyExpr:
    def __init__(self, name):
        self.name = name

    def eval(self, ctx):
        arr, kind = ctx.key(self.name)
        if kind == "str":
            raise SelectionError(
                f"selection keyword '{self.name}' with string type cannot be used in a "
                "numeric expression"
            )
        return arr.astype(np.float64)


class _Neg:
    def __init__(self, sub):
        self.sub = sub

    def eval(self, ctx):
        return -self.sub.eval(ctx)


class _Func:
    def __init__(self, func, sub):
        self.func, self.sub = func, sub

    def eval(self, ctx):
        return self.func(self.sub.eval(ctx))


class _Bin:
    def __init__(self, op, lhs, rhs):
        self.op, self.lhs, self.rhs = op, lhs, rhs

    def eval(self, ctx):
        a, b = self.lhs.eval(ctx), self.rhs.eval(ctx)
        with np.errstate(all="ignore"):
            if self.op == "ADD":
                return a + b
            if self.op == "SUB":
                return a - b
            if self.op == "MUL":
                return a * b
            if self.op == "DIV":
                return a / b
            if self.op == "MOD":  # C semantics: integer remainder, truncated
                return np.fmod(a.astype(np.int64), b.astype(np.int64)).astype(np.float64)
            return np.power(a, b)


class _Cmp:
    def __init__(self, op, lhs, rhs):
        self.op, self.lhs, self.rhs = op, lhs, rhs

    def eval(self, ctx, m):
        with np.errstate(invalid="ignore"):
            return m & _CMP[self.op](self.lhs.eval(ctx), self.rhs.eval(ctx))


# string functions: smarts, paramtype, sequence


def _smarts(ctx, args, m):
    """Atoms matched by any SMARTS pattern within the molecules of the current atoms (RDKit)."""
    try:
        from rdkit import Chem
    except ImportError:
        raise SelectionError("smarts selections need rdkit") from None
    from .chem import to_rdkit

    s = ctx.s
    hit = np.zeros(ctx.n, bool)
    atoms = np.flatnonzero(np.isin(s.fragids, np.unique(s.fragids[m])))
    if len(atoms) == 0:
        return hit
    mol = to_rdkit(s, atoms, sanitize=False, conformer=False, stereo=False, residue_info=False)
    Chem.SanitizeMol(mol, catchErrors=True)
    for pattern in args:
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            raise SelectionError(f"bad SMARTS pattern {pattern!r}")
        for match in mol.GetSubstructMatches(query, uniquify=False, maxMatches=10_000_000):
            hit[atoms[list(match)]] = True
    return hit


def _paramtype(ctx, args, m=None):
    if not args:
        raise SelectionError("paramtype selection requires table name as first argument")
    table = ctx.s.tables.get(args[0])
    if table is None:
        raise SelectionError(f"paramtype selection references nonexistent table '{args[0]}'")
    if "type" not in table.params.props:
        raise SelectionError(f"paramtype selection references table '{args[0]}' with no 'type'")
    wanted = np.flatnonzero(_str_isin(table.params["type"], args[1:]))
    terms = np.isin(table.param_ids, wanted)
    hit = np.zeros(ctx.n, bool)
    hit[table.atoms[terms].ravel()] = True
    return hit


_ONE_LETTER = {
    "GLY": "G", "ALA": "A", "VAL": "V", "PHE": "F", "PRO": "P", "MET": "M", "ILE": "I",
    "LEU": "L", "ASP": "D", "GLU": "E", "LYS": "K", "ARG": "R", "SER": "S", "THR": "T",
    "TYR": "Y", "HIS": "H", "CYS": "C", "ASN": "N", "GLN": "Q", "TRP": "W", "HSE": "H",
    "HSD": "H", "HSP": "H", "HID": "H", "HIP": "H", "CYX": "C", "LYP": "K", "ADE": "A",
    "A": "A", "THY": "T", "T": "T", "CYT": "C", "C": "C", "GUA": "G", "G": "G",
}  # fmt: skip


def _sequence(ctx, args, m=None):
    s = ctx.s
    try:
        regs = [re.compile(p) for p in args]
    except re.error as e:
        raise SelectionError(f"bad regular expression: {e}") from None
    restype, _ = ctx.types()
    resnames = s._residues.column("name").tolist()
    hit_res = np.zeros(s.nresidues, bool)
    for c in range(s.nchains):
        residues = s.chain_residues(c)
        codes = "".join(
            "X" if restype[r] == analyze.RES_WATER else _ONE_LETTER.get(resnames[r], "X")
            for r in residues.tolist()
        )
        for reg in regs:
            for match in reg.finditer(codes):
                hit_res[residues[match.start() : match.end()]] = True
    return hit_res[s._atoms.column("residue")]


_STRFUNC = {"smarts": _smarts, "paramtype": _paramtype, "sequence": _sequence}


def parse(text: str, system):
    """Parse a selection into an evaluable predicate."""
    if not text or not text.strip():
        raise SelectionError("empty selection")
    return _Parser(text, system).parse()


def select(system, text: str, pos=None, cell=None) -> np.ndarray:
    """Indices of the atoms matching ``text``.

    ``pos`` replaces the system positions (e.g. a trajectory frame) for
    coordinate keywords and distance searches; ``cell`` replaces the unit
    cell used by ``pbwithin``/``pbnearest``.
    """
    pred = parse(text, system)
    ctx = _Ctx(system, pos, cell)
    return np.flatnonzero(pred.eval(ctx, ctx.full()))
