"""Parser for the Maestro (MAE/CMS) text format.

A file is a sequence of blocks.  Each block has a schema (typed keys such as
``s_m_title`` or ``r_chorus_box_ax``), a ``:::`` separator, one value per key,
then sub-blocks.  Array blocks (``m_atom[1234] { ... }``) have a schema and
rows of values.  Array rows make up nearly all of a file, so they are split
with one regex call per block and converted column-wise with numpy.

Tokenizing follows msys: ``#`` starts a comment (up to the next ``#`` or end
of line), strings may be double-quoted with backslash escapes, and ``<>`` is
a null value.
"""

from __future__ import annotations

import bz2
import gzip
import re
import warnings

import numpy as np

from .._columns import STR

_SKIP = re.compile(r"(?:\s+|#[^#\n]*#?)*")
_STRUCT = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]]|[^\s{}\[\]"]+')
_VALUE = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
_ROW = re.compile(r'"(?:[^"\\]|\\.)*"|#[^#\n]*#?|\S+')
_END_ROWS = re.compile(r"(?<!\S):::(?!\S)")
_ESCAPE = re.compile(r"\\(.)", re.DOTALL)
_LEADING_INT = re.compile(r"\s*[+-]?\d+")
_LEADING_FLOAT = re.compile(r"\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

NULL = "<>"


class MaeError(ValueError):
    """Malformed MAE content."""


def _unquote(tok: str) -> str:
    if len(tok) > 1 and tok[0] == '"' and tok[-1] == '"':
        return _ESCAPE.sub(r"\1", tok[1:-1])
    return tok


def _atoi(tok: str) -> int:
    m = _LEADING_INT.match(tok)
    return int(m.group()) if m else 0


def _atof(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        m = _LEADING_FLOAT.match(tok)
        return float(m.group()) if m else 0.0


def _scalar(tok: str, kind: str):
    if tok == NULL:
        return None
    tok = _unquote(tok)
    if kind == "s":
        return tok
    if kind == "r":
        return _atof(tok)
    if kind == "b":
        return bool(_atoi(tok))
    return _atoi(tok)


def _parse_numbers(toks: list[str], dtype, conv) -> np.ndarray:
    """Parse a column of numeric tokens; C-level fast path, per-token fallback."""
    if not toks:
        return np.empty(0, dtype)
    try:
        with warnings.catch_warnings():
            # a token numpy cannot read (e.g. "1.0" in an int column) warns and stops early
            warnings.simplefilter("error")
            out = np.fromstring(" ".join(toks), dtype=dtype, sep=" ")
        if len(out) == len(toks):
            return out
    except (ValueError, DeprecationWarning):
        pass
    return np.array([conv(t) for t in toks], dtype=dtype)


class MaeArray:
    """Columns of an array block; ``nulls[key]`` marks ``<>`` entries when present."""

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.cols: dict[str, np.ndarray] = {}
        self.kinds: dict[str, str] = {}
        self.nulls: dict[str, np.ndarray] = {}

    def __contains__(self, key) -> bool:
        return key in self.cols

    def get(self, key, default=None):
        return self.cols.get(key, default)

    def __getitem__(self, key) -> np.ndarray:
        return self.cols[key]

    def add(self, key: str, kind: str, toks: list[str], quoted: bool = True) -> None:
        null = None
        if NULL in toks:
            null = np.array([t == NULL for t in toks])
            self.nulls[key] = null
            toks = ["" if t == NULL else t for t in toks]
        self.kinds[key] = kind
        if kind == "s":
            arr = np.array(toks, dtype=STR) if toks else np.array([], dtype=STR)
            if quoted and len(arr):
                # names repeat heavily, so unquote each distinct value once
                uniq, inv = np.unique(arr, return_inverse=True)
                words = uniq.tolist()
                if any(w[:1] == '"' for w in words):
                    arr = np.array([_unquote(w) for w in words], dtype=STR)[inv.reshape(-1)]
            self.cols[key] = arr
            return
        if quoted:
            toks = [_unquote(t) if t[:1] == '"' else t for t in toks]
        if null is not None:
            toks = [("0" if n else t) for t, n in zip(toks, null.tolist(), strict=True)]
        dtype = np.float64 if kind == "r" else np.int64
        self.cols[key] = _parse_numbers(toks, dtype, _atof if kind == "r" else _atoi)

    def __repr__(self) -> str:
        return f"<MaeArray {self.name}[{self.size}] {list(self.cols)}>"


class _Scanner:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _skip(self) -> None:
        self.pos = _SKIP.match(self.text, self.pos).end()

    def line(self) -> int:
        return self.text.count("\n", 0, self.pos) + 1

    def peek(self) -> str | None:
        self._skip()
        m = _STRUCT.match(self.text, self.pos)
        return m.group() if m else None

    def take(self) -> str:
        self._skip()
        m = _STRUCT.match(self.text, self.pos)
        if m is None:
            raise MaeError(f"premature end of file at line {self.line()}")
        self.pos = m.end()
        return m.group()

    def expect(self, tok: str) -> None:
        got = self.take()
        if got != tok:
            raise MaeError(f"line {self.line()}: expected '{tok}', have '{got}'")

    def value(self) -> str:
        self._skip()
        m = _VALUE.match(self.text, self.pos)
        if m is None:
            raise MaeError(f"premature end of file at line {self.line()}")
        tok = m.group()
        if tok in (":::", "}"):
            raise MaeError(f"line {self.line()}: unexpected '{tok}'")
        self.pos = m.end()
        return tok

    def schema(self) -> list[tuple[str, str]]:
        keys = []
        while True:
            if self.peek() is None:
                raise MaeError("premature end of file in schema")
            self._skip()
            tok = _VALUE.match(self.text, self.pos).group()
            self.pos += len(tok)
            if tok == ":::":
                return keys
            if len(tok) < 3 or tok[1] != "_" or tok[0] not in "birs":
                raise MaeError(f"line {self.line()}: invalid schema entry '{tok}'")
            keys.append((tok[2:], tok[0]))


def _values(sc: _Scanner, block: dict) -> None:
    for key, kind in sc.schema():
        block[key] = _scalar(sc.value(), kind)


def _block_body(sc: _Scanner, block: dict) -> None:
    sc.expect("{")
    _values(sc, block)
    while True:
        tok = sc.peek()
        if tok is None:
            raise MaeError("premature end of file inside a block")
        if tok == "}":
            sc.take()
            return
        name = sc.take()
        if not (name[:1].isalpha() or name[:1] == "_"):
            raise MaeError(f"line {sc.line()}: expected a block name, have '{name}'")
        if sc.peek() == "[":
            block[name] = _array_body(sc, name)
        else:
            sub: dict = {}
            _block_body(sc, sub)
            block[name] = sub


def _array_body(sc: _Scanner, name: str) -> MaeArray:
    sc.expect("[")
    sc.take()  # declared row count; the rows themselves are authoritative
    sc.expect("]")
    sc.expect("{")
    keys = sc.schema()
    text = sc.text
    end = text.find(":::", sc.pos)
    while end >= 0 and not (
        text[end - 1].isspace() and (end + 3 == len(text) or text[end + 3].isspace())
    ):
        end = text.find(":::", end + 1)
    if end < 0:
        raise MaeError(f"array {name}: missing ':::' after rows")
    chunk = text[sc.pos : end]
    quoted = '"' in chunk
    if quoted or "#" in chunk:
        toks = [t for t in _ROW.findall(chunk) if t[0] != "#"]
    else:
        toks = chunk.split()
    sc.pos = end + 3
    ncol = len(keys) + 1  # the first column of every row is its index
    if len(toks) % ncol:
        raise MaeError(f"array {name}: {len(toks)} values do not fill rows of {ncol}")
    arr = MaeArray(name, len(toks) // ncol)
    for c, (key, kind) in enumerate(keys):
        arr.add(key, kind, toks[c + 1 :: ncol], quoted)
    tok = sc.take()
    if tok != "}":  # one level of nested array is tolerated, and skipped (as msys does)
        warnings.warn(f"skipping nested array '{tok}' in {name}", stacklevel=2)
        _array_body(sc, tok)
        sc.expect("}")
    return arr


def parse_mae(text: str) -> list[dict]:
    """Parse MAE text into a list of top-level blocks (dicts with ``__name__``)."""
    sc = _Scanner(text)
    blocks = []
    while True:
        tok = sc.peek()
        if tok is None:
            return blocks
        if tok == "{":  # unnamed header block, e.g. the m2io version
            sc.take()
            _values(sc, {})
            sc.expect("}")
            continue
        name = sc.take()
        block: dict = {"__name__": name}
        _block_body(sc, block)
        blocks.append(block)


def read_text(path: str) -> str:
    """File contents, transparently decompressing gzip or bzip2."""
    with open(path, "rb") as f:
        head = f.read(3)
    if head[:2] == b"\x1f\x8b":
        opener = gzip.open
    elif head == b"BZh":
        opener = bz2.open
    else:
        opener = open
    with opener(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")
