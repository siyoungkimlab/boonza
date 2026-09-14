"""Growable column storage shared by every boonza table.

Each table (atoms, bonds, residues, chains, term tables, param tables) is a
set of equally long numpy arrays.  Arrays are over-allocated so that adding
rows one at a time stays cheap; ``column()`` returns a view of the live rows.
Rows past the live count always hold the column default, so appending never
has to reset storage.
"""

from __future__ import annotations

import numpy as np

STR = np.dtypes.StringDType()

KINDS = ("int", "float", "str")
_KIND_DTYPE = {"int": np.dtype(np.int64), "float": np.dtype(np.float64), "str": STR}
_KIND_DEFAULT = {"int": 0, "float": 0.0, "str": ""}
_ALIASES = {
    int: "int",
    float: "float",
    str: "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "real": "float",
    "str": "str",
    "string": "str",
    "text": "str",
}


def as_kind(kind) -> str:
    """Normalize a user-facing type (``int``, ``"float"``, ``"text"``...) to a kind."""
    try:
        return _ALIASES[kind]
    except (KeyError, TypeError):
        raise ValueError(f"unknown value type {kind!r}; use int, float or str") from None


def kind_of(dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == STR or dtype.kind in "UOS":
        return "str"
    if dtype.kind in "iub":
        return "int"
    if dtype.kind == "f":
        return "float"
    raise ValueError(f"cannot store values of dtype {dtype} in a boonza column")


def infer_kind(value) -> str:
    """Kind of a python scalar or array-like."""
    if isinstance(value, bool | int | np.integer):
        return "int"
    if isinstance(value, float | np.floating):
        return "float"
    if isinstance(value, str):
        return "str"
    return kind_of(np.asarray(value).dtype)


def default_for(kind: str):
    return _KIND_DEFAULT[kind]


def dtype_for(kind: str) -> np.dtype:
    return _KIND_DTYPE[kind]


def scalar(x):
    """Convert a numpy scalar to the matching python value."""
    return x.item() if isinstance(x, np.generic) else x


def _row_index(rows, n: int) -> np.ndarray:
    rows = np.asarray(rows)
    if rows.dtype == bool:
        if rows.shape != (n,):
            raise IndexError(f"boolean mask has shape {rows.shape}, expected ({n},)")
        return np.flatnonzero(rows)
    return rows.astype(np.int64, copy=False).reshape(-1)


class ColumnTable:
    """Equal-length columns with amortized O(1) row appends."""

    __slots__ = ("_data", "_spec", "_n", "_cap", "builtin")

    def __init__(self, builtin=()):
        self._data: dict[str, np.ndarray] = {}
        self._spec: dict[str, tuple[str, tuple, object]] = {}
        self._n = 0
        self._cap = 0
        self.builtin = frozenset(builtin)

    # ---- schema -------------------------------------------------------
    def add_column(self, name: str, kind, shape=(), default=None) -> None:
        kind = as_kind(kind)
        shape = tuple(shape)
        if name in self._spec:
            old_kind, old_shape, _ = self._spec[name]
            if (old_kind, old_shape) != (kind, shape):
                raise ValueError(
                    f"column {name!r} already exists with type {old_kind}{list(old_shape) or ''}"
                )
            return
        if default is None:
            default = _KIND_DEFAULT[kind]
        self._data[name] = np.full((self._cap, *shape), default, dtype=_KIND_DTYPE[kind])
        self._spec[name] = (kind, shape, default)

    def del_column(self, name: str) -> None:
        if name in self.builtin:
            raise ValueError(f"cannot delete built-in column {name!r}")
        del self._data[name]
        del self._spec[name]

    def kind(self, name: str) -> str:
        return self._spec[name][0]

    def spec(self, name: str):
        return self._spec[name]

    @property
    def names(self) -> list[str]:
        return list(self._spec)

    @property
    def props(self) -> list[str]:
        """Columns that are not built in."""
        return [n for n in self._spec if n not in self.builtin]

    def __contains__(self, name) -> bool:
        return name in self._spec

    def __len__(self) -> int:
        return self._n

    # ---- data ---------------------------------------------------------
    def column(self, name: str) -> np.ndarray:
        return self._data[name][: self._n]

    def set(self, name: str, values, rows=slice(None)) -> None:
        if name not in self._spec:
            raise KeyError(name)
        self._data[name][: self._n][rows] = values

    def _reserve(self, need: int) -> None:
        if need <= self._cap:
            return
        cap = max(need, 2 * self._cap, 8)
        for name, arr in self._data.items():
            _, shape, default = self._spec[name]
            new = np.full((cap, *shape), default, dtype=arr.dtype)
            new[: self._n] = arr[: self._n]
            self._data[name] = new
        self._cap = cap

    def append(self, count: int = 1, values: dict | None = None) -> int:
        """Add ``count`` default rows, optionally filled from ``values``; return first index."""
        start = self._n
        self._reserve(start + count)
        self._n += count
        if values:
            rows = slice(start, start + count)
            try:
                for name, v in values.items():
                    if name not in self._spec:
                        raise KeyError(name)
                    self._data[name][rows] = v
            except BaseException:
                self._reset(start, start + count)
                self._n = start
                raise
        return start

    def _reset(self, lo: int, hi: int) -> None:
        for name, arr in self._data.items():
            arr[lo:hi] = self._spec[name][2]

    def keep(self, rows) -> None:
        """Keep only ``rows`` (indices in the desired order, or a bool mask)."""
        rows = _row_index(rows, self._n)
        n_old, m = self._n, len(rows)
        for arr in self._data.values():
            arr[:m] = arr[:n_old][rows]
        self._reset(m, n_old)
        self._n = m

    def take(self, rows=None) -> ColumnTable:
        """New table holding copies of ``rows`` (all rows when None)."""
        out = ColumnTable(self.builtin)
        out._spec = dict(self._spec)
        if rows is None:
            out._data = {k: v[: self._n].copy() for k, v in self._data.items()}
            count = self._n
        else:
            rows = _row_index(rows, self._n)
            out._data = {k: v[: self._n][rows] for k, v in self._data.items()}
            count = len(rows)
        out._n = out._cap = count
        return out

    def extend(self, other: ColumnTable, rows=None, overrides: dict | None = None) -> int:
        """Append rows of ``other``; missing columns are added.  Return first new index."""
        for name, (kind, shape, default) in other._spec.items():
            self.add_column(name, kind, shape, default)
        if rows is not None:
            rows = _row_index(rows, other._n)
        count = other._n if rows is None else len(rows)
        start = self.append(count)
        sl = slice(start, start + count)
        overrides = overrides or {}
        for name in other._spec:
            if name in overrides:
                self._data[name][sl] = overrides[name]
            else:
                col = other.column(name)
                self._data[name][sl] = col if rows is None else col[rows]
        return start

    def copy(self) -> ColumnTable:
        return self.take()
