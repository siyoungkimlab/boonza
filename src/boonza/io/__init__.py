"""File readers and writers, dispatched by extension."""

from __future__ import annotations

from pathlib import Path

from ._maeparse import MaeError
from .amber import load_prmtop, read_amber_coordinates
from .cif import CifError, load_cif, save_cif
from .dms import DMSError, load_dms, save_dms
from .gro import load_gro, save_gro
from .mae import load_mae
from .mae_writer import save_mae
from .pdb import load_pdb, save_pdb
from .psf import load_psf
from .sdf import load_sdf, save_sdf

_READERS = {
    ".dms": load_dms,
    ".mae": load_mae,
    ".cms": load_mae,
    ".pdb": load_pdb,
    ".prmtop": load_prmtop,
    ".parm7": load_prmtop,
    ".psf": load_psf,
    ".sdf": load_sdf,
    ".mol": load_sdf,
    ".gro": load_gro,
    ".cif": load_cif,
    ".mmcif": load_cif,
    ".pdbx": load_cif,
}
_WRITERS = {
    ".dms": save_dms,
    ".mae": save_mae,
    ".cms": save_mae,
    ".pdb": save_pdb,
    ".sdf": save_sdf,
    ".mol": save_sdf,
    ".gro": save_gro,
    ".cif": save_cif,
    ".mmcif": save_cif,
}
_COMPRESSION = (".gz", ".bz2")
_ALIASES = {".maegz": ".mae", ".cmsgz": ".cms"}


def _format(path, fmt: str | None) -> str:
    if fmt:
        fmt = fmt.lower() if fmt.startswith(".") else "." + fmt.lower()
        return _ALIASES.get(fmt, fmt)
    name = Path(path).name.lower()
    for ext in _COMPRESSION:
        name = name.removesuffix(ext)
    suffix = Path(name).suffix
    return _ALIASES.get(suffix, suffix)


def load(path, format: str | None = None, **kwargs):
    """Read a system from ``path``; the format comes from the extension unless given."""
    fmt = _format(path, format)
    try:
        reader = _READERS[fmt]
    except KeyError:
        raise ValueError(f"cannot read {fmt!r} files yet; supported: {sorted(_READERS)}") from None
    return reader(path, **kwargs)


def save(system, path, format: str | None = None, **kwargs) -> None:
    """Write ``system`` to ``path``; the format comes from the extension unless given."""
    fmt = _format(path, format)
    try:
        writer = _WRITERS[fmt]
    except KeyError:
        raise ValueError(f"cannot write {fmt!r} files yet; supported: {sorted(_WRITERS)}") from None
    writer(system, path, **kwargs)


__all__ = [
    "CifError",
    "DMSError",
    "MaeError",
    "load",
    "load_cif",
    "load_dms",
    "load_gro",
    "load_mae",
    "load_pdb",
    "load_prmtop",
    "load_psf",
    "read_amber_coordinates",
    "load_sdf",
    "save",
    "save_cif",
    "save_dms",
    "save_gro",
    "save_mae",
    "save_pdb",
    "save_sdf",
]
