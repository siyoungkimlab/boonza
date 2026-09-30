"""SIRAH: a coarse-grained force field that keeps its parameters in its topology.

SIRAH maps a protein to a few beads a residue, keeps explicit charges on them
and runs with PME, unlike Martini's reaction field.  boonza reads a SIRAH
topology and runs it; the settings here are the ones SIRAH's own mdp files
use (``tutorial/7/md_CGPROT.mdp`` of the GROMACS distribution):

* a 20 fs step,
* PME with a 1.2 nm cut-off for both Coulomb and Lennard-Jones,
* Lennard-Jones shifted to zero at the cut-off, as GROMACS's Verlet scheme
  does by default, and no dispersion correction,
* the medium's own permittivity, since the charges are explicit.

Reference: Machado, Barrera, Klein, Sóñora, Silva and Pantano, J. Chem.
Theory Comput. 15, 2719-2733 (2019), and https://www.sirahff.com.
"""

from __future__ import annotations

import zipfile
from functools import cache
from pathlib import Path

#: The force field boonza carries, as its July 2020 release ships it.
PARAMETERS = Path(__file__).resolve().parent.parent / "data" / "sirah" / "sirah_x2.2.zip"


@cache
def _archive() -> zipfile.ZipFile:
    return zipfile.ZipFile(PARAMETERS)


def contents() -> list[str]:
    """The files the carried force field holds."""
    return sorted(_archive().namelist())


def read(name: str) -> str:
    """One file of the carried force field, as text.

    ``name`` is its name in the release (``aminoacids.rtp``), with or without
    the folders the mapping files sit in.
    """
    here = {Path(n).name: n for n in _archive().namelist()}
    if name in here:
        name = here[name]
    try:
        return _archive().read(name).decode()
    except KeyError:
        raise FileNotFoundError(
            f"{name} is not in the carried SIRAH force field; it holds {', '.join(contents())}"
        ) from None


#: What a built topology includes; :func:`unpack` writes these and whatever
#: they include in turn.
INCLUDED = ("forcefield.itp", "solv.itp")


def needed(names=INCLUDED) -> list[str]:
    """``names`` and every file they ``#include``, in the order a run needs them."""
    import re

    here = {Path(n).name: n for n in _archive().namelist()}
    out: list[str] = []
    todo = [n for n in names]
    while todo:
        name = todo.pop(0)
        if name in out or name not in here:
            continue
        out.append(name)
        todo += re.findall(r'#include\s+"([^"]+)"', read(name))
    return out


def unpack(directory, everything: bool = False) -> Path:
    """Write the force field a run needs into ``directory``/sirah.ff; return it.

    A topology that includes its parameters needs them on disk, and a run
    directory that carries its own is one that moves -- in GROMACS as well as
    here.  What goes out is what the topology includes and what those files
    include in turn, which is the parameters and the solvent: the rest of the
    release is read from the archive where boonza needs it (the residue
    libraries when it builds a topology, the maps when it places beads, the
    water box when it fills one) and has no business in a run directory.

    ``everything`` writes the release as it ships instead, documentation and
    all, for running SIRAH's own tools beside it -- pdb2gmx, say, which reads
    the residue libraries.
    """
    out = Path(directory) / "sirah.ff"
    out.mkdir(parents=True, exist_ok=True)
    names = _archive().namelist() if everything else needed()
    here = {Path(n).name: n for n in _archive().namelist()}
    for name in names:
        (out / Path(name).name).write_bytes(_archive().read(here.get(name, name)))
    return out


#: What :func:`boonza.to_openmm` needs to reproduce GROMACS for SIRAH.
OPENMM_OPTIONS = {
    "nonbonded_method": "PME",
    "cutoff": 12.0,
    "dispersion_correction": False,
    "lj_shift": True,
}
#: SIRAH's water: four beads standing for about eleven waters.
WATER = "WT4"
#: Its electrolytes, which stand for solvated ions.
IONS = ("NaW", "KW", "ClW")
#: One ion pair for this many waters is about 0.15 M (Machado et al. 2019).
WATERS_PER_ION_PAIR = 34
#: The backbone beads of a SIRAH residue: amide N, alpha C and carbonyl O.
BACKBONE = ("GN", "GC", "GO")


def __getattr__(name):  # the builder is heavier than the settings above
    if name in ("Sirahized", "sirahize", "solvate", "map_structure", "read_residues"):
        from . import build

        return getattr(build, name)
    raise AttributeError(name)


__all__ = ["BACKBONE", "INCLUDED", "IONS", "OPENMM_OPTIONS", "PARAMETERS", "WATER",
           "WATERS_PER_ION_PAIR", "Sirahized", "contents", "map_structure", "needed", "read",
           "read_residues", "sirahize", "solvate", "unpack"]  # fmt: skip
