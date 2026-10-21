"""A protein as beads: positions, force-field types, sizes and polarity.

An all-atom structure is mapped and typed by boonza: Martini 2 and 3 by
``martinize`` (each bead's type as the parameterized system carries it), SIRAH
by SIRAH's own map (``boonza.pockets.coarse_grain``, which keeps insertion
codes) with each bead's type from ``sirahize`` -- matched by chain, residue and
bead name, so the termini get their own types -- or, where sirahize fails on a
chain, from SIRAH's residue library.  A structure that is coarse-grained
already must carry its types (a .dms, or a .gro read with its topology).

Polarity is what ``boonza pockets`` gives fpocket: Martini by type class (with
the N class per model), SIRAH backbone by name and side chains by charge.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .forcefield import pair_table

MODELS = ("martini2", "martini3", "sirah")
#: Martini's N-class beads count as polar for Martini 2, apolar for Martini 3 (the fpocket presets)
N_POLAR = {"martini2": True, "martini3": False}


@dataclass
class Beads:
    """The protein beads of one structure."""

    model: str
    xyz: np.ndarray  #: (n, 3) A
    types: np.ndarray  #: (n,) force-field type names
    type_index: np.ndarray  #: (n,) index into pair_table(model).types
    radius: np.ndarray  #: (n,) sigma/2 of each bead's self-pair, A
    polar: np.ndarray  #: (n,) bool

    def __len__(self) -> int:
        return len(self.xyz)


def from_arrays(model: str, xyz, types, polar) -> Beads:
    table = pair_table(model)
    k = table.index(types)
    return Beads(model, np.asarray(xyz, float), np.asarray(types, str), k,
                 table.sigma[k, k] / 2, np.asarray(polar, bool))  # fmt: skip


def _sirah_types(system, cg, atoms) -> list[str]:
    from .. import sirah
    from ..sirah.build import read_residues

    names = np.asarray(cg.atoms["name"])
    res = np.asarray(cg.atoms["residue"])
    key = [(str(cg.chains["name"][cg.residues["chain"][r]]).strip(), int(cg.residues["resid"][r]),
            str(cg.residues["insertion"][r]).strip(), str(n))
           for n, r in zip(names, res, strict=True)]  # fmt: skip
    found = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            built = sirah.sirahize(system, atoms).system()  # its termini, as a run has them
        bres = np.asarray(built.atoms["residue"])
        for n, r, t in zip(built.atoms["name"], bres, built.atoms["type"], strict=True):
            found[(str(built.chains["name"][built.residues["chain"][r]]).strip(),
                   int(built.residues["resid"][r]), str(built.residues["insertion"][r]).strip(),
                   str(n))] = str(t)  # fmt: skip
    except (IndexError, KeyError, ValueError):
        pass  # sirahize fails on some chains (1IGJ): the residue library below
    lib = read_residues()[0]
    out = []
    for (_, _, _, name), r in zip(key, res, strict=True):
        t = found.get(key[len(out)])
        if t is None:
            entry = lib.get(str(cg.residues["name"][r]).strip())
            t = {n: k for n, k, _ in entry.atoms}.get(name) if entry else None
        if t is None:
            raise ValueError(f"no SIRAH type for bead {name} of {cg.residues['name'][r]}")
        out.append(t)
    return out


def coarse_grain(system, model: str, atoms: str | None = None, with_system: bool = False):
    """The protein of an all-atom ``system`` as ``model``'s beads, typed by boonza; with
    ``with_system`` also the coarse-grained System they come from (for drawing)."""
    from ..pockets import beads as pb
    from ..pockets import prepare as pp

    atoms = atoms or f"protein and {pp.NOT_PROBES}"
    cg = pp.coarse_grain(system, model, atoms)
    if model == "sirah":
        types = _sirah_types(system, cg, atoms)
        polar = [pb.sirah_polar(n, q) for n, q in zip(cg.atoms["name"], cg.atoms["charge"],
                                                      strict=True)]  # fmt: skip
    else:
        types = [str(t) for t in cg.atoms["type"]]
        polar = [pb.martini_polar(t, N_POLAR[model]) for t in types]
    beads = from_arrays(model, cg.positions, types, polar)
    return (beads, cg) if with_system else beads


def from_system(system, model: str, ids=None) -> Beads:
    """Beads of a coarse-grained ``system`` that carries its types (a run's .dms)."""
    from ..pockets import beads as pb
    from ..pockets import prepare as pp

    ids = pp.protein_ids(system) if ids is None else np.asarray(ids)
    types = [str(t) for t in np.asarray(system.atoms["type"])[ids]]
    if model == "sirah":
        polar = [pb.sirah_polar(n, q) for n, q in zip(np.asarray(system.atoms["name"])[ids],
                                                      np.asarray(system.atoms["charge"])[ids],
                                                      strict=True)]  # fmt: skip
    else:
        polar = [pb.martini_polar(t, N_POLAR[model]) for t in types]
    return from_arrays(model, np.asarray(system.positions)[ids], types, polar)
