"""The protein a pocket is searched in: its atoms or beads, never the probes.

A structure comes in all-atom or coarse-grained.  An all-atom one searched as a
bead model is mapped onto that model's beads first, by boonza's own mappers
(:func:`boonza.martinize`, :func:`boonza.sirah.map_structure`); a coarse-grained
one is taken as it is.  Either way, the probes a swim puts in chain ``LIG`` --
ligand copies, often amino acids themselves -- are left out, since the protein
keyword and the residue names would both take them.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

from ..md.cgswim import PROBE_CHAIN
from .beads import AMINO_ACIDS, FORCEFIELDS, MODELS, guess_model

#: a selection of everything but the probes
NOT_PROBES = f'not chain "{PROBE_CHAIN}"'

__all__ = ["NOT_PROBES", "coarse_grain", "probe_ids", "protein_ids", "searchable"]


def probe_ids(system) -> np.ndarray:
    """The atoms or beads of the probe chain."""
    chain_of = np.asarray(system.residues["chain"])[np.asarray(system.atoms["residue"])]
    names = np.array([str(c).strip() for c in system.chains["name"]])
    return np.flatnonzero(names[chain_of] == PROBE_CHAIN) if len(names) else np.array([], int)


def protein_ids(system, selection: str | None = None) -> np.ndarray:
    """The protein's atoms or beads, or those of ``selection``, never the probes.

    A coarse-grained system the ``protein`` keyword finds nothing in is taken by
    its residue names (the twenty amino acids, their protonation variants and
    SIRAH's names).
    """
    if selection:
        ids = system.select(selection).ids
    else:
        ids = system.select("protein").ids
        if not len(ids):
            res = np.asarray(system.atoms["residue"])
            names = np.asarray(system.residues["name"])
            ids = np.flatnonzero(np.isin(names[res], list(AMINO_ACIDS)))
    return np.setdiff1d(ids, probe_ids(system))


def searchable(system, model: str, selection: str | None = None):
    """``(system, ids)``: what fpocket is to read, in ``model``'s beads.

    An all-atom structure searched as a bead model is mapped first, and the ids
    are then every bead of the mapped system.  A coarse-grained one must already
    be in ``model``'s beads.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    ids = protein_ids(system, selection)
    if not len(ids):
        raise ValueError("no protein atoms or beads selected")
    names = np.asarray(system.atoms["name"])[ids]
    resnames = np.asarray(system.residues["name"])[np.asarray(system.atoms["residue"])[ids]]
    found = guess_model(names, resnames)
    if found == "aa" and model != "aa":
        cg = coarse_grain(system, model, f"({selection or 'protein'}) and {NOT_PROBES}")
        return cg, np.arange(cg.natoms)
    if (found == "sirah") != (model == "sirah") or (found == "martini") != model.startswith(
            "martini"):  # fmt: skip
        raise ValueError(f"the structure looks like {found}, not {model}")
    return system, ids


def coarse_grain(system, model: str, atoms: str = f"protein and {NOT_PROBES}",
                 drop_unknown: bool = True):  # fmt: skip
    """``atoms`` of an all-atom ``system`` mapped onto ``model``'s beads, as a System.

    ``aa`` returns the heavy atoms themselves.  The beads keep the chain names
    and residue numbers of the atoms they were mapped from.  ``drop_unknown``
    leaves out, with a warning, residues the model has no mapping for (a D-amino
    acid, a modified residue) rather than refusing the structure.
    """
    if model == "aa":
        return system.select(f"({atoms}) and not hydrogen").clone()
    dropped: list[str] = []
    while True:
        names = " ".join(f'"{d}"' for d in dropped)  # quoted: a residue name can be numeric
        selection = atoms if not dropped else f"({atoms}) and not resname {names}"
        try:
            cg = _map(system, model, selection)
            break
        except ValueError as e:
            unknown = _unknown_residues(str(e))
            if not drop_unknown or not unknown or set(unknown) <= set(dropped):
                raise
            dropped += [u for u in unknown if u not in dropped]
    if dropped:
        warnings.warn(f"{model}: left out residues it cannot map: {' '.join(dropped)}",
                      stacklevel=2)  # fmt: skip
    if model != "sirah":  # SIRAH beads carry their residues' numbers already
        _renumber_like(cg, system, selection)
    return cg


def _map(system, model, atoms):
    from ..martini import martinize

    if model == "sirah":
        return _sirah_beads(system, atoms)
    if model not in FORCEFIELDS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return martinize(system, atoms, forcefield=FORCEFIELDS[model], cofactors=False).system()


def _sirah_beads(system, atoms):
    """SIRAH beads as a System, straight from SIRAH's map, with SIRAH's charges.

    The beads and residues come from the map, not from the built topology: the
    topology keys beads by residue number alone, so its residues cannot carry
    insertion codes (77, 77A), which crystal structures of serine proteases and
    others are full of.  ``sirahize`` is run only for the partial charges, which
    decide the polarity of side-chain beads.
    """
    from .. import sirah
    from ..system import System
    from .beads import sirah_charges

    beads = sirah.map_structure(system, atoms)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            built = sirah.sirahize(system, atoms, termini="Neutral")
        charge = {
            (str(b.chain).strip(), int(b.resid), str(b.insertion).strip(), b.name): q
            for mol in built.molecules
            for b, q in zip(mol.beads, mol.charges, strict=True)
        }
    except (IndexError, KeyError, ValueError):
        # sirahize fails on some chains (1IGJ); the residue library gives the
        # same side-chain charges, and backbone polarity goes by name
        lib = sirah_charges([b.name for b in beads], [b.residue for b in beads])
        charge = {(str(b.chain).strip(), int(b.resid), str(b.insertion).strip(), b.name): q
                  for b, q in zip(beads, lib, strict=True)}  # fmt: skip
    s = System("sirah")
    chain, residue, key = None, None, None
    for b in beads:
        if chain is None or b.chain != s.chains["name"][chain.id]:
            chain = s.add_chain(name=b.chain)
            key = None
        if (b.resid, b.insertion, b.residue) != key:
            residue = s.add_residue(
                chain, name=b.residue, resid=int(b.resid), insertion=b.insertion
            )
            key = (b.resid, b.insertion, b.residue)
        bead = (str(b.chain).strip(), int(b.resid), str(b.insertion).strip(), b.name)
        s.add_atom(residue, name=b.name, pos=tuple(float(x) for x in b.position),
                   charge=float(charge.get(bead, 0.0)))  # fmt: skip
    return s


def _unknown_residues(message: str) -> list[str]:
    """The residue names a mapper's error says it has no mapping for."""
    m = re.search(r"(?:protein residues|map has no)[:]?\s+([A-Za-z0-9_, ]+?);", message)
    return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []


def _renumber_like(cg, system, atoms) -> None:
    """Give the beads' residues the chain names and numbers of the all-atom residues.

    Both mappers keep residues in order, one bead residue per all-atom residue,
    so the k-th of the beads is the k-th all-atom residue of ``atoms``.
    """
    ids = system.select(atoms).ids
    res = np.unique(np.asarray(system.atoms["residue"])[ids])
    if len(res) != cg.nresidues:
        return  # the mapper merged or split residues; keep its numbering
    resid = np.asarray(system.residues["resid"])[res]
    ins = np.asarray(system.residues["insertion"])[res]
    chain_names = np.asarray(system.chains["name"])[np.asarray(system.residues["chain"])[res]]
    for r in range(cg.nresidues):
        cg.residues["resid"][r] = int(resid[r])
        cg.residues["insertion"][r] = str(ins[r])
    cg_chain_of = np.asarray(cg.residues["chain"])
    for c in range(cg.nchains):
        first = np.flatnonzero(cg_chain_of == c)
        if len(first):
            cg.chains["name"][c] = str(chain_names[first[0]])
