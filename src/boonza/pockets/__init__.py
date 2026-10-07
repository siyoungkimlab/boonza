"""Pockets of coarse-grained proteins: fpocket with presets tuned for the bead models.

    from boonza.pockets import write_fpocket_pdb, run_fpocket, preset
    write_fpocket_pdb(beads, "cg.pdb", model="martini3")
    pockets = run_fpocket("cg.pdb", preset("martini3")["flags"])

A pocket here is the shape of the protein, found with no ligand at all, where
:mod:`boonza.sites` maps where a ligand actually went.  The two share their
measures: :func:`ppc` and :func:`moc` take a pocket and a ligand as
:func:`boonza.sites.dca` does.

fpocket itself is not part of boonza: the presets need the build at
https://github.com/siyoungkimlab/fpocket, found by :func:`find_fpocket`.
"""

from .beads import (
    bead_elements,
    bead_types,
    guess_model,
    martini_polar,
    sirah_charges,
    sirah_polar,
    standard_resname,
    write_fpocket_pdb,
)
from .consensus import (
    CONSENSUS_CUTOFF,
    ConsensusPocket,
    FramePocket,
    consensus_pockets,
    frame_pockets,
)
from .core import bead_radii, enclosed_core
from .overlap import (
    grid_pocket,
    ligand_atoms_within,
    moc,
    point_pocket,
    ppc,
    sphere_pocket,
    spheres_within,
    volume_overlap,
)
from .prepare import coarse_grain, probe_ids, protein_ids, searchable
from .run import (
    Pocket,
    PocketError,
    find_fpocket,
    merge_flags,
    preset,
    presets,
    read_pockets,
    run_fpocket,
    run_mdpocket,
)

__all__ = [
    "CONSENSUS_CUTOFF",
    "ConsensusPocket",
    "FramePocket",
    "Pocket",
    "PocketError",
    "bead_elements",
    "bead_radii",
    "bead_types",
    "coarse_grain",
    "consensus_pockets",
    "enclosed_core",
    "find_fpocket",
    "frame_pockets",
    "grid_pocket",
    "guess_model",
    "ligand_atoms_within",
    "martini_polar",
    "merge_flags",
    "moc",
    "volume_overlap",
    "point_pocket",
    "ppc",
    "preset",
    "presets",
    "probe_ids",
    "protein_ids",
    "read_pockets",
    "run_fpocket",
    "run_mdpocket",
    "searchable",
    "sirah_charges",
    "sirah_polar",
    "sphere_pocket",
    "spheres_within",
    "standard_resname",
    "write_fpocket_pdb",
]
