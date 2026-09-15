"""boonza: a molecular reader, editor and writer for humans and AI.

It connects three views of a molecule: 1-D text (files, sequences, selections),
the 2-D graph (atoms, bonds, residues, molecules) and 3-D positions that obey
the rules of physics (coordinates, boxes, force fields).  The name stands for
Biomolecular Objects & Operations, Numba-accelerated, Zero-copy Arrays.
"""

__version__ = "0.1.0.dev0"

from . import (  # noqa: E402
    analysis,
    pbc,
    viparr,  # noqa: E402
)
from .align import kabsch, rmsd, superpose  # noqa: E402
from .analysis import (  # noqa: E402
    PCA,
    BlockAverage,
    block_average,
    contact_frequency,
    native_contacts,
    pca,
    radius_of_gyration,
    rdf,
    residue_contacts,
    rmsd_trajectory,
    rmsf,
    sasa,
)
from .build import (  # noqa: E402
    from_smiles,
    neutralize,
    peptide,
    repartition_hydrogen_masses,
    solvate,
)
from .ce import CEResult, ce_align, cealign  # noqa: E402
from .chem import assign_bond_orders, fragments_to_rdkit, from_rdkit, to_rdkit  # noqa: E402
from .describe import ForceFieldReport, describe, topological_distances  # noqa: E402
from .diff import Difference, canonical_forcefield, diff  # noqa: E402
from .exclusions import update_exclusions  # noqa: E402
from .ffxml import OpenMMForcefield, load_openmm_forcefield, parameterize_openmm  # noqa: E402
from .glue import Glue, make_whole  # noqa: E402
from .handles import Atom, AtomSel, Bond, Chain, Ct, Residue, StaleHandleError  # noqa: E402
from .hbonds import HBonds, baker_hubbard, hbonds, wernet_nilsson  # noqa: E402
from .io import DMSError, load, save  # noqa: E402
from .matchmaker import MatchResult, matchmaker, needleman_wunsch  # noqa: E402
from .molecules import distinct_fragments  # noqa: E402
from .omm import from_openmm, openmm_energies, to_openmm  # noqa: E402
from .rings import ring_systems, sssr  # noqa: E402
from .schemas import NONBONDED_SCHEMAS, TERM_SCHEMAS  # noqa: E402
from .secondary import backbone_dihedrals, backbone_hbonds, chimerax_ss, dssp  # noqa: E402
from .sequence import (  # noqa: E402
    SequenceAlignment,
    align_sequences,
    identity_matrix,
    sequence,
)
from .summary import Summary, summarize  # noqa: E402
from .symmetry import (  # noqa: E402
    DRMSD,
    LigandRMSD,
    SymmetryRMSD,
    drmsd,
    ligand_rmsd,
    symmetry_rmsd,
)
from .system import NonbondedInfo, System  # noqa: E402
from .terms import OverrideTable, Param, ParamTable, Term, TermTable  # noqa: E402
from .trajectory import Frame, Frames, Trajectory, open_trajectory, open_writer  # noqa: E402
from .validate import Problem, find_knots, validate  # noqa: E402
from .view import MoleculeView, view  # noqa: E402
from .viparr import (  # noqa: E402
    ViparrForcefield,
    build_constraints,
    load_forcefield,
    merge_forcefields,
    parameterize,
)

__all__ = [
    "Summary",
    "summarize",
    "from_smiles",
    "peptide",
    "neutralize",
    "repartition_hydrogen_masses",
    "solvate",
    "viparr",
    "ViparrForcefield",
    "build_constraints",
    "load_forcefield",
    "merge_forcefields",
    "parameterize",
    "OpenMMForcefield",
    "load_openmm_forcefield",
    "parameterize_openmm",
    "PCA",
    "BlockAverage",
    "MoleculeView",
    "block_average",
    "contact_frequency",
    "native_contacts",
    "pca",
    "topological_distances",
    "view",
    "Atom",
    "AtomSel",
    "Bond",
    "Chain",
    "Ct",
    "DMSError",
    "Frame",
    "ForceFieldReport",
    "Frames",
    "Glue",
    "describe",
    "Difference",
    "diff",
    "canonical_forcefield",
    "assign_bond_orders",
    "fragments_to_rdkit",
    "from_rdkit",
    "to_rdkit",
    "from_openmm",
    "openmm_energies",
    "to_openmm",
    "analysis",
    "radius_of_gyration",
    "rdf",
    "residue_contacts",
    "rmsd_trajectory",
    "rmsf",
    "sasa",
    "HBonds",
    "baker_hubbard",
    "hbonds",
    "wernet_nilsson",
    "SequenceAlignment",
    "LigandRMSD",
    "SymmetryRMSD",
    "ligand_rmsd",
    "drmsd",
    "DRMSD",
    "symmetry_rmsd",
    "align_sequences",
    "identity_matrix",
    "sequence",
    "CEResult",
    "MatchResult",
    "ce_align",
    "cealign",
    "chimerax_ss",
    "matchmaker",
    "needleman_wunsch",
    "backbone_dihedrals",
    "backbone_hbonds",
    "dssp",
    "kabsch",
    "make_whole",
    "pbc",
    "Problem",
    "find_knots",
    "distinct_fragments",
    "ring_systems",
    "update_exclusions",
    "sssr",
    "validate",
    "rmsd",
    "superpose",
    "Trajectory",
    "open_trajectory",
    "open_writer",
    "NONBONDED_SCHEMAS",
    "NonbondedInfo",
    "OverrideTable",
    "Param",
    "ParamTable",
    "Residue",
    "StaleHandleError",
    "System",
    "TERM_SCHEMAS",
    "Term",
    "TermTable",
    "load",
    "save",
]
