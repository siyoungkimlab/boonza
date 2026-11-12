"""CTMD: ranking docked poses by the reversible work it takes to break them.

A docking score says little about which of its own top hits will bind, and a
converged free energy says a great deal but costs too much to ask of hundreds of
ligands.  CTMD (Adury, Tiwary, Gu and Shekhar, J. Chem. Inf. Model. 2026, 66,
11929) sits between them: a few short well-tempered metadynamics runs push the
ligand off its pose, and the reversible work c(t) the bias had to do -- Tiwary and
Parrinello's estimator, J. Phys. Chem. B 2015, 119, 736 -- is the score.  Nothing
has to converge; what is compared is how hard each pose was to break.

    boonza ctmd run complex.dms --reference docked.mae --workdir out
    boonza ctmd score out/* --hits hits.csv

The published protocol biases the RMSD of the ligand after superposing the protein.
Here the coordinate is the pocket-ligand drmsd (:mod:`boonza.ctmd.cv`), which says
the same thing without a superposition, since OpenMM cannot align on one set of
atoms and measure another.  It is the one deliberate difference from the paper, and
it means the published width and range of the coordinate are worth checking before
trusting a ranking (see the guide).
"""

from .cv import (
    CUTOFF,
    MIN_POCKET,
    POCKET,
    CVError,
    drmsd_force,
    drmsd_of,
    drmsd_pairs,
    drmsd_response,
)
from .meta import Settings, bias_grid, reversible_work, run
from .score import (
    Pose,
    Replica,
    bedroc,
    enrichment_factor,
    escape_index,
    rank,
    read_colvar,
    read_replicas,
    replica_score,
    score_pose,
    score_run,
)

__all__ = [
    "CUTOFF",
    "MIN_POCKET",
    "POCKET",
    "CVError",
    "Pose",
    "Replica",
    "Settings",
    "bedroc",
    "bias_grid",
    "drmsd_force",
    "drmsd_of",
    "drmsd_pairs",
    "drmsd_response",
    "enrichment_factor",
    "escape_index",
    "rank",
    "read_colvar",
    "read_replicas",
    "replica_score",
    "reversible_work",
    "run",
    "score_pose",
    "score_run",
]
