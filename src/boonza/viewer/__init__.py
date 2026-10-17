"""An interactive molecular viewer in the browser, with boonza underneath.

    boonza view complex.mae run.xtc                  # serves http://127.0.0.1:8765
    boonza view solvated.dms run.dcd -s "not water" --fit "protein and name CA"
    boonza view MODES_OUT                            # boonza modes, triage, sitemap
    boonza view a.mae b.mae c.mae                    # several models

    from boonza import viewer
    viewer.serve(system, trajectory=traj)            # the same, from Python

session.py is one model, outputs.py many (and the folders boonza's commands
write), server.py what the page asks for.
"""

from .outputs import Model, Viewer, fetch_pdb, from_folder, from_manifest, from_paths, is_pdb_id
from .server import make_server, serve
from .session import (
    CG_RADIUS,
    LIGAND,
    MemoryTrajectory,
    Session,
    bead_radii,
    default_reps,
    is_coarse_grained,
)

__all__ = ["CG_RADIUS", "LIGAND", "MemoryTrajectory", "Model", "Session", "Viewer", "bead_radii",
           "default_reps", "fetch_pdb", "from_folder", "from_manifest", "from_paths",
           "is_coarse_grained", "is_pdb_id", "make_server", "serve"]  # fmt: skip
