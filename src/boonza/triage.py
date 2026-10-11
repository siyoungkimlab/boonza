"""Triage many runs of one system: which pose each took, and how many took it.

A thousand simulations of one protein and one ligand from a thousand starting
poses ask a different question from one simulation.  Not "how good is this
run", but "how many different things happened, and how many runs found each".

Three kinds of run come out of such a set:

* the ligand leaves, and the run stops (``boonza md --early-stop``).  Nothing
  here has to look at those: their ``status.json`` says ``detached``.
* the ligand sits where it started.
* the ligand starts badly, leaves, and settles somewhere else.  This is the
  interesting kind and the easiest to lose: every measure taken against the
  *starting* pose scores it badly, because it was right to move.  So the second
  half of each run is measured on its own as well, and a run that moved and
  then held one pose is visible as a large :attr:`Triage.drifted` with a large
  :attr:`Triage.late_population`.

What the runs are then grouped by is a **pharmacophore fingerprint**: for each
of RDKit's feature families, how near that kind of group comes to each residue.
A plain contact fingerprint cannot tell a flipped ligand from the pose it came
from -- the same residues are touched either way -- and this one can, because
the flip puts a donor where an acceptor was.  It is also what makes grouping a
thousand runs affordable: a full pairwise dRMSD is tens of minutes, these
vectors are a cosine away from each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interactions import interaction_fingerprints
from .pharmacophore import FAMILIES, ligand_features

__all__ = ["Modes", "Triage", "frame_system", "modes", "pharmacophore_fingerprint",
           "triage"]  # fmt: skip

#: Two poses nearer than this in cosine distance are one mode.  A guess that
#: wants checking against :attr:`Modes.spread` on a real set, not a constant
#: anybody measured.
MODE_CUTOFF = 0.25


@dataclass
class Fingerprint:
    """``values`` is (frames, families x residues); the columns are fixed."""

    values: np.ndarray
    residues: object
    families: tuple


def pharmacophore_fingerprint(system, positions=None, ligand="not (polymer or water or ions) "
                              "and noh", protein="protein and noh",
                              families=FAMILIES) -> Fingerprint:  # fmt: skip
    """How near each kind of chemical group comes to each residue, per frame.

    One contact fingerprint per feature family, over that family's atoms alone,
    laid end to end.  The residue columns are taken once and reused for every
    family, so the vector is the same width in the same order for every run of
    one system -- which is what lets them be compared at all.
    """
    base = interaction_fingerprints(system, positions=positions, ligand=ligand, protein=protein)
    values = np.asarray(base.values)
    frames, width = values.shape[0], values.shape[-1]
    atoms_of: dict = {}
    for copy in ligand_features(system, ligand=ligand, families=tuple(families)):
        for family, atoms in copy:
            atoms_of.setdefault(family, set()).update(int(a) for a in atoms)
    rows = []
    for family in families:
        want = sorted(atoms_of.get(family, ()))
        if not want:  # the ligand has none of this kind: a column of zeros, not a gap
            rows.append(np.zeros((frames, width)))
            continue
        fp = interaction_fingerprints(system, positions=positions, protein=protein,
                                      ligand="index " + " ".join(map(str, want)),
                                      residues=base.residues)  # fmt: skip
        v = np.asarray(fp.values)
        rows.append(v.reshape(frames, -1)[:, :width])
    return Fingerprint(np.concatenate(rows, axis=1), base.residues, tuple(families))


def cosine_distance(a, b) -> float:
    """1 - cos(a, b), and 0 for two vectors that are both nothing."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0 if (na > 1e-9) != (nb > 1e-9) else 0.0
    return float(np.clip(1.0 - np.dot(a, b) / (na * nb), 0.0, 2.0))


@dataclass
class Triage:
    """What one run did, in the numbers a thousand of them can be sorted by."""

    frames: int = 0
    #: dRMSD to the first production frame: all of it, its second half, its worst
    from_start: float = float("nan")
    drifted: float = float("nan")
    from_start_max: float = float("nan")
    #: and to the last frame, and to the run's own most populated pose
    from_end: float = float("nan")
    from_pose: float = float("nan")
    pose_frame: int = 0
    population: float = 0.0
    n_poses: int = 0
    #: the second half on its own, which is where a late settler shows itself
    late_population: float = 0.0
    late_from_pose: float = float("nan")
    #: how far the contacts travelled, first half's mean against the second's
    moved: float = float("nan")
    fingerprint: np.ndarray | None = None
    note: str = ""

    @property
    def settled(self) -> bool:
        """It left where it started, and then held one pose."""
        return bool(self.drifted > 2.0 and self.late_population >= 0.5)


@dataclass
class Modes:
    """Runs grouped into binding modes, the most populated first."""

    labels: np.ndarray
    distance: np.ndarray
    members: list = field(default_factory=list)
    medoids: list = field(default_factory=list)
    spread: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.members)


def modes(fingerprints, cutoff: float = MODE_CUTOFF) -> Modes:
    """Group poses into modes by average linkage on cosine distance.

    Cosine because a fingerprint is a direction in contact space: two poses
    that touch the same things with the same kinds of group point the same way,
    however deeply buried either happens to sit.  Each mode is reported by its
    medoid, so what comes back is a pose that was simulated, and by its widest
    internal pair -- a large one says the cutoff has merged two things.
    """
    v = np.asarray(fingerprints, float)
    if v.ndim != 2 or not len(v):
        raise ValueError("fingerprints must be (poses, width) and not empty")
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    unit = v / np.where(norm < 1e-9, 1.0, norm)
    d = np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    groups = [[i] for i in range(len(v))]
    while len(groups) > 1:
        best, where = np.inf, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                gap = float(d[np.ix_(groups[a], groups[b])].mean())
                if gap < best:
                    best, where = gap, (a, b)
        if best > cutoff:
            break
        a, b = where
        groups[a] += groups.pop(b)
    groups.sort(key=len, reverse=True)
    labels = np.zeros(len(v), int)
    out = Modes(labels, d)
    for n, members in enumerate(groups):
        for i in members:
            labels[i] = n
        sub = d[np.ix_(members, members)]
        out.members.append(sorted(members))
        out.medoids.append(int(members[int(np.argmin(sub.mean(1)))]))
        out.spread.append(float(sub.max()) if len(members) > 1 else 0.0)
    return out


def frame_system(system, positions, k: int, glue=None):
    """``system`` at frame ``k``: whole, centred, and in that frame's own box.

    A trajectory is wrapped, so a molecule sits in pieces across the cell and a
    pocket can be a box-length from its ligand.  Under NPT the box drifts from
    the system's own cell as well, and a minimum image taken in the wrong box
    puts atoms where they are not.
    """
    from .glue import Glue

    out = system.clone()
    frame = positions[k]
    xyz = getattr(frame, "positions", frame)
    box = getattr(frame, "box", None)
    if box is None or not np.any(box):
        box = system.cell
    if glue is None:
        glue = Glue(system, glue="not (water or ions)", center="protein")
    out.positions, out.cell = glue(xyz, box)
    return out


def triage(system, positions, *, ligand="not (polymer or water or ions) and noh",
           protein="protein and noh", pocket="protein and name CA",
           cutoff: float = 1.5, fingerprint: bool = True) -> Triage:  # fmt: skip
    """Measure one run the way a thousand of them want measuring.

    Every dRMSD reference is asked for on its own.  A ligand that has wandered
    off has no pocket within reach of the last frame and dRMSD against *that*
    frame cannot be defined -- but the rest of the run still can be, and
    dropping it over one reference loses exactly the runs worth looking at.
    """
    from .glue import Glue
    from .symmetry import drmsd

    out = Triage(frames=len(positions))
    n = out.frames
    if n < 2:
        out.note = "fewer than two frames"
        return out

    def against(reference):
        try:
            return drmsd(system, reference, ligand=ligand, protein=pocket,
                         positions=positions).drmsd  # fmt: skip
        except ValueError:  # no pocket left around that frame's ligand
            return None

    def group(which):
        from . import poses

        try:
            return poses(system, which, None, cutoff=cutoff, ligand=ligand, protein=pocket)
        except ValueError:
            return []

    half = n // 2
    # one spanning tree for the run, not one per frame asked for
    glue = Glue(system, glue="not (water or ions)", center="protein")
    d0 = against(None)
    if d0 is not None:
        out.from_start, out.drifted = float(d0.mean()), float(d0[half:].mean())
        out.from_start_max = float(d0.max())
    last = frame_system(system, positions, n - 1, glue)
    dl = against(last)
    if dl is not None:
        out.from_end = float(dl.mean())

    found = group(positions)
    out.n_poses = len(found)
    if len(found):
        out.pose_frame, out.population = int(found[0].center), float(found[0].population)
    elif dl is not None:  # no pose held two frames: it never sat still
        out.pose_frame = int(np.argmin(dl))
    dr = against(frame_system(system, positions, out.pose_frame, glue))
    if dr is not None:
        out.from_pose = float(dr.mean())

    if n - half >= 2:  # the second half alone: where a late settler shows itself
        tail = np.array([getattr(positions[k], "positions", positions[k])
                         for k in range(half, n)])  # fmt: skip
        late = group(tail)
        if len(late):
            out.late_population = float(late[0].population)
            dlate = against(frame_system(system, positions, half + int(late[0].center), glue))
            if dlate is not None:
                out.late_from_pose = float(dlate[half:].mean())

    if fingerprint:
        try:
            fp = pharmacophore_fingerprint(system, positions, ligand=ligand, protein=protein)
            out.fingerprint = fp.values[half:].mean(0)
            out.moved = cosine_distance(fp.values[:half].mean(0), fp.values[half:].mean(0))
        except Exception as e:  # RDKit cannot type every ligand, and beads not at all
            out.note = f"no fingerprint: {type(e).__name__}: {e}"[:90]
    return out
