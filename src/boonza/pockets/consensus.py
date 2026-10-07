"""Consensus pockets over a trajectory: fpocket on every frame, pockets merged by place.

fpocket runs on each frame with the model's preset, and every pocket it finds
gets a probability, ``p = 1 / (1 + exp(-score))`` (the refitted score is a
logistic model of whether a ligand sits there), and its burial
(:func:`boonza.sites.burial` averaged over its alpha-sphere centres).  Pockets
of all frames are then grouped greedily, best first: a pocket joins the
consensus pocket whose centroid lies within ``cutoff`` of its own centre, or
starts one.  Each frame counts once per consensus pocket, with its best member
(the pocket's representative in that frame).

Three rankings:

- persistence: the mean of p over all frames, 0 where the pocket is absent;
- quality: the ``QUALITY_PERCENTILE``th percentile of p over the frames it is
  open in, so a pocket that opens rarely but well ranks high; pockets open in
  fewer than ``MIN_OCCUPANCY`` of the frames rank after the others;
- quality x burial: quality times the mean burial over those frames, which
  favours enclosed pockets over grooves.

Each consensus pocket also gets its enclosed core on its best frame
(:func:`boonza.pockets.enclosed_core`), a compact, ligand-sized site; the
rankings do not use it.

Measured on 2 x 40 coarse-grained apo trajectories of the Schrodinger set, with
the holo ligand as the answer, a 6 A cutoff halves the splitting of one site
into several consensus pockets against 4 A and leaves the rest within noise;
7 to 10 A start merging neighbouring sites.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .run import Pocket, find_fpocket, run_fpocket

#: a frame's pocket joins a consensus pocket whose centroid is this close, A
CONSENSUS_CUTOFF = 6.0
#: cryptic: no pocket of the apo crystal structure within this of the consensus centre, A
CRYPTIC_CUTOFF = 4.0
#: quality is ranked first among pockets open in at least this share of frames
MIN_OCCUPANCY = 0.05
#: quality: this percentile of a pocket's p over the frames it is open in
QUALITY_PERCENTILE = 90

__all__ = [
    "CONSENSUS_CUTOFF",
    "ConsensusPocket",
    "FramePocket",
    "consensus_pockets",
    "frame_pockets",
]


@dataclass
class FramePocket:
    """One frame's fpocket pocket."""

    frame: int  #: frame index in the trajectory read
    rank: int  #: fpocket's rank in that frame
    pocket: Pocket
    burial: float

    @property
    def p(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self.pocket.score)))

    @property
    def centre(self) -> np.ndarray:
        return self.pocket.centre


@dataclass
class ConsensusPocket:
    """Pockets of different frames at one place."""

    members: list  #: every FramePocket that joined, best first
    frames: int  #: frames in the trajectory read
    open: list = field(default_factory=list)  #: the best member of each frame it is open in
    cryptic: bool | None = None
    core: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rank_quality: int = 0
    rank_persistence: int = 0
    rank_quality_burial: int = 0

    def __post_init__(self):
        per_frame: dict[int, FramePocket] = {}
        for m in self.members:
            per_frame.setdefault(m.frame, m)
        self.open = list(per_frame.values())

    @property
    def occupancy(self) -> float:
        """The share of frames it is open in."""
        return len(self.open) / self.frames

    @property
    def persistence(self) -> float:
        return float(sum(m.p for m in self.open) / self.frames)

    @property
    def quality(self) -> float:
        return float(np.percentile([m.p for m in self.open], QUALITY_PERCENTILE))

    @property
    def burial(self) -> float:
        return float(np.mean([m.burial for m in self.open]))

    @property
    def quality_burial(self) -> float:
        return self.quality * self.burial

    @property
    def centre(self) -> np.ndarray:
        """The mean of its representatives' centres."""
        return np.mean([m.centre for m in self.open], axis=0)

    @property
    def best(self) -> FramePocket:
        """Its representative with the highest p: the frame it is drawn and measured in."""
        return max(self.open, key=lambda m: (m.p, -m.frame))


def frame_pockets(pdb, frames, flags=(), fpocket=None):
    """fpocket on each of ``frames`` (positions of the atoms in ``pdb``, in order).

    Yields each frame's pockets as FramePockets, with their burial against that
    frame.  ``pdb`` is the structure fpocket reads (from
    :func:`boonza.pockets.write_fpocket_pdb`); only its coordinates change.
    """
    from ..sites import burial

    exe = find_fpocket(fpocket)
    lines = Path(pdb).read_text().splitlines()
    atom_lines = [k for k, line in enumerate(lines) if line.startswith("ATOM")]
    with tempfile.TemporaryDirectory() as tmp:
        frame_pdb = Path(tmp) / "frame.pdb"
        for f, xyz in enumerate(frames):
            xyz = np.asarray(xyz, float)
            out = list(lines)
            for k, (x, y, z) in zip(atom_lines, xyz, strict=True):
                out[k] = f"{lines[k][:30]}{x:8.3f}{y:8.3f}{z:8.3f}{lines[k][54:]}"
            frame_pdb.write_text("\n".join(out) + "\n")
            found = run_fpocket(frame_pdb, flags, exe, quiet=True)
            yield [FramePocket(f, r, p, float(burial(p.centres, xyz).mean()))
                   for r, p in enumerate(found, 1)]  # fmt: skip


def consensus_pockets(per_frame, cutoff: float = CONSENSUS_CUTOFF, crystal=None,
              ) -> list[ConsensusPocket]:  # fmt: skip
    """Consensus pockets from each frame's pockets (``per_frame``, a list of lists of
    FramePocket), ranked; returned in quality order.

    ``crystal``: the pocket centres of the apo crystal structure in the same frame,
    to flag each consensus pocket cryptic when none lies within ``CRYPTIC_CUTOFF``
    of it (left as None without them).
    """
    n = len(per_frame)
    ordered = sorted((m for frame in per_frame for m in frame), key=lambda m: -m.p)
    sums, members = [], []
    for m in ordered:
        best, dist = None, cutoff
        for g, s in enumerate(sums):
            d = float(np.linalg.norm(s / len(members[g]) - m.centre))
            if d < dist:
                best, dist = g, d
        if best is None:
            sums.append(np.zeros(3))
            members.append([])
            best = len(sums) - 1
        sums[best] = sums[best] + m.centre
        members[best].append(m)
    pockets = [ConsensusPocket(group, n) for group in members]
    for key in ("persistence", "quality", "quality_burial"):
        floor = key != "persistence"
        order = sorted(range(len(pockets)), key=lambda i: (
            floor and pockets[i].occupancy < MIN_OCCUPANCY, -getattr(pockets[i], key)))  # fmt: skip
        for r, i in enumerate(order, 1):
            setattr(pockets[i], f"rank_{key}", r)
    if crystal is not None:
        crystal = np.asarray(crystal, float).reshape(-1, 3)
        for q in pockets:
            q.cryptic = not len(crystal) or bool(
                np.linalg.norm(crystal - q.centre, axis=1).min() >= CRYPTIC_CUTOFF
            )
    return sorted(pockets, key=lambda q: q.rank_quality)


def write_frame_pockets(path, pockets: list[ConsensusPocket]) -> None:
    """Every frame's pocket with the consensus pocket it joined, so rankings and merges
    can be worked out again without running fpocket.

    One row per frame's pocket: ``consensus`` (its consensus pocket's quality
    rank), ``frame``, ``rank`` (fpocket's, in that frame), ``p``, ``score``,
    ``burial``, ``center`` (n, 3), ``block`` (fpocket's info text); its alpha
    spheres are ``sphere_centers``/``sphere_radii`` rows ``offsets[i]:offsets[i + 1]``.
    """
    rows = [(q.rank_quality, m) for q in pockets for m in q.members]
    spheres = [m.pocket.centres for _, m in rows]
    np.savez_compressed(
        path,
        consensus=np.array([c for c, _ in rows], int),
        frame=np.array([m.frame for _, m in rows], int),
        rank=np.array([m.rank for _, m in rows], int),
        p=np.array([m.p for _, m in rows]),
        score=np.array([m.pocket.score for _, m in rows]),
        burial=np.array([m.burial for _, m in rows]),
        center=np.array([m.centre for _, m in rows]).reshape(-1, 3),
        offsets=np.cumsum([0] + [len(c) for c in spheres]),
        sphere_centers=np.vstack(spheres) if spheres else np.zeros((0, 3)),
        sphere_radii=np.concatenate([m.pocket.radii for _, m in rows]) if rows else np.zeros(0),
        block=np.array([m.pocket.info for _, m in rows], dtype=str),
    )
