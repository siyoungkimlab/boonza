"""Scoring and ranking, as the CTMD authors' analysis scripts do it.

One replica's score is c(t) at the moment the ligand leaves, or at the end of the
run when it never does: the work the bias had to do to get it out.  A pose's score is
the lowest of its replicas', since the least work is the one that matters -- the
easiest way out is the one a ligand takes.  With more replicas than the three the
protocol asks for, the published analysis bootstraps: it draws three at a time,
takes their lowest, and averages those, which smooths the answer without pretending
to more data than there is.

Ligands are then ranked by that score, highest first: a higher score is a pose that
took more work to break, and a ligand that leaves on its own during the unbiased
filter scores 0 and so ranks last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Replica:
    """One biased trajectory: the coordinate, the work, and where it ended."""

    index: int
    time: np.ndarray  # ps
    drmsd: np.ndarray  # nm
    work: np.ndarray  # c(t), kJ/mol
    escaped_at: float | None  # ps, or None when the ligand stayed
    score: float  # c(t) where it ended

    @property
    def escaped(self) -> bool:
        return self.escaped_at is not None


@dataclass
class Pose:
    """What CTMD makes of one pose: its score, and the replicas behind it."""

    name: str
    score: float
    error: float = 0.0
    exit_time: float = 0.0  # ps, the secondary score: how soon it left
    replicas: list[Replica] = field(default_factory=list)
    filtered: bool = False  # it left during the unbiased run, so it scores 0

    def __str__(self) -> str:
        if self.filtered:
            return f"{self.name}: 0 (left the site unbiased)"
        return (f"{self.name}: c(t) {self.score:.1f} +/- {self.error:.1f} kJ/mol from "
                f"{len(self.replicas)} replica(s), out at {self.exit_time:g} ps")  # fmt: skip


def read_colvar(path):
    """``(time, drmsd, bias, work)`` from a PLUMED-style COLVAR file.

    The first and second columns are the time and the coordinate and the last is
    c(t), which is how the authors' files are laid out (``metad.rct`` last), so their
    data and ours read alike.  An empty file gives empty arrays: a replica that never
    started is not a replica that scored 0.
    """
    lines = [x for x in Path(path).read_text().splitlines()
             if x.strip() and not x.lstrip().startswith("#")]  # fmt: skip
    if not lines:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    rows = np.loadtxt(lines, usecols=(0, 1, -2, -1), ndmin=2)
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]


def escape_index(time, drmsd, cutoff: float, commit: float):
    """The first frame from which the ligand stayed past ``cutoff`` for ``commit`` ps.

    None when it never did.  The published code counts frames; this counts the time
    they span, so a file written at a different stride means the same thing.
    """
    out = np.asarray(drmsd, float) > cutoff
    if not out.any():
        return None
    time = np.asarray(time, float)
    start = None
    for k, gone in enumerate(out):
        if not gone:
            start = None
            continue
        if start is None:
            start = k
        if time[k] - time[start] >= commit:
            return start
    return None


def replica_score(time, drmsd, work, cutoff: float = 0.6, commit: float = 200.0):
    """``(score, escaped_at)``: c(t) where the ligand left, or at the end of the run."""
    if not len(time):
        return float("nan"), None
    where = escape_index(time, drmsd, cutoff, commit)
    if where is None:
        return float(work[-1]), None
    return float(work[where]), float(time[where])


def read_replicas(folder, cutoff: float = 0.6, commit: float = 200.0,
                  until: float | None = None) -> list[Replica]:  # fmt: skip
    """Every ``colvar_K.dat`` or ``COLVAR_K`` of ``folder``, scored.

    ``until``: read only this many ps of each, to ask what a shorter run would have
    said.
    """
    out = []
    folder = Path(folder)
    found = sorted(folder.glob("colvar_*.dat")) + sorted(folder.glob("COLVAR_*"))
    for path in found:
        time, drmsd, bias, work = read_colvar(path)
        if until is not None and len(time):
            keep = time <= until
            time, drmsd, bias, work = time[keep], drmsd[keep], bias[keep], work[keep]
        score, escaped = replica_score(time, drmsd, work, cutoff, commit)
        index = int("".join(c for c in path.stem.split("_")[-1] if c.isdigit()) or len(out) + 1)
        out.append(Replica(index, time, drmsd, work, escaped, score))
    return out


def score_pose(folder, name: str | None = None, *, replicas: int = 3, cutoff: float = 0.6,
               commit: float = 200.0, bootstrap: int = 250, until: float | None = None,
               seed: int = 0) -> Pose:  # fmt: skip
    """One pose's score, from the COLVAR files in ``folder``.

    With exactly ``replicas`` usable trajectories the score is their lowest.  With
    more, ``bootstrap`` draws of ``replicas`` are taken and their lowest scores
    averaged, as the published analysis does; the spread of those draws is the error.
    A folder with a ``filter.dat`` whose coordinate ends past ``cutoff`` is a ligand
    that left unbiased, and scores 0.
    """
    folder = Path(folder)
    label = name or folder.name
    runs = read_replicas(folder, cutoff, commit, until)
    usable = [r for r in runs if np.isfinite(r.score)]
    filtered = folder / "filter.dat"
    if filtered.exists() and not usable:
        time, drmsd, _, _ = read_colvar(filtered)
        if escape_index(time, drmsd, cutoff, commit) is not None:
            return Pose(label, 0.0, filtered=True)
    if not usable:
        raise ValueError(f"{folder}: no COLVAR file with any data in it")
    scores = np.array([r.score for r in usable], float)
    exits = np.array([r.escaped_at if r.escaped else r.time[-1] for r in usable], float)
    if len(usable) <= replicas:
        score, error = float(scores.min()), 0.0
        exit_time = float(exits.min())
    else:
        rng = np.random.default_rng(seed)
        picks = [rng.choice(len(usable), replicas, replace=False) for _ in range(bootstrap)]
        lows = np.array([scores[p].min() for p in picks])
        score, error = float(lows.mean()), float(lows.std())
        exit_time = float(np.mean([exits[p].min() for p in picks]))
    return Pose(label, score, error, exit_time, usable)


def score_run(folder, settings=None, *, log=print) -> Pose:
    """:func:`score_pose` with a run's own :class:`boonza.ctmd.Settings`."""
    cfg = settings
    kwargs = {}
    if cfg is not None:
        kwargs = {"replicas": cfg.replicas, "cutoff": cfg.cutoff, "commit": cfg.commit}
    pose = score_pose(folder, **kwargs)
    if log is not None:
        log(str(pose))
    return pose


def rank(poses) -> list[Pose]:
    """The poses, best first: by score, then by how long the ligand lasted."""
    return sorted(poses, key=lambda p: (-p.score, -p.exit_time, p.name))


def enrichment_factor(hits, scores, fraction: float = 0.3) -> float:
    """How many more hits the top ``fraction`` holds than chance would give.

    ``hits`` is 1 for a binder and 0 for a non-binder, ``scores`` is what ranks them
    (higher first).  1.0 is chance.
    """
    hits = np.asarray(hits, float)
    scores = np.asarray(scores, float)
    if hits.shape != scores.shape or not hits.size:
        raise ValueError("hits and scores must be the same non-empty shape")
    total, actives = len(hits), hits.sum()
    if not actives:
        raise ValueError("no hits among these ligands")
    top = max(1, int(round(fraction * total)))
    order = np.argsort(-scores, kind="stable")
    found = hits[order][:top].sum()
    return float((found / top) / (actives / total))


def bedroc(hits, scores, alpha: float = 20.0) -> float:
    """BEDROC (Truchon and Bayly 2007): the ROC area with the top weighted.

    ``alpha`` says how steeply: 20, the usual choice, puts about 80% of the weight in
    the first 8%.  1.0 is a perfect ranking, 0.0 the reverse of one.
    """
    hits = np.asarray(hits, float)
    scores = np.asarray(scores, float)
    if hits.shape != scores.shape or not hits.size:
        raise ValueError("hits and scores must be the same non-empty shape")
    n, total = hits.sum(), len(hits)
    if not n or n == total:
        raise ValueError("BEDROC needs both hits and non-hits")
    ratio = n / total
    order = np.argsort(-scores, kind="stable")
    ranks = np.flatnonzero(hits[order]) + 1  # 1-based, as the formula has them
    rie_sum = np.exp(-alpha * ranks / total).sum() / n
    rie = rie_sum / ((1 - np.exp(-alpha)) / (total * (np.exp(alpha / total) - 1)))
    return float(rie * ratio * np.sinh(alpha / 2) / (np.cosh(alpha / 2)
                 - np.cosh(alpha / 2 - alpha * ratio))
                 + 1 / (1 - np.exp(alpha * (1 - ratio))))  # fmt: skip
