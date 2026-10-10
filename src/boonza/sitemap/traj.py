"""Sites through a coarse-grained trajectory, and the pockets they make up.

Per frame: the protein's beads (never the probe chain LIG, which
boonza.pockets.prepare.protein_ids leaves out), made whole and fitted on the
backbone (boonza's Glue; the first frame is the reference), and their sites
under the model's preset (presets.find), each with the residues lining it.

Pockets move and change shape through a run, so a pocket is not a place: it is
named by the residues that line it.  A site's lining residues are those with a
bead within that bead's radius + LINING of a site point.  The sites of every
frame, best SiteScore first, are grouped (consensus):

  1. each site joins the pocket whose core -- the residues lining at least half
     of that pocket's sites so far -- it is most similar to, by Jaccard
     similarity of residue sets, if that is at least ``similarity``; otherwise
     it starts a pocket
  2. reassign: each site moves to the pocket whose core it is now most similar
     to (if at least ``similarity``), the cores are recomputed, until no site
     moves -- so a site's pocket does not depend on the order of step 1
  3. merge: two pockets whose cores are at least ``similarity`` alike and that
     are present together in at most ``cooccur`` of the smaller one's frames
     are one pocket seen in different states, and become one; then step 2 again
  4. absorb: a fragment -- open in under FRAGMENT_OCC of frames, or of median
     volume under FRAGMENT_VOL -- joins the nearest other pocket whose sites
     come within FRAGMENT_GAP of its own (SiteMap's rule that small pieces may
     merge, across frames).  On the Schrodinger apo trajectories this cut the
     pockets per run by ~40% (Martini 2 26.5 -> 16.6, Martini 3 30.9 -> 18.1)
     with Top-k, MOc and fill unchanged within a case

Comparing with a pocket's core rather than with any one member keeps a chain of
slightly overlapping sites from growing into one pocket, and two pockets that
are open at the same time are never merged.  A pocket may hold two sites of one
frame (a pocket split for a moment); its occupancy counts frames.
"""

from __future__ import annotations

import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

LINING = 3.0  # A beyond a bead's radius
#: ways to rank pockets, each a key of Pocket.summary (higher first):
#: p_max the best frame, p_mean_open the mean while open, p_mean_all the mean over every
#: frame (0 where absent: occupancy x p_mean_open), occ_x_pmax occupancy x the best
#: frame, p_q90 the 90th percentile over every frame, occupancy how often it is open
RANKINGS = ("p_max", "p_mean_open", "p_mean_all", "occ_x_pmax", "p_q90", "occupancy")
SIMILARITY = 0.3  # Jaccard of lining residues to join a pocket
CORE = 0.5  # share of a pocket's sites a residue lines to be in its core
COOCCUR = 0.1  # pockets present together in at most this share of frames may merge
ROUNDS = 20  # reassignment rounds at most
#: a fragment -- open in under FRAGMENT_OCC of frames or of median volume under FRAGMENT_VOL
#: A^3 -- may join the nearest pocket whose sites come within FRAGMENT_GAP A of its own
FRAGMENT_OCC, FRAGMENT_VOL, FRAGMENT_GAP = 0.1, 40.0, 8.0


@dataclass
class FrameSite:
    frame: int
    rank: int  #: in its frame, by score
    score: float
    props: dict
    xyz: np.ndarray  #: site points, in the fitted frame
    residues: frozenset  #: residue indices of the system


@dataclass
class Pocket:
    sites: list = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    @property
    def core(self) -> frozenset:
        need = CORE * len(self.sites)
        return frozenset(r for r, c in self.counts.items() if c >= need)

    def add(self, s: FrameSite) -> None:
        self.sites.append(s)
        self.counts.update(s.residues)

    @property
    def frames(self) -> np.ndarray:
        return np.unique([s.frame for s in self.sites])

    @property
    def best(self) -> FrameSite:
        return max(self.sites, key=lambda s: s.score)

    def per_frame_best(self) -> dict:
        out = {}
        for s in self.sites:
            if s.frame not in out or s.score > out[s.frame].score:
                out[s.frame] = s
        return out

    def summary(self, n_frames: int) -> dict:
        """Occupancy, scores, volumes, and the RANKINGS values (p: the SiteScore as the
        probability it is a ligand site, 1 / (1 + exp(-score)), as its logistic fit means)."""
        best = self.per_frame_best()
        scores = np.array([s.score for s in best.values()])
        vols = np.array([s.props["n"] for s in best.values()])
        p = 1 / (1 + np.exp(-scores))
        every = np.zeros(n_frames)  # p per frame, 0 where the pocket is absent
        every[: len(p)] = np.sort(p)[::-1]
        occ = len(best) / n_frames
        return {"occupancy": occ, "frames": len(best),
                "best_score": float(scores.max()), "mean_score": float(scores.mean()),
                "p_max": float(p.max()), "p_mean_open": float(p.mean()),
                "p_mean_all": float(every.mean()), "occ_x_pmax": float(occ * p.max()),
                "p_q90": float(np.quantile(every, 0.9)),
                "median_volume": float(np.median(vols)), "max_volume": float(vols.max()),
                "best_frame": int(self.best.frame), "core_residues": len(self.core)}  # fmt: skip


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _pockets(groups) -> list[Pocket]:
    out = []
    for g in groups:
        if g:
            p = Pocket()
            for s in g:
                p.add(s)
            out.append(p)
    return out


def _reassign(pockets: list[Pocket], similarity: float) -> list[Pocket]:
    for _ in range(ROUNDS):
        cores = [p.core for p in pockets]
        groups, moved = [[] for _ in pockets], 0
        for k, p in enumerate(pockets):
            for s in p.sites:
                sims = [_jaccard(s.residues, c) for c in cores]
                j = int(np.argmax(sims))
                to = j if sims[j] >= similarity and sims[j] > sims[k] else k
                moved += to != k
                groups[to].append(s)
        pockets = _pockets(groups)
        if not moved:
            break
    return pockets


def _cooccur(a: Pocket, b: Pocket) -> float:
    fa, fb = set(a.frames.tolist()), set(b.frames.tolist())
    return len(fa & fb) / min(len(fa), len(fb))


def _gap(a: Pocket, b: Pocket) -> float:
    """The closest approach of two pockets' sites: within a frame both are open in if
    there is one, else between their sites of any frames."""
    from ..spatial import min_dist2

    fa, fb = a.per_frame_best(), b.per_frame_best()
    both = set(fa) & set(fb)
    pairs = [(fa[f].xyz, fb[f].xyz) for f in both] or [
        (np.vstack([s.xyz for s in a.sites]), np.vstack([s.xyz for s in b.sites]))
    ]
    return float(np.sqrt(min(min_dist2(x, y, FRAGMENT_GAP).min() for x, y in pairs)))


def _absorb(pockets: list[Pocket], n_frames: int) -> list[Pocket]:
    """Fragments join their nearest non-fragment pocket within FRAGMENT_GAP, smallest first."""

    def small(p):
        s = p.summary(n_frames)
        return s["occupancy"] < FRAGMENT_OCC or s["median_volume"] < FRAGMENT_VOL

    keep = [p for p in pockets if not small(p)]
    frags = sorted((p for p in pockets if small(p)), key=lambda p: len(p.sites))
    for f in frags:
        gaps = [_gap(f, k) for k in keep]
        j = int(np.argmin(gaps)) if gaps else -1
        if j >= 0 and gaps[j] <= FRAGMENT_GAP:
            for s in f.sites:
                keep[j].add(s)
        else:
            keep.append(f)
    return keep


def consensus(sites: list[FrameSite], similarity: float = SIMILARITY,
              cooccur: float = COOCCUR, absorb: bool = True) -> list[Pocket]:  # fmt: skip
    """Pockets of the frame sites (module docstring); ``absorb``: then let fragments join
    a neighbouring pocket (_absorb)."""
    if not sites:  # no frame has a site: no pockets
        return []
    pockets: list[Pocket] = []
    for s in sorted(sites, key=lambda s: -s.score):
        sims = [_jaccard(s.residues, p.core) for p in pockets]
        k = int(np.argmax(sims)) if sims else -1
        if k >= 0 and sims[k] >= similarity:
            pockets[k].add(s)
        else:
            pockets.append(Pocket())
            pockets[-1].add(s)
    pockets = _reassign(pockets, similarity)
    while True:
        cores = [p.core for p in pockets]
        best = None
        for a in range(len(pockets)):
            for b in range(a + 1, len(pockets)):
                sim = _jaccard(cores[a], cores[b])
                if (sim >= similarity and (best is None or sim > best[0])
                        and _cooccur(pockets[a], pockets[b]) <= cooccur):  # fmt: skip
                    best = (sim, a, b)
        if best is None:
            if absorb:
                pockets = _absorb(pockets, max(s.frame for s in sites) + 1)
            return pockets
        _, a, b = best
        merged = pockets[a].sites + pockets[b].sites
        pockets = _reassign(_pockets([merged] + [p.sites for k, p in enumerate(pockets)
                                                 if k not in (a, b)]), similarity)  # fmt: skip


def load_run(workdir=None, system=None, trajectory=None, selection: str | None = None,
             frames: slice = slice(None), probes: bool = False):  # fmt: skip
    """``(system, ids, coords, fit_rmsd)``: the protein beads ``ids`` of a run, every frame
    made whole and fitted on the backbone beads (BB, or SIRAH's GC) onto the first, and
    each frame's backbone RMSD from the first after the fit (A).

    ``probes``: also ``(probe_ids, probe_coords)`` at the end -- the probe chain's beads,
    each probe made whole, in the same fitted frames.  Only for showing them: site
    finding never uses probes."""
    from pathlib import Path

    from ..glue import Glue
    from ..io import load
    from ..pockets.prepare import probe_ids, protein_ids
    from ..trajectory import open_trajectory

    if workdir is not None:
        system = load(str(Path(workdir) / "solvated.dms"))
        trajectory = Path(workdir) / "trajectory.dcd"
    traj = open_trajectory(str(trajectory), system)[frames]
    ids = protein_ids(system, selection)
    names = np.asarray(system.atoms["name"])
    fit = ids[np.isin(names[ids], ["BB", "GC"])]
    pids = probe_ids(system) if probes else np.zeros(0, int)
    whole = np.r_[ids, pids] if system.nbonds else None
    glue = Glue(system, glue=[ids], center=ids, fit=fit, whole=whole)
    coords, pcoords = [], []
    for block in glue.frames(traj):
        for k in range(len(block)):
            coords.append(np.asarray(block.positions[k][ids], np.float32))
            pcoords.append(np.asarray(block.positions[k][pids], np.float32))
    coords = np.stack(coords)
    where = np.searchsorted(ids, fit)
    rmsd = np.sqrt(((coords[:, where] - coords[0, where]) ** 2).sum(-1).mean(1))
    if probes:
        return system, ids, coords, rmsd, pids, np.stack(pcoords)
    return system, ids, coords, rmsd


_W: dict = {}


def _init(beads, residue):
    _W["beads"], _W["residue"] = beads, residue


def _frame(job):
    from dataclasses import replace

    from ..spatial import min_dist2
    from .presets import find

    k, xyz = job
    beads = replace(_W["beads"], xyz=np.asarray(xyz, float))
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found, _ = find(beads)
    for rank, (score, s) in enumerate(found, 1):
        reach = beads.radius + LINING
        d2 = min_dist2(beads.xyz, s.xyz, float(reach.max())).astype(float)
        lining = frozenset(np.unique(_W["residue"][d2 <= reach**2]).tolist())
        out.append(FrameSite(k, rank, score, s.props, s.xyz.astype(np.float32), lining))
    return out


def frame_sites(system, ids, coords, model: str, jobs: int = 4,
                progress: bool = True) -> list[FrameSite]:  # fmt: skip
    """The sites of every frame under ``model``'s preset, with their lining residues;
    ``progress`` reports frames done, the rate and the time left about every 10%."""
    import sys
    import time

    from .beads import from_system

    beads = from_system(system, model, ids)
    residue = np.asarray(system.atoms["residue"])[ids]
    out, t0, step = [], time.time(), max(1, len(coords) // 10)
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(beads, residue)) as pool:
        for k, found in enumerate(pool.map(_frame, enumerate(coords), chunksize=4), 1):
            out.extend(found)
            if progress and (k % step == 0 or k == len(coords)):
                rate = k / (time.time() - t0)
                print(f"  {k}/{len(coords)} frames, {rate:.1f} frames/s, "
                      f"~{(len(coords) - k) / rate / 60:.1f} min left", file=sys.stderr,
                      flush=True)  # fmt: skip
    return out
