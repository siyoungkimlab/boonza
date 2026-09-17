"""Representative poses: the frame a trajectory stands for, and how much of it it does.

    poses = boonza.poses(system, traj, ligand="chain L")
    poses[0].center, poses[0].population, poses[0].spread

Frames are compared by the distances between pocket atoms and ligand atoms
(:func:`boonza.drmsd`'s measure), so no superposition is needed and a protein
that breathes or tumbles does not look like a ligand that moved.  Ligand
symmetry is taken out once per frame, against the reference, so a flipped ring
is not a second pose.

The frames are grouped by average linkage, which builds the whole merge tree
in one pass: ``cutoff`` only decides where to cut it, and
:meth:`PoseSet.sweep` reads every other cutoff off the same tree, so the
choice can be shown rather than trusted.  A pose is reported by its medoid --
a real frame, never an average -- with the share of frames it holds and how
tightly they sit around it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit

from .pbc import distances
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids, _prepare

MAX_FRAMES = 20000


@dataclass
class Pose:
    """One group of frames: its medoid, its share of the frames, its spread."""

    center: int  # the medoid, as an index into the frames analysed
    frames: np.ndarray  # the members, in time order
    population: float  # members / frames analysed
    spread: float  # mean dRMSD of the members to the centre (A)

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class PoseSet:
    """The poses of a trajectory, most populated first."""

    poses: list[Pose]
    labels: np.ndarray  # pose of every frame, -1 for frames of too small a group
    distances: np.ndarray  # (nframes, nframes) dRMSD in A
    merges: np.ndarray  # (nframes - 1, 3): the two frames joined, and at what dRMSD
    cutoff: float
    min_population: float

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, k) -> Pose:
        return self.poses[k]

    def __iter__(self):
        return iter(self.poses)

    def sweep(self, cutoffs=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(cutoffs, population of the largest pose, number of poses)``.

        The tree is already built, so this costs nothing but the cutting: a
        pose that holds its share while the cutoff doubles is a real one.

        The share only grows with the cutoff, but the count need not fall:
        it counts poses that clear ``min_population``, and a wider cutoff can
        gather scattered frames into a group that then clears it.
        """
        if cutoffs is None:
            top = float(self.merges[:, 2].max())
            cutoffs = np.linspace(top / 20.0, top / 2.0, 12)
        cutoffs = np.asarray(cutoffs, float)
        n = len(self.distances)
        share, count = [], []
        for c in cutoffs:
            labels = _cut(self.merges, n, float(c))
            sizes = np.bincount(labels)
            share.append(sizes.max() / n)
            count.append(int((sizes >= max(1, round(self.min_population * n))).sum()))
        return cutoffs, np.array(share), np.array(count, int)

    def summary(self) -> str:
        rows = [f"{len(self.poses)} poses of {len(self.distances)} frames "
                f"(cut at {self.cutoff:g} A)"]  # fmt: skip
        for k, p in enumerate(self.poses):
            rows.append(f"  pose {k}: frame {p.center}, {100 * p.population:.1f}% of frames, "
                        f"spread {p.spread:.2f} A")  # fmt: skip
        return "\n".join(rows)


def pocket_contacts(system, positions=None, ligand: str = DEFAULT_LIGAND,
                    protein: str = "protein and name CA", cutoff: float = 5.0,
                    periodic: bool = True) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``(protein atoms in contact per frame, share of frames each is in contact)``.

    One chunked pass, reading only the ligand and the candidate protein
    atoms.  The per-frame count says whether the ligand is bound at all,
    without caring which pose it is in; the per-atom share says which atoms a
    pocket is really made of, over the whole run rather than in one frame.
    """
    lig = _ids(system, ligand)
    lig = lig[system.atoms["anum"][lig] > 1]
    if not len(lig):
        raise ValueError(f"ligand {ligand!r} selects no heavy atoms")
    prot = _ids(system, protein)
    if not len(prot):
        raise ValueError(f"protein {protein!r} selects no atoms")
    need = np.union1d(prot, lig)
    pl, ll = np.searchsorted(need, prot), np.searchsorted(need, lig)
    own = np.isin(prot, lig)
    blocks, _ = _boxed_blocks(system, positions, need)
    counts, hits = [], np.zeros(len(prot))
    for xyz, boxes in blocks:
        for X, box in zip(xyz, boxes, strict=True):
            near = distances(X[pl], X[ll], box if periodic else None).min(1) <= cutoff
            near &= ~own
            counts.append(int(near.sum()))
            hits += near
    if not counts:
        raise ValueError("no frames")
    return np.array(counts), hits / len(counts)


def bound_frame(counts) -> int:
    """A typical bound frame: the median of the frames that touch the protein.

    The *most* contacting frame is by construction an outlier -- the one where
    a loop happened to close in -- and letting it decide the pocket lets one
    frame speak for the run.
    """
    counts = np.asarray(counts)
    bound = np.flatnonzero(counts > 0)
    if not len(bound):
        raise ValueError("the ligand never touches the protein: is it bound at all?")
    return int(bound[np.abs(counts[bound] - np.median(counts[bound])).argmin()])


def _check_pocket(pocket, positions) -> None:
    """A pocket of three atoms cannot tell a pose from its mirror image."""
    if len(pocket) < 4:
        warnings.warn(f"the pocket has {len(pocket)} atoms: fewer than four cannot fix a "
                      "position in space, so a pose and its mirror image measure the same. "
                      "Widen pocket_cutoff or the protein selection.", stacklevel=3)  # fmt: skip
        return
    spread = np.linalg.svd(positions - positions.mean(0), compute_uv=False)
    if spread[-1] < 0.5:
        warnings.warn(f"the pocket is nearly flat (thickness {spread[-1]:.2f} A): poses on "
                      "either side of it measure almost the same. Widen pocket_cutoff or "
                      "the protein selection.", stacklevel=3)  # fmt: skip


def pose_distances(system, positions=None, reference=None, ligand: str = DEFAULT_LIGAND,
                   protein: str = "protein and name CA", pocket_cutoff: float = 5.0,
                   pocket=None, symmetry: bool = True, heavy_only: bool = True,
                   bond_orders: bool = False, periodic: bool = True) -> np.ndarray:  # fmt: skip
    """The (nframes, nframes) dRMSD matrix of a trajectory, in A.

    Every frame becomes the matrix of distances between the pocket atoms --
    the ``protein`` atoms within ``pocket_cutoff`` A of the ligand in the
    reference -- and the ligand atoms, and two frames are compared by the
    RMS difference of those distances.

    ``reference``: a System (a crystal structure, or a frame where the ligand
    is bound), or None for the first frame.  It decides which atoms the
    pocket is made of, so a run that starts with the ligand elsewhere -- out
    in bulk, or not yet settled -- wants one rather than its own first frame.
    :func:`pocket_contacts` and :func:`bound_frame` find a fair one.

    ``pocket``: the pocket atoms themselves, as a selection or atom indices,
    when you would rather say than have it worked out -- from the atoms a
    site contacts over many runs, say.  ``protein``, ``pocket_cutoff`` and
    the reference then do not enter into which atoms are used.

    With ``symmetry`` each frame's ligand
    atoms are first matched to the first frame's, so equivalent atoms do not
    count as motion.  That mapping is chosen once per frame rather than once
    per pair, which is what keeps this quadratic in frames but linear in
    symmetry searches; when two frames would rather be compared through
    different mappings the distance between them is an upper bound.
    """
    ref_sys = system if reference is None else reference
    matcher, mids, rids, _ = _prepare(system, ref_sys, ligand, None, heavy_only, bond_orders)
    if pocket is not None:  # given outright: protein and pocket_cutoff decide nothing
        prot = _ids(system, pocket) if isinstance(pocket, str) else np.asarray(pocket, np.int64)
        prot = np.setdiff1d(prot, mids)
        if not len(prot):
            raise ValueError("pocket selects no atoms outside the ligand")
        rprot = prot
    else:
        prot, rprot = _ids(system, protein), _ids(ref_sys, protein)
        if len(prot) != len(rprot):
            raise ValueError(f"{protein!r} selects {len(prot)} atoms here and {len(rprot)} in "
                             "the reference; they are paired in order")  # fmt: skip
    need = np.union1d(prot, mids)
    blocks, _ = _boxed_blocks(system, positions, need)
    pl, ll = np.searchsorted(need, prot), np.searchsorted(need, mids)

    rows, keep, dref = [], None, None
    if pocket is not None:
        keep = pl
        if reference is not None:
            rbox = np.asarray(reference.cell, np.float64) if periodic else None
            dref = distances(reference.positions[prot], reference.positions[rids], rbox)
            _check_pocket(prot, reference.positions[prot])
    elif reference is not None:
        rpos = reference.positions
        rbox = np.asarray(reference.cell, np.float64) if periodic else None
        near = distances(rpos[rprot], rpos[rids], rbox).min(1) <= pocket_cutoff
        near &= ~np.isin(rprot, rids)
        if not near.any():
            raise ValueError(f"no {protein!r} atoms within {pocket_cutoff} A of the "
                             "reference ligand")  # fmt: skip
        keep, dref = pl[near], distances(rpos[rprot[near]], rpos[rids], rbox)
        _check_pocket(keep, rpos[rprot[near]])
    for xyz, boxes in blocks:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic else None
            if keep is None:
                near = distances(X[pl], X[ll], box).min(1) <= pocket_cutoff
                near &= ~np.isin(prot, mids)
                if not near.any():
                    raise ValueError(f"no {protein!r} atoms within {pocket_cutoff} A "
                                     "of the ligand in the first frame")  # fmt: skip
                keep = pl[near]
            d = distances(X[keep], X[ll], box)
            if dref is None:
                _check_pocket(keep, X[keep])
                dref = d
            elif symmetry:
                # cost[r, m]: the deviation if this frame's atom m plays reference atom r
                _, mapping = matcher.best(((dref.T[:, None, :] - d.T[None, :, :]) ** 2).sum(-1))
                d = d[:, mapping]
            rows.append(d.ravel())
            if len(rows) > MAX_FRAMES:
                raise ValueError(f"more than {MAX_FRAMES} frames: take every nth frame instead "
                                 "(the distance matrix is quadratic in frames)")  # fmt: skip
    if len(rows) < 2:
        raise ValueError("a trajectory of at least two frames is needed")
    a = np.array(rows)
    sq = np.einsum("ij,ij->i", a, a)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (a @ a.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2 / a.shape[1])


@njit(cache=True)
def _nn_chain(d):
    """Average-linkage merges by Mullner's nearest-neighbour chain.

    Returns (n - 1, 3): the two rows joined, and the distance between them.
    Row indices are original frames, so union-find over them cuts the tree.
    """
    n = d.shape[0]
    D = d.copy()
    size = np.ones(n, np.int64)
    alive = np.ones(n, np.bool_)
    chain = np.empty(n + 1, np.int64)
    out = np.empty((n - 1, 3), np.float64)
    length, done = 0, 0
    while done < n - 1:
        if length == 0:
            for i in range(n):
                if alive[i]:
                    chain[0] = i
                    length = 1
                    break
        a = chain[length - 1]
        b, best = -1, np.inf
        for x in range(n):
            if alive[x] and x != a and D[a, x] < best:
                best, b = D[a, x], x
        if length > 1 and b == chain[length - 2]:
            length -= 2  # a and b are each other's nearest: they merge
            if size[b] > size[a] or (size[b] == size[a] and b < a):
                a, b = b, a
            out[done, 0], out[done, 1], out[done, 2] = a, b, best
            done += 1
            na, nb = size[a], size[b]
            for x in range(n):
                if alive[x] and x != a and x != b:
                    v = (na * D[a, x] + nb * D[b, x]) / (na + nb)
                    D[a, x] = v
                    D[x, a] = v
            size[a] = na + nb
            alive[b] = False
        else:
            chain[length] = b
            length += 1
    return out


def _cut(merges, n: int, height: float) -> np.ndarray:
    """Flat groups from the merge tree: every join closer than ``height``, applied."""
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(merges[:, 2], kind="stable")
    for k in order:
        a, b, h = merges[k]
        if h > height:
            break
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[rb] = ra
    labels = np.array([root(i) for i in range(n)])
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(-1)


def poses(system, positions=None, reference=None, cutoff: float = 1.5,
          min_population: float = 0.02,
          ligand: str = DEFAULT_LIGAND, protein: str = "protein and name CA",
          pocket_cutoff: float = 5.0, pocket=None, symmetry: bool = True,
          heavy_only: bool = True, bond_orders: bool = False,
          periodic: bool = True) -> PoseSet:  # fmt: skip
    """The poses a trajectory holds, most populated first.

    Frames within ``cutoff`` A dRMSD of one another (average linkage) are one
    pose; a pose holding less than ``min_population`` of the frames, or fewer
    than two of them, is left out of the list and its frames are labelled -1.
    Each pose is reported by its medoid -- the member with the smallest mean
    distance to the others -- so what comes back is always a frame that was
    simulated.
    """
    d = pose_distances(system, positions, reference, ligand=ligand, protein=protein,
                       pocket_cutoff=pocket_cutoff, pocket=pocket, symmetry=symmetry,
                       heavy_only=heavy_only,
                       bond_orders=bond_orders, periodic=periodic)  # fmt: skip
    merges = _nn_chain(d)
    groups = _cut(merges, len(d), float(cutoff))
    smallest = max(2, round(min_population * len(d)))  # one frame is not a state
    found = []
    for g in range(groups.max() + 1):
        members = np.flatnonzero(groups == g)
        if len(members) < smallest:
            continue
        within = d[np.ix_(members, members)]
        centre = int(members[within.sum(1).argmin()])
        found.append(Pose(center=centre, frames=members, population=len(members) / len(d),
                          spread=float(d[centre, members].mean())))  # fmt: skip
    found.sort(key=lambda p: (-len(p.frames), p.center))
    labels = np.full(len(d), -1)
    for k, p in enumerate(found):
        labels[p.frames] = k
    return PoseSet(poses=found, labels=labels, distances=d, merges=merges, cutoff=float(cutoff),
                   min_population=float(min_population))  # fmt: skip
