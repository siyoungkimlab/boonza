"""Symmetry-corrected RMSD, for ligand poses and other molecules with equivalent atoms.

    r = boonza.symmetry_rmsd(pose, crystal, "resname LIG")      # in place, no fitting
    r.rmsd, r.plain_rmsd, r.mapping
    r = boonza.symmetry_rmsd(conf_a, conf_b, superpose=True)     # compare conformers
    rmsd_per_frame = boonza.symmetry_rmsd(system, reference, "resname LIG", positions=frames).rmsd

    r = boonza.ligand_rmsd(docked, crystal)   # superpose the proteins, then the ligand RMSD
    d = boonza.drmsd(system, positions=traj)  # pocket-ligand distance RMSD, no fitting

Equivalent atoms (the two oxygens of a carboxylate, the methyls of a
t-butyl group, a flipped phenyl ring) make a plain atom-by-atom RMSD depend
on atom naming.  The symmetry-corrected RMSD is the smallest RMSD over every
mapping of reference atoms onto mobile atoms that preserves elements and
bonds (a graph isomorphism), as spyrmsd and RDKit's CalcRMS compute it.  The
mobile and reference atoms may be listed in different orders.

The best mapping is found by an exact branch-and-bound search: mappings are
extended atom by atom along the bond graph, trying close atoms first, and a
partial mapping is abandoned as soon as it cannot beat the best RMSD found.
This stays fast even for very symmetric molecules, where enumerating every
isomorphism (thousands for a few t-butyl groups) would not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .align import kabsch

DEFAULT_LIGAND = "not (polymer or water or ions) and noh"
DEFAULT_FIT = "protein and name CA and not resname NMA NME ACE"


@dataclass
class SymmetryRMSD:
    """Outcome of ``symmetry_rmsd``."""

    rmsd: float | np.ndarray  # symmetry-corrected (one value per frame with ``positions``)
    plain_rmsd: float | np.ndarray | None  # atoms paired in listed order; None if elements differ
    mapping: np.ndarray  # mobile atom matched to each reference atom (best mapping, first frame)
    mobile_atoms: np.ndarray
    reference_atoms: np.ndarray
    isomorphisms: int | None = None  # mappings enumerated (``superpose=True``)
    truncated: bool = False  # enumeration stopped at ``max_isomorphisms``


@dataclass
class LigandRMSD:
    """Outcome of ``ligand_rmsd``: the protein fit and the ligand RMSD in that frame."""

    rmsd: float | np.ndarray  # symmetry-corrected ligand RMSD (per frame with ``positions``)
    plain_rmsd: float | np.ndarray | None
    fit_rmsd: float | np.ndarray  # RMSD of the fitted protein atoms
    rotation: np.ndarray
    translation: np.ndarray
    mapping: np.ndarray
    mobile_ligand: np.ndarray
    reference_ligand: np.ndarray


@dataclass
class DRMSD:
    """Outcome of ``drmsd``: pocket-ligand distance RMSD (per frame with ``positions``)."""

    drmsd: float | np.ndarray  # symmetry-corrected (best ligand atom mapping in each frame)
    plain_drmsd: float | np.ndarray | None  # ligand atoms paired in order; None if elements differ
    pocket: np.ndarray  # pocket atoms in the system
    reference_pocket: np.ndarray  # the same atoms in the reference
    mapping: np.ndarray  # mobile ligand atom matched to each reference ligand atom (first frame)
    mobile_ligand: np.ndarray
    reference_ligand: np.ndarray
    reference_distances: np.ndarray  # (npocket, nligand), Å


def _ids(system, sel) -> np.ndarray:
    if sel is None:
        return np.arange(system.natoms)
    if isinstance(sel, str):
        return system.select(sel).ids
    return np.asarray(system._ids("atom", sel), dtype=np.int64)


class _Graph:
    """Element-labelled bond graph of a set of atoms."""

    def __init__(self, system, ids, bond_orders):
        self.ids = ids
        n = len(ids)
        local = np.full(system.natoms, -1, np.int64)
        local[ids] = np.arange(n)
        bi, bj = local[system.bonds["i"]], local[system.bonds["j"]]
        keep = (bi >= 0) & (bj >= 0)
        self.adj = [set() for _ in range(n)]
        self.order = {}
        orders = system.bonds["order"][keep].tolist() if bond_orders else None
        for k, (a, b) in enumerate(zip(bi[keep].tolist(), bj[keep].tolist(), strict=True)):
            self.adj[a].add(b)
            self.adj[b].add(a)
            if orders is not None:
                self.order[(a, b)] = self.order[(b, a)] = orders[k]
        self.anum = system.atoms["anum"][ids].astype(np.int64)
        self.nedges = int(keep.sum())


def _joint_colors(g1: _Graph, g2: _Graph) -> tuple[np.ndarray, np.ndarray]:
    """Isomorphism-invariant atom colors, comparable between the two graphs."""
    labels = [[(int(z), len(a)) for z, a in zip(g.anum, g.adj, strict=True)] for g in (g1, g2)]
    table: dict = {}
    colors = [np.array([table.setdefault(x, len(table)) for x in lab]) for lab in labels]
    for _ in range(len(g1.anum)):
        table = {}
        new = []
        for g, c in zip((g1, g2), colors, strict=True):
            keys = [(int(c[k]), tuple(sorted(int(c[x]) for x in g.adj[k]))) for k in range(len(c))]
            new.append(np.array([table.setdefault(key, len(table)) for key in keys]))
        if all(len(set(n)) == len(set(o)) for n, o in zip(new, colors, strict=True)):
            colors = new
            break
        colors = new
    return colors[0], colors[1]


class _Matcher:
    """Graph isomorphisms reference -> mobile, best-first with pruning."""

    def __init__(self, ref: _Graph, mob: _Graph, bond_orders: bool):
        n = len(ref.anum)
        if n != len(mob.anum) or ref.nedges != mob.nedges or \
                sorted(ref.anum.tolist()) != sorted(mob.anum.tolist()):  # fmt: skip
            raise ValueError(
                f"the two atom sets are not the same molecule: {n} vs {len(mob.anum)} atoms, "
                f"{ref.nedges} vs {mob.nedges} bonds (check the selections, and that both "
                "include or both exclude hydrogens)"
            )
        self.ref, self.mob, self.bond_orders = ref, mob, bond_orders
        cr, cm = _joint_colors(ref, mob)
        self.cand = [np.flatnonzero(cm == cr[r]) for r in range(n)]
        if any(len(c) == 0 for c in self.cand) or \
                sorted(cr.tolist()) != sorted(cm.tolist()):  # fmt: skip
            raise ValueError("the two atom sets are not the same molecule (bond graphs differ)")
        # visit atoms along bonds, starting where the choice is smallest
        order, seen = [], np.zeros(n, bool)
        while len(order) < n:
            rest = np.flatnonzero(~seen)
            start = int(rest[np.argmin([len(self.cand[r]) for r in rest])])
            seen[start] = True
            queue = [start]
            while queue:
                r = queue.pop(0)
                order.append(r)
                for x in sorted(ref.adj[r], key=lambda x: len(self.cand[x])):
                    if not seen[x]:
                        seen[x] = True
                        queue.append(x)
        self.order = order

    def _extensions(self, r, mapping, used):
        """Mobile atoms that can take reference atom ``r`` given the mapping so far."""
        ref, mob = self.ref, self.mob
        mapped = [x for x in ref.adj[r] if mapping[x] >= 0]
        for m in self.cand[r]:
            m = int(m)
            if used[m]:
                continue
            nbrs = mob.adj[m]
            if any(mapping[x] not in nbrs for x in mapped):
                continue
            if sum(1 for y in nbrs if used[y]) != len(mapped):
                continue
            if self.bond_orders and any(ref.order[(r, x)] != mob.order[(m, mapping[x])]
                                        for x in mapped):  # fmt: skip
                continue
            yield m

    def best(self, D2: np.ndarray) -> tuple[float, np.ndarray]:
        """Mapping minimizing the sum of squared distances ``D2[ref, mob]``."""
        n = len(self.order)
        lower = np.array([D2[r, self.cand[r]].min() for r in self.order])
        bound = np.concatenate([np.cumsum(lower[::-1])[::-1], [0.0]])
        mapping = np.full(n, -1, np.int64)
        used = np.zeros(n, bool)
        best = [math.inf, None]
        tol = 1e-12

        def search(depth, total):
            if total + bound[depth] >= best[0] - tol:
                return
            if depth == n:
                best[0], best[1] = total, mapping.copy()
                return
            r = self.order[depth]
            options = list(self._extensions(r, mapping, used))
            options.sort(key=lambda m: D2[r, m])
            for m in options:
                mapping[r] = m
                used[m] = True
                search(depth + 1, total + D2[r, m])
                mapping[r] = -1
                used[m] = False

        search(0, 0.0)
        if best[1] is None:
            raise ValueError("the two atom sets are not the same molecule (no isomorphism)")
        return best[0], best[1]

    def all(self, limit: int) -> tuple[list[np.ndarray], bool]:
        """Every isomorphism (up to ``limit``)."""
        n = len(self.order)
        mapping = np.full(n, -1, np.int64)
        used = np.zeros(n, bool)
        found: list[np.ndarray] = []

        def search(depth):
            if len(found) >= limit:
                return
            if depth == n:
                found.append(mapping.copy())
                return
            r = self.order[depth]
            for m in list(self._extensions(r, mapping, used)):
                mapping[r] = m
                used[m] = True
                search(depth + 1)
                mapping[r] = -1
                used[m] = False

        search(0)
        return found, len(found) >= limit


def _coordinate_blocks(mobile, positions, atoms):
    """(blocks of ``atoms`` coordinates, one (nframes, len(atoms), 3) array per chunk; frames?).

    ``positions``: None (the system's coordinates), one frame, an (nframes,
    natoms, 3) array, a Frames block, or a Trajectory (read chunk by chunk,
    only the needed atoms).
    """
    if positions is None:
        return [mobile.positions[atoms][None]], False
    if hasattr(positions, "chunks") and not hasattr(positions, "positions"):  # a Trajectory
        blocks = (np.asarray(b.positions, np.float64) for b in positions.chunks(512, atoms=atoms))
        return blocks, True
    xyz = np.asarray(getattr(positions, "positions", positions), dtype=np.float64)
    if xyz.ndim == 2:
        return [xyz[atoms][None]], False
    return [xyz[:, atoms]], True


def _first_frame(mobile, positions) -> np.ndarray:
    if positions is None:
        return mobile.positions
    if hasattr(positions, "chunks") and not hasattr(positions, "positions"):
        return np.asarray(positions[0].positions, np.float64)
    xyz = np.asarray(getattr(positions, "positions", positions), dtype=np.float64)
    return xyz if xyz.ndim == 2 else xyz[0]


def _prepare(mobile, reference, atoms, reference_atoms, heavy_only, bond_orders):
    mids = _ids(mobile, atoms)
    rids = _ids(reference, reference_atoms if reference_atoms is not None else atoms)
    if heavy_only:
        mids = mids[mobile.atoms["anum"][mids] > 1]
        rids = rids[reference.atoms["anum"][rids] > 1]
    if not len(rids):
        raise ValueError("no atoms selected")
    matcher = _Matcher(_Graph(reference, rids, bond_orders), _Graph(mobile, mids, bond_orders),
                       bond_orders)  # fmt: skip
    same_order = mobile.atoms["anum"][mids].tolist() == reference.atoms["anum"][rids].tolist()
    return matcher, mids, rids, same_order


def _rmsds(matcher, R, blocks, superpose, same_order, max_isomorphisms):
    """Symmetry-corrected and plain RMSD of every frame in ``blocks`` (mobile atoms in order)."""
    n = len(R)
    out, plain, first = [], [], None
    perms, count, truncated = None, None, False
    if superpose:
        maps, truncated = matcher.all(max_isomorphisms)
        perms, count = np.array(maps), len(maps)
    for block in blocks:
        for M in block:
            if superpose:
                total, mapping = math.inf, None
                for p in perms:
                    rot, t = kabsch(M[p], R)
                    v = float(((M[p] @ rot.T + t - R) ** 2).sum())
                    if v < total:
                        total, mapping = v, p
            else:
                total, mapping = matcher.best(((R[:, None, :] - M[None, :, :]) ** 2).sum(-1))
            out.append(math.sqrt(total / n))
            if first is None:
                first = mapping
            if same_order:
                X = M
                if superpose:
                    rot, t = kabsch(M, R)
                    X = M @ rot.T + t
                plain.append(math.sqrt(((X - R) ** 2).sum() / n))
    if not out:
        raise ValueError("no frames")
    return np.array(out), (np.array(plain) if same_order else None), first, count, truncated


def symmetry_rmsd(mobile, reference, atoms=None, reference_atoms=None, positions=None,
                  superpose: bool = False, heavy_only: bool = True, bond_orders: bool = False,
                  max_isomorphisms: int = 100_000) -> SymmetryRMSD:  # fmt: skip
    """Smallest RMSD (Å) over atom mappings that preserve elements and bonds.

    ``atoms`` selects the molecule in ``mobile`` (default: every atom) and
    ``reference_atoms`` in ``reference`` (default: the same selection).
    ``positions``: mobile coordinates to use instead of ``mobile.positions``:
    one frame, an (nframes, natoms, 3) array, a Frames block or a Trajectory
    (one RMSD per frame; trajectories are read chunk by chunk).
    ``superpose=False`` (default) compares coordinates as they are, as for a
    docked pose in the protein frame; ``superpose=True`` fits each mapping
    first (the RMSD between conformers).  ``heavy_only`` drops hydrogens and
    pseudo particles.  ``bond_orders=True`` also requires bond orders to match.
    Connectivity alone is the default, as in most docking benchmarks: with
    Kekulé bond orders a flipped phenyl ring or a carboxylate's two oxygens
    would no longer count as equivalent.
    """
    matcher, mids, rids, same = _prepare(mobile, reference, atoms, reference_atoms, heavy_only,
                                         bond_orders)  # fmt: skip
    blocks, many = _coordinate_blocks(mobile, positions, mids)
    out, plain, first, count, truncated = _rmsds(matcher, reference.positions[rids], blocks,
                                                 superpose, same, max_isomorphisms)  # fmt: skip
    rmsd = out if many else float(out[0])
    plain = None if plain is None else (plain if many else float(plain[0]))
    return SymmetryRMSD(rmsd, plain, mids[first], mids, rids, count, truncated)


def ligand_rmsd(mobile, reference, ligand: str = DEFAULT_LIGAND, reference_ligand=None,
                fit: str = DEFAULT_FIT, reference_fit=None, align: str | None = "order",
                positions=None, heavy_only: bool = True, bond_orders: bool = False,
                apply: bool = False) -> LigandRMSD:  # fmt: skip
    """Superpose ``mobile`` onto ``reference`` by protein atoms, then the ligand RMSD.

    The docking-pose RMSD: ``fit`` atoms (C-alpha by default) are superposed,
    the transform moves the whole mobile system (ligand included), and the
    ligand RMSD is the symmetry-corrected, in-place RMSD between the ligand
    atoms (``ligand``/``reference_ligand`` selections; by default everything
    that is not protein, nucleic acid, water or ions, without hydrogens).
    ``align``: "order" pairs the fit atoms in order (same protein);
    "sequence" uses ``boonza.matchmaker`` (different sequences or numbering);
    None when the two are already in one frame.

    ``positions``: mobile coordinates instead of ``mobile.positions``: one
    frame, an (nframes, natoms, 3) array, a Frames block or a Trajectory.
    Every frame is fitted on its own and the results are arrays with one value
    per frame (rotations (nframes, 3, 3)); trajectories are read chunk by
    chunk, only the fit and ligand atoms.  With ``align="sequence"`` the
    residue pairing comes from the first frame.  ``apply`` moves ``mobile``
    (single structures only).
    """
    matcher, mids, rids, same = _prepare(mobile, reference, ligand, reference_ligand, heavy_only,
                                         bond_orders)  # fmt: skip
    if align == "order":
        mfit, rfit = _ids(mobile, fit), _ids(reference, reference_fit or fit)
        if len(mfit) != len(rfit):
            raise ValueError(f"fit selections have {len(mfit)} and {len(rfit)} atoms; "
                             "use align='sequence' for different proteins")  # fmt: skip
    elif align == "sequence":
        from .matchmaker import matchmaker

        first = mobile.copy()
        first.positions = _first_frame(mobile, positions)
        mm = matchmaker(first, reference, apply=False)
        mfit, rfit = mm.mobile_atoms[mm.kept], mm.reference_atoms[mm.kept]
    elif align is None:
        mfit = rfit = np.empty(0, np.int64)
    else:
        raise ValueError("align must be 'order', 'sequence' or None")
    need = np.union1d(mfit, mids)
    fit_local, lig_local = np.searchsorted(need, mfit), np.searchsorted(need, mids)
    target = reference.positions[rfit]
    blocks, many = _coordinate_blocks(mobile, positions, need)
    rots, trans, fits = [], [], []

    def fitted_ligand():
        for block in blocks:
            out = np.empty((len(block), len(mids), 3))
            for k, X in enumerate(block):
                if len(mfit):
                    rot, t = kabsch(X[fit_local], target)
                    d = X[fit_local] @ rot.T + t - target
                    fits.append(float(np.sqrt((d * d).sum(1).mean())))
                else:
                    rot, t = np.eye(3), np.zeros(3)
                    fits.append(0.0)
                rots.append(rot)
                trans.append(t)
                out[k] = X[lig_local] @ rot.T + t
            yield out

    out, plain, first_map, _, _ = _rmsds(matcher, reference.positions[rids], fitted_ligand(),
                                         False, same, 0)  # fmt: skip
    if apply:
        if many:
            raise ValueError("apply=True needs a single structure, not frames")
        mobile.positions = _first_frame(mobile, positions) @ rots[0].T + trans[0]
        if mobile.cell.any():
            mobile.cell = mobile.cell @ rots[0].T
    if many:
        return LigandRMSD(out, plain, np.array(fits), np.array(rots), np.array(trans),
                          mids[first_map], mids, rids)  # fmt: skip
    return LigandRMSD(float(out[0]), None if plain is None else float(plain[0]), fits[0],
                      rots[0], trans[0], mids[first_map], mids, rids)  # fmt: skip


def _boxed_blocks(system, positions, atoms):
    """Blocks of (positions of ``atoms`` (nframes, n, 3), boxes (nframes, 3, 3)); frames?"""
    cell = np.asarray(system.cell, np.float64)
    if positions is None:
        return [(system.positions[atoms][None], cell[None])], False
    if hasattr(positions, "chunks") and not hasattr(positions, "positions"):  # a Trajectory
        blocks = ((np.asarray(b.positions, np.float64), np.asarray(b.boxes, np.float64))
                  for b in positions.chunks(512, atoms=atoms))  # fmt: skip
        return blocks, True
    xyz = np.asarray(getattr(positions, "positions", positions), dtype=np.float64)
    boxes = getattr(positions, "boxes", None)
    if xyz.ndim == 2:
        return [(xyz[atoms][None], cell[None] if boxes is None else np.asarray(boxes)[None])], False
    if boxes is None:
        boxes = np.broadcast_to(cell, (len(xyz), 3, 3))
    return [(xyz[:, atoms], np.asarray(boxes, np.float64))], True


def drmsd(system, reference=None, ligand: str = DEFAULT_LIGAND,
          protein: str = "protein and name CA", cutoff: float = 5.0, positions=None,
          reference_ligand=None, reference_protein=None,
          symmetry: bool = True, heavy_only: bool = True, bond_orders: bool = False,
          periodic: bool = True) -> DRMSD:  # fmt: skip
    """Distance RMSD (Å) between pocket atoms and ligand atoms, against a reference.

    The pocket is the ``protein`` atoms within ``cutoff`` Å of the ligand in
    the reference.  For every frame, the (pocket x ligand) distance matrix
    ``d`` is compared with the reference matrix: ``sqrt(mean((d - d_ref)**2))``.
    No superposition is needed, and the ligand's internal distances are not
    included.

    ``reference``: a System (a crystal structure, say), or None for the first
    frame of ``positions`` (the system's own coordinates without it), as in a
    "drift from frame 0" analysis.  Protein atoms are paired between the two
    systems in selection order (``reference_protein`` defaults to
    ``protein``; the counts must agree).  Ligand atoms are paired by symmetry:
    with ``symmetry=True`` each frame uses the element- and bond-preserving
    atom mapping with the smallest dRMSD (exact, like ``symmetry_rmsd``), so a
    flipped ring or swapped carboxylate oxygens do not count as motion;
    ``plain_drmsd`` pairs the ligand atoms in listed order.

    ``positions``: one frame, an (nframes, natoms, 3) array, a Frames block or
    a Trajectory (read chunk by chunk, only pocket and ligand atoms).  With
    ``periodic=True`` distances use the minimum image of each frame's box (the
    system's cell for plain arrays).
    """
    from .pbc import distances

    ref_sys = system if reference is None else reference
    matcher, mids, rids, same = _prepare(system, ref_sys, ligand, reference_ligand, heavy_only,
                                         bond_orders)  # fmt: skip
    mprot = _ids(system, protein)
    rprot = _ids(ref_sys, reference_protein or protein)
    if len(mprot) != len(rprot):
        raise ValueError(f"protein selections have {len(mprot)} and {len(rprot)} atoms; "
                         "they are paired in order")  # fmt: skip
    if reference is None:
        blocks, _ = _boxed_blocks(system, positions, np.arange(system.natoms))
        rpos, rbox = next(iter(blocks))
        rpos, rbox = rpos[0], rbox[0]
    else:
        rpos, rbox = reference.positions, np.asarray(reference.cell, np.float64)
    rbox = rbox if periodic else None
    keep = distances(rpos[rprot], rpos[rids], rbox).min(1) <= cutoff
    keep &= ~np.isin(rprot, rids)
    if not keep.any():
        raise ValueError(f"no protein atoms within {cutoff} A of the reference ligand")
    mpocket, rpocket = mprot[keep], rprot[keep]
    dref = distances(rpos[rpocket], rpos[rids], rbox)
    npocket, nlig = dref.shape

    need = np.union1d(mpocket, mids)
    pl, ll = np.searchsorted(need, mpocket), np.searchsorted(need, mids)
    blocks, many = _boxed_blocks(system, positions, need)
    out, plain, first = [], [], None
    for xyz, boxes in blocks:
        for X, box in zip(xyz, boxes, strict=True):
            d = distances(X[pl], X[ll], box if periodic else None)
            if same:
                plain.append(math.sqrt(((d - dref) ** 2).sum() / d.size))
            if symmetry:
                # cost[r, m]: squared deviations if mobile atom m plays reference atom r
                cost = ((dref.T[:, None, :] - d.T[None, :, :]) ** 2).sum(-1)
                total, mapping = matcher.best(cost)
                out.append(math.sqrt(total / (npocket * nlig)))
            else:
                if not same:
                    raise ValueError("symmetry=False needs the ligand atoms in the same order")
                out.append(plain[-1])
                mapping = np.arange(nlig)
            if first is None:
                first = mapping
    if not out:
        raise ValueError("no frames")
    res = np.array(out)
    pl_arr = np.array(plain) if same else None
    if not many:
        res = float(res[0])
        pl_arr = None if pl_arr is None else float(pl_arr[0])
    return DRMSD(res, pl_arr, mpocket, rpocket, mids[first], mids, rids, dref)
