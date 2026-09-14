"""Trajectory analysis: RMSD, RMSF, radius of gyration, RDF, residue contacts and SASA.

Each function takes a System plus optional frames (one (natoms, 3) array,
an (nframes, natoms, 3) array, or a Frames block; the system's own positions
and cell by default) and follows a reference implementation exactly:

* ``rmsd_trajectory`` and ``rmsf``: MDAnalysis RMSD and RMSF
* ``radius_of_gyration``: MDAnalysis (mass weighted) or mdtraj (``weights=None``)
* ``rdf``: MDAnalysis InterRDF (normalization, exclusion blocks)
* ``residue_contacts``: mdtraj compute_contacts
* ``sasa``: mdtraj shrake_rupley (same float32 sphere points and radii)
* ``native_contacts``: MDAnalysis Contacts (hard_cut, soft_cut, radius_cut)
* ``contact_frequency``: fraction of frames each residue (or atom) pair is in contact
* ``pca``: MDAnalysis PCA (alignment to the first frame, same variances and components)
* ``block_average``: Flyvbjerg-Petersen blocking, the standard error of a correlated series

Lengths are in Å and areas in Å^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from . import pbc
from .align import kabsch
from .hbonds import _frames


def _ids(system, sel):
    if sel is None or (isinstance(sel, str) and sel == "all"):
        return np.arange(system.natoms)
    if isinstance(sel, str):
        return system.select(sel).ids
    return np.asarray(system._ids("atom", sel), dtype=np.int64)


def _weights(system, ids, weights):
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights != "mass":
            raise ValueError("weights must be None, 'mass' or an array")
        w = system.atoms["mass"][ids].astype(np.float64)
        if not w.any():  # e.g. PDB files: guess element masses, as MDAnalysis does
            from .elements import guessed_masses

            w = guessed_masses(system.atoms["anum"][ids])
    else:
        w = np.asarray(weights, dtype=np.float64)
    if not w.sum() > 0:
        raise ValueError("weights sum to zero (does the system have masses?)")
    return w


# ---------------------------------------------------------------------------
# RMSD, RMSF, radius of gyration


def rmsd_trajectory(system, positions=None, sel="all", reference=None, weights=None,
                    fit: bool = True) -> np.ndarray:  # fmt: skip
    """RMSD (Å) of the ``sel`` atoms in each frame to ``reference`` (MDAnalysis RMSD).

    ``reference``: positions of all atoms or of the selected atoms (default:
    the first frame).  With ``fit`` each frame is optimally superposed first.
    ``weights``: None, "mass" or per-atom weights.
    """
    xyz, _ = _frames(system, positions, None)
    ids = _ids(system, sel)
    ref = xyz[0][ids] if reference is None else np.asarray(reference, np.float64).reshape(-1, 3)
    if len(ref) == system.natoms and len(ids) != system.natoms:
        ref = ref[ids]
    w = _weights(system, ids, weights)
    wn = np.ones(len(ids)) if w is None else w
    out = np.empty(len(xyz))
    for f, frame in enumerate(xyz):
        P = frame[ids]
        if fit:
            R, t = kabsch(P, ref, w)
            P = P @ R.T + t
        out[f] = math.sqrt((wn * ((P - ref) ** 2).sum(1)).sum() / wn.sum())
    return out


def rmsf(system, positions=None, sel="all") -> np.ndarray:
    """Root mean square fluctuation (Å) of each ``sel`` atom about its mean position.

    As MDAnalysis RMSF, frames are used as given: superpose them first
    (e.g. with ``boonza.Glue(fit=...)``) to remove overall motion.
    """
    xyz, _ = _frames(system, positions, None)
    p = xyz[:, _ids(system, sel)]
    return np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))


def radius_of_gyration(system, positions=None, sel="all", weights="mass") -> np.ndarray:
    """Radius of gyration (Å) per frame.

    ``weights="mass"``: about the center of mass, as MDAnalysis;
    ``weights=None``: equal weights about the centroid, as mdtraj's default.
    Systems without masses (e.g. from PDB files) use element masses as
    MDAnalysis guesses them.
    """
    xyz, _ = _frames(system, positions, None)
    ids = _ids(system, sel)
    p = xyz[:, ids]
    w = _weights(system, ids, weights)
    if w is None:
        w = np.ones(len(ids))
    w = w / w.sum()
    center = np.einsum("i,fij->fj", w, p)
    return np.sqrt(np.einsum("i,fi->f", w, ((p - center[:, None]) ** 2).sum(2)))


# ---------------------------------------------------------------------------
# radial distribution function


def rdf(system, g1, g2, positions=None, nbins: int = 75, range=(0.0, 15.0), norm: str = "rdf",
        exclusion_block=None, exclude_same=None):  # fmt: skip
    """Radial distribution function between groups ``g1`` and ``g2`` (MDAnalysis InterRDF).

    Returns (bin centers, rdf, edges, counts).  ``norm``: "rdf" (g(r), using
    the average box volume), "density" (single-particle density) or "none"
    (counts).  ``exclusion_block=(nA, nB)`` drops pairs within consecutive
    blocks of nA and nB atoms (e.g. the same molecule); ``exclude_same`` drops
    pairs in the same "residue", "chain" or "fragment".
    """
    if norm not in ("rdf", "density", "none"):
        raise ValueError("norm must be 'rdf', 'density' or 'none'")
    if exclude_same is not None and exclusion_block is not None:
        raise ValueError("use either exclusion_block or exclude_same")
    xyz, boxes = _frames(system, positions, None)
    a, b = _ids(system, g1), _ids(system, g2)
    count, edges = np.histogram([-1], bins=nbins, range=range)
    count = count.astype(np.float64) * 0
    if exclude_same is not None:
        key = {"residue": system.atoms["residue"],
               "chain": system.residues["chain"][system.atoms["residue"]],
               "fragment": system.fragids}[exclude_same]  # fmt: skip
    volume = 0.0
    for frame, cell in zip(xyz, boxes, strict=True):
        i, j, d = pbc.capped_distances(frame[a], frame[b], float(range[1]), cell)
        if exclusion_block is not None:
            keep = (i // exclusion_block[0]) != (j // exclusion_block[1])
            d = d[keep]
        elif exclude_same is not None:
            d = d[key[a[i]] != key[b[j]]]
        count += np.histogram(d, bins=nbins, range=range)[0]
        if norm == "rdf":
            if cell is None:
                raise ValueError("norm='rdf' needs a periodic box")
            volume += abs(np.linalg.det(cell))
    nframes = len(xyz)
    scale = np.full(nbins, float(nframes))
    if norm in ("rdf", "density"):
        scale *= 4 / 3 * np.pi * np.diff(edges**3)
    if norm == "rdf":
        n = len(a) * len(b)
        if exclusion_block is not None:
            xa, xb = exclusion_block
            n -= xa * xb * (len(a) / xa)
        scale *= n / (volume / nframes)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, count / scale, edges, count


# ---------------------------------------------------------------------------
# residue contacts


def residue_contacts(system, positions=None, contacts="all", scheme: str = "closest-heavy",
                     ignore_nonprotein: bool = True, periodic: bool = True):  # fmt: skip
    """Residue-residue distances (Å) per frame, as mdtraj compute_contacts.

    ``contacts="all"``: pairs of residues in the same chain at least three
    apart (only residues with a CA atom when ``ignore_nonprotein``), or an
    (n, 2) array of residue indices.  ``scheme``: "ca" (CA-CA), "closest"
    (closest atoms) or "closest-heavy" (closest non-hydrogen atoms).
    Returns (distances of shape (nframes, npairs), residue pairs).
    """
    xyz, boxes = _frames(system, positions, None)
    s = system
    names = np.strings.lower(s.atoms["name"])
    res_of = s.atoms["residue"]
    nres = s.nresidues
    has_ca = np.zeros(nres, bool)
    has_ca[res_of[names == "ca"]] = True
    if isinstance(contacts, str):
        if contacts.lower() != "all":
            raise ValueError(f"({contacts}) is not a valid contacts specifier")
        chain = s.residues["chain"]
        ok = has_ca if ignore_nonprotein else np.ones(nres, bool)
        pairs = [(i, j) for i in np.flatnonzero(ok).tolist() for j in range(i + 3, nres)
                 if ok[j] and chain[i] == chain[j]]  # fmt: skip
        pairs = np.array(pairs, np.int64).reshape(-1, 2)
        if not len(pairs):
            raise ValueError("No acceptable residue pairs found")
    else:
        pairs = np.asarray(contacts, np.int64).reshape(-1, 2)
    scheme = scheme.lower()
    if scheme == "ca":
        ca = np.full(nres, -1, np.int64)
        count = np.bincount(res_of[names == "ca"], minlength=nres)
        if (count > 1).any():
            raise ValueError("More than 1 alpha carbon detected in a residue")
        ca[res_of[names == "ca"]] = np.flatnonzero(names == "ca")
        pairs = pairs[(ca[pairs[:, 0]] >= 0) & (ca[pairs[:, 1]] >= 0)]
        ai, aj = ca[pairs[:, 0]], ca[pairs[:, 1]]
        starts = np.arange(len(pairs))
    elif scheme in ("closest", "closest-heavy"):
        member = np.ones(s.natoms, bool) if scheme == "closest" else s.atoms["anum"] != 1
        order = np.flatnonzero(member)
        order = order[np.argsort(res_of[order], kind="stable")]
        start = np.searchsorted(res_of[order], np.arange(nres + 1))
        ai_l, aj_l, starts = [], [], np.empty(len(pairs), np.int64)
        total = 0
        for k, (r0, r1) in enumerate(pairs.tolist()):
            m0, m1 = order[start[r0] : start[r0 + 1]], order[start[r1] : start[r1 + 1]]
            starts[k] = total
            ai_l.append(np.repeat(m0, len(m1)))
            aj_l.append(np.tile(m1, len(m0)))
            total += len(m0) * len(m1)
        ai, aj = np.concatenate(ai_l), np.concatenate(aj_l)
    else:
        raise ValueError("scheme must be one of ca, closest, closest-heavy")
    out = np.empty((len(xyz), len(pairs)))
    for f, (frame, cell) in enumerate(zip(xyz, boxes, strict=True)):
        d = pbc.paired_distances(frame[ai], frame[aj], cell if periodic else None)
        out[f] = np.minimum.reduceat(d, starts) if len(starts) else d[:0]
    return out, pairs


# ---------------------------------------------------------------------------
# solvent accessible surface area (Shrake-Rupley, as mdtraj)

# mdtraj's radii (nm) for the elements proteins and common ligands contain
_SASA_RADII = {
    "H": 0.120, "He": 0.140, "Li": 0.076, "Be": 0.059, "B": 0.192, "C": 0.170, "N": 0.155,
    "O": 0.152, "F": 0.147, "Ne": 0.154, "Na": 0.102, "Mg": 0.086, "Al": 0.184, "Si": 0.210,
    "P": 0.180, "S": 0.180, "Cl": 0.181, "Ar": 0.188, "K": 0.138, "Ca": 0.114, "Sc": 0.211,
    "Ti": 0.200, "V": 0.200, "Cr": 0.200, "Mn": 0.200, "Fe": 0.200, "Co": 0.200, "Ni": 0.163,
    "Cu": 0.140, "Zn": 0.139, "Ga": 0.187, "Ge": 0.211, "As": 0.185, "Se": 0.190,
    "Br": 0.185, "Kr": 0.202, "Rb": 0.303, "Sr": 0.249, "Y": 0.200, "Zr": 0.200,
}  # fmt: skip


def _sphere_points(n: int) -> np.ndarray:
    """mdtraj's golden-section spiral, with its float/double promotions."""
    inc = np.float32(math.pi * (3.0 - math.sqrt(5.0)))
    offset = np.float32(2.0 / n)
    pts = np.empty((n, 3), np.float32)
    for i in range(n):
        y = np.float32(float(np.float32(i * offset)) - 1.0 + float(offset) / 2.0)
        r = np.float32(math.sqrt(1.0 - float(np.float32(y * y))))
        phi = np.float32(np.float32(i) * inc)
        pts[i] = (np.float32(math.cos(float(phi)) * float(r)), y,
                  np.float32(math.sin(float(phi)) * float(r)))  # fmt: skip
    return pts


@njit(parallel=True, cache=True)
def _sasa_frame(xyz, radii, points, noff, nbr, selected):
    n = xyz.shape[0]
    npts = points.shape[0]
    constant = np.float32(4.0 * math.pi / npts)
    areas = np.zeros(n, np.float32)
    for i in prange(n):
        if not selected[i]:
            continue
        ri = radii[i]
        count = 0
        k0 = 0
        m = noff[i + 1] - noff[i]
        for p in range(npts):
            px = xyz[i, 0] + ri * points[p, 0]
            py = xyz[i, 1] + ri * points[p, 1]
            pz = xyz[i, 2] + ri * points[p, 2]
            accessible = True
            for kk in range(k0, k0 + m):
                j = nbr[noff[i] + kk % m]
                dx, dy, dz = px - xyz[j, 0], py - xyz[j, 1], pz - xyz[j, 2]
                rj = radii[j]
                if dx * dx + dy * dy + dz * dz < rj * rj:
                    k0 = kk
                    accessible = False
                    break
            if accessible:
                count += 1
        areas[i] = np.float32(count) * constant * ri * ri
    return areas


def sasa(system, positions=None, probe_radius: float = 1.4, n_sphere_points: int = 960,
         mode: str = "atom", atoms=None, radii=None) -> np.ndarray:  # fmt: skip
    """Solvent accessible surface area (Å^2) per atom or residue, per frame (mdtraj).

    ``radii``: {element symbol: radius in Å} overriding mdtraj's defaults.
    ``atoms``: compute only these atoms (others report -1, as mdtraj);
    they are still occluded by every atom.  Frames are not made whole.
    """
    from .elements import msys_symbol

    if mode not in ("atom", "residue"):
        raise ValueError('mode must be one of "residue", "atom"')
    xyz, _ = _frames(system, positions, None)
    table = {k: v * 10 for k, v in _SASA_RADII.items()}
    if radii:
        table.update(radii)
    symbols = [msys_symbol(int(z)) for z in system.atoms["anum"].tolist()]
    try:
        vdw = np.array([table[sym] for sym in symbols], np.float64)
    except KeyError as e:
        raise KeyError(f"no SASA radius for element {e.args[0]}; pass radii=") from None
    # mdtraj works in float32 nm; keep its arithmetic so sphere points agree
    r_nm = (np.float32(vdw / 10) + np.float32(probe_radius / 10)).astype(np.float32)
    points = _sphere_points(int(n_sphere_points))
    selected = np.ones(system.natoms, bool) if atoms is None else np.zeros(system.natoms, bool)
    if atoms is not None:
        selected[_ids(system, atoms)] = True
    groups = np.arange(system.natoms) if mode == "atom" else system.atoms["residue"]
    ngroups = system.natoms if mode == "atom" else system.nresidues
    out = np.zeros((len(xyz), ngroups), np.float64)
    if atoms is not None:
        out[:] = -1
        out[:, groups[selected]] = 0
    for f, frame in enumerate(xyz):
        nm = (frame * 0.1).astype(np.float32)
        pi, pj, d2 = _neighbors(nm, r_nm)
        order = np.argsort(pi, kind="stable")
        noff = np.zeros(system.natoms + 1, np.int64)
        np.cumsum(np.bincount(pi, minlength=system.natoms), out=noff[1:])
        areas = _sasa_frame(nm, r_nm, points, noff, pj[order], selected)
        np.add.at(out[f], groups[selected], areas[selected].astype(np.float64) * 100.0)
    return out


def _neighbors(nm, r_nm):
    """Directed pairs (i, j), i != j, with |ri - rj|^2 < (radius_i + radius_j)^2 in float32."""
    from .spatial import pairs_within

    i, j, _ = pairs_within(nm.astype(np.float64), 2 * float(r_nm.max()) + 1e-4)
    diff = nm[i] - nm[j]
    r2 = (diff * diff).sum(1, dtype=np.float32)
    cut = r_nm[i] + r_nm[j]
    keep = r2 < cut * cut
    i, j = i[keep], j[keep]
    return np.concatenate([i, j]), np.concatenate([j, i]), None


# ---------------------------------------------------------------------------
# native contacts, contact frequencies, PCA, block averaging


def native_contacts(system, sel1, sel2, positions=None, reference=None, radius: float = 4.5,
                    method: str = "hard_cut", beta: float = 5.0, lambda_constant: float = 1.8,
                    periodic: bool = True) -> np.ndarray:  # fmt: skip
    """Fraction of native contacts Q in each frame (MDAnalysis Contacts).

    Native contacts are the ``sel1`` x ``sel2`` atom pairs within ``radius``
    Å in ``reference`` (a System with the same selections; default the
    system's own coordinates), with reference distances r0.  ``method``:
    "hard_cut" counts r <= r0, "radius_cut" counts r <= radius, "soft_cut"
    is Best, Hummer and Eaton's 1 / (1 + exp(beta (r - lambda_constant r0)))
    with beta in 1/Å.  With ``periodic``, distances use each frame's box (and
    the reference cell).
    """
    if method not in ("hard_cut", "soft_cut", "radius_cut"):
        raise ValueError("method must be 'hard_cut', 'soft_cut' or 'radius_cut'")
    a, b = _ids(system, sel1), _ids(system, sel2)
    ref = system if reference is None else reference
    ra, rb = (a, b) if reference is None else (_ids(ref, sel1), _ids(ref, sel2))
    if (len(ra), len(rb)) != (len(a), len(b)):
        raise ValueError("the reference selections have different sizes")
    rbox = ref.cell if (periodic and ref.cell.any()) else None
    d0 = pbc.distances(ref.positions[ra], ref.positions[rb], rbox)
    ci, cj = np.nonzero(d0 <= radius)
    if not len(ci):
        raise ValueError(f"no {sel1!r} - {sel2!r} pairs within {radius} A in the reference")
    r0 = d0[ci, cj]
    xyz, boxes = _frames(system, positions, system.cell if periodic else None)
    out = np.empty(len(xyz))
    for f, X in enumerate(xyz):
        r = pbc.paired_distances(X[a[ci]], X[b[cj]], boxes[f] if periodic else None)
        if method == "hard_cut":
            out[f] = (r <= r0).mean()
        elif method == "radius_cut":
            out[f] = (r <= radius).mean()
        else:
            out[f] = (1.0 / (1.0 + np.exp(beta * (r - lambda_constant * r0)))).sum() / len(r0)
    return out


def contact_frequency(system, sel1, sel2=None, positions=None, cutoff: float = 4.5,
                      level: str = "residue", periodic: bool = True):  # fmt: skip
    """How often each pair is in contact: (rows, cols, fraction of frames).

    A residue pair (``level="residue"``) is in contact in a frame when any of
    their atoms are within ``cutoff`` Å; ``level="atom"`` uses atom pairs.
    ``rows`` and ``cols`` are residue (or atom) indices of ``sel1`` and
    ``sel2``, and the matrix has one fraction per (row, col).  Without
    ``sel2`` the pairs are within ``sel1``, excluding a residue (atom) with
    itself.  With ``periodic``, distances use each frame's box.
    """
    if level not in ("residue", "atom"):
        raise ValueError("level must be 'residue' or 'atom'")
    a = _ids(system, sel1)
    b = a if sel2 is None else _ids(system, sel2)
    res = system.atoms["residue"]
    ka, kb = (res[a], res[b]) if level == "residue" else (a, b)
    rows, la = np.unique(ka, return_inverse=True)
    cols, lb = np.unique(kb, return_inverse=True)
    la, lb = la.reshape(-1), lb.reshape(-1)
    counts = np.zeros((len(rows), len(cols)))
    xyz, boxes = _frames(system, positions, system.cell if periodic else None)
    for f, X in enumerate(xyz):
        i, j, d = pbc.capped_distances(X[a], X[b], cutoff, boxes[f] if periodic else None)
        keep = d <= cutoff
        i, j = i[keep], j[keep]
        if sel2 is None:
            other = ka[i] != kb[j]
            i, j = i[other], j[other]
        hit = np.zeros(counts.shape, bool)
        hit[la[i], lb[j]] = True
        counts += hit
    return rows, cols, counts / len(xyz)


@dataclass
class PCA:
    """Principal components of atomic fluctuations (see ``pca``)."""

    atoms: np.ndarray
    mean: np.ndarray  # (natoms, 3), in the reference frame when aligned
    variance: np.ndarray  # (ncomponents,) Å^2, largest first
    components: np.ndarray  # (ncomponents, 3 natoms) unit vectors
    projections: np.ndarray  # (nframes, ncomponents): the frames along each component
    reference: np.ndarray | None  # coordinates every frame was fitted onto (aligned PCA)

    @property
    def cumulated_variance(self) -> np.ndarray:
        """Fraction of the total variance in the first 1, 2, ... components."""
        return np.cumsum(self.variance) / self.variance.sum()

    def transform(self, system, positions=None, n_components=None) -> np.ndarray:
        """Project frames of ``system`` onto the components (fitted like the analysis)."""
        xyz, _ = _frames(system, positions, None)
        X = _fit_frames(xyz[:, self.atoms], self.reference)
        k = len(self.variance) if n_components is None else n_components
        return (X - self.mean).reshape(len(X), -1) @ self.components[:k].T


def _fit_frames(X, reference):
    if reference is None:
        return X
    out = np.empty_like(X)
    for f, x in enumerate(X):
        rot, t = kabsch(x, reference)
        out[f] = x @ rot.T + t
    return out


def pca(system, positions=None, sel="name CA", align: bool = True,
        n_components=None) -> PCA:  # fmt: skip
    """Principal component analysis of the ``sel`` atoms over frames (MDAnalysis PCA).

    With ``align`` every frame is first superposed on the first frame (as
    MDAnalysis does); the covariance of the fitted coordinates is divided by
    nframes - 1.  Components come from a singular value decomposition of the
    centered frames, so at most min(nframes, 3 natoms) are returned (the rest
    have zero variance), each with its largest element positive.
    """
    ids = _ids(system, sel)
    xyz, _ = _frames(system, positions, None)
    if len(xyz) < 2:
        raise ValueError("PCA needs at least two frames")
    X = xyz[:, ids]
    reference = X[0].copy() if align else None
    X = _fit_frames(X, reference)
    mean = X.mean(axis=0)
    D = (X - mean).reshape(len(X), -1)
    _, sv, vt = np.linalg.svd(D, full_matrices=False)
    variance = sv * sv / (len(X) - 1)
    k = len(variance) if n_components is None else min(int(n_components), len(variance))
    comps = vt[:k]
    flip = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
    comps = comps * np.where(flip == 0, 1.0, flip)[:, None]
    return PCA(ids, mean, variance[:k], comps, D @ comps.T, reference)


@dataclass
class BlockAverage:
    """Standard error of the mean from blocks of increasing size (see ``block_average``)."""

    mean: float
    block_sizes: np.ndarray  # frames per block: 1, 2, 4, ...
    sem: np.ndarray  # standard error of the mean estimated with each block size
    sem_error: np.ndarray  # uncertainty of each estimate, sem / sqrt(2 (nblocks - 1))

    @property
    def estimate(self) -> float:
        """The plateau: the first block size whose SEM the next size no longer exceeds by
        more than its error bar (the largest block size if it keeps rising)."""
        for k in range(len(self.sem) - 1):
            if self.sem[k + 1] - self.sem[k] <= self.sem_error[k]:
                return float(self.sem[k])
        return float(self.sem[-1])

    @property
    def statistical_inefficiency(self) -> float:
        """(estimate / naive SEM)^2: frames per independent sample."""
        return float((self.estimate / self.sem[0]) ** 2)


def block_average(values, min_blocks: int = 4) -> BlockAverage:
    """Flyvbjerg-Petersen blocking of a time series (for example an RMSD or Q per frame).

    For blocks of 1, 2, 4, ... frames (at least ``min_blocks`` blocks; a
    remainder is dropped), the standard error of the mean is estimated from
    the spread of the block means.  It grows with the block size until blocks
    are longer than the correlation time, then levels off at the true error
    (``estimate``).
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if len(x) < 2 * min_blocks:
        raise ValueError(f"need at least {2 * min_blocks} values")
    sizes, sem, err = [], [], []
    size = 1
    while len(x) // size >= min_blocks:
        nb = len(x) // size
        means = x[: nb * size].reshape(nb, size).mean(axis=1)
        s = means.std(ddof=1) / math.sqrt(nb)
        sizes.append(size)
        sem.append(s)
        err.append(s / math.sqrt(2.0 * (nb - 1)))
        size *= 2
    return BlockAverage(float(x.mean()), np.array(sizes), np.array(sem), np.array(err))
