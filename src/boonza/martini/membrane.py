"""Martini lipid bilayers, after insane: lipids on a grid in two leaflets,
water above and below, ions, and optionally proteins across the membrane."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .build import Martinized
from .solvate import CLASH, SEAM, _tile

BEAD = 4.7  # Å between bonded beads in a template, about a Martini bond
LEVEL = 3.3  # Å between bead levels in a template, as insane stacks them
SPREAD = 4.0  # Å between a lipid's tails in a template
ROTATIONS = 12  # turns tried for each lipid
CORE = 2.5  # Å from the membrane's midplane to the ends of the lipids


@dataclass
class Lipid:
    """A lipid's beads and a straight template of them, head up, from its topology."""

    name: str
    names: list
    xyz: np.ndarray  # Å, the deepest bead at z = 0, the head up
    charge: float


def lipid_templates(itps, names=None) -> dict[str, Lipid]:
    """Templates for the molecule types in ``itps`` (Martini lipid topologies).

    A template is built from the molecule's graph (its bonds, constraints and
    virtual-site parents), as insane builds lipids: bead levels 3.3 Å apart
    down from the first bead (the head), chains zig-zagging so that bonded
    beads stay about a bond apart, branches (a phospholipid's two tails)
    side by side.  Constraints are then brought to their lengths, virtual
    sites put where their parents place them, and the molecule turned so its
    long axis is along z, head up.  Minimization does the rest.
    """
    from ..io.gromacs import _Preprocessor, _Topology

    pre = _Preprocessor(None, [os.fspath(Path(p).parent) for p in itps])
    for p in itps:
        pre.read(os.fspath(p))
    top = _Topology(pre.records)
    out = {}
    for name, mol in top.molecules.items():
        if names is not None and name not in names:
            continue
        atoms = mol.lines("atoms")
        adj = [set() for _ in atoms]
        edges = [(int(t[0]) - 1, int(t[1]) - 1) for s in ("bonds", "constraints")
                 for t in mol.lines(s)]  # fmt: skip
        sites = _virtual_sites(mol)
        edges += [(site, p) for site, parents, _ in sites for p in parents]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        xyz = _straight(adj)
        constraints = [(int(t[0]) - 1, int(t[1]) - 1, float(t[3]) * 10)
                       for t in mol.lines("constraints") if len(t) > 3]  # fmt: skip
        xyz = _place(xyz, constraints, sites)
        charge = sum(float(t[6]) for t in atoms if len(t) > 6)
        out[name] = Lipid(name, [t[4] for t in atoms], xyz, charge)
    missing = set(names or ()) - set(out)
    if missing:
        raise ValueError(f"no topology for {', '.join(sorted(missing))} in the given files")
    return out


def _virtual_sites(mol) -> list:
    """(site, parents, construction) for each virtual site, in file order."""
    out = []
    for t in mol.lines("virtual_sitesn"):
        funct = int(t[1])
        if funct == 3:
            parents = [int(x) - 1 for x in t[2::2]]
            w = [float(x) for x in t[3::2]]
        else:  # geometry or mass: the template does not tell them apart
            parents = [int(x) - 1 for x in t[2:]]
            w = [1.0] * len(parents)
        out.append((int(t[0]) - 1, parents, ("average", np.array(w) / sum(w))))
    for t in mol.lines("virtual_sites2"):
        out.append((int(t[0]) - 1, [int(t[1]) - 1, int(t[2]) - 1], ("lc2", float(t[4]))))
    for t in mol.lines("virtual_sites3"):
        parents = [int(x) - 1 for x in t[1:4]]
        abc = [float(x) for x in t[5:8]]
        out.append((int(t[0]) - 1, parents, ("lc3" if t[4] == "1" else "out3", abc)))
    return out


def _straight(adj) -> np.ndarray:
    """Beads down a tree from bead 0, a level at a time; branches side by side."""
    n = len(adj)
    parent, order, seen = {0: None}, [0], {0}
    for i in order:  # breadth first
        for j in sorted(adj[i]):
            if j not in seen:
                seen.add(j)
                parent[j] = i
                order.append(j)
    for k in range(n):  # a piece not connected to the head: hang it under the head
        if k not in seen:
            seen.add(k)
            parent[k] = 0
            order.append(k)
    children = {k: [j for j in order if parent.get(j) == k] for k in range(n)}
    zig = np.sqrt(BEAD**2 - LEVEL**2)
    xyz, level = np.zeros((n, 3)), np.zeros(n, int)
    for i in order[1:]:
        p = parent[i]
        sibs = children[p]
        level[i] = level[p] + 1
        offset = (sibs.index(i) - (len(sibs) - 1) / 2) * SPREAD
        xyz[i] = xyz[p] + [offset, zig * (1 if level[i] % 2 else -1) / 2, -LEVEL]
    return xyz


def _place(xyz, constraints, sites) -> np.ndarray:
    # beads the layout put on one spot have no direction to be pushed apart in
    xyz = xyz + np.random.default_rng(0).normal(0, 0.3, xyz.shape)
    for _ in range(500):  # constraints to their lengths, pair by pair
        worst = 0.0
        for i, j, d in constraints:
            v = xyz[j] - xyz[i]
            r = np.linalg.norm(v)
            worst = max(worst, abs(r - d))
            corr = (r - d) / 2 * v / max(r, 1e-6)
            xyz[i] += corr
            xyz[j] -= corr
        if worst < 1e-4:
            break
    for site, parents, (kind, c) in sites:
        p = xyz[parents]
        if kind == "average":
            xyz[site] = c @ p
        elif kind == "lc2":
            xyz[site] = (1 - c) * p[0] + c * p[1]
        else:
            rij, rik = p[1] - p[0], p[2] - p[0]
            xyz[site] = p[0] + c[0] * rij + c[1] * rik
            if kind == "out3":
                xyz[site] += c[2] / 10 * np.cross(rij, rik)  # c per nm; Å here
    # the long axis along z, the head up, the deepest bead at z = 0
    centered = xyz - xyz.mean(0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    if centered[0] @ axis < 0:
        axis = -axis
    z = np.array([0.0, 0.0, 1.0])
    v, cos = np.cross(axis, z), float(axis @ z)
    if np.linalg.norm(v) > 1e-9:
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + k + k @ k / (1 + cos)
    else:
        rot = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    out = centered @ rot.T
    return out - [0.0, 0.0, out[:, 2].min()]


def _counts(fractions: dict, total: int) -> list[str]:
    """``total`` lipid names in the ratios of ``fractions``, largest remainders first."""
    names = list(fractions)
    w = np.array([float(fractions[k]) for k in names])
    exact = w / w.sum() * total
    n = np.floor(exact).astype(int)
    for k in np.argsort(-(exact - n), kind="stable")[: total - n.sum()]:
        n[k] += 1
    return [name for name, c in zip(names, n, strict=True) for _ in range(c)]


def bilayer(lipid_itps, upper: dict, lower: dict | None = None, size=100.0,
            area_per_lipid: float = 60.0, water: float = 25.0, salt: float = 0.15,
            protein: Martinized | None = None, protein_origin: bool = False,
            protein_shift: float = 0.0, martini: int = 3,
            lipid_clash: float = 4.5, seed: int = 0) -> Martinized:  # fmt: skip
    """A Martini lipid bilayer in water, optionally around proteins, as insane builds one.

    ``upper``/``lower``: lipid name -> share in each leaflet (``lower`` is
    ``upper`` by default), for molecule types in ``lipid_itps`` (Martini's
    lipid and sterol topologies).  The membrane is ``size`` Å square (or
    (x, y)) with lipids on the lattice nearest ``area_per_lipid`` Å² per
    lipid (counts in the ratios given, rounded), lies in the xy plane with
    its midplane at the box's center, and has ``water`` Å of water beyond
    its lipids on each side (and beyond the protein).  Each lipid is a
    straight template from its topology, turned about z the way, of 12 tried
    at random, that keeps it farthest from the lipids already around it.

    ``protein``: a martinized system (from :func:`boonza.martinize`),
    oriented with the membrane normal along z.  Its center goes to the
    middle of the box; along z, its center goes to the midplane, or with
    ``protein_origin`` its z = 0 does (as OPM orients structures), then it
    moves by ``protein_shift`` Å.  Lipids with a bead within
    ``lipid_clash`` Å of a protein bead are left out.  Water and ions
    are then added as :func:`boonza.martini.solvate` adds them.

    ``martini``: which Martini the lipid topologies are, 3 or 2.  It picks
    the water and ions written beside them and the parameter file the
    topology includes; it does not change how the bilayer is built, which
    reads every bead and bond from ``lipid_itps``.
    """
    from . import VERSIONS

    if martini not in VERSIONS:
        raise ValueError(f"martini must be one of {', '.join(map(str, VERSIONS))}, not {martini!r}")
    if protein is not None and protein.martini != martini:
        raise ValueError(f"the protein is Martini {protein.martini}, the membrane Martini "
                         f"{martini}; they cannot be mixed in one system")  # fmt: skip
    from ..spatial import min_dist2, pairs_within

    lower = upper if lower is None else lower
    templates = lipid_templates(lipid_itps, set(upper) | set(lower))
    rng = np.random.default_rng(seed)
    lx, ly = np.broadcast_to(np.asarray(size, float), (2,))
    per_row_x = max(1, round(lx / np.sqrt(area_per_lipid)))
    per_row_y = max(1, round(lx * ly / area_per_lipid / per_row_x))
    gx, gy = lx / per_row_x, ly / per_row_y
    sites = np.array([((i + 0.5) * gx, (j + 0.25 + 0.5 * (i % 2)) * gy)
                      for i in range(per_row_x) for j in range(per_row_y)])  # fmt: skip
    reach = max(t.xyz[:, 2].max() for t in templates.values()) + CORE  # from the midplane
    prot = None
    if protein is not None:
        if protein.solvent:
            raise ValueError("the protein is already solvated")
        prot = np.asarray(protein.positions, float)
        center = prot.mean(0)
        if protein_origin:
            center[2] = 0.0
        prot = prot - center + [lx / 2, ly / 2, protein_shift]  # midplane at z = 0 for now
        reach = max(reach, float(np.abs(prot[:, 2]).max()))
    height = 2 * (reach + water)
    cell = np.diag([lx, ly, height])
    mid = height / 2
    if prot is not None:
        prot[:, 2] += mid

    # grid sites within two spacings of each other, across the periodic edges
    dxy = sites[:, None] - sites[None]
    dxy -= np.round(dxy / [lx, ly]) * [lx, ly]
    near = [np.flatnonzero(row) for row in (np.hypot(dxy[..., 0], dxy[..., 1])
                                            < 2.1 * max(gx, gy))]  # fmt: skip
    box = np.array([lx, ly, height])
    leaflets, occupied = [], {}  # (leaflet, site) -> beads
    for leaf, (sign, fractions) in enumerate(((1, upper), (-1, lower))):
        names = _counts(fractions, len(sites))
        rng.shuffle(names)
        placed = []
        for k, (name, (x, y)) in enumerate(zip(names, sites, strict=True)):
            t = templates[name]
            others = [occupied[key] for key in ((f, q) for f in (0, 1) for q in near[k])
                      if key in occupied]  # fmt: skip
            others = np.vstack(others) if others else np.zeros((0, 3))
            best, best_d = None, -1.0
            for a in rng.uniform(0, 2 * np.pi, ROTATIONS):  # the turn clearing the neighbors most
                rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                xyz = t.xyz @ rot.T
                xyz[:, 2] += CORE
                xyz[:, 2] *= sign  # the lower leaflet upside down
                xyz += [x, y, mid]
                if not len(others):
                    best = xyz
                    break
                d = xyz[:, None] - others[None]
                d -= np.round(d / box) * box
                closest = float(np.sqrt((d**2).sum(-1).min()))
                if closest > best_d:
                    best, best_d = xyz, closest
            if prot is not None and (min_dist2(best, prot, lipid_clash, cell=cell)
                                     < lipid_clash**2).any():  # fmt: skip
                continue
            occupied[(leaf, k)] = best
            placed.append((name, best))
        leaflets.append(placed)

    lipids = [xyz for leaflet in leaflets for _, xyz in leaflet]
    solute = np.vstack(([prot] if prot is not None else []) + lipids)
    top_z, bottom_z = max(x[:, 2].max() for x in lipids), min(x[:, 2].min() for x in lipids)
    w = _tile(np.array([lx, ly, height]))
    w = w[(w[:, 2] > top_z) | (w[:, 2] < bottom_z)]  # no water inside the membrane
    w = w[min_dist2(w, solute, CLASH, cell=cell) > CLASH**2]
    i, j, _ = pairs_within(w, SEAM, cell=cell)
    gone = np.zeros(len(w), bool)
    for a, b in zip(i.tolist(), j.tolist(), strict=True):
        if not gone[a] and not gone[b]:
            gone[b] = True
    w = w[~gone]

    charge = 0.0 if protein is None else sum(float(n["charge"]) for mol in protein.molecules
                                             for n in mol.nodes)  # fmt: skip
    charge += sum(templates[name].charge for leaflet in leaflets for name, _ in leaflet)
    net = round(charge)
    if abs(charge - net) > 1e-6:
        raise ValueError(f"the system carries a charge of {charge:g}, not a whole number")
    pairs = int(salt / 55.345 * (4 * len(w) - abs(net))) if salt > 0 else 0
    n_na, n_cl = pairs + max(-net, 0), pairs + max(net, 0)
    far = np.flatnonzero(min_dist2(w, solute, 5.0, cell=cell) > 25.0)
    if len(far) < n_na + n_cl:
        raise ValueError(f"not enough water for {n_na + n_cl} ions")
    picked = rng.permutation(far)[: n_na + n_cl]
    na, cl = w[picked[:n_na]], w[picked[n_na:]]
    w = np.delete(w, picked, axis=0)

    out = copy.deepcopy(protein) if protein is not None else Martinized([], np.zeros((0, 3)),
                                                                        None)  # fmt: skip
    if protein is not None:
        shift = (prot.mean(0) - np.asarray(protein.positions).mean(0)) / 10  # nm
        for mol in out.molecules:
            mol.positions = [np.asarray(p) + shift for p in mol.positions]
    blocks, groups = [] if prot is None else [prot], []
    for leaflet in leaflets:  # lipids of one kind together, leaflet by leaflet
        for name in dict.fromkeys(n for n, _ in leaflet):
            same = [xyz for n, xyz in leaflet if n == name]
            blocks += same
            groups.append((name, len(same), templates[name].names))
    out.positions = np.vstack([*blocks, w, na, cl])
    out.cell = cell
    out.lipids = groups
    out.includes = [Path(p).resolve() for p in lipid_itps]
    out.solvent = [("W", len(w)), ("NA", len(na)), ("CL", len(cl))]
    out.martini = martini
    return out
