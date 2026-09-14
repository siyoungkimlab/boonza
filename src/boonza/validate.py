"""Sanity checks for chemical systems, after msys dms-validate and dms-find-knot.

    problems = boonza.validate(system)               # basic checks
    problems = boonza.validate(system, strict=True)  # plus simulation-readiness checks
    for p in problems:
        print(p)
    knots = boonza.find_knots(system)                # bonds threaded through rings

Basic checks (msys "basic" plus force-field consistency): every particle
has nonbonded parameters, every bond between real atoms has a stretch or
constraint term, every molecule has an integer net charge (virtual sites
count with their host), no bond passes through a ring of at most 10 atoms,
the cell has positive volume, real atoms have mass, and no virtual site sits
in two virtual tables.  Strict checks add: hydrogens are constrained, the
cell is orthorhombic, no unbonded atoms are closer than 1 Å (periodic),
atoms of one element share one mass, every pair within three bonds is
excluded and no excluded pair is farther apart, bonded terms connect bonded
atoms, and each water has its own residue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .elements import msys_symbol
from .rings import sssr
from .spatial import pairs_within, within

_SHOW = 10  # atoms or items quoted per message


@dataclass(frozen=True)
class Problem:
    """One failed check: its name, a message and the atoms involved."""

    check: str
    message: str
    atoms: tuple[int, ...] = ()

    def __str__(self) -> str:
        return f"{self.check}: {self.message}"


def _quote(items) -> str:
    items = list(items)
    text = " ".join(map(str, items[:_SHOW]))
    return text + (" ..." if len(items) > _SHOW else "")


# ---------------------------------------------------------------------------
# knots


@njit(cache=True)
def _cross(a0, a1, a2, b0, b1, b2):
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


@njit(cache=True)
def _segments_hit(r, s, a, b, c):
    """For each segment r[k]-s[k], whether it crosses triangle abc (msys line_intersects_tri)."""
    out = np.zeros(len(r), np.bool_)
    n0, n1, n2 = _cross(b[0] - a[0], b[1] - a[1], b[2] - a[2],
                        c[0] - b[0], c[1] - b[1], c[2] - b[2])  # fmt: skip
    tri = (a, b, c)
    for k in range(len(r)):
        rs0, rs1, rs2 = s[k, 0] - r[k, 0], s[k, 1] - r[k, 1], s[k, 2] - r[k, 2]
        nd = n0 * rs0 + n1 * rs1 + n2 * rs2
        if nd == 0:  # parallel to the triangle
            continue
        t = -(n0 * (r[k, 0] - a[0]) + n1 * (r[k, 1] - a[1]) + n2 * (r[k, 2] - a[2])) / nd
        if t <= 0 or t >= 1:  # crossing beyond the segment ends
            continue
        i0, i1, i2 = r[k, 0] + rs0 * t, r[k, 1] + rs1 * t, r[k, 2] + rs2 * t
        neg = 0
        ok = True
        for e in range(3):
            t0, t1 = tri[e], tri[(e + 1) % 3]
            c0, c1, c2 = _cross(t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2],
                                i0 - t0[0], i1 - t0[1], i2 - t0[2])  # fmt: skip
            area = c0 * n0 + c1 * n1 + c2 * n2
            if area == 0:
                ok = False
                break
            neg += area < 0
        out[k] = ok and (neg == 0 or neg == 3)
    return out


def line_intersects_triangle(r, s, a, b, c) -> bool:
    """True if segment rs crosses triangle abc."""
    seg = np.asarray(r, dtype=np.float64).reshape(1, 3)
    end = np.asarray(s, dtype=np.float64).reshape(1, 3)
    tri = [np.asarray(x, dtype=np.float64).reshape(3) for x in (a, b, c)]
    return bool(_segments_hit(seg, end, *tri)[0])


def _msys_wrap(system):
    """Make molecules whole and put their centers in the box centered at the origin.

    This is msys ``Wrapper(system).wrap()`` (pfx without a center): each
    molecule moves by whole box vectors so its center has fractional
    coordinates in [-0.5, 0.5].
    """
    from .glue import Glue

    glue = Glue(system, wrap=False)
    frag = system.fragids
    nfrag = int(frag.max()) + 1 if len(frag) else 0
    size = np.bincount(frag, minlength=nfrag)

    def wrap(pos, box):
        pos, _ = glue(pos, box)
        if nfrag and box.any():
            centers = np.column_stack([np.bincount(frag, pos[:, d], nfrag) for d in range(3)])
            shift = -np.rint((centers / size[:, None]) @ np.linalg.inv(box))
            pos = pos + (shift @ box)[frag]
        return pos

    return wrap


def find_knots(system, max_cycle_size=None, selection="all", ignore_excluded_knots=False,
               positions=None, cell=None) -> list[tuple]:  # fmt: skip
    """Bonds passing through rings, as msys dms-find-knot.

    Returns ``(ring, bond, idx)`` tuples: ``bond`` (i, j) crosses the
    triangle ring[0], ring[idx], ring[idx + 1].  Rings come from ``sssr``
    over the selected atoms, limited to ``max_cycle_size`` atoms; bonds
    within 10 Å of a ring are tested.  The search is repeated with the
    system shifted by half a box to catch knots across periodic boundaries.
    ``ignore_excluded_knots`` skips bonds whose atoms are excluded from every
    ring atom (needs an exclusion table).
    """
    s = system
    excluded = None
    if ignore_excluded_knots:
        t = s.tables.get("exclusion")
        if t is None:
            raise ValueError("cannot ignore excluded knots without an exclusion table")
        excluded = set(map(tuple, np.sort(t.atoms, axis=1).tolist()))
    ids = np.arange(s.natoms) if selection in (None, "all") else s.select(selection).ids
    rings = sssr(s, ids)
    if max_cycle_size is not None:
        rings = [r for r in rings if len(r) <= int(max_cycle_size)]
    if not rings:
        return []
    pos = np.array(s.positions if positions is None else positions, dtype=np.float64)
    box = np.array(s.cell if cell is None else cell, dtype=np.float64).reshape(3, 3)
    wrap = _msys_wrap(s)
    bi, bj = s.bonds["i"], s.bonds["j"]
    hi, lo = np.maximum(bi, bj), np.minimum(bi, bj)
    results, found = [], set()
    for shift in (False, True):
        if shift:
            pos = pos + 0.5 * box.sum(axis=0)
        pos = wrap(pos, box)
        for ring in rings:
            r = np.asarray(ring, dtype=np.int64)
            near = np.zeros(s.natoms, bool)
            near[ids] = within(pos[ids], pos[r], 10.0)
            near[r] = False
            cand = np.flatnonzero(near[hi] & near[lo])
            if not len(cand):
                continue
            ph, pl, cp = pos[hi[cand]], pos[lo[cand]], pos[r]
            for idx in range(1, len(r) - 1):
                for k in np.flatnonzero(_segments_hit(ph, pl, cp[0], cp[idx], cp[idx + 1])):
                    ai, aj = int(hi[cand[k]]), int(lo[cand[k]])
                    if (ai, aj, idx) in found:
                        continue
                    if excluded is not None and all(
                        (min(x, y), max(x, y)) in excluded for x in ring for y in (ai, aj)
                    ):
                        continue
                    results.append((tuple(ring), (ai, aj), idx))
                    found.add((ai, aj, idx))
    return results


# ---------------------------------------------------------------------------
# checks


@njit(cache=True)
def _pairs_within_bonds(n, off, nbr, maxd):
    """Pairs i < j separated by 1..maxd bonds."""
    mark = np.full(n, -1, np.int64)
    depth = np.zeros(n, np.int64)
    queue = np.empty(max(n, 1), np.int64)
    out_i = np.empty(0, np.int64)
    out_j = np.empty(0, np.int64)
    count = 0
    for fill in range(2):
        if fill == 1:
            out_i = np.empty(count, np.int64)
            out_j = np.empty(count, np.int64)
            count = 0
        for i in range(n):
            stamp = fill * n + i
            mark[i] = stamp
            depth[i] = 0
            queue[0] = i
            head, tail = 0, 1
            while head < tail:
                v = queue[head]
                head += 1
                if depth[v] == maxd:
                    continue
                for k in range(off[v], off[v + 1]):
                    w = nbr[k]
                    if mark[w] != stamp:
                        mark[w] = stamp
                        depth[w] = depth[v] + 1
                        queue[tail] = w
                        tail += 1
                        if w > i:
                            if fill == 1:
                                out_i[count] = i
                                out_j[count] = w
                            count += 1
    return out_i, out_j


def _pair_keys(a, b, n) -> np.ndarray:
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    return np.minimum(a, b) * n + np.maximum(a, b)


def _covered_bonds(s) -> np.ndarray:
    """Keys of atom pairs held by a stretch or constraint term."""
    n, keys = s.natoms, []
    for name, t in s.tables.items():
        if not len(t):
            continue
        a = t.atoms
        if t.category == "bond" and t.natoms == 2 and "stretch" in name:
            keys.append(_pair_keys(a[:, 0], a[:, 1], n))
        elif t.category == "constraint":
            if name.startswith("constraint_ah") and not name.endswith("R"):
                pairs = [(0, k) for k in range(1, t.natoms)]
            elif name == "constraint_hoh":
                pairs = [(0, 1), (0, 2)]
            else:  # rigid groups: every pair
                pairs = [(x, y) for x in range(t.natoms) for y in range(x + 1, t.natoms)]
            keys += [_pair_keys(a[:, x], a[:, y], n) for x, y in pairs]
    return np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)


def _consistency(s) -> list[Problem]:
    """Bonds without a stretch/constraint term and molecules with a non-integer charge."""
    from .exclusions import _hosts

    out = []
    n, anum = s.natoms, s.atoms["anum"]
    if s.tables:
        bi, bj = s.bonds["i"], s.bonds["j"]
        real = (anum[bi] > 0) & (anum[bj] > 0)
        bi, bj = bi[real], bj[real]
        missing = ~np.isin(_pair_keys(bi, bj, n), _covered_bonds(s))
        if missing.any():
            pairs = np.column_stack([bi[missing], bj[missing]])
            text = (f"{len(pairs)} bonds have no stretch or constraint term: "
                    f"{_quote(f'{a}-{b}' for a, b in pairs.tolist())}")  # fmt: skip
            out.append(Problem("stretch", text, tuple(np.unique(pairs).tolist())))
    frag = s.fragids.copy()
    host = _hosts(s)
    sites = np.flatnonzero((anum == 0) & (host >= 0))
    frag[sites] = frag[host[sites]]
    charge = s.atoms["charge"].astype(np.float64)
    net = np.bincount(frag, weights=charge, minlength=s.nfragments)
    bad = np.flatnonzero(np.abs(net - np.round(net)) > 1e-3)
    if len(bad):
        text = (f"{len(bad)} molecules have a non-integer net charge: "
                f"{_quote(f'fragment {f} ({net[f]:+.4f})' for f in bad.tolist())}")  # fmt: skip
        out.append(Problem("charge", text, tuple(np.flatnonzero(np.isin(frag, bad)).tolist())))
    return out


def _term_topology(s) -> list[Problem]:
    """Bonded terms whose atoms are not connected as the term requires."""
    n = s.natoms
    bonded = np.unique(_pair_keys(s.bonds["i"], s.bonds["j"], n))
    off, nbr, _ = s._adjacency()
    ni, nj = _pairs_within_bonds(n, off, nbr, 2)
    near2 = np.unique(_pair_keys(ni, nj, n))  # 1-2 and 1-3 (Urey-Bradley, H-H constraints)

    def b(a, x, y):
        return np.isin(_pair_keys(a[:, x], a[:, y], n), bonded)

    def path(a, first):
        return b(a, first, first + 1) & b(a, first + 1, first + 2) & b(a, first + 2, first + 3)

    out = []
    for name, t in sorted(s.tables.items()):
        if t.category not in ("bond", "constraint") or not len(t) or name.startswith("pair"):
            continue
        a = t.atoms
        if t.category == "constraint" or ("stretch" in name and t.natoms == 2):
            partners = range(1, t.natoms) if t.category == "constraint" else [1]
            ok = np.ones(len(t), bool)
            for k in partners:
                ok &= np.isin(_pair_keys(a[:, 0], a[:, k], n), near2)
        elif name.startswith(("angle", "alchemical_angle")) and t.natoms == 3:
            ok = b(a, 0, 1) & b(a, 1, 2)
        elif t.natoms == 4 and ("dihedral" in name or "improper" in name):
            center = np.zeros(len(t), bool)  # one atom bonded to the other three
            for c in range(4):
                others = [k for k in range(4) if k != c]
                center |= b(a, c, others[0]) & b(a, c, others[1]) & b(a, c, others[2])
            ok = path(a, 0) | center
        elif "cmap" in name and t.natoms == 8:
            ok = path(a, 0) & path(a, 4)
        else:
            continue
        wrong = np.flatnonzero(~ok)
        if len(wrong):
            where = _quote(tuple(a[k].tolist()) for k in wrong.tolist())
            text = (f"{name}: {len(wrong)} terms whose atoms are not bonded as the term "
                    f"requires: {where}")  # fmt: skip
            out.append(Problem("term_topology", text, tuple(np.unique(a[wrong]).tolist())))
    return out


def _extra_exclusions(s) -> list[Problem]:
    """Excluded pairs more than three bonds apart (virtual sites counted at their host)."""
    from .exclusions import _hosts, _real_csr

    excl = s.tables.get("exclusion")
    if excl is None or not len(excl):
        return []
    n = s.natoms
    host = _hosts(s)
    a = excl.atoms
    ha, hb = host[a[:, 0]], host[a[:, 1]]
    roff, rnbr = _real_csr(s)
    wi, wj = _pairs_within_bonds(n, roff, rnbr, 3)
    far = (ha >= 0) & (hb >= 0) & (ha != hb)
    far &= ~np.isin(_pair_keys(ha, hb, n), _pair_keys(wi, wj, n))
    far = np.flatnonzero(far)
    if not len(far):
        return []
    text = (f"{len(far)} excluded pairs are more than three bonds apart: "
            f"{_quote(f'{x}-{y}' for x, y in a[far].tolist())}")  # fmt: skip
    return [Problem("extra_exclusions", text, tuple(np.unique(a[far]).tolist()))]


def _basic(s, max_ring) -> list[Problem]:
    out = []
    n = s.natoms
    anum, mass = s.atoms["anum"], s.atoms["mass"]
    nb = s.tables.get("nonbonded")
    if s.tables and nb is None:
        out.append(Problem("nonbonded", "the system has force-field tables but no nonbonded table"))
    if nb is not None:
        have = np.zeros(n, bool)
        have[nb.atoms[nb.param_ids >= 0, 0]] = True
        missing = np.flatnonzero(~have)
        if len(missing):
            text = f"{len(missing)} particles have no nonbonded parameters: {_quote(missing)}"
            out.append(Problem("nonbonded", text, tuple(missing.tolist())))
    out += _consistency(s)
    knots = find_knots(s, max_ring, "atomicnumber > 0",
                       ignore_excluded_knots="exclusion" in s.tables)  # fmt: skip
    if knots:
        where = [f"bond {b[0]}-{b[1]} through ring {'-'.join(map(str, r))}" for r, b, _ in knots]
        atoms = sorted({a for _, b, _ in knots for a in b})
        out.append(Problem("knot", f"{len(knots)} bonds pass through rings of at most "
                           f"{max_ring} atoms: {_quote(where)}", tuple(atoms)))  # fmt: skip
    volume = float(np.linalg.det(s.cell))
    if not volume > 0:
        out.append(Problem("box", f"the cell must have positive volume, got {volume:g}"))
    massless = np.flatnonzero((anum > 0) & (mass == 0))
    if len(massless):
        out.append(Problem("mass", f"{len(massless)} atoms with atomic number > 0 have no mass: "
                           f"{_quote(massless)}", tuple(massless.tolist())))  # fmt: skip
    pseudo = anum == 0
    tables_of = np.zeros(n, np.int64)
    for name, t in s.tables.items():
        if name.startswith("virtual_") and len(t):
            touched = np.zeros(n, bool)
            touched[t.atoms.ravel()] = True
            tables_of += touched & pseudo
    multi = np.flatnonzero(tables_of > 1)
    if len(multi):
        out.append(Problem("virtual", f"{len(multi)} virtual sites belong to several virtual "
                           f"tables: {_quote(multi)}", tuple(multi.tolist())))  # fmt: skip
    groups = s.atoms["interaction_grp"] if "interaction_grp" in s.atoms.props else None
    if groups is not None or "modified_interaction" in s.aux_tables:
        used = set() if groups is None else set(np.unique(groups).tolist()) - {""}
        tab = s.aux_tables.get("modified_interaction")
        named = set() if tab is None else set(tab["g0"].tolist()) | set(tab["g1"].tolist())
        if used != named:
            only_atoms, only_table = sorted(used - named), sorted(named - used)
            text = (f"interaction_grp names and the modified_interaction table disagree: "
                    f"only in atoms {only_atoms}, only in table {only_table}")  # fmt: skip
            out.append(Problem("modified_interaction", text))
    return out


def _strict(s) -> list[Problem]:
    out = []
    n = s.natoms
    anum, mass = s.atoms["anum"], s.atoms["mass"]
    hydrogens = anum == 1
    constraint_tables = [t for t in s.tables.values() if t.category == "constraint"]
    if hydrogens.any():
        if not sum(len(t) for t in constraint_tables):
            out.append(Problem("constraints", "the system has hydrogens but no constraint terms"))
        constrained = np.zeros(n, bool)
        for t in constraint_tables:
            constrained[t.atoms.ravel()] = True
        free = np.flatnonzero(hydrogens & ~constrained)
        if len(free):
            out.append(Problem("constrained_hydrogens", f"{len(free)} hydrogens are not "
                               f"constrained: {_quote(free)}", tuple(free.tolist())))  # fmt: skip
    cell = s.cell
    if (cell != np.diag(np.diag(cell))).any():
        out.append(Problem("cell", "the unit cell is not diagonal"))
    periodic = cell if np.linalg.det(cell) > 0 else None
    i, j, d2 = pairs_within(s.positions, 1.0, periodic)
    if len(i):
        bonded = set(zip(np.minimum(s.bonds["i"], s.bonds["j"]).tolist(),
                         np.maximum(s.bonds["i"], s.bonds["j"]).tolist(), strict=True))  # fmt: skip
        close = [(a, b, float(np.sqrt(d))) for a, b, d in zip(i.tolist(), j.tolist(), d2.tolist(),
                                                                strict=True)
                 if (a, b) not in bonded]  # fmt: skip
        if close:
            pairs = _quote(f"{a}-{b} ({d:.3f})" for a, b, d in close)
            atoms = tuple(sorted({x for a, b, _ in close for x in (a, b)}))
            text = f"{len(close)} unbonded atom pairs are within 1 Å: {pairs}"
            out.append(Problem("contacts", text, atoms))
    real = anum > 0
    for z in np.unique(anum[real]).tolist():
        values = np.unique(mass[anum == z])
        if len(values) > 1:
            out.append(Problem("masses", f"{msys_symbol(z)} atoms have {len(values)} different "
                               f"masses: {_quote(values.tolist())}"))  # fmt: skip
    excl = s.tables.get("exclusion")
    if excl is not None:
        off, nbr, _ = s._adjacency()
        pi, pj = _pairs_within_bonds(n, off, nbr, 3)
        have = set(map(tuple, np.sort(excl.atoms, axis=1).tolist()))
        missing = [(a, b) for a, b in zip(pi.tolist(), pj.tolist(), strict=True)
                   if (a, b) not in have]  # fmt: skip
        if missing:
            out.append(Problem("exclusions", f"{len(missing)} atom pairs within three bonds are "
                               f"not excluded: {_quote(f'{a}-{b}' for a, b in missing)}",
                               tuple(sorted({x for p in missing for x in p}))))  # fmt: skip
    out += _extra_exclusions(s)
    out += _term_topology(s)
    waters = s.select("water").ids
    if len(waters):
        frag, res = s.fragids[waters], s.atoms["residue"][waters]
        pairs = np.unique(np.column_stack([res, frag]), axis=0)
        shared = np.unique(pairs[:, 0])[np.bincount(pairs[:, 0])[np.unique(pairs[:, 0])] > 1]
        if len(shared):
            out.append(Problem("waters", f"{len(shared)} residues hold more than one water "
                               f"molecule: {_quote(shared)}"))  # fmt: skip
    return out


def validate(system, strict: bool = False, max_ring: int = 10) -> list[Problem]:
    """Problems found in ``system`` (empty when it passes); see the module docstring."""
    problems = _basic(system, max_ring)
    if strict:
        problems += _strict(system)
    return problems
