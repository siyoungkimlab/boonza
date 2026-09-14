"""Exclusions and scaled 1-4 pairs regenerated from the bond graph.

    boonza.update_exclusions(system)                           # exclusions only
    boonza.update_exclusions(system, pair_scales=(0.5, 1 / 1.2))  # plus Amber 1-4 pairs

Atom pairs up to ``separation`` bonds apart (1-2, 1-3 and 1-4 by default)
are excluded from the nonbonded interaction.  Pseudo particles (virtual
sites, lone pairs, Drude-like particles) are excluded like the atom that
hosts them: their first parent in a ``virtual_*`` table, otherwise their
bonded real atom.  With ``pair_scales=(lj, es)`` the pairs exactly
``separation`` bonds apart (by shortest path) get ``pair_12_6_es`` terms
holding ``lj`` times their combined Lennard-Jones interaction and ``es``
times their Coulomb interaction; force fields with special 1-4
Lennard-Jones parameters (CHARMM) need those written separately.
"""

from __future__ import annotations

import math

import numpy as np

from .validate import _pairs_within_bonds


def _real_csr(system):
    """CSR adjacency of the bond graph restricted to real atoms."""
    n = system.natoms
    anum = system.atoms["anum"]
    bi, bj = system.bonds["i"], system.bonds["j"]
    keep = (anum[bi] > 0) & (anum[bj] > 0)
    src = np.concatenate([bi[keep], bj[keep]])
    dst = np.concatenate([bj[keep], bi[keep]])
    order = np.argsort(src, kind="stable")
    off = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=off[1:])
    return off, dst[order].astype(np.int64)


def _hosts(system) -> np.ndarray:
    """The real atom each atom is excluded like (itself for real atoms, -1 if none)."""
    anum = system.atoms["anum"]
    host = np.where(anum > 0, np.arange(system.natoms), -1)
    for name, t in system.tables.items():
        if name.startswith("virtual_") and len(t):
            host[t.atoms[:, 0]] = t.atoms[:, 1]
    free = np.flatnonzero(host < 0)
    if len(free):
        bi, bj = system.bonds["i"], system.bonds["j"]
        for a, b in ((bi, bj), (bj, bi)):
            real = anum[b] > 0
            hit = np.isin(a, free) & real
            host[a[hit]] = b[hit]
    # a site hosted by another site takes that site's host
    for _ in range(3):
        nested = (host >= 0) & (anum == 0)
        nested[nested] = anum[host[nested]] == 0
        if not nested.any():
            break
        host[nested] = host[host[nested]]
    return host


def _expand(pairs_i, pairs_j, host, n):
    """Real-atom pairs, and every pair of atoms sharing a host, carried over to hosted sites."""
    members: dict[int, list[int]] = {}
    for a, h in enumerate(host.tolist()):
        if h >= 0:
            members.setdefault(h, []).append(a)
    out = set()
    if all(len(m) == 1 for m in members.values()):
        return set(zip(pairs_i.tolist(), pairs_j.tolist(), strict=True))
    for a, b in zip(pairs_i.tolist(), pairs_j.tolist(), strict=True):
        for x in members.get(a, (a,)):
            for y in members.get(b, (b,)):
                out.add((min(x, y), max(x, y)))
    for group in members.values():
        for k, x in enumerate(group):
            for y in group[k + 1 :]:
                out.add((min(x, y), max(x, y)))
    return out


def update_exclusions(system, separation: int = 3, pair_scales=None) -> None:
    """Replace the exclusion table (and, with ``pair_scales``, the 1-4 pairs) from the bonds."""
    s = system
    n = s.natoms
    off, nbr = _real_csr(s)
    host = _hosts(s)
    within_i, within_j = _pairs_within_bonds(n, off, nbr, int(separation))
    excluded = sorted(_expand(within_i, within_j, host, n))
    if "exclusion" in s.tables:
        s.del_table("exclusion")
    t = s.add_table_from_schema("exclusion")
    if excluded:
        t.add_terms(np.array(excluded, dtype=np.int64))
    if pair_scales is None:
        return
    lj_scale, es_scale = pair_scales
    near_i, near_j = _pairs_within_bonds(n, off, nbr, int(separation) - 1)
    near = set(zip(near_i.tolist(), near_j.tolist(), strict=True))
    ends = [(a, b) for a, b in zip(within_i.tolist(), within_j.tolist(), strict=True)
            if (a, b) not in near]  # fmt: skip
    if "pair_12_6_es" in s.tables:
        s.del_table("pair_12_6_es")
    pairs = s.add_table_from_schema("pair_12_6_es")
    if not ends:
        return
    nb = s.tables.get("nonbonded")
    rule = s.nonbonded_info.vdw_rule.lower()
    charge = s.atoms["charge"]
    ptype = np.full(n, -1, np.int64)
    if nb is not None:
        ptype[nb.atoms[:, 0]] = nb.param_ids
        sig, eps = nb.params["sigma"], nb.params["epsilon"]
    index: dict = {}
    atoms, pids = [], []
    for a, b in ends:
        aij = bij = 0.0
        if nb is not None and ptype[a] >= 0 and ptype[b] >= 0:
            pa, pb = int(ptype[a]), int(ptype[b])
            ov = nb.overrides.get(pa, pb)
            if ov is not None:
                sg, ep = float(ov["sigma"]), float(ov["epsilon"])
            else:
                sa, sb = float(sig[pa]), float(sig[pb])
                sg = math.sqrt(sa * sb) if rule == "geometric" else 0.5 * (sa + sb)
                ep = math.sqrt(float(eps[pa]) * float(eps[pb]))
            aij, bij = lj_scale * 4 * ep * sg**12, lj_scale * 4 * ep * sg**6
        qij = es_scale * float(charge[a]) * float(charge[b])
        key = (aij, bij, qij)
        pid = index.get(key)
        if pid is None:
            pid = index[key] = pairs.params.add_param(aij=aij, bij=bij, qij=qij)
        atoms.append((a, b))
        pids.append(pid)
    pairs.add_terms(np.array(atoms, dtype=np.int64), pids)
