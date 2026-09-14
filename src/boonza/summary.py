"""Plain-text summaries of structures, for people and language models.

    print(boonza.summarize(s))                                  # the whole system
    print(boonza.summarize(s, focus="resname HEM and chain A"))  # one site
    data = boonza.summarize(s).to_dict()                        # the same, as plain data

A language model reads text, not coordinates.  A summary states in words and
numbers what the 3D structure holds: what the system contains, each chain's
sequence and secondary structure, what is bound where and how tightly it
is surrounded, which residues touch across chains, and what looks wrong.
Every number is computed from the structure.  Without hydrogens, polar
contacts are reported as N/O pairs within 3.5 Å rather than as hydrogen
bonds, since the angles cannot be checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .elements import msys_symbol

POLAR_CONTACT = 3.5  # Å between N/O atoms
# nonmetals and metalloids; every other element counts as a metal
_NONMETALS = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36, 51, 52, 53, 54,
              85, 86}  # fmt: skip
CLASH = 2.2  # Å between nonbonded heavy atoms more than two bonds apart


@dataclass
class Summary:
    """A structure summary: ``str()`` for text (Markdown), ``to_dict()`` for data."""

    title: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        out = [f"# {self.title}"]
        for heading, lines in self.sections:
            out += ["", f"## {heading}", *(f"- {line}" for line in lines)]
        return "\n".join(out) + "\n"

    def _repr_markdown_(self) -> str:
        return str(self)

    def to_dict(self) -> dict:
        return {"title": self.title, **self.data}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# helpers


def _residue_label(s, r: int) -> str:
    chain = str(s.chains["name"][s.residues["chain"][r]]).strip()
    ins = str(s.residues["insertion"][r]).strip()
    return f"{s.residues['name'][r]} {chain}{int(s.residues['resid'][r])}{ins}"


def _atom_label(s, a: int) -> str:
    return f"{_residue_label(s, int(s.atoms['residue'][a]))} {s.atoms['name'][a]}"


def _formula(anum) -> str:
    """Hill formula: C, then H, then the others alphabetically."""
    counts: dict[str, int] = {}
    for z in np.asarray(anum).tolist():
        if z > 0:
            sym = msys_symbol(int(z))
            counts[sym] = counts.get(sym, 0) + 1
    order = [e for e in ("C", "H") if e in counts] if "C" in counts else []
    order += sorted(e for e in counts if e not in order)
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


def _mask(s, sel: str) -> np.ndarray:
    m = np.zeros(s.natoms, bool)
    try:
        m[s.select(sel).ids] = True
    except Exception:  # a keyword the selection engine does not know
        pass
    return m


def _box(s):
    return s.cell if s.cell.any() else None


def _pairs(s, a_ids, b_ids, cutoff):
    """(i of a, j of b, distance) within ``cutoff``, periodic when the system has a cell."""
    from .pbc import capped_distances

    if not len(a_ids) or not len(b_ids):
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0)
    pos = s.positions
    i, j, d = capped_distances(pos[a_ids], pos[b_ids], cutoff, _box(s))
    keep = d <= cutoff
    return a_ids[i[keep]], b_ids[j[keep]], d[keep]


def _kinds(s) -> dict[str, np.ndarray]:
    polymer = _mask(s, "protein") | _mask(s, "nucleic")
    water = _mask(s, "water")
    ions = _mask(s, "ions") & ~water & ~polymer
    other = ~(polymer | water | ions)
    return {"polymer": polymer, "protein": _mask(s, "protein"), "nucleic": _mask(s, "nucleic"),
            "water": water, "ions": ions, "other": other}  # fmt: skip


def _other_groups(s, kinds) -> list[np.ndarray]:
    """Ligands, cofactors and other non-polymer, non-water, non-ion molecules, as atom
    arrays.  Residues are joined only through covalent bonds between such residues:
    a cofactor bonded to a protein (a heme's iron to its histidine) and a ligand
    coordinated to a metal (O2 on the heme's iron) stay separate molecules, and
    those bonds are reported instead."""
    other = kinds["other"]
    res = s.atoms["residue"]
    residues = sorted(set(res[other].tolist()))
    parent = {r: r for r in residues}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    i, j = s.bonds["i"], s.bonds["j"]
    anum = s.atoms["anum"]
    metal = (anum > 0) & ~np.isin(anum, list(_NONMETALS))
    both = other[i] & other[j] & (res[i] != res[j]) & ~metal[i] & ~metal[j]
    for a, b in zip(res[i[both]].tolist(), res[j[both]].tolist(), strict=True):
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for r in residues:
        groups.setdefault(find(r), []).append(r)
    out = [np.flatnonzero(other & np.isin(res, members)) for members in groups.values()]
    return sorted(out, key=lambda ids: int(ids[0]))


def _has_polar_hydrogens(s) -> bool:
    anum = s.atoms["anum"]
    i, j = s.bonds["i"], s.bonds["j"]
    polar = np.isin(anum, (7, 8))
    return bool((((anum[i] == 1) & polar[j]) | ((anum[j] == 1) & polar[i])).any())


# ---------------------------------------------------------------------------
# sections


def _composition(s, kinds) -> tuple[list[str], dict]:
    from .io.pdb import lengths_angles_from_cell

    frag = np.asarray(s.fragids)
    lines = [f"{s.natoms} atoms in {s.nresidues} residues, {s.nchains} chains and "
             f"{s.nfragments} molecules"]  # fmt: skip
    data: dict = {"atoms": s.natoms, "residues": s.nresidues, "chains": s.nchains,
                  "molecules": s.nfragments}  # fmt: skip
    if s.cell.any():
        a, b, c, al, be, ga = lengths_angles_from_cell(s.cell)
        lines.append(f"periodic box {a:.1f} x {b:.1f} x {c:.1f} Å "
                     f"(angles {al:.0f}, {be:.0f}, {ga:.0f})")  # fmt: skip
        data["box"] = [round(x, 3) for x in (a, b, c, al, be, ga)]
    polymer_chains = sorted(set(s.residues["chain"][s.atoms["residue"][kinds["polymer"]]].tolist()))
    nwater = len(set(frag[kinds["water"]].tolist()))
    if polymer_chains:
        names = [str(s.chains["name"][c]) or "(unnamed)" for c in polymer_chains]
        lines.append(f"{len(polymer_chains)} polymer chain(s): " + ", ".join(names))
    if nwater:
        lines.append(f"{nwater} water molecules")
    ions: dict[str, int] = {}
    for a in np.flatnonzero(kinds["ions"]).tolist():
        q = int(s.atoms["formal_charge"][a])
        key = msys_symbol(int(s.atoms["anum"][a])) + ("+" * q if q > 0 else "-" * -q)
        ions[key] = ions.get(key, 0) + 1
    if ions:
        lines.append("ions: " + ", ".join(f"{k} x {v}" for k, v in sorted(ions.items())))
    other = {}
    for ids in _other_groups(s, kinds):
        names = sorted(set(s.residues["name"][s.atoms["residue"][ids]].tolist()))
        key = "+".join(names) or "(unnamed)"
        other[key] = other.get(key, 0) + 1
    if other:
        lines.append("other molecules: " + ", ".join(f"{k} x {v}" for k, v in other.items()))
    qf = int(s.atoms["formal_charge"].sum())
    lines.append(f"net formal charge {qf:+d}")
    data.update(polymer_chains=[str(s.chains["name"][c]) for c in polymer_chains],
                waters=nwater, ions=ions, other_molecules=other, formal_charge=qf)  # fmt: skip
    if "nonbonded" in s.tables:
        q = float(s.atoms["charge"].sum())
        lines.append(f"force field: {len(s.tables)} tables ({', '.join(sorted(s.tables))}); "
                     f"net partial charge {q:+.3f}")  # fmt: skip
        data["force_field"] = sorted(s.tables)
        data["partial_charge"] = round(q, 4)
    else:
        lines.append("no force-field parameters")
    return lines, data


def _chains(s, kinds, max_items) -> tuple[list[str], list[dict]]:
    from .secondary import dssp
    from .sequence import sequence

    lines, data = [], []
    try:
        ss = dssp(s, simplified=True)[0]
    except Exception:
        ss = np.full(s.nresidues, "NA")
    res_chain = s.residues["chain"]
    chains = sorted(set(res_chain[s.atoms["residue"][kinds["polymer"]]].tolist()))
    for c in chains[:max_items]:
        try:
            seq = sequence(s, chain=int(c))
        except Exception:
            seq = ""
        rows = np.flatnonzero(res_chain == c)
        polymer_rows = [r for r in rows.tolist() if kinds["polymer"][s.atoms["residue"] == r].any()]
        if not polymer_rows:
            continue
        name = str(s.chains["name"][c]) or f"#{c}"
        first, last = polymer_rows[0], polymer_rows[-1]
        codes = [str(ss[r]) for r in polymer_rows if str(ss[r]) != "NA"]
        entry = {"chain": name, "residues": len(polymer_rows),
                 "first": _residue_label(s, first), "last": _residue_label(s, last),
                 "sequence": seq}  # fmt: skip
        lines.append(f"chain {name}: {len(polymer_rows)} residues "
                     f"({_residue_label(s, first)} to {_residue_label(s, last)})")  # fmt: skip
        if seq:
            lines.append(f"chain {name} sequence: {seq}")
        if codes:
            helix = codes.count("H") / len(codes)
            strand = codes.count("E") / len(codes)
            string = "".join(codes)
            lines.append(f"chain {name} secondary structure (DSSP; H helix, E strand, C coil): "
                         f"{helix:.0%} helix, {strand:.0%} strand: {string}")  # fmt: skip
            entry.update(helix=round(helix, 3), strand=round(strand, 3), dssp=string)
        breaks = _chain_breaks(s, polymer_rows)
        if breaks:
            lines.append(f"chain {name} breaks (C-N over 2.5 Å) after: "
                         + ", ".join(_residue_label(s, r) for r in breaks[:max_items]))  # fmt: skip
        entry["breaks"] = [_residue_label(s, r) for r in breaks]
        data.append(entry)
    if len(chains) > max_items:
        lines.append(f"... and {len(chains) - max_items} more chains")
    bridges = _disulfides(s)
    if bridges:
        lines.append(f"{len(bridges)} disulfide bond(s): " + ", ".join(bridges))
    return lines, data


def _chain_breaks(s, rows) -> list[int]:
    names, res = s.atoms["name"], s.atoms["residue"]
    pos = s.positions
    atom = {}
    for key in ("N", "C"):
        ids = np.flatnonzero(names == key)
        atom[key] = dict(zip(res[ids].tolist(), ids.tolist(), strict=True))
    out = []
    for r, nxt in zip(rows[:-1], rows[1:], strict=True):
        c, n = atom["C"].get(r), atom["N"].get(nxt)
        if c is not None and n is not None and np.linalg.norm(pos[c] - pos[n]) > 2.5:
            out.append(r)
    return out


def _disulfides(s) -> list[str]:
    names = s.atoms["name"]
    i, j = s.bonds["i"], s.bonds["j"]
    pick = (names[i] == "SG") & (names[j] == "SG")
    return [f"{_residue_label(s, int(s.atoms['residue'][a]))}-"
            f"{_residue_label(s, int(s.atoms['residue'][b]))}"
            for a, b in zip(i[pick].tolist(), j[pick].tolist(), strict=True)]  # fmt: skip


def _site(s, ids: np.ndarray, kinds, cutoff: float, max_items: int, what: str):
    """Lines and data describing the surroundings of the atoms ``ids``."""
    anum = s.atoms["anum"]
    res = s.atoms["residue"]
    inside = np.zeros(s.natoms, bool)
    inside[ids] = True
    heavy = ids[anum[ids] > 1]
    others = np.flatnonzero(~inside & (anum > 1))
    li, oj, d = _pairs(s, heavy, others, cutoff)
    lines, data = [], {}
    # closest approach per neighboring residue
    best: dict[int, tuple[float, int, int]] = {}
    for a, b, dist in zip(li.tolist(), oj.tolist(), d.tolist(), strict=True):
        r = int(res[b])
        if r not in best or dist < best[r][0]:
            best[r] = (dist, a, b)
    near = sorted(best.items(), key=lambda kv: kv[1][0])
    polymer_near = [(r, v) for r, v in near if kinds["polymer"][v[2]]]
    water_near = {r for r, v in near if kinds["water"][v[2]]}
    items = [f"{_residue_label(s, r)} {dist:.1f} Å ({s.atoms['name'][a]}-{s.atoms['name'][b]})"
             for r, (dist, a, b) in polymer_near[:max_items]]  # fmt: skip
    if polymer_near:
        more = f" ... and {len(polymer_near) - max_items} more" \
            if len(polymer_near) > max_items else ""  # fmt: skip
        lines.append(f"{what}: {len(polymer_near)} polymer residues within {cutoff:g} Å, "
                     f"closest first: " + "; ".join(items) + more)  # fmt: skip
    else:
        lines.append(f"{what}: no polymer residues within {cutoff:g} Å")
    others_near = [(r, v) for r, v in near if kinds["other"][v[2]] or kinds["ions"][v[2]]]
    if others_near:
        lines.append(f"{what}: other molecules within {cutoff:g} Å: "
                     + "; ".join(f"{_residue_label(s, r)} {v[0]:.1f} Å"
                                 for r, v in others_near[:max_items]))  # fmt: skip
    if water_near:
        lines.append(f"{what}: {len(water_near)} water molecules within {cutoff:g} Å")
    data["neighbors"] = [{"residue": _residue_label(s, r), "distance": round(dist, 2),
                          "atoms": [str(s.atoms["name"][a]), str(s.atoms["name"][b])]}
                         for r, (dist, a, b) in polymer_near]  # fmt: skip
    data["waters_nearby"] = len(water_near)

    # bonds from the site to the rest (covalent links, metal coordination)
    i, j = s.bonds["i"], s.bonds["j"]
    cross = inside[i] != inside[j]
    links = []
    for a, b in zip(i[cross].tolist(), j[cross].tolist(), strict=True):
        a, b = (a, b) if inside[a] else (b, a)
        dist = float(np.linalg.norm(s.positions[a] - s.positions[b]))
        links.append(f"{s.atoms['name'][a]} to {_atom_label(s, b)} ({dist:.2f} Å)")
    if links:
        lines.append(f"{what}: bonded to " + "; ".join(links[:max_items]))
    data["bonds_out"] = links

    # polar contacts or hydrogen bonds
    polar_in = ids[np.isin(anum[ids], (7, 8))]
    polar_out = np.flatnonzero(~inside & np.isin(anum, (7, 8)) & ~kinds["water"])
    pi, pj, pd = _pairs(s, polar_in, polar_out, POLAR_CONTACT)
    contacts = sorted(zip(pd.tolist(), pi.tolist(), pj.tolist(), strict=True))
    kind = "polar contacts (N/O within 3.5 Å)"
    if _has_polar_hydrogens(s) and len(contacts):
        kind = "hydrogen-bond candidates (N/O within 3.5 Å, hydrogens present)"
    if contacts:
        lines.append(f"{what}: {len(contacts)} {kind}: "
                     + "; ".join(f"{s.atoms['name'][a]}-{_atom_label(s, b)} {dist:.2f} Å"
                                 for dist, a, b in contacts[:max_items]))  # fmt: skip
    data["polar_contacts"] = [{"atom": str(s.atoms["name"][a]), "partner": _atom_label(s, b),
                               "distance": round(dist, 2)} for dist, a, b in contacts]  # fmt: skip

    # burial: surface area in place vs. alone
    buried = _burial(s, ids)
    if buried is not None:
        lines.append(f"{what}: {buried:.0%} of its solvent-accessible surface is buried")
        data["buried_fraction"] = round(buried, 3)
    return lines, data


def _burial(s, ids) -> float | None:
    from .analysis import sasa

    try:
        near = np.unique(np.concatenate([ids, _pairs(s, ids, np.arange(s.natoms), 10.0)[1]]))
        env = s.clone(near)
        local = np.searchsorted(near, ids)
        waters = _mask(env, "water")
        keep = np.flatnonzero(~waters | np.isin(np.arange(env.natoms), local))
        env = env.clone(keep)
        local = np.searchsorted(keep, local)
        together = sasa(env, atoms=local)[0][local].sum()
        alone = sasa(s.clone(ids))[0].sum()
    except Exception:
        return None
    if alone <= 0:
        return None
    return float(max(0.0, 1.0 - together / alone))


def _ligands(s, kinds, cutoff, max_items):
    lines, data = [], []
    groups = _other_groups(s, kinds)
    for ids in groups[:max_items]:
        residues = sorted(set(s.atoms["residue"][ids].tolist()))
        label = ", ".join(_residue_label(s, r) for r in residues[:5]) + (
            " ..." if len(residues) > 5 else "")  # fmt: skip
        anum = s.atoms["anum"][ids]
        q = int(s.atoms["formal_charge"][ids].sum())
        entry = {"molecule": label, "formula": _formula(anum), "heavy_atoms": int((anum > 1).sum()),
                 "formal_charge": q}  # fmt: skip
        head = f"{label}: {_formula(anum)}, {entry['heavy_atoms']} heavy atoms, charge {q:+d}"
        smiles = _smiles(s, ids)
        if smiles:
            head += f", SMILES {smiles}"
            entry["smiles"] = smiles
        lines.append(head)
        site_lines, site = _site(s, ids, kinds, cutoff, max_items, label)
        lines += site_lines
        entry.update(site)
        data.append(entry)
    if len(groups) > max_items:
        lines.append(f"... and {len(groups) - max_items} more molecules")
    return lines, data


def _smiles(s, ids) -> str | None:
    try:
        from rdkit import Chem

        mol = s.to_rdkit(ids)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def _interfaces(s, kinds, cutoff, max_items):
    res = s.atoms["residue"]
    res_chain = s.residues["chain"]
    heavy = kinds["polymer"] & (s.atoms["anum"] > 1)
    chains = sorted(set(res_chain[res[heavy]].tolist()))
    lines, data = [], []
    for x, c1 in enumerate(chains):
        a1 = np.flatnonzero(heavy & (res_chain[res] == c1))
        for c2 in chains[x + 1 :]:
            a2 = np.flatnonzero(heavy & (res_chain[res] == c2))
            i, j, _ = _pairs(s, a1, a2, cutoff)
            if not len(i):
                continue
            pairs = sorted(set(zip(res[i].tolist(), res[j].tolist(), strict=True)))
            r1 = sorted({p[0] for p in pairs})
            r2 = sorted({p[1] for p in pairs})
            n1, n2 = str(s.chains["name"][c1]), str(s.chains["name"][c2])
            lines.append(f"chains {n1} and {n2}: {len(pairs)} residue pairs within {cutoff:g} Å; "
                         f"{n1}: " + ", ".join(_residue_label(s, r) for r in r1[:max_items])
                         + f"; {n2}: " + ", ".join(_residue_label(s,
                             r) for r in r2[:max_items]))  # fmt: skip
            data.append({"chains": [n1, n2], "residue_pairs": len(pairs),
                         "residues": [[_residue_label(s, r) for r in r1],
                                      [_residue_label(s, r) for r in r2]]})  # fmt: skip
    return lines, data


def _clashes(s, max_items):
    heavy = np.flatnonzero(s.atoms["anum"] > 1)
    i, j, d = _pairs(s, heavy, heavy, CLASH)
    keep = i < j
    i, j, d = i[keep], j[keep], d[keep]
    if not len(i):
        return [], []
    nbr: dict[int, set] = {}
    for a, b in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True):
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    out = []
    for a, b, dist in sorted(zip(i.tolist(), j.tolist(), d.tolist(), strict=True),
                             key=lambda t: t[2]):  # fmt: skip
        na, nb = nbr.get(a, set()), nbr.get(b, set())
        if b in na or na & nb:  # bonded, or two bonds apart
            continue
        out.append((a, b, dist))
    lines = [f"{len(out)} clashes (nonbonded heavy atoms closer than {CLASH} Å): "
             + "; ".join(f"{_atom_label(s, a)} - {_atom_label(s, b)} {dist:.2f} Å"
                         for a, b, dist in out[:max_items])] if out else []  # fmt: skip
    data = [{"atoms": [_atom_label(s, a), _atom_label(s, b)], "distance": round(dist, 2)}
            for a, b, dist in out]  # fmt: skip
    return lines, data


def summarize(system, focus=None, cutoff: float = 4.0, max_items: int = 20,
              title: str | None = None) -> Summary:  # fmt: skip
    """A text summary of ``system`` (see the module docstring).

    ``focus``: a selection (string, indices or AtomSel) to describe in
    detail: its surroundings within ``cutoff`` Å, bonds, polar contacts and
    burial.  ``max_items`` caps every list; longer lists say how many more.
    """
    s = system
    kinds = _kinds(s)
    out = Summary(title or (s.name or "system"))
    lines, data = _composition(s, kinds)
    out.sections.append(("Composition", lines))
    out.data["composition"] = data
    lines, data = _chains(s, kinds, max_items)
    if lines:
        out.sections.append(("Chains", lines))
    out.data["chains"] = data
    lines, data = _ligands(s, kinds, cutoff, max_items)
    if lines:
        out.sections.append(("Other molecules and their sites", lines))
    out.data["molecules"] = data
    lines, data = _interfaces(s, kinds, cutoff, max_items)
    if lines:
        out.sections.append(("Contacts between chains", lines))
    out.data["interfaces"] = data
    if focus is not None:
        ids = s.select(focus).ids if isinstance(focus, str) else \
            np.asarray(s._ids("atom", focus), dtype=np.int64)  # fmt: skip
        if not len(ids):
            raise ValueError(f"focus {focus!r} selects no atoms")
        what = focus if isinstance(focus, str) else f"{len(ids)} atoms"
        residues = sorted(set(s.atoms["residue"][ids].tolist()))
        head = [f"{len(ids)} atoms in {len(residues)} residue(s): "
                + ", ".join(_residue_label(s, r) for r in residues[:max_items])]  # fmt: skip
        site_lines, site = _site(s, ids, kinds, cutoff, max_items, "focus")
        out.sections.append((f"Focus: {what}", head + site_lines))
        out.data["focus"] = {"selection": what, "atoms": len(ids), **site}
    from .validate import validate

    checks, data = _clashes(s, max_items)
    try:
        problems = validate(s)
    except Exception:
        problems = []
    checks += [str(p) for p in problems[:max_items]]
    if not checks:
        checks = ["no clashes, and no problems found by boonza.validate"]
    out.sections.append(("Checks", checks))
    out.data["clashes"] = data
    out.data["problems"] = [str(p) for p in problems]
    return out
