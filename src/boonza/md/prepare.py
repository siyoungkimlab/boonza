"""Building the simulation system: force fields, GAFF2 ligands, water and ions."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from .. import gaff, viparr
from ..build import neutralize, repartition_hydrogen_masses, solvate
from ..io import load, save
from ..system import System
from .config import HYDROGEN_MASS_AMU, XML_FAMILIES


def load_input(path) -> System:
    """The input structure, with ``md_index`` (1-based) to follow its atoms."""
    s = load(path)
    if s.natoms == 0:
        raise ValueError(f"{path} has no atoms")
    if not (s.atoms["anum"] == 1).any():
        raise ValueError(f"{path} has no hydrogens: add them, with protonation states, first")
    s.atoms["md_index"] = np.arange(1, s.natoms + 1, dtype=np.int64)
    return s


def forcefields(args):
    """("viparr", [force fields]) or ("xml", OpenMM force field)."""
    if args.proteinff is not None:
        from ..ffxml import load_openmm_forcefield

        protein, directory, models = XML_FAMILIES[args.proteinff]
        return "xml", load_openmm_forcefield(protein, f"{directory}/{models[args.waterff]}")
    ffs = []
    for base, *patches in args.forcefields:
        ff = viparr.load_forcefield(base)
        for patch in patches:
            ff = viparr.merge_forcefields(ff, patch)
        ffs.append(ff)
    return "viparr", ffs


def components(s: System, groups=()) -> dict:
    """The molecules of the input: ``component-N`` by lowest atom index,
    ``ligand-N`` for those made entirely of GAFF2 residues; waters and ions
    are counted."""
    new = {r for g in groups for r in g}
    water = np.zeros(s.natoms, bool)
    water[s.select("water").ids] = True
    frag = np.asarray(s.fragids)
    atoms_of: dict[int, list[int]] = {}
    for a, f in enumerate(frag.tolist()):
        atoms_of.setdefault(f, []).append(a)
    out, nwater, nion = [], 0, 0
    chains, residues = s.chains["name"], s.residues
    res_of = s.atoms["residue"]
    for f in sorted(atoms_of, key=lambda f: atoms_of[f][0]):
        atoms = atoms_of[f]
        if water[atoms].all():
            nwater += 1
            continue
        if len(atoms) == 1:
            nion += 1
            continue
        rs = list(dict.fromkeys(res_of[atoms].tolist()))
        flags = [r in new for r in rs]
        kind = "ligand" if all(flags) else "modified" if any(flags) else "standard"
        out.append(
            {
                "id": f"component-{len(out)}",
                "ligand_id": None,
                "classification": kind,
                "chains": sorted({str(chains[residues["chain"][r]]) for r in rs}),
                "residues": [
                    {
                        "chain": str(chains[residues["chain"][r]]),
                        "name": str(residues["name"][r]),
                        "resid": int(residues["resid"][r]),
                        "insertion": str(residues["insertion"][r]),
                        "gaff2": r in new,
                    }
                    for r in rs
                ],
                "input_atom_indices": atoms,
            }
        )
    k = 0
    for c in out:
        if c["classification"] == "ligand":
            c["ligand_id"] = f"ligand-{k}"
            k += 1
    return {"components": out, "water_molecules": nwater, "ions": nion}


def component_table(info: dict) -> str:
    """``--list-components``: each molecule and the selector that names it."""
    comps = info["components"]
    per_chain = Counter(ch for c in comps for ch in c["chains"])
    rows = [("ID", "LIGAND ID", "CLASSIFICATION", "CHAINS", "RESIDUES", "COMPOSITION")]
    hints = [""]
    for c in comps:
        names = Counter(r["name"] for r in c["residues"])
        comp = ", ".join(n for n, _ in names.most_common(4)) + (", ..." if len(names) > 4 else "")
        rows.append(
            (
                c["id"],
                c["ligand_id"] or "-",
                c["classification"],
                ",".join(c["chains"]) or "-",
                str(len(c["residues"])),
                comp,
            )
        )
        if c["ligand_id"]:
            hints.append(f"--early-stop --monitor-ligand {c['ligand_id']}")
        elif len(c["chains"]) == 1 and per_chain[c["chains"][0]] == 1 and c["chains"][0]:
            hints.append(f"--early-stop --monitor-chain {c['chains'][0]}")
        else:
            hints.append(f"--early-stop --monitor-component {c['id']}")
    widths = [max(len(r[k]) for r in rows) for k in range(len(rows[0]))]
    lines = []
    for row, hint in zip(rows, hints, strict=True):
        lines.append("  ".join(x.ljust(w) for x, w in zip(row, widths, strict=True)).rstrip())
        if hint:
            lines.append("    " + hint)
    lines.append(
        f"({info['water_molecules']} water molecules and {info['ions']} ions "
        "are counted, not listed)"
    )
    return "\n".join(lines)


def _groups(s, kind, ffs) -> list[list[int]]:
    return gaff.find_unmatched(s, ffs) if kind == "viparr" else []


def _label(s: System, groups) -> str:
    res = s.residues
    return "; ".join("+".join(f"{res['name'][r]}{res['resid'][r]}" for r in g) for g in groups)


def describe_components(args) -> str:
    """The component table of ``args.input_structure``; nothing is built."""
    if args.input_structure is None:
        raise ValueError("give INPUT_STRUCTURE to list its components")
    s = load_input(args.input_structure)
    kind, ffs = forcefields(args)
    return f"Components of {args.input_structure}:\n\n" + component_table(
        components(s, _groups(s, kind, ffs))
    )


def _solute_charge(p: System) -> float:
    """Net charge of a parameterized input, without Na+/Cl- ions (which
    ``neutralize`` counts itself)."""
    frag = np.asarray(p.fragids)
    single = np.bincount(frag)[frag] == 1
    ion = single & np.isin(p.atoms["anum"], (11, 17))
    return float(p.atoms["charge"][~ion].sum())


def build_system(args, workdir: Path, log=print) -> tuple[System, dict]:
    """The solvated, neutralized, parameterized system and the input's components.

    GAFF2 templates are made for what the force fields cannot match
    (``ligand_mode = "auto"``) and kept in ``workdir/gaff2_patch``. Water
    comes from the bundled TIP3P box (4- and 5-site models add their sites
    from their templates); counterions and NaCl follow msys, counting salt
    against the number of waters.
    """
    s = load_input(args.input_structure)
    kind, ff = forcefields(args)
    groups = _groups(s, kind, ff)
    if groups:
        if args.ligand_mode == "disabled":
            raise ValueError(
                f"no force field has templates for {_label(s, groups)}; "
                "ligand_mode = 'auto' makes GAFF2 templates for them"
            )
        log(f"GAFF2 {args.ligandff} templates (AM1-BCC charges) for {_label(s, groups)}")
        patch = gaff.gaff2_patch(s, ff, charges=args.ligand_charges, workdir=workdir / "gaff2")
        viparr.write_forcefield(patch, workdir / "gaff2_patch")
        host = gaff.host_index(ff)
        ff[host] = viparr.merge_forcefields(ff[host], patch)
    info = components(s, groups)

    def parameterize(system: System) -> System:
        if kind == "viparr":
            with warnings.catch_warnings():
                # a single-atom ion that two ion sets both have: the first listed wins, by design
                warnings.filterwarnings(
                    "ignore",
                    r"fragment \d+ \([A-Z][a-z]?\) was matched by multiple",
                    viparr.ViparrWarning,
                )
                out = viparr.parameterize(system, ff)
            if args.hmr:
                out = repartition_hydrogen_masses(out, "not water", HYDROGEN_MASS_AMU)
            return out
        from ..ffxml import parameterize_openmm

        return parameterize_openmm(
            system,
            ff,
            constraints="hbonds",
            rigid_water=True,
            hydrogen_mass=HYDROGEN_MASS_AMU if args.hmr else None,
        )

    charge = _solute_charge(parameterize(s))
    box = solvate(s, thickness=10.0 * args.padding_nm)
    box = neutralize(box, cation="Na", anion="Cl", charge=charge, concentration=args.saltM)
    out = parameterize(box)
    where = {int(k) - 1: i for i, k in enumerate(out.atoms["md_index"].tolist()) if k > 0}
    for c in info["components"]:
        c["production_atom_indices"] = [where[a] for a in c["input_atom_indices"]]
    nwater = len(set(out.atoms["residue"][out.select("water").ids].tolist()))
    log(
        f"Solvated: {out.natoms} particles, {nwater} waters, box "
        + " x ".join(f"{x / 10:.2f}" for x in np.diag(out.cell))
        + " nm; "
        f"input charge {charge:+.2f}"
    )
    return out, info


def write_components(path, info: dict, args) -> None:
    doc = {
        "forcefields": (
            [list(x) for x in args.forcefields]
            if args.proteinff is None
            else {"proteinff": args.proteinff, "waterff": args.waterff}
        ),
        "ligand_mode": args.ligand_mode,
        "ligandff": args.ligandff,
        **info,
    }
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def save_structure(s: System, path, positions=None, box=None) -> None:
    """``s`` without its force field (atoms, bonds, cell), for viewing trajectories."""
    frame = s.copy()
    for name in list(frame.table_names):
        frame.del_table(name)
    if positions is not None:
        frame.positions = positions
    if box is not None:
        frame.cell = box
    save(frame, path)
