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
from .config import HYDROGEN_MASS_AMU, describe_forcefields, forcefield_kind, ion_water_mismatch

#: Residue names of Martini water: a bead is four waters, and carries no
#: element that the all-atom ``water`` selection could find it by.
MARTINI_WATER = ("W", "WF")


def load_input(path, log=None, hydrogens: bool = True) -> System:
    """The input structure, with ``md_index`` (1-based) to follow its atoms.

    ``hydrogens``: require them, as an all-atom force field does.  Martini
    maps heavy atoms, so its route asks for none.
    """
    s = load(path)
    if s.natoms == 0:
        raise ValueError(f"{path} has no atoms")
    if hydrogens and not (s.atoms["anum"] == 1).any():
        raise ValueError(f"{path} has no hydrogens: add them, with protonation states, first")
    s.atoms["md_index"] = np.arange(1, s.natoms + 1, dtype=np.int64)
    _gather_residues(s, path, log)
    return s


def _gather_residues(s: System, path, log=None) -> None:
    """Put the atoms of each residue back together.

    Preparation tools often write the hydrogens they add at the end of the
    file rather than beside the heavy atoms they belong to, which leaves a
    residue's atoms in two pieces. OpenMM refuses such a topology ("All atoms
    within a residue must be contiguous"), so sort the atoms by residue. The
    sort is stable, so everything else keeps the order the file gave it, and
    ``md_index`` still points into the input file.
    """
    res = s.atoms["residue"]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(res) != 0) + 1])
    _, pieces = np.unique(res[starts], return_counts=True)
    split = int(np.count_nonzero(pieces > 1))
    if not split:
        return
    s.reorder_atoms(np.argsort(res, kind="stable"))
    if log is not None:
        log(
            f"Gathered the atoms of {split} residue(s) that {path} keeps in "
            "pieces (added hydrogens written at the end of the file, most likely)"
        )


def forcefields(args):
    """("viparr", [force fields]) or ("xml", OpenMM force field)."""
    if forcefield_kind(args.forcefields) == "xml":
        from ..ffxml import load_openmm_forcefield

        return "xml", load_openmm_forcefield(*(entry[0] for entry in args.forcefields))
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
    # a Martini water bead is four waters, with no element to recognize it by
    names = s.residues["name"][s.atoms["residue"]]
    water |= np.isin(names, MARTINI_WATER)
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
    if getattr(args, "model", "aa") != "aa":
        # nothing is matched against templates, so nothing is a GAFF2 ligand
        s = load_input(args.input_structure, hydrogens=False)
        return f"Components of {args.input_structure}:\n\n" + component_table(components(s))
    s = load_input(args.input_structure)
    kind, ffs = forcefields(args)
    return f"Components of {args.input_structure}:\n\n" + component_table(
        components(s, _groups(s, kind, ffs))
    )


def select_atoms(s: System, text: str) -> dict:
    """The atoms of the input a selection names (an early-stop target)."""
    try:
        ids = s.select(text).ids
    except Exception as e:  # noqa: BLE001 - report any selection error the same way
        raise ValueError(f"monitor_selection {text!r}: {e}") from None
    if not len(ids) or not (s.atoms["anum"][ids] > 1).any():
        raise ValueError(f"monitor_selection {text!r} selects no heavy atoms of the input")
    res = sorted(set(s.atoms["residue"][ids].tolist()))
    chains = sorted({str(s.chains["name"][s.residues["chain"][r]]) for r in res})
    return {"selection": text, "chains": chains, "input_atom_indices": ids.tolist()}


def _solute_charge(p: System) -> float:
    """Net charge of a parameterized input, without Na+/Cl- ions (which
    ``neutralize`` counts itself)."""
    frag = np.asarray(p.fragids)
    single = np.bincount(frag)[frag] == 1
    ion = single & np.isin(p.atoms["anum"], (11, 17))
    return float(p.atoms["charge"][~ion].sum())


def _own_cell(s: System, mode: str, path) -> None:
    """``solvate`` 'fill' and 'none' keep the input's own box, so it must have one."""
    cell = np.asarray(s.cell, dtype=float)
    if not cell.any():
        raise ValueError(
            f"solvate = '{mode}' needs a periodic cell, and {path} has none "
            "(a DMS, MAE, GRO or CIF file of a built system carries one)"
        )
    if mode == "fill" and np.abs(cell - np.diag(np.diag(cell))).max() > 1e-6:
        raise ValueError(f"solvate = 'fill' needs a rectangular cell; {path} has a triclinic one")


def build_system(args, workdir: Path, log=print, check=None) -> tuple[System, dict]:
    """The solvated, neutralized, parameterized system and the input's components.

    GAFF2 templates are made for what the force fields cannot match
    (``ligand_mode = "auto"``) and kept in ``workdir/gaff2_patch``. Water
    comes from the bundled TIP3P box (4- and 5-site models add their sites
    from their templates); counterions and NaCl follow msys, counting salt
    against the number of waters.

    A Martini model takes the other route, :func:`build_martini_system`:
    coarse-grain the input and read the parameters off the beads.
    """
    if getattr(args, "model", "aa") != "aa":
        return build_martini_system(args, workdir, log, check)
    s = load_input(args.input_structure, log)
    kind, ff = forcefields(args)
    log(f"Force fields: {describe_forcefields(args.forcefields)}")
    if kind == "xml":
        from ..ffxml import bundled_version

        where = ""
        if any(f.startswith("bundled:") for f in ff.files):
            where = f" (bundled: {' '.join(bundled_version().split()[:2])})"
        log(f"  read {', '.join(ff.files)}{where}")
    note = ion_water_mismatch(args.forcefields)
    if note:
        log(f"Warning: {note}")
    groups = _groups(s, kind, ff)
    info = components(s, groups)
    if getattr(args, "monitor_selection", None) is not None:
        info["selection"] = select_atoms(s, args.monitor_selection)
    if check is not None:
        check(info)  # e.g. the early-stop target, before AmberTools spends minutes
    if groups:
        if args.ligand_mode == "disabled":
            raise ValueError(
                f"no force field has templates for {_label(s, groups)}; "
                "ligand_mode = 'auto' makes GAFF2 templates for them"
            )
        log(f"GAFF2 {args.ligandff} templates (AM1-BCC charges) for {_label(s, groups)}")
        patch = gaff.gaff2_patch(
            s,
            ff,
            charges=args.ligand_charges,
            parents=args.parents,
            workdir=workdir / "gaff2",
            protein_extent=args.protein_extent,
            draw=workdir,
        )
        viparr.write_forcefield(patch, workdir / "gaff2_patch")
        info["parents"] = patch.parents
        for p in patch.parents:
            log(f"{p['residue']} ({p['chain']}): parent {p['parent']} ({p['source']}); "
                f"{p['protein_heavy_atoms']} of {p['heavy_atoms']} heavy atoms keep "
                "protein types")  # fmt: skip
        for png in sorted(workdir.glob("covalent_*.png")):
            log(f"Covalent adduct drawn in {png.name} (blue: protein types, orange: GAFF2)")
        host = gaff.host_index(ff)
        ff[host] = viparr.merge_forcefields(ff[host], patch)

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

    mode = getattr(args, "solvate", "box")
    mode = {True: "box", False: "none"}.get(mode, mode)
    if mode != "box":
        _own_cell(s, mode, args.input_structure)
        given = getattr(args, "specified", ())
        keys = ("padding_nm", "box_nm", "saltM") if mode == "none" else ("padding_nm", "box_nm")
        unused = [k for k in keys if k in given]
        if unused:
            log(f"Warning: solvate = '{mode}', so {', '.join(unused)} is not used")
    if mode == "none":  # the input is the system: its water, ions and box, as they are
        out = parameterize(s)
        charge = float(out.atoms["charge"].sum())
        if abs(charge) > 1e-3:
            log(f"Warning: the system's charge is {charge:+.2f}, and nothing is added to "
                "neutralize it; add the ions yourself, or solvate")  # fmt: skip
    else:
        charge = _solute_charge(parameterize(s))
        if mode == "fill":  # the input's own cell, its empty space filled
            before = s.natoms
            box = solvate(s, box=np.diag(np.asarray(s.cell, dtype=float)).copy(),
                          center_selection="none", remove_buried=True)  # fmt: skip
            log(f"Filled the input's box with {box.natoms - before} solvent atoms "
                "(hydrophobic voids left dry)")  # fmt: skip
        elif getattr(args, "box_nm", None) is not None:  # a fixed cubic box (boonza swim)
            box = solvate(s, box=10.0 * args.box_nm)
        else:
            box = solvate(s, thickness=10.0 * args.padding_nm)
        box = neutralize(box, cation="Na", anion="Cl", charge=charge, concentration=args.saltM)
        out = parameterize(box)
    # md_index is the atom's line in the input file; the input's own order may
    # differ from it, because load_input gathers residues the file split.
    where = {int(k): i for i, k in enumerate(out.atoms["md_index"].tolist()) if k > 0}
    md = s.atoms["md_index"]
    for c in [*info["components"], *([info["selection"]] if "selection" in info else [])]:
        c["production_atom_indices"] = [where[int(md[a])] for a in c["input_atom_indices"]]
    nwater = len(set(out.atoms["residue"][out.select("water").ids].tolist()))
    log(
        f"{'System' if mode == 'none' else 'Solvated'}: {out.natoms} particles, "
        f"{nwater} waters, box "
        + " x ".join(f"{x / 10:.2f}" for x in np.diag(out.cell))
        + f" nm; charge {charge:+.2f}"
    )
    return out, info


def write_components(path, info: dict, args) -> None:
    doc = {
        "forcefields": [list(x) for x in args.forcefields],
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


def _coordinates_beside(top: Path) -> Path:
    """The coordinates of a Martini topology: ``<stem>.gro``, or the ``cg.gro``
    that :meth:`boonza.martini.Martinized.save` writes beside ``topol.top``."""
    for name in (top.with_suffix(".gro").name, "cg.gro", "cg.pdb"):
        here = top.parent / name
        if here.is_file():
            return here
    raise ValueError(f"{top} is a topology, which carries no coordinates; put its .gro "
                     f"beside it as {top.with_suffix('.gro').name} or cg.gro")  # fmt: skip


def _composition(text: str) -> dict:
    """``POPC:7,CHOL:3`` as shares per lipid."""
    out = {}
    for part in str(text).split(","):
        name, _, share = part.partition(":")
        name = name.strip().upper()
        if not name:
            raise ValueError(f"cannot read lipids from {text!r}: use POPC:7,CHOL:3")
        try:
            out[name] = float(share) if share else 1.0
        except ValueError:
            raise ValueError(f"{part!r}: the share after ':' must be a number") from None
    return out


def build_martini_system(args, workdir: Path, log=print, check=None) -> tuple[System, dict]:
    """The coarse-grained system: the input martinized, then water and ions or
    a bilayer around it, or a topology that was already built.

    Martini takes its parameters from the topology rather than from a force
    field, so nothing here matches templates: the beads carry their own.
    """
    from .. import martini as mt

    version = int(args.model.removeprefix("martini"))
    mode = getattr(args, "solvate", "box")
    mode = {True: "box", False: "none"}.get(mode, mode)
    path = Path(args.input_structure) if args.input_structure else None
    lipid_itps = [str(p) for p in mt.parameters(*mt.LIPIDS_FOR[version])]

    if path is not None and path.suffix.lower() in (".top", ".itp"):
        if "solvate" in getattr(args, "specified", ()) and mode != "none":
            raise ValueError(f"{path.name} is a topology, which is already built; "
                             "it runs with solvate = 'none'")  # fmt: skip
        gro = _coordinates_beside(path)
        log(f"Martini {version}: {path.name} with {gro.name}, as built")
        s = load(path, coordinates=gro)
        s.atoms["md_index"] = np.arange(1, s.natoms + 1, dtype=np.int64)
        info = components(s, [])
        if getattr(args, "monitor_selection", None) is not None:
            info["selection"] = select_atoms(s, args.monitor_selection)
        if check is not None:
            check(info)
        _log_built(log, s, "System")
        return _with_production_indices(s, s, info)

    protein = None
    if path is not None:
        if version != 3:
            raise ValueError(f"model = '{args.model}' cannot coarse-grain {path.name}: boonza "
                             "martinizes proteins as Martini 3.  Give a Martini 2 topology "
                             "(.top) instead, or build a membrane without a protein")  # fmt: skip
        aa = load_input(path, log, hydrogens=False)
        protein = mt.martinize(aa, "protein", elastic=bool(getattr(args, "elastic", False)))
        log(f"Martinized: {protein.nbeads} beads in {len(protein.molecules)} molecule(s)"
            f"{', elastic network' if args.elastic else ''}")  # fmt: skip
    elif mode != "membrane":
        raise ValueError("give a structure to coarse-grain, or solvate = 'membrane' "
                         "with upper = 'POPC:7,CHOL:3' to build a bilayer on its own")  # fmt: skip

    if mode == "membrane":
        upper = _composition(args.upper)
        lower = _composition(args.lower) if args.lower else None
        size = 10.0 * args.box_nm if getattr(args, "box_nm", None) else 100.0
        m = mt.bilayer(lipid_itps, upper, lower, size=size, martini=version,
                       area_per_lipid=args.area_per_lipid, salt=args.saltM, protein=protein,
                       seed=args.seed)  # fmt: skip
        total: dict[str, int] = {}  # m.lipids is per leaflet; the log wants the system
        for name, count, _ in m.lipids:
            total[name] = total.get(name, 0) + count
        log("Bilayer: " + ", ".join(f"{c} {n}" for n, c in total.items()))
    elif mode == "none":
        m = protein
    else:
        box = 10.0 * args.box_nm if getattr(args, "box_nm", None) else None
        m = mt.solvate(protein, padding=10.0 * args.padding_nm, box=box, salt=args.saltM,
                       seed=args.seed)  # fmt: skip

    m.save(workdir / "martini")
    s = m.system()
    s.atoms["md_index"] = np.arange(1, s.natoms + 1, dtype=np.int64)
    info = components(s, [])
    if getattr(args, "monitor_selection", None) is not None:
        info["selection"] = select_atoms(s, args.monitor_selection)
    if check is not None:
        check(info)
    _log_built(log, s, "Bilayer" if mode == "membrane" else "Solvated")
    return _with_production_indices(s, s, info)


def _log_built(log, s: System, what: str) -> None:
    cell = np.diag(np.asarray(s.cell, dtype=float))
    log(f"{what}: {s.natoms} beads, box " + " x ".join(f"{x / 10:.2f}" for x in cell) + " nm")


def _with_production_indices(s: System, out: System, info: dict) -> tuple[System, dict]:
    where = {int(k): i for i, k in enumerate(out.atoms["md_index"].tolist()) if k > 0}
    md = s.atoms["md_index"]
    for c in [*info["components"], *([info["selection"]] if "selection" in info else [])]:
        c["production_atom_indices"] = [where[int(md[a])] for a in c["input_atom_indices"]]
    return out, info
