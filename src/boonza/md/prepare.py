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
    if getattr(args, "model", "aa") == "sirah":
        return build_sirah_system(args, workdir, log, check)
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


def secondary_beside(top) -> str | None:
    """The DSSP codes boonza wrote beside a built Martini topology, if any."""
    path = Path(top).parent / "secondary.txt"
    return path.read_text().strip() if path.is_file() else None


#: Where a topology's coordinates are looked for, in this order.  A .dms or a
#: .mae holds them as they are; a .gro rounds to 0.001 nm and a .pdb to
#: 0.001 A, so they come last, and a file of the topology's own name comes
#: before the cg.* that :meth:`boonza.martini.Martinized.save` writes.
COORDINATE_SUFFIXES = (".dms", ".mae", ".gro", ".pdb")


def _coordinates_beside(top: Path) -> Path:
    """The coordinates of a topology that is already built, the least rounded first."""
    names = [top.with_suffix(s).name for s in COORDINATE_SUFFIXES]
    names += [f"cg{s}" for s in COORDINATE_SUFFIXES]
    for name in names:
        here = top.parent / name
        if here.is_file():
            return here
    raise ValueError(f"{top} is a topology, which carries no coordinates; put them beside it as "
                     f"{top.stem}.dms, .mae or .gro (or cg.gro)")  # fmt: skip


#: The beads that say a coarse-grained file of each model is one boonza built.
BUILT_BEADS = {"martini": ("BB",), "sirah": ("GN", "GC", "GO")}


def _already_built(s: System, model: str, path: Path) -> bool:
    """Whether the input is a coarse-grained system boonza built: beads with a
    force field on them, which run as they are.

    A ``cg.dms`` carries every parameter the topology gave it -- boonza's own
    format holds them all -- so a run needs neither the topology nor the force
    field files beside it.  A structure is mapped as before: without a force
    field, or with hydrogens, there is nothing built to run.  The file written
    for viewing is refused rather than run, since its backbone bead is named CA
    and a Martini view carries no elastic network, so running it would quietly
    let a fold go.
    """
    from ..martini.build import VIEWING_MARK

    kind = "sirah" if model == "sirah" else "martini"
    if bool((s.atoms["anum"] == 1).any()):
        return False  # a structure with hydrogens: one to map
    here = lambda beads: all(len(s.select(f"name {bead}").ids) for bead in beads)  # noqa: E731
    if "nonbonded" not in s.table_names:
        if here(BUILT_BEADS[kind]):
            raise ValueError(f"{path.name} is coarse-grained already, but carries no parameters: "
                             "it is a snapshot of a run (its coordinates and bonds, for looking "
                             "at and for measuring).  Run that run's solvated.dms, which carries "
                             "them, or the topology beside it")  # fmt: skip
        return False  # a structure to map, not a system to run
    if any(str(s.ct(c).name) == VIEWING_MARK for c in range(s.ncts)):
        raise ValueError(f"{path.name} is the file written for viewing, which has had the elastic "
                         "network taken out of it (its cts say so), so a run of it would let the "
                         "fold go; run the cg.dms beside it")  # fmt: skip
    return here(BUILT_BEADS[kind])


def _write_built(m, directory: Path, system: System, gromacs: bool, log=print,
                 **save_kwargs) -> Path:  # fmt: skip
    """Write the built coarse-grained system to ``directory``; return the file a
    run reads.

    ``cg.dms`` is the system itself, parameters and all, and is what boonza
    runs.  With ``gromacs`` the same thing goes out in GROMACS's form beside it
    -- ``topol.top``, an ``.itp`` per molecule, ``cg.gro``, and for SIRAH the
    force field the topology includes -- for running or checking it there.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if not gromacs:
        save(system, directory / "cg.dms")
        return directory / "cg.dms"
    m.save(directory, system=system, **save_kwargs)
    log(f"Wrote the GROMACS form of the system too: {directory.name}/topol.top")
    return directory / "cg.dms"


def _views_beside(where: Path, workdir: Path, log=print) -> int:
    """Copy the view files written beside a built system into the run.

    ``boonza swim`` writes ``view.dms`` and ``view.mae`` where it builds each
    simulation -- the beads to look at, the backbone named CA and no elastic
    network -- and what the run reads, a ``cg.dms`` or a topology, carries no
    such thing.  A system built somewhere else has none, and none is written.
    """
    import shutil

    copied = []
    for suffix in (".dms", ".mae"):
        here = where.parent / f"view{suffix}"
        if here.is_file() and here.resolve() != where.resolve():
            shutil.copyfile(here, workdir / f"view{suffix}")
            copied.append(here.name)
    if copied:
        log(f"Copied {' and '.join(copied)} from beside {where.name}")
    return len(copied)


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


def _check_nothing_is_dropped(s: System, args, path: Path, log=print) -> None:
    """Refuse to coarse-grain only part of the input by accident.

    ``cg_selection`` is ``protein`` by default, and Martini has parameters for
    nothing else here, so a ligand beside the protein is simply not mapped --
    the run would then be of the protein alone, which is never what a complex
    was given for.  Choosing the selection says it was meant, and the note
    still names what was left behind.
    """
    from ..martini.build import _RESNAMES, force_field

    mappable = set(force_field("martini3001").blocks)
    kept = set(s.select(args.cg_selection).ids.tolist())
    residue = np.asarray(s.atoms["residue"])
    names = np.asarray(s.residues["name"])
    chains = np.asarray(s.chains["name"])
    of_chain = np.asarray(s.residues["chain"])
    solvent = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "SPC"}
    left: dict[str, set] = {}
    fragments: set[str] = set()
    for r in range(s.nresidues):
        name = str(names[r]).strip().upper()
        atoms = np.flatnonzero(residue == r)
        if name in solvent or not len(atoms) or not set(atoms.tolist()).isdisjoint(kept):
            continue
        if _RESNAMES.get(name, name) in mappable:
            # a residue Martini knows, which the selection did not take whole:
            # a fragment of the structure rather than a molecule of its own
            fragments.add(f"{str(chains[of_chain[r]]).strip() or '-'}/{name}{r + 1}")
            continue
        left.setdefault(str(chains[of_chain[r]]).strip() or "-", set()).add(name)
    if fragments:
        log(f"Note: not coarse-grained, being too little of a residue to map: "
            f"{', '.join(sorted(fragments)[:5])}"
            f"{f' and {len(fragments) - 5} more' if len(fragments) > 5 else ''}")  # fmt: skip
    if not left:
        return
    what = "; ".join(f"chain {c}: {', '.join(sorted(n))}" for c, n in sorted(left.items()))
    if "cg_selection" in getattr(args, "specified", ()):
        log(f"Note: left out of the coarse-grained system by cg_selection "
            f"{args.cg_selection!r} -- {what}")  # fmt: skip
        return
    raise ValueError(
        f"{path.name} holds more than cg_selection {args.cg_selection!r} maps -- {what}.  Martini "
        "has parameters for proteins here and none for a ligand, so those would be dropped and "
        "the run would be of the rest alone.  Give cg_selection to say that is meant, or bring a "
        "topology that already holds them (boonza md runs a .top as it is)"
    )


def build_sirah_system(args, workdir: Path, log=print, check=None) -> tuple[System, dict]:
    """A SIRAH system from a topology that is already built.

    SIRAH keeps its parameters in the topology, as Martini does, so nothing
    here matches templates: a structure is mapped onto beads and filled with
    WT4 water here (:func:`boonza.sirah.sirahize`, :func:`boonza.sirah.solvate`),
    and a topology built by SIRAH's own tools (cgconv.pl, then pdb2gmx or
    tleap) runs as it is.
    """
    path = Path(args.input_structure) if args.input_structure else None
    if path is None:
        raise ValueError("model = 'sirah' needs a topology: INPUT_STRUCTURE")
    mode = getattr(args, "solvate", "box")
    mode = {True: "box", False: "none"}.get(mode, mode)
    if path.suffix.lower() not in (".top", ".itp"):
        from ..sirah import sirahize

        if mode not in ("none", "box"):
            raise ValueError(f"model = 'sirah' fills a box with WT4 water or none at all, "
                             f"not solvate = {mode!r}")  # fmt: skip
        aa = load_input(path, log, hydrogens=False)
        if _already_built(aa, "sirah", path):
            return _run_as_built(aa, args, path, workdir, "SIRAH", log, check)
        if not (aa.atoms["anum"] == 1).any():
            raise ValueError(f"{path} has no hydrogens: SIRAH puts beads on named hydrogens "
                             "(serine's HG, tryptophan's HE1), so add them, with protonation "
                             "states, first")  # fmt: skip
        _check_nothing_is_dropped(aa, args, path, log)
        built = sirahize(aa, args.cg_selection, termini=args.termini, log=log)
        log(f"SIRAH: {built.nbeads} beads in {len(built.molecules)} molecule(s)")
        if mode == "box":
            from ..sirah.build import solvate as solvate_sirah

            built = solvate_sirah(built, padding=10.0 * args.padding_nm, salt=args.saltM,
                                  seed=args.seed, log=log)  # fmt: skip
        elif built.cell is None or not np.any(built.cell):
            # no water, but a periodic box all the same: the run needs one
            edge = float(np.ptp(built.positions, axis=0).max()) + 20.0 * args.padding_nm
            built.positions = built.positions - built.positions.mean(0) + edge / 2
            built.cell = np.diag(np.full(3, edge))
            log(f"Box: {edge / 10:.2f} nm a side, {args.padding_nm:g} nm around the beads")
        s = built.system()
        _write_built(built, workdir / "sirah", s, bool(args.gromacs), log)
        # no view file: SIRAH holds its fold with torsion terms rather than an
        # elastic network, so there is nothing to leave out of one, and beads
        # renamed for a viewer are beads a viewer mis-bonds (sirah/cg.dms is
        # what to open)
        s.atoms["md_index"] = np.arange(1, s.natoms + 1, dtype=np.int64)
        info = components(s, [])
        if getattr(args, "monitor_selection", None) is not None:
            info["selection"] = select_atoms(s, args.monitor_selection)
        if check is not None:
            check(info)
        _log_built(log, s, "System")
        return _with_production_indices(s, s, info)
    if "solvate" in getattr(args, "specified", ()) and mode != "none":
        raise ValueError(f"{path.name} is a topology, which is already built; "
                         "it runs with solvate = 'none'")  # fmt: skip
    gro = _coordinates_beside(path)
    log(f"SIRAH: {path.name} with {gro.name}, as built")
    s = load(path, coordinates=gro)
    _views_beside(path, workdir, log)
    s.atoms["md_index"] = np.arange(1, s.natoms + 1, dtype=np.int64)
    info = components(s, [])
    if getattr(args, "monitor_selection", None) is not None:
        info["selection"] = select_atoms(s, args.monitor_selection)
    if check is not None:
        check(info)
    _log_built(log, s, "System")
    return _with_production_indices(s, s, info)


def _run_as_built(s: System, args, path: Path, workdir: Path, model: str, log, check):
    """Run a coarse-grained system boonza built, as it is: it carries its own
    parameters, so there is nothing to map and nothing to resolve."""
    mode = getattr(args, "solvate", "box")
    mode = {True: "box", False: "none"}.get(mode, mode)
    if "solvate" in getattr(args, "specified", ()) and mode != "none":
        raise ValueError(f"{path.name} is a coarse-grained system that is already built; "
                         "it runs with solvate = 'none'")  # fmt: skip
    log(f"{model}: {path.name}, as built ({s.natoms} beads with their parameters)")
    _views_beside(path, workdir, log)
    info = components(s, [])
    if getattr(args, "monitor_selection", None) is not None:
        info["selection"] = select_atoms(s, args.monitor_selection)
    if check is not None:
        check(info)
    _log_built(log, s, "System")
    return _with_production_indices(s, s, info)


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
    lipid_itps = args.lipid_itp or [str(p) for p in mt.parameters(*mt.LIPIDS_FOR[version])]

    if path is not None and path.suffix.lower() in (".top", ".itp"):
        if "solvate" in getattr(args, "specified", ()) and mode != "none":
            raise ValueError(f"{path.name} is a topology, which is already built; "
                             "it runs with solvate = 'none'")  # fmt: skip
        gro = _coordinates_beside(path)
        log(f"Martini {version}: {path.name} with {gro.name}, as built")
        s = load(path, coordinates=gro)
        _views_beside(path, workdir, log)
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
        aa = load_input(path, log, hydrogens=False)
        if _already_built(aa, args.model, path):  # beads with their parameters on them
            return _run_as_built(aa, args, path, workdir, f"Martini {version}", log, check)
        if version != 3:
            raise ValueError(f"model = '{args.model}' cannot coarse-grain {path.name}: boonza "
                             "martinizes proteins as Martini 3.  Give a Martini 2 topology "
                             "(.top) or a built Martini 2 system (cg.dms) instead, or build a "
                             "membrane without a protein")  # fmt: skip
        _check_nothing_is_dropped(aa, args, path, log)
        protein = mt.martinize(aa, args.cg_selection, elastic=bool(args.elastic),
                               elastic_selection=args.elastic_selection,
                               neutral_termini=bool(args.neutral_termini))  # fmt: skip
        log(f"Martinized: {protein.nbeads} beads in {len(protein.molecules)} molecule(s)"
            f"{', elastic network' if args.elastic else ''}")  # fmt: skip
    elif mode != "membrane":
        raise ValueError("give a structure to coarse-grain, or solvate = 'membrane' "
                         "with upper = 'POPC:7,CHOL:3' to build a bilayer on its own")  # fmt: skip

    if mode == "membrane":
        upper = _composition(args.upper)
        lower = _composition(args.lower) if args.lower else None
        size = [10.0 * v for v in args.size_nm] if args.size_nm else [100.0]
        m = mt.bilayer(lipid_itps, upper, lower, size=size[0] if len(size) == 1 else tuple(size),
                       martini=version, area_per_lipid=args.area_per_lipid,
                       water=10.0 * args.water_nm, salt=args.saltM, protein=protein,
                       protein_origin=bool(args.opm), protein_shift=10.0 * args.shift_nm,
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

    s = m.system(args.martini_itp)
    _write_built(m, workdir / "martini", s, bool(args.gromacs), log,
                 martini_itp=args.martini_itp)  # fmt: skip
    if m.ss:  # DSSP cannot read beads: dihedral_restraint = 'ss' reads this back
        (workdir / "martini" / "secondary.txt").write_text(m.ss + "\n")
    # what to open in a viewer: the backbone bead named CA, so a chain is
    # traced, and no elastic network, which a viewer would draw as a hairball.
    # Same atoms in the same order, so a trajectory still lines up with it.
    bands = m.elastic_bonds()
    viewing = m.for_viewing(s)
    for suffix in (".dms", ".mae"):
        save(viewing, workdir / f"view{suffix}")
    log("Wrote view.dms and view.mae: what to open in a viewer"
        + (f", without the {len(bands)} rubber bands" if bands else ""))  # fmt: skip
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
