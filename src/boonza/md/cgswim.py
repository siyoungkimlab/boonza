"""``boonza swim`` coarse-grained: dipeptide probes swimming around a protein.

Neither Martini nor SIRAH has a general way to parameterize a small
molecule, so the probes are dipeptides, which need nothing new (see
:mod:`boonza.martini.probes` and :mod:`boonza.sirah.probes`).  Each
simulation holds a group of probe types, several copies each, around the
mapped protein in that model's water.  The probes are the same 105 in either
model, so a surface mapped in one reads against the other.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..io import save

CLEARANCE = 5.0  # Å between a placed probe and the protein or another probe


def groups_of(sequences, types: int) -> list[list[str]]:
    """Probes dealt into simulations of about ``types`` each, round robin, so
    that every simulation holds a spread of side chains."""
    n = max(1, round(len(sequences) / max(1, types)))
    out: list[list[str]] = [[] for _ in range(n)]
    for k, seq in enumerate(sequences):
        out[k % n].append(seq)
    return out


def _rotation(rng) -> np.ndarray:
    """A turn drawn evenly from all of them, as a matrix."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def place(protein_xyz, probes, copies: int, box, rng, clearance: float = CLEARANCE):
    """Copies of each probe, turned and placed at random in ``box`` (Å), clear
    of the protein and of one another; returns one array per probe type."""
    from ..spatial import min_dist2

    cell = np.diag(box)
    taken = [np.asarray(protein_xyz, float)]
    out = []
    for m in probes:
        xyz = np.asarray(m.positions, float)
        placed = []
        for _ in range(copies):
            for attempt in range(1000):
                candidate = xyz @ _rotation(rng).T + rng.uniform(0, 1, 3) * box
                near = np.vstack(taken)
                if (min_dist2(candidate, near, clearance, cell=cell) > clearance**2).all():
                    placed.append(candidate)
                    taken.append(candidate)
                    break
                if attempt == 999:
                    raise ValueError("no room for another probe: give a larger box "
                                     "(padding_nm) or fewer copies")  # fmt: skip
        out.append(np.vstack(placed))
    return out


def build(protein, probes, copies: int, box, rng, salt: float = 0.15,
          clearance: float = CLEARANCE):  # fmt: skip
    """One simulation's system: the martinized protein, ``copies`` of each probe,
    then water and ions."""
    from ..martini import solvate
    from ..martini.build import Martinized

    protein_xyz = np.asarray(protein.positions, float)
    protein_xyz = protein_xyz - protein_xyz.mean(0) + np.asarray(box, float) / 2
    placed = place(protein_xyz, probes, copies, np.asarray(box, float), rng, clearance)
    molecules = [*protein.molecules, *(m.molecules[0] for m in probes)]
    names = [*protein.names, *(m.names[0] for m in probes)]
    counts = [*([1] * len(protein.molecules)), *([copies] * len(probes))]
    system = Martinized(molecules, np.vstack([protein_xyz, *placed]), np.diag(box), names,
                        protein.ss, copies=counts, martini=protein.martini)  # fmt: skip
    return solvate(system, box=box, salt=salt, seed=int(rng.integers(1 << 30)))


def build_sirah(protein, probes, copies: int, box, rng, salt: float = 0.15,
                clearance: float = CLEARANCE, log=None):  # fmt: skip
    """One simulation's system: the sirahized protein, ``copies`` of each probe,
    then WT4 water and its ions."""
    from ..sirah.build import Sirahized, solvate

    protein_xyz = np.asarray(protein.positions, float)
    protein_xyz = protein_xyz - protein_xyz.mean(0) + np.asarray(box, float) / 2
    placed = place(protein_xyz, probes, copies, np.asarray(box, float), rng, clearance)
    molecules = [*protein.molecules, *(m.molecules[0] for m in probes)]
    counts = [*([1] * len(protein.molecules)), *([copies] * len(probes))]
    system = Sirahized(molecules, np.vstack([protein_xyz, *placed]), np.diag(box),
                       protein.nrexcl, copies=counts)  # fmt: skip
    # the box grows to whole tiles of SIRAH's water box, as it does for a run
    # of one protein; the solute moves with it, probes and all
    return solvate(system, box=box, salt=salt, seed=int(rng.integers(1 << 30)), log=log)


def _built_system(system, martini_itp=None):
    """The mapped system with its parameters on it, or None when they are named
    to be resolved somewhere else (a bare ``--martini-itp``) and cannot be read
    here.  Without them there is no ``cg.dms``: the topology is what runs."""
    from ..io import GromacsError

    try:
        return system.system() if martini_itp is None else system.system(martini_itp)
    except (GromacsError, FileNotFoundError):
        return None


def prepare(args, sequences=None, types: int = 10, copies: int = 5,
            clearance: float = CLEARANCE, log=print, repel: bool = True,
            elastic: bool = True) -> list[Path]:  # fmt: skip
    """Write one ``boonza md`` coarse-grained simulation per group of probes into
    ``args.workdir``; returns their directories.

    Martini or SIRAH, by ``args.model``: each maps the protein with its own
    tool, places its own probes and fills the box with its own water.
    """
    from .config import ALL_ATOM_ONLY, MARTINI_ONLY, settings_of, write_settings
    from .prepare import _check_nothing_is_dropped, load_input

    sirah = args.model == "sirah"
    if sirah:
        from ..sirah.probes import probe, probe_sequences
    else:
        from ..martini.probes import probe, probe_sequences

    if args.input_structure is None:
        raise ValueError("give the protein structure")
    if args.model == "martini2":
        raise ValueError("model = 'martini2' cannot coarse-grain a protein: boonza martinizes "
                         "as Martini 3, so a coarse-grained swim takes model = 'martini3' or "
                         "'sirah'")  # fmt: skip
    sequences = list(sequences) if sequences else probe_sequences()
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    # SIRAH maps polar hydrogens onto beads of their own; Martini maps none
    aa = load_input(args.input_structure, log, hydrogens=sirah)
    _check_nothing_is_dropped(aa, args, Path(args.input_structure), log)
    if sirah:
        from ..sirah import sirahize

        protein = sirahize(aa, args.cg_selection, termini=args.termini, log=log)
        log(f"SIRAH: {protein.nbeads} beads in {len(protein.molecules)} molecule(s)")
    else:
        from ..martini import martinize

        protein = martinize(aa, args.cg_selection, elastic=elastic,
                            elastic_selection=args.elastic_selection if elastic else None,
                            neutral_termini=bool(args.neutral_termini))  # fmt: skip
        log(f"Martinized: {protein.nbeads} beads in {len(protein.molecules)} molecule(s)"
            f"{', elastic network' if elastic else ''}")  # fmt: skip
    extent = float((protein.positions.max(0) - protein.positions.min(0)).max())
    edge = extent + 20.0 * args.padding_nm
    box = np.full(3, edge)
    parts = groups_of(sequences, types)
    log(f"{len(sequences)} dipeptide probes in {len(parts)} simulations of "
        f"{min(len(g) for g in parts)}-{max(len(g) for g in parts)} types, {copies} copies each, "
        f"in a {edge / 10:.1f} nm box")  # fmt: skip
    with (root / "assignment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["simulation", "probe", "residues", "copies"])
        for s, group in enumerate(parts):
            for seq in group:
                w.writerow([f"sim_{s:03d}", seq, "-".join(seq), copies])
    sims = []
    for s, group in enumerate(parts):
        d = root / f"sim_{s:03d}"
        sims.append(d)
        if (d / "md").is_dir() and any((d / "md").iterdir()):
            continue  # started: leave it be
        d.mkdir(exist_ok=True)
        rng = np.random.default_rng([int(args.seed), s])
        probes = [probe(q) for q in group]
        if sirah:
            system = build_sirah(protein, probes, copies, box, rng, args.saltM, clearance,
                                 log=log if s == 0 else None)  # fmt: skip
            built = d / "sirah"
            whole = _built_system(system)
            system.save(built, system=whole)  # the force field beside it, .dms and .gro both
        else:
            system = build(protein, probes, copies, box, rng, args.saltM, clearance)
            built = d / "martini"
            whole = _built_system(system, args.martini_itp)
            system.save(built, martini_itp=args.martini_itp, system=whole)  # .dms and .gro
            if protein.ss:  # DSSP cannot read beads: dihedral_restraint = 'ss' reads this back
                (built / "secondary.txt").write_text(protein.ss + "\n")
        if whole is not None:
            # what to open in a viewer: the backbone named CA and no rubber bands
            viewing = (system.for_viewing(whole) if sirah else
                       system.for_viewing(whole, martini_itp=args.martini_itp))  # fmt: skip
            for suffix in (".dms", ".mae"):
                save(viewing, built / f"view{suffix}")
        # the run reads cg.dms, which carries every parameter the topology gave
        # it, so it needs neither the topology nor the force field beside it;
        # with parameters named to be resolved elsewhere there is no cg.dms to
        # read, and the run reads the topology as GROMACS would
        run_from = built / ("cg.dms" if whole is not None else "topol.top")
        settings = {**settings_of(args), "model": args.model, "solvate": "none",
                    "input_structure": str(run_from.resolve()),
                    "workdir": str((d / "md").resolve())}  # fmt: skip
        # what built the system, and what only an all-atom run has, are not its
        # settings; nor is the other model's (a SIRAH run takes no elastic network)
        for key in ("upper", "lower", "size_nm", "opm", "shift_nm", "elastic",
                    "elastic_selection", "cg_selection", "termini",
                    "neutral_termini", "lipid_itp", *ALL_ATOM_ONLY,
                    *(MARTINI_ONLY if sirah else ())):  # fmt: skip
            settings.pop(key, None)
        probes_are = "resname " + " ".join(group)
        if repel and "repulsion_selection" not in args.specified:
            settings["repulsion_selection"] = probes_are  # probes apart
        if (settings.get("dihedral_restraint", "none") != "none"
                and "dihedral_restraint_selection" not in args.specified):  # fmt: skip
            settings["dihedral_restraint_selection"] = f"not ({probes_are})"  # probes swim
        write_settings(d / "md.toml", settings)
        # what the analysis needs to know about a coarse-grained run
        (d / "probes.json").write_text(json.dumps(
            {"probes": list(group), "copies": copies, "ligand": "resname " + " ".join(group),
             "align": "name GC" if sirah else "name BB"}, indent=1) + "\n")  # fmt: skip
    (root / "simulations.txt").write_text(
        "".join(f"boonza md --config {(d / 'md.toml').resolve()}\n" for d in sims)
    )
    log(f"{len(sims)} simulations in {root}: run each line of {root / 'simulations.txt'} "
        "(a job array), or pass --run")  # fmt: skip
    return sims
