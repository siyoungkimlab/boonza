"""``boonza swim`` in Martini: dipeptide probes swimming around a coarse-grained protein.

Martini has no general way to parameterize a small molecule, so the probes
are dipeptides, which need nothing new (see :mod:`boonza.martini.probes`).
Each simulation holds a group of probe types, several copies each, around
the martinized protein in water.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

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


def prepare(args, sequences=None, types: int = 10, copies: int = 5,
            clearance: float = CLEARANCE, log=print, repel: bool = True) -> list[Path]:  # fmt: skip
    """Write one ``boonza md`` Martini simulation per group of probes into
    ``args.workdir``; returns their directories."""
    from ..martini import martinize
    from ..martini.probes import probe, probe_sequences
    from .config import ALL_ATOM_ONLY, settings_of, write_settings
    from .prepare import load_input

    if args.input_structure is None:
        raise ValueError("give the protein structure")
    sequences = list(sequences) if sequences else probe_sequences()
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    aa = load_input(args.input_structure, log, hydrogens=False)
    elastic = args.elastic if "elastic" in args.specified else True
    protein = martinize(aa, args.cg_selection, elastic=elastic,
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
        system = build(protein, [probe(q) for q in group], copies, box, rng, args.saltM,
                       clearance)  # fmt: skip
        system.save(d / "martini", martini_itp=args.martini_itp)
        settings = {**settings_of(args), "model": args.model, "solvate": "none",
                    "input_structure": str((d / "martini" / "topol.top").resolve()),
                    "workdir": str((d / "md").resolve())}  # fmt: skip
        # what built the system, and what only an all-atom run has, are not its settings
        for key in ("upper", "lower", "size_nm", "opm", "shift_nm", "elastic", "cg_selection",
                    "neutral_termini", "lipid_itp", *ALL_ATOM_ONLY):  # fmt: skip
            settings.pop(key, None)
        if repel and "repulsion_selection" not in args.specified:
            settings["repulsion_selection"] = "resname " + " ".join(group)  # probes apart
        write_settings(d / "md.toml", settings)
        # what the analysis needs to know about a coarse-grained run
        (d / "probes.json").write_text(json.dumps(
            {"probes": list(group), "copies": copies, "ligand": "resname " + " ".join(group),
             "align": "name BB"}, indent=1) + "\n")  # fmt: skip
    (root / "simulations.txt").write_text(
        "".join(f"boonza md --config {(d / 'md.toml').resolve()}\n" for d in sims)
    )
    log(f"{len(sims)} simulations in {root}: run each line of {root / 'simulations.txt'} "
        "(a job array), or pass --run")  # fmt: skip
    return sims
