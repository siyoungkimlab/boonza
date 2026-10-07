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
from .swim import concentration_of, copies_for

CLEARANCE = 5.0  # Å between a placed probe and the protein or another probe
#: The chain the probes go in, as an all-atom swim gives its ligand library one
#: of its own.  Theirs was the protein's chain once, which made every
#: chain-based selection ambiguous.
PROBE_CHAIN = "LIG"


def _chains_in(mapped) -> set[str]:
    """The chains the mapped protein's beads carry, Martini's or SIRAH's."""
    out: set[str] = set()
    for mol in mapped.molecules:
        beads = getattr(mol, "beads", None)
        out |= ({str(b.chain or "") for b in beads} if beads is not None
                else {str(n.get("chain", "") or "") for n in mol.nodes})  # fmt: skip
    return out


def probe_chain(mapped) -> str:
    """A chain for the probes that the protein does not use already."""
    used = _chains_in(mapped)
    return next(c for c in (PROBE_CHAIN, *(f"{PROBE_CHAIN}{k}" for k in range(2, 1000)))
                if c not in used)  # fmt: skip


def in_chain(probe, chain: str):
    """``probe`` with every bead in ``chain``; returns it."""
    for mol in probe.molecules:
        beads = getattr(mol, "beads", None)
        if beads is not None:
            for bead in beads:
                bead.chain = chain
        else:
            for node in mol.nodes:
                node["chain"] = chain
    return probe


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


def _quiet(*_args, **_kwargs) -> None:
    """Say nothing: what the first simulation logged holds for the rest."""


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
            elastic: bool = True, conc_mM: float | None = None) -> list[Path]:  # fmt: skip
    """Write one ``boonza md`` coarse-grained simulation per group of probes into
    ``args.workdir``; returns their directories.

    Martini or SIRAH, by ``args.model``: each maps the protein with its own
    tool, places its own probes and fills the box with its own water.
    """
    from .config import ALL_ATOM_ONLY, MARTINI_ONLY, settings_of, write_settings
    from .prepare import _check_nothing_is_dropped, _write_built, load_input

    gromacs = bool(getattr(args, "gromacs", False))

    sirah = args.model == "sirah"
    if sirah:
        from ..sirah.probes import probe, probe_sequences
    else:
        from ..martini.probes import probe, probe_sequences

    if args.input_structure is None:
        raise ValueError("give the protein structure")
    sequences = list(sequences) if sequences else probe_sequences()
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    # SIRAH maps polar hydrogens onto beads of their own; Martini maps none
    aa = load_input(args.input_structure, log, hydrogens=sirah)
    _check_nothing_is_dropped(aa, args, Path(args.input_structure), log)
    if sirah:
        from ..sirah import sirahize

        protein = sirahize(aa, args.cg_selection, termini=args.termini, log=log,
                           strict=bool(getattr(args, "strict_mapping", False)))  # fmt: skip
        made_of = {}  # SIRAH has one version
        log(f"SIRAH: {protein.nbeads} beads in {len(protein.molecules)} molecule(s)"
            f"{', elastic network' if elastic else ''}")  # fmt: skip
    else:
        from ..martini import FORCEFIELD_FOR, martinize

        version = int(args.model.removeprefix("martini"))
        made_of = {"forcefield": FORCEFIELD_FOR[version]}  # the probes are made of it too
        protein = martinize(aa, args.cg_selection, elastic=elastic,
                            elastic_selection=args.elastic_selection if elastic else None,
                            elastic_fc=float(args.elastic_kJ),
                            elastic_upper=10.0 * float(args.elastic_nm),  # nm to A
                            elastic_lower=10.0 * float(args.elastic_lower_nm),
                            neutral_termini=bool(args.neutral_termini),
                            **made_of)  # fmt: skip
        log(f"Martinized as Martini {version}: {protein.nbeads} beads in "
            f"{len(protein.molecules)} molecule(s)"
            f"{', elastic network' if elastic else ''}")  # fmt: skip
    chain = probe_chain(protein)  # the probes', free of the protein's
    extent = float((protein.positions.max(0) - protein.positions.min(0)).max())
    edge = extent + 20.0 * args.padding_nm
    box = np.full(3, edge)
    parts = groups_of(sequences, types)
    sizes = [len(g) for g in parts]
    if conc_mM is not None:
        # one number of copies for every type and every simulation, so the
        # simulations that hold a type more land a little above the asked for
        copies = copies_for(conc_mM, edge, sum(sizes) / len(sizes))
        got = [concentration_of(copies, edge, n) for n in (min(sizes), max(sizes))]
        asked = f"{conc_mM:g} mM asked"
        if copies == 1 and got[0] > 1.2 * conc_mM:
            log(f"One copy of each type is {got[0]:.0f}-{got[1]:.0f} mM, which is more than the "
                f"{conc_mM:g} mM asked for: fewer types a simulation (--types) or a larger box "
                "(--padding-nm) is what lowers it")  # fmt: skip
        else:
            log(f"{copies} copies of each type: {got[0]:.0f}-{got[1]:.0f} mM ({asked})")
    kind = "amino acid" if max(len(q) for q in sequences) == 1 else "dipeptide"
    log(f"{len(sequences)} {kind} probes in {len(parts)} simulations of "
        f"{min(sizes)}-{max(sizes)} types, {copies} copies each, "
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
        # the probes are of the protein's own Martini: the two meet through
        # their bead types, and those are not the same between the versions
        probes = [in_chain(probe(q, **made_of), chain) for q in group]
        if sirah:
            system = build_sirah(protein, probes, copies, box, rng, args.saltM, clearance,
                                 log=log if s == 0 else None)  # fmt: skip
            built = d / "sirah"
            whole = _built_system(system)
            _write_built(system, built, whole, gromacs or whole is None,
                         log if s == 0 else _quiet)  # fmt: skip
            if protein.ss:  # DSSP cannot read beads: dihedral_restraint = 'ss' reads this back
                (built / "secondary.txt").write_text(protein.ss + "\n")
        else:
            system = build(protein, probes, copies, box, rng, args.saltM, clearance)
            built = d / "martini"
            whole = _built_system(system, args.martini_itp)
            _write_built(system, built, whole, gromacs or whole is None,
                         log if s == 0 else _quiet, martini_itp=args.martini_itp)  # fmt: skip
            if protein.ss:  # DSSP cannot read beads: dihedral_restraint = 'ss' reads this back
                (built / "secondary.txt").write_text(protein.ss + "\n")
        if whole is not None and not sirah:
            # what to open in a viewer: the system without its rubber bands,
            # which a viewer would otherwise draw as a hairball.  The beads keep
            # their names: a viewer that knows amino acids reads a renamed
            # "GLU: CA SC1" as a broken residue and draws its own bonds over it.
            # SIRAH has no network to leave out, so cg.dms is what to open.
            viewing = system.for_viewing(whole, martini_itp=args.martini_itp)
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
        # settings; nor is the other model's.  'elastic' goes with it: it is
        # answered here, by martinize's bands or by the springs written below
        for key in ("upper", "lower", "size_nm", "opm", "shift_nm", "elastic",
                    "elastic_selection", "cg_selection", "termini",
                    "neutral_termini", "lipid_itp", *ALL_ATOM_ONLY,
                    *(MARTINI_ONLY if sirah else ())):  # fmt: skip
            settings.pop(key, None)
        if repel and "repulsion_selection" not in args.specified:
            settings["repulsion_selection"] = f"chain {chain}"  # probes apart
        if sirah and elastic and "elastic_network_selection" not in args.specified:
            # what Martini's rubber bands do for the fold, SIRAH has to be given:
            # springs between the alpha carbons of the protein, never the probes'
            settings["elastic_network_selection"] = f"name GC and not chain {chain}"
        if (settings.get("dihedral_restraint", "none") != "none"
                and "dihedral_restraint_selection" not in args.specified):  # fmt: skip
            settings["dihedral_restraint_selection"] = f"not chain {chain}"  # probes swim
        write_settings(d / "md.toml", settings)
        # what the analysis needs to know about a coarse-grained run
        (d / "probes.json").write_text(json.dumps(
            {"probes": list(group), "copies": copies, "chain": chain,
             "concentration_mM": round(concentration_of(copies, edge, len(group)), 1),
             "ligand": "resname " + " ".join(group),
             "align": "name GC" if sirah else "name BB"}, indent=1) + "\n")  # fmt: skip
    (root / "simulations.txt").write_text(
        "".join(f"boonza md --config {(d / 'md.toml').resolve()}\n" for d in sims)
    )
    log(f"{len(sims)} simulations in {root}: run each line of {root / 'simulations.txt'} "
        "(a job array), or pass --run")  # fmt: skip
    return sims
