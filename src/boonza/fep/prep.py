"""boonza fep prep: from a receptor and posed ligands to hybrid systems to run.

Ligands come from an SDF, posed in the receptor's frame.  Each is given GAFF2
types and AM1-BCC charges once (at a reference conformer, so in any pose the
same; boonza.gaff), and every pair the network chooses (boonza.fep.network)
gets two legs: in the receptor (complex) and in water (solvent); without a
receptor, in water and in vacuum (a relative hydration free energy), or in
water and in 1-octanol, dry or wet (a relative water-to-octanol transfer free
energy, boonza.fep.octanol).

A leg is solvated with both ligands in place, so the waters avoid both; state
A is that box without ligand B, state B without ligand A, and the two share
every other atom in order -- what a hybrid topology needs.  The hybrid
(boonza.fep.hybrid) is written beside the settings it runs with, and
``commands.txt`` lists one ``boonza fep run`` per leg and repeat.

Layout::

    WORKDIR/ligands.csv          the ligands, their charges and SMILES
    WORKDIR/ligands.sdf          the ligands as given, every field kept (result adds its own)
    WORKDIR/receptor.EXT         the receptor the complex legs are built in (boonza view
                                 WORKDIR); EXT the input's: mae, dms, cif or pdb (mae
                                 for any other, which keeps bond orders)
    WORKDIR/edges.csv            the pairs, how many atoms map, their scores
    WORKDIR/forcefield/          the GAFF2 patch (every ligand, any cofactor)
    WORKDIR/edges/A~B/complex/   system.xml.gz, state.npz, leg.json
    WORKDIR/edges/A~B/solvent/
    WORKDIR/edges/A~B/vacuum/      (no receptor: hydration)
    WORKDIR/edges/A~B/octanol/     (no receptor, --phase octanol: transfer)
    WORKDIR/octanol/box.dms      the equilibrated octanol box the octanol legs tile
    WORKDIR/receptor_rem.EXT, rem.csv, rem_residues.csv   (--rem) the receptor relaxed
    WORKDIR/commands.txt
    WORKDIR/commands_new.txt     (--extend) the runs of the edges just added

With ``extend``, ligands are added to a work directory prepared before: the SDF
holds the first prep's ligands (the same molecules in the same poses) and the
new ones, and only edges that join the new ones are built, in the receptor the
first edges were built in (receptor_rem.EXT, if --rem made one), with the first
prep's templates and charges kept; its settings must be the same.  Every edge
already run is reused, and result analyzes them all as one network.
"""

from __future__ import annotations

import csv
import gzip
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import gaff, viparr
from ..system import System
from . import lambdas, network
from .hybrid import _md

#: the solvent leg's padding (openfe: 1.5 nm), the complex leg's is boonza md's
SOLVENT_PADDING_NM = 1.5
#: an alchemical water is at least this far from the solute (openfe: 0.8 nm)
ALCHEMICAL_WATER_NM = 0.8
#: a change of net charge scores this times LOMAP's score, so a network takes
#: one only to join ligands of different charge
CHARGE_CHANGE_PENALTY = 0.1
#: and runs openfe's longer protocol (its industry benchmark's settings)
CHARGE_WINDOWS = 22
CHARGE_PRODUCTION_NS = 20.0
#: an edge where either ligand has at least this many atoms of its own (not mapped
#: to the other's) runs boonza.fep.lambdas.dense(); 0 never
LARGE_CHANGE_ATOMS = 8
#: without a receptor, what the water leg is set against (boonza fep prep --phase)
PHASES = ("vacuum", "octanol", "wet-octanol")
#: what the receptor is written as (receptor.EXT, receptor_rem.EXT): the input's
#: format if one of these, else the first, which keeps bond orders as PDB does not
RECEPTOR_FORMATS = (".mae", ".dms", ".cif", ".pdb")


def receptor_format(path) -> str:
    """The extension to write a receptor read from ``path`` with (RECEPTOR_FORMATS)."""
    from ..io import _format

    fmt = {".cms": ".mae", ".mmcif": ".cif"}.get(_format(path, None), _format(path, None))
    return fmt if fmt in RECEPTOR_FORMATS else RECEPTOR_FORMATS[0]


def load_ligands(path) -> tuple[list, list[System], list[str]]:
    """RDKit molecules, boonza systems (atoms in the same order) and names of the
    ligands in an SDF, each with all its hydrogens and its pose."""
    from rdkit import Chem

    from ..io import load

    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False)]
    if any(m is None for m in mols):
        raise ValueError(f"{path}: RDKit cannot read every record")
    if not all(any(a.GetAtomicNum() == 1 for a in m.GetAtoms()) for m in mols):
        raise ValueError(f"{path}: the ligands need all their hydrogens")
    s = load(path)
    if s.ncts != len(mols):
        raise ValueError(f"{path}: {s.ncts} records read as systems, {len(mols)} by RDKit")
    systems, names = [], []
    for k, m in enumerate(mols):
        lig = s.clone(s.ct_atoms(k))
        if lig.natoms != m.GetNumAtoms():
            raise ValueError(f"{path}: record {k + 1} differs between readers")
        lig.residues["name"] = [f"L{k:02d}"] * lig.nresidues  # one template each
        systems.append(lig)
        name = m.GetProp("_Name").strip() if m.HasProp("_Name") else ""
        names.append(name or f"ligand{k + 1}")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: the ligands' names must differ: {names}")
    return mols, systems, names


@dataclass
class Previous:
    """What a work directory prepared before holds (boonza fep prep --extend)."""

    names: list  # the ligands, in their order (their indices in leg.json)
    pairs: list  # (a, b) of every edge
    rows: list  # edges.csv
    commands: list  # commands.txt
    metas: list  # every leg.json
    receptor: Path | None  # receptor_rem.EXT, if --rem relaxed the receptor


def previous(work: Path, mols, names) -> Previous:
    """The prep already in ``work``, checked against the ligands now given: each of
    its ligands must be among them, the same molecule (SMILES) in the same pose
    (ligands.sdf, which prep keeps)."""
    from rdkit import Chem

    work = Path(work)
    if not (work / "ligands.csv").exists() or not (work / "edges").is_dir():
        raise ValueError(f"--extend adds to a prep already in --workdir: {work} has none")
    table = list(csv.DictReader((work / "ligands.csv").open()))
    old = [r["name"] for r in table]
    missing = [n for n in old if n not in names]
    if missing:
        raise ValueError(f"--extend needs the first prep's ligands in the SDF too; missing: "
                         f"{', '.join(missing)}")  # fmt: skip
    by_name = dict(zip(names, mols, strict=True))
    for r in table:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(by_name[r["name"]]))
        if smiles != r["smiles"]:
            raise ValueError(f"{r['name']} is not the molecule the first prep had: {smiles}, "
                             f"not {r['smiles']}")  # fmt: skip
    kept = work / "ligands.sdf"
    if kept.exists():
        for m in Chem.SDMolSupplier(str(kept), removeHs=False):
            name = m.GetProp("_Name").strip()
            now = by_name[name].GetConformer().GetPositions()
            was = m.GetConformer().GetPositions()
            if now.shape != was.shape or np.abs(now - was).max() > 1e-3:
                raise ValueError(f"{name} is not in the pose the first prep had (ligands.sdf)")
    with (work / "edges.csv").open() as fh:
        rows = [r for r in csv.reader(fh)][1:]
    index = {n: k for k, n in enumerate(old)}
    pairs = [(index[r[0]], index[r[1]]) for r in rows]
    commands = (work / "commands.txt").read_text().splitlines()
    metas = [json.loads(p.read_text()) for p in sorted((work / "edges").glob("*/*/leg.json"))]
    if not metas:
        raise ValueError(f"{work}: no legs to add to")
    rem = next((work / f"receptor_rem{s}" for s in RECEPTOR_FORMATS
                if (work / f"receptor_rem{s}").is_file()), None)  # fmt: skip
    return Previous(old, pairs, rows, [c for c in commands if c.strip()], metas, rem)


def schedule(n_old: int, n_new: int, charged: bool, windows: int, charge_windows: int,
             large_change_atoms: int = LARGE_CHANGE_ATOMS) -> np.ndarray:  # fmt: skip
    """The master lambdas of an edge whose ligands have ``n_old`` and ``n_new`` atoms
    of their own: ``charge_windows`` for a change of charge, the dense schedule for
    a large group (``large_change_atoms``), else ``windows`` evenly spaced."""
    if charged:
        return lambdas.windows(charge_windows)
    if large_change_atoms and max(n_old, n_new) >= large_change_atoms:
        return lambdas.dense()
    return lambdas.windows(windows)


def same_settings(before: Previous, args, repeats, windows, iteration_ps, phase,
                  charge_windows, charge_production_ns,
                  large_change_atoms: int = LARGE_CHANGE_ATOMS) -> None:  # fmt: skip
    """Raise unless the new edges would be built as the first prep's were."""
    differ = []
    for meta in before.metas:
        charged = bool(meta.get("charge_change"))
        own = meta.get("unique_old", 0), meta.get("unique_new", 0)
        want_windows = len(schedule(*own, charged, windows, charge_windows, large_change_atoms))
        want = {"repeats": repeats, "iteration_ps": iteration_ps, "phase": phase,
                "temperature": args.temperature, "pressure": args.pressure,
                "timestep_fs": args.integration_fs,
                "equilibration_ns": args.equilibration_ns,
                "production_ns": charge_production_ns if charged else args.production_ns,
                "windows": want_windows}  # fmt: skip
        had = dict(meta, windows=len(meta["lambdas"]))
        for k, v in want.items():
            if k not in had:
                continue
            same = had[k] == v if isinstance(v, str) else np.isclose(float(had[k]), float(v))
            if not same:
                differ.append(f"{k} {had[k]} (now {v})")
    if differ:
        raise ValueError("--extend builds the new edges as the first ones were; the first "
                         "prep had " + ", ".join(sorted(set(differ))))  # fmt: skip


def _not_kept(patch, kept):
    """``patch`` without the parameter rows ``kept`` has already, whichever way round
    their types are written (two GAFF2 runs write one improper ha-ca-ca-ca and
    ca-ca-ca-ha, which viparr then matches twice): the first prep's rows stay."""

    def keys(t: str) -> tuple[tuple, tuple]:
        mode = ""
        if t.startswith("__mode__"):  # a row's mode, written before its types
            mode, _, t = t.partition(" ")
        words = tuple(t.split())
        return (mode, words), (mode, words[::-1])

    out = patch.copy()
    out.templates = list(patch.templates)
    for table, rows in patch.params.items():
        have = {keys(r.type)[0] for r in kept.params.get(table, [])}
        out.params[table] = [r for r in rows if not set(keys(r.type)) & have]
    return out


def write_ligands(path: Path, mols, names) -> None:
    """The ligands as an SDF, each named and with every field it came with."""
    from rdkit import Chem

    with Chem.SDWriter(str(path)) as w:
        for m, n in zip(mols, names, strict=True):
            m = Chem.Mol(m)
            m.SetProp("_Name", n)
            w.write(m)


def _parameterize(system: System, ff, hmr: bool) -> System:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"fragment \d+ \([A-Z][a-z]?\) was matched by multiple",
                                viparr.ViparrWarning)  # fmt: skip
        out = viparr.parameterize(system, ff)
    if hmr:
        from ..build import repartition_hydrogen_masses
        from ..md.config import HYDROGEN_MASS_AMU

        out = repartition_hydrogen_masses(out, "not water", HYDROGEN_MASS_AMU)
    return out


def _by_tag(s: System, n: int) -> np.ndarray:
    """Where each tagged atom (fep_index 1..n) ended up after parameterizing."""
    tag = np.asarray(s.atoms["fep_index"], dtype=np.int64)
    where = np.full(n, -1, dtype=np.int64)
    where[tag[tag > 0] - 1] = np.flatnonzero(tag > 0)
    return where


@dataclass
class Leg:
    """One leg's two end states (boonza systems and OpenMM systems) and their hybrid."""

    hybrid: object
    old: System
    new: System
    old_system: object
    new_system: object
    dropped: int  # atoms left out of the core because their constraints differ
    alchemical_waters: list = None  # hybrid indices of the waters that become ions
    #: B's long-range dispersion coefficient less A's (dispersion_coefficient), for
    #: a periodic leg; the hybrid runs without one, so result adds it back
    dispersion_dK: float = 0.0


def build_leg(env: System | None, lig_a: System, lig_b: System, core: dict, ff, args,
              padding_nm: float, log=print, charge_change: int = 0,
              solvent: System | None = None):  # fmt: skip
    """The two end states of one leg and their hybrid: ``env`` (the receptor, or
    None in water) with ligand A, and with ligand B; ``core`` maps A's atoms to
    B's.  ``padding_nm`` None: in vacuum, no water and no cutoff.  ``solvent``:
    a periodic box (octanol) to tile in whole copies instead of TIP3P, with no
    ions, cleared where its heavy atoms come near the ligands.

    ``charge_change``, B's net charge less A's, is balanced by as many waters
    far from the solute becoming ions in B (openfe's, after perses): Cl- for a
    gain, Na+ for a loss, so every state's box is neutral.  The water keeps its
    atoms, masses and constraints; its oxygen takes the ion's charge and
    Lennard-Jones, its hydrogens lose their charge."""
    from ..build import neutralize, solvate
    from ..omm import to_openmm
    from .hybrid import build

    parts = ([env] if env is not None else []) + [lig_a, lig_b]
    combined = parts[0].copy()
    for p in parts[1:]:
        combined.append(p)
    ne = env.natoms if env is not None else 0
    na, nb = lig_a.natoms, lig_b.natoms
    a_ids = np.arange(ne, ne + na)
    b_ids = np.arange(ne + na, ne + na + nb)
    # the counterions state A needs; B carries the same charge (network.edges)
    solute_a = combined.clone(np.arange(ne + na))
    charge = float(_parameterize(solute_a, ff, False).atoms["charge"].sum())
    if padding_nm is None:  # in vacuum: the two ligands alone
        box = combined.copy()
    elif solvent is not None:  # octanol: a box of whole copies, so no seam
        from .octanol import OCTANOL_RADIUS

        if charge_change or abs(charge) > 1e-3:
            raise ValueError("an octanol leg takes neutral ligands: it has no ions to "
                             "balance a charge")  # fmt: skip
        from ..build import _tiled

        edge = np.diag(np.asarray(solvent.cell, float))
        pos = np.asarray(combined.positions, float)
        need = float((pos.max(0) - pos.min(0)).max()) + 2 * 10.0 * padding_nm
        need = max(need, 2 * 10.0 * args.cutoff_nm * 1.05)  # PME: wider than two cutoffs
        # the copies tiled first, so the box is exactly as wide as they are
        n = np.ceil(need / edge).astype(int)
        cells = _tiled(solvent, n, np.zeros(3), edge)
        cells.cell = np.diag(n * edge)
        box = solvate(combined, solvent=cells, box=np.diag(cells.cell),
                      solvent_selection="noh", min_solute_dist=OCTANOL_RADIUS)  # fmt: skip
    else:
        box = solvate(combined, thickness=10.0 * padding_nm)
        box = neutralize(box, cation="Na", anion="Cl", charge=charge, concentration=args.saltM)
    box.atoms["fep_index"] = np.arange(1, box.natoms + 1)
    old_ids = np.setdiff1d(np.arange(box.natoms), b_ids)  # in order: env, A, solvent
    new_ids = np.setdiff1d(np.arange(box.natoms), a_ids)
    old = _parameterize(box.clone(old_ids), ff, args.hmr)
    new = _parameterize(box.clone(new_ids), ff, args.hmr)
    where_old, where_new = _by_tag(old, box.natoms), _by_tag(new, box.natoms)
    shared = np.setdiff1d(np.arange(box.natoms), np.concatenate([a_ids, b_ids]))
    environment = {int(where_old[t]): int(where_new[t]) for t in shared}
    core_idx = {int(where_old[ne + i]): int(where_new[ne + na + j]) for i, j in core.items()}
    options = dict(nonbonded_method="PME" if padding_nm is not None else "NoCutoff",
                   cutoff=10.0 * args.cutoff_nm, constraints=True,
                   dispersion_correction=False)  # fmt: skip
    _, sys_old, _ = to_openmm(old, **options)
    _, sys_new, _ = to_openmm(new, **options)
    core_idx, dropped = _constraint_fix(sys_old, sys_new, core_idx)
    waters = []
    if charge_change:
        if padding_nm is None:
            raise ValueError("a change of net charge needs water to put its ion in")
        for res in _alchemical_waters(box, abs(charge_change)):
            tags = np.flatnonzero(box.atoms["residue"] == res)
            _water_to_ion(sys_new, where_new[tags], box.atoms["anum"][tags],
                          _ion_parameters("Cl" if charge_change > 0 else "Na", ff))  # fmt: skip
            for t in tags:
                core_idx[int(where_old[t])] = environment.pop(int(where_old[t]))
            waters.append(int(where_old[tags[box.atoms["anum"][tags] == 8][0]]))
        log(f"  net charge {charge_change:+d}: {len(waters)} water(s) become "
            f"{'Cl-' if charge_change > 0 else 'Na+'}")  # fmt: skip
    hyb = build(sys_old, sys_new, np.asarray(old.positions) / 10.0,
                np.asarray(new.positions) / 10.0, core_idx, environment)  # fmt: skip
    dk = 0.0
    if padding_nm is not None:
        rc = 10.0 * args.cutoff_nm
        dk = dispersion_coefficient(new, rc) - dispersion_coefficient(old, rc)
    return Leg(hyb, old, new, sys_old, sys_new, dropped, waters, dk)


def dispersion_coefficient(s: System, cutoff: float) -> float:
    """K of a periodic system's long-range dispersion correction, E = K / V
    (K in kcal/mol Å^3, V in Å^3): the homogeneous Lennard-Jones tail past
    ``cutoff`` (Å), 4π over every pair of A/(9 rc^9) - B/(3 rc^3), summed by
    atom type, every pair once and no atom with itself.  OpenMM's
    NonbondedForce counts an atom with itself too, which moves K by 1/N of
    itself; between a leg's two end states, which share every atom but the
    ligands', that offset all but cancels, so the difference agrees to 1/N.
    A hybrid cannot carry the correction, so a leg's two end states' K are
    taken here and their difference reported by result."""
    from ..omm import _combine

    nb = s.tables.get("nonbonded")
    if nb is None or not len(nb.params):
        return 0.0
    rule = s.nonbonded_info.vdw_rule.lower() or "arithmetic/geometric"
    ps = np.asarray(nb.params["sigma"], float)
    pe = np.asarray(nb.params["epsilon"], float)
    ss, ee = _combine(rule, ps[:, None], pe[:, None], ps[None, :], pe[None, :])
    ss, ee = np.array(ss, float), np.array(ee, float)
    for (p, q), vals in nb.overrides.items():
        ss[p, q] = ss[q, p] = vals["sigma"]
        ee[p, q] = ee[q, p] = vals["epsilon"]
    t = 4 * ee * ss**12 / (9 * cutoff**9) - 4 * ee * ss**6 / (3 * cutoff**3)
    n = np.bincount(np.asarray(nb.param_ids), minlength=len(ps)).astype(float)
    pairs = 0.5 * (n @ t @ n - n @ np.diag(t))  # i < j: every pair once, no atom with itself
    return float(4 * np.pi * pairs)


def _alchemical_waters(box: System, n: int) -> list[int]:
    """``n`` water residues of ``box`` to become ions: the farthest from the
    solute (both ligands, the receptor: all but water and ions), at least ALCHEMICAL_WATER_NM
    from it and 1 nm from each other."""
    from ..spatial import min_dist2

    oxygens = box.select("water and noh").ids
    frag = np.asarray(box.fragids)
    lone = np.bincount(frag)[frag] == 1  # ions
    others = np.flatnonzero(~np.isin(np.arange(box.natoms), box.select("water").ids) & ~lone)
    pos = np.asarray(box.positions, float)
    reach = 30.0  # Å; farther counts as this far
    d = np.sqrt(np.minimum(min_dist2(pos[oxygens], pos[others], reach, cell=box.cell), reach**2))
    order = np.argsort(-d, kind="stable")
    chosen = []
    for k in order:
        if d[k] < 10.0 * ALCHEMICAL_WATER_NM or len(chosen) == n:
            break
        if chosen and np.sqrt(min_dist2(pos[oxygens[[k]]], pos[oxygens[chosen]], 10.0,
                                        cell=box.cell))[0] < 10.0:  # fmt: skip
            continue
        chosen.append(int(k))
    if len(chosen) < n:
        raise ValueError(f"no {n} waters {ALCHEMICAL_WATER_NM} nm from the solute (and 1 nm "
                         "apart) to turn into ions: give the box more padding")  # fmt: skip
    return [int(box.atoms["residue"][oxygens[k]]) for k in chosen]


_ION_PARAMETERS: dict = {}


def _ion_parameters(name: str, ff) -> tuple[float, float, float]:
    """The charge (e), sigma (nm) and epsilon (kJ/mol) ``ff`` gives a lone ion."""
    key = (name, id(ff))
    if key not in _ION_PARAMETERS:
        from ..build import _ion
        from ..elements import MSYS_MASSES
        from ..omm import to_openmm

        x = _ion(name)
        ion = System.from_arrays(np.zeros((1, 3)), names=[x["name"]], anum=[x["anum"]],
                                 resnames=[""], resids=[1], chains=["ION"],
                                 charge=[x["charge"]], formal_charge=[int(x["charge"])],
                                 mass=[float(MSYS_MASSES[x["anum"]])])  # fmt: skip
        _, s, _ = to_openmm(_parameterize(ion, ff, False), nonbonded_method="NoCutoff",
                            dispersion_correction=False)  # fmt: skip
        nb = next(f for f in s.getForces() if type(f).__name__ == "NonbondedForce")
        _ION_PARAMETERS[key] = tuple(_md(v) for v in nb.getParticleParameters(0))
    return _ION_PARAMETERS[key]


def _water_to_ion(system, atoms, anum, ion) -> None:
    """In ``system``, the water ``atoms`` (elements ``anum``) become the ion:
    the oxygen takes its charge, sigma and epsilon; the hydrogens lose their charge."""
    nb = next(f for f in system.getForces() if type(f).__name__ == "NonbondedForce")
    if sorted(anum.tolist()) != [1, 1, 8]:
        raise ValueError("an alchemical water must be three atoms (no virtual sites)")
    for i, z in zip(atoms.tolist(), anum.tolist(), strict=True):
        if z == 8:
            nb.setParticleParameters(i, *ion)
        else:
            _, sigma, epsilon = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, sigma, epsilon)


def _constraint_fix(sys_old, sys_new, core: dict) -> tuple[dict, int]:
    """Leave out of the core any atom whose constrained bond differs between the
    two states (in length, or constrained in one only): a hybrid holds one length."""

    def constraints(s):
        out = {}
        for k in range(s.getNumConstraints()):
            i, j, d = s.getConstraintParameters(k)
            out[min(i, j), max(i, j)] = _md(d)
        return out

    co, cn = constraints(sys_old), constraints(sys_new)
    core = dict(core)
    dropped = 0
    changed = True
    while changed:
        changed = False
        for (i, j), d in list(co.items()):
            if i in core and j in core:
                key = (min(core[i], core[j]), max(core[i], core[j]))
                if key not in cn or abs(cn[key] - d) > 1e-6:
                    light = i if sys_old.getParticleMass(i) <= sys_old.getParticleMass(j) else j
                    core.pop(light)
                    dropped += 1
                    changed = True
        inv = {v: k for k, v in core.items()}
        for i, j in list(cn):
            if i in inv and j in inv:
                key = (min(inv[i], inv[j]), max(inv[i], inv[j]))
                if key not in co:
                    light = inv[i] if sys_new.getParticleMass(i) <= sys_new.getParticleMass(j) \
                        else inv[j]  # fmt: skip
                    core.pop(light, None)
                    dropped += 1
                    changed = True
                    inv = {v: k for k, v in core.items()}
    return core, dropped


def _finish(hyb, old: System, args, complex_leg: bool, log=print):
    """What boonza md would add: the barostat, and (complex) its restraints."""
    import openmm

    if hyb.system.usesPeriodicBoundaryConditions():  # not in vacuum
        hyb.system.addForce(openmm.MonteCarloBarostat(args.pressure, args.temperature, 25))
    if complex_leg and args.dihedral_restraint != "none":
        from ..md.restraints import add_dihedral_restraints

        records, what = add_dihedral_restraints(
            hyb.system, old, args.dihedral_restraint, args.dihedral_restraint_kJ,
            getattr(args, "dihedral_restraint_selection", None), None,
        )  # fmt: skip
        log(f"  dihedral restraints: {len(records)} backbone torsions over {what}")


def write_leg(directory: Path, hyb, old: System, meta: dict) -> None:
    import openmm

    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / "system.xml.gz", "wt") as fh:
        fh.write(openmm.XmlSerializer.serialize(hyb.system))
    np.savez(directory / "state.npz", positions=hyb.positions,
             box=np.asarray(old.cell, float) / 10.0, core=np.asarray(hyb.core),
             unique_old=np.asarray(hyb.unique_old),
             unique_new=np.asarray(hyb.unique_new))  # fmt: skip
    (directory / "leg.json").write_text(json.dumps(meta, indent=1))


def prep(args, ligands_path, repeats: int = 3, windows: int = 11, iteration_ps: float = 2.5,
         cofactors=(), log=print, charge_windows: int = CHARGE_WINDOWS,
         charge_production_ns: float = CHARGE_PRODUCTION_NS, phase: str = "vacuum",
         octanol_box=None, extend: bool = False,
         large_change_atoms: int = LARGE_CHANGE_ATOMS) -> list[Path]:  # fmt: skip
    """Build every leg of every edge in ``args.workdir``; the run directories.
    An edge that changes the net charge (only with a receptor) gets
    ``charge_windows`` windows and ``charge_production_ns`` of production; one
    where either ligand has ``large_change_atoms`` or more atoms of its own, the
    dense schedule (boonza.fep.lambdas.dense; 0 never).
    ``phase``, without a receptor, what the water leg is set against: vacuum
    (hydration), octanol or wet-octanol (transfer); ``octanol_box`` a box built
    before (boonza.fep.octanol.box), else one is built in the work directory.
    ``extend``: add ligands to the prep already in the work directory (the module
    docstring)."""
    from ..md.prepare import forcefields, load_input

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    mols, ligs, names = load_ligands(ligands_path)
    log(f"{len(ligs)} ligands from {ligands_path}")
    before, new = None, []
    if extend:  # the first prep's ligands keep their indices; the new ones follow
        before = previous(work, mols, names)
        order = ([names.index(n) for n in before.names]
                 + [k for k, n in enumerate(names) if n not in before.names])  # fmt: skip
        mols, ligs, names = ([x[k] for k in order] for x in (mols, ligs, names))
        for k, lig in enumerate(ligs):
            lig.residues["name"] = [f"L{k:02d}"] * lig.nresidues
        new = list(range(len(before.names), len(names)))
        if not new:
            raise ValueError("--extend: every ligand in the SDF is in the first prep already")
        log(f"  adding {len(new)} to the {len(before.names)} of the first prep: "
            + ", ".join(names[k] for k in new))  # fmt: skip
        same_settings(before, args, repeats, windows, iteration_ps, phase, charge_windows,
                      charge_production_ns, large_change_atoms)  # fmt: skip
    if phase not in PHASES:
        raise ValueError(f"--phase is one of {', '.join(PHASES)}, not {phase!r}")
    if phase != "vacuum" and args.input_structure:
        raise ValueError("--phase sets the leg a water leg is compared with, without a "
                         "receptor; with one the legs are complex and solvent")  # fmt: skip
    receptor = load_input(args.input_structure, log) if args.input_structure else None
    if cofactors and receptor is None:
        raise ValueError("cofactors go with a receptor")
    for k, path in enumerate(cofactors):  # each record its own template, as the ligands
        _, extra, _ = load_ligands(path)
        for j, c in enumerate(extra):
            c.residues["name"] = [f"C{k}{j}"] * c.nresidues
            receptor.append(c)
        log(f"{len(extra)} cofactor(s) from {path} added to the receptor")
    if before is not None and before.receptor is not None:  # what the first edges met
        from ..io import load

        relaxed = load(before.receptor)
        if (relaxed.natoms != receptor.natoms
                or list(relaxed.atoms["name"]) != list(receptor.atoms["name"])):  # fmt: skip
            raise ValueError(f"{before.receptor} is not this receptor (with its cofactors)")
        receptor.positions = np.asarray(relaxed.positions, float)
        log(f"the receptor as the first prep relaxed it ({before.receptor})")
    kind, ff = forcefields(args)
    if kind != "viparr":
        raise ValueError("boonza fep builds its systems with viparr force fields (-f "
                         "aa.amber.ff19SB ...), not OpenMM XML")  # fmt: skip
    # one GAFF2 patch for every ligand and cofactor, charged once
    everything = receptor.copy() if receptor is not None else ligs[0].copy()
    for lig in ligs if receptor is not None else ligs[1:]:
        everything.append(lig)
    if phase != "vacuum":  # and octanol, one template for every molecule of the box
        from . import octanol

        everything.append(octanol.molecule())
    host = gaff.host_index(ff)
    kept = None
    if before is not None:  # the first prep's templates as they were: only the new charged
        kept = viparr.load_forcefield(str(work / "forcefield"), require_rules=False)
        ff[host] = viparr.merge_forcefields(ff[host], kept)
        log("GAFF2 templates of the first prep kept; the new ligands' AM1-BCC charges at "
            "reference conformers")  # fmt: skip
    else:
        log("GAFF2 templates, AM1-BCC charges at reference conformers")
    patch = gaff.gaff2_patch(everything, ff, charges=args.ligand_charges, parents=args.parents,
                             protein_extent=args.protein_extent,
                             charge_geometry="reference")  # fmt: skip
    if kept is not None:
        patch = _not_kept(patch, kept)
    viparr.write_forcefield(patch if kept is None else viparr.merge_forcefields(kept, patch),
                            work / "forcefield")  # fmt: skip
    ff[host] = viparr.merge_forcefields(ff[host], patch)
    # Amber's (and GAFF2's) Lennard-Jones has a long-range dispersion correction,
    # which the hybrid cannot carry: each leg's end states' difference is kept
    # for result.  CHARMM36 has none, but a force switch boonza fep does not build.
    ff_names = [n.lower() for entry in args.forcefields for n in entry]
    tail = not any("charmm" in n for n in ff_names)
    if not tail:
        log("  warning: CHARMM36 was parameterized force-switched from 1.0 to 1.2 nm; "
            "boonza fep's hybrids are cut plainly")  # fmt: skip
    if receptor is not None:  # poses that clash: relieved with --rem, else named
        from . import relief

        if getattr(args, "rem", False) and before is None:
            log("REM, then EM: the receptor relaxed around every ligand's pose (the "
                "ligands held as posed, the residues within 5 Å of one free)")  # fmt: skip
            relief.write(work, receptor, *relief.relieve(receptor, ligs, names, ff, args, log),
                         suffix=receptor_format(args.input_structure))  # fmt: skip
        else:
            close = relief.clashes(receptor, ligs, names)
            if close:
                log(f"  warning: {len(close)} ligand(s) within {relief.CLASH_A} Å of the "
                    "receptor (heavy atoms): " + ", ".join(f"{n} {d:.2f} Å" for n, d in close)
                    + ("; --rem relieves them first" if before is None else "; the receptor "
                       "stays the one the first prep's edges were built in"))  # fmt: skip
    solvent_box = None
    if phase != "vacuum":
        solvent_box = octanol.box(work, ff, args, wet=phase == "wet-octanol", path=octanol_box,
                                  log=log)  # fmt: skip
    from rdkit import Chem

    from .experiment import experimental_dG

    set_kind = ("binding" if receptor is not None else
                "hydration" if phase == "vacuum" else "transfer")  # fmt: skip
    charges = [Chem.GetFormalCharge(m) for m in mols]
    penalty = CHARGE_CHANGE_PENALTY if receptor is not None else 0.0
    if before is None:
        edges = network.edges(mols, charge_change=penalty)
        log(f"Network: {len(edges)} edges over {len(ligs)} ligands")
    else:
        edges = network.extend(mols, before.pairs, new, charge_change=penalty)
        log(f"Network: {len(edges)} new edges, beside the {len(before.pairs)} of the first "
            f"prep, over {len(ligs)} ligands")  # fmt: skip
    with (work / "ligands.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "name", "atoms", "charge", "smiles", "exp_dG", "exp_field"])
        for k, (m, n) in enumerate(zip(mols, names, strict=True)):
            # a measured value, if the record has one (boonza.fep.experiment)
            got = experimental_dG(m.GetPropsAsDict(), n, args.temperature, set_kind)
            w.writerow([k, n, m.GetNumAtoms(), Chem.GetFormalCharge(m),
                        Chem.MolToSmiles(Chem.RemoveHs(m)),
                        "" if got is None else round(got[0], 4),
                        "" if got is None else got[1]])  # fmt: skip
    write_ligands(work / "ligands.sdf", mols, names)
    if receptor is not None:
        from ..io import save

        save(receptor, work / f"receptor{receptor_format(args.input_structure)}")
    runs, rows = [], []
    for e in edges:
        a, b = names[e.a], names[e.b]
        edge_dir = work / "edges" / f"{a}~{b}"
        legs = (("complex", receptor, args.padding_nm), ("solvent", None, SOLVENT_PADDING_NM))
        if receptor is None:  # a relative hydration free energy: water, and vacuum
            legs = (("solvent", None, SOLVENT_PADDING_NM), ("vacuum", None, None))
        if solvent_box is not None:  # a relative transfer free energy: water, and octanol
            legs = (("solvent", None, SOLVENT_PADDING_NM), ("octanol", None, SOLVENT_PADDING_NM))
        dq = charges[e.b] - charges[e.a]
        production_ns = charge_production_ns if dq else args.production_ns
        for leg, env, padding in legs:
            log(f"{a} -> {b}: {leg}")
            built = build_leg(env, ligs[e.a], ligs[e.b], e.mapping, ff, args, padding, log,
                              charge_change=dq,
                              solvent=solvent_box if leg == "octanol" else None)  # fmt: skip
            hyb, old, dropped = built.hybrid, built.old, built.dropped
            n_old, n_new = len(hyb.unique_old), len(hyb.unique_new)
            schedule_ = schedule(n_old, n_new, bool(dq), windows, charge_windows,
                                 large_change_atoms)  # fmt: skip
            if len(schedule_) != (charge_windows if dq else windows):
                log(f"  {max(n_old, n_new)} atoms of one ligand's own: {len(schedule_)} "
                    "windows, dense where their Lennard-Jones is switched")  # fmt: skip
            _finish(hyb, old, args, leg == "complex", log)
            meta = {"edge": [a, b], "ligands": [e.a, e.b], "leg": leg, "phase": phase,
                    "score": e.score,
                    "lambdas": schedule_.tolist(), "charge_change": dq,
                    "alchemical_waters": built.alchemical_waters,
                    "temperature": args.temperature, "pressure": args.pressure,
                    "timestep_fs": args.integration_fs, "iteration_ps": iteration_ps,
                    "equilibration_ns": args.equilibration_ns,
                    "production_ns": production_ns, "repeats": repeats,
                    "core": len(hyb.core), "unique_old": len(hyb.unique_old),
                    "unique_new": len(hyb.unique_new),
                    "unmapped_by_constraints": dropped}  # fmt: skip
            if tail and built.dispersion_dK:
                # kcal/mol Å^3: result adds dK / <V> to this leg's ΔG
                meta["dispersion_dK"] = built.dispersion_dK
            write_leg(edge_dir / leg, hyb, old, meta)
            for r in range(1, repeats + 1):
                runs.append((edge_dir / leg, r))
            if dropped:
                log(f"  {dropped} atoms left out of the core: their constrained bonds differ")
        rows.append([a, b, len(e.mapping), round(e.score, 3), dq])
    with (work / "edges.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ligand_a", "ligand_b", "mapped_atoms", "score", "charge_change"])
        w.writerows((before.rows if before else []) + rows)
    lines = [f"boonza fep run {d.resolve()} --repeat {r}" for d, r in runs]
    (work / "commands.txt").write_text("\n".join((before.commands if before else []) + lines)
                                       + "\n")  # fmt: skip
    if before is not None:
        (work / "commands_new.txt").write_text("\n".join(lines) + "\n")
        log(f"{len(runs)} new runs listed in {work / 'commands_new.txt'}; commands.txt "
            "lists every run")  # fmt: skip
    else:
        log(f"{len(runs)} runs listed in {work / 'commands.txt'}")
    return [d for d, _ in runs]
