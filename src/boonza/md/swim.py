"""``boonza swim``: ligands swimming around a protein, in many simulations.

A library of N ligands (SDF, or DMS with or without a force field) is split
into simulations of ``types`` ligands each; each simulation holds the
protein and ``copies`` copies of each of its ligands, placed at random in
the water around it. With m = N // types simulations, the ligands left over
join the last simulations, one each. Ligands are dealt deterministically so
that similar ones (by Morgan fingerprints) land in different simulations.

Every simulation is a ``boonza md`` run with the same settings (backbone
restraints included): ``boonza swim`` writes ``sim_000/`` ... each with its
``input.dms`` and ``md.toml``, and ``simulations.txt`` with one ``boonza md``
command per simulation, for job arrays; ``--run`` runs them here.

Each ligand is parameterized once: a force field it already has is kept; a
peptide of standard residues takes the protein force field; the rest is
GAFF2 (AM1-BCC, AmberTools), in parallel with ``--jobs``.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import gaff, viparr
from ..io import load, save
from ..system import System

_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def code(k: int) -> str:
    """The residue name of library ligand ``k``: L000, L001, ..., LZZZ."""
    if not 0 <= k < 36**3:
        raise ValueError("a library holds at most 46656 ligands")
    return "L" + _BASE36[k // 1296] + _BASE36[k // 36 % 36] + _BASE36[k % 36]


@dataclass
class Ligand:
    code: str
    name: str
    system: System  # as placed: named residues, no force field
    source: System | None  # the input with its force field, when it has one
    smiles: str  # without stereochemistry: ligands sharing it share a bond graph
    mol: object  # RDKit molecule, for fingerprints


# ---- which ligands go together ----------------------------------------------------


def simulation_sizes(n: int, types: int) -> list[int]:
    """Ligands per simulation: ``n // types`` simulations of ``types``, the
    ligands left over added one each to the last ones."""
    if types < 1:
        raise ValueError("types per simulation must be at least 1")
    if n < 1:
        raise ValueError("no ligands")
    m = max(1, n // types)
    sizes = [min(types, n)] * m
    extra = n - sum(sizes)
    for j in range(extra):
        sizes[(m - extra + j) % m] += 1
    return sizes


def fingerprints(mols) -> list:
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(m) for m in mols]


def deal(mols, types: int, keys=None) -> list[list[int]]:
    """Split ligands into simulations of ``types``, similar ones apart.

    The ligands are ordered by Butina clustering of Morgan fingerprints
    (similar ligands side by side) and dealt round-robin, so a cluster is
    spread over as many simulations as it has members. Deterministic.
    Ligands with the same ``keys`` (the same bond graph) never share a
    simulation, since their templates could not be told apart.
    """
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    n = len(mols)
    sizes = simulation_sizes(n, types)
    fps = fingerprints(mols)
    dists: list[float] = []
    for i in range(1, n):
        dists.extend(1.0 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, n, 0.65, isDistData=True, reordering=True)
    order = [j for cluster in clusters for j in cluster]
    groups: list[list[int]] = [[] for _ in sizes]
    p = 0
    for j in order:
        while len(groups[p]) >= sizes[p]:
            p = (p + 1) % len(groups)
        groups[p].append(j)
        p = (p + 1) % len(groups)
    if 1 < len(groups) and n <= _REFINE_MAX:
        _refine(groups, fps)
    if keys is not None:
        _separate(groups, keys)
    return groups


_REFINE_MAX = 5000  # the similarity matrix of this many ligands takes 100 MB


def _refine(groups: list[list[int]], fps, passes: int = 50) -> None:
    """Swap ligands between simulations while a swap lowers the total similarity
    within simulations; each ligand in turn takes its best swap (the lowest
    index on ties), so the result is deterministic."""
    from rdkit import DataStructs

    n = len(fps)
    sim = np.zeros((n, n), np.float32)
    for i in range(n):
        sim[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    np.fill_diagonal(sim, 0.0)
    where = np.empty(n, np.int64)
    for s, g in enumerate(groups):
        where[g] = s
    # c[i, s]: similarity of ligand i to the ligands of simulation s
    c = np.stack([sim[:, g].sum(1) for g in groups], axis=1)
    for _ in range(passes):
        improved = False
        for a in range(n):
            s = where[a]
            gain = c[np.arange(n), s] + c[a, where] - 2 * sim[a] - c[a, s] - c[np.arange(n), where]
            gain[where == s] = np.inf
            b = int(np.argmin(gain))
            if gain[b] >= -1e-6:
                continue
            t = where[b]
            groups[s][groups[s].index(a)] = b
            groups[t][groups[t].index(b)] = a
            where[a], where[b] = t, s
            c[:, s] += sim[:, b] - sim[:, a]
            c[:, t] += sim[:, a] - sim[:, b]
            improved = True
        if not improved:
            break
    for g in groups:
        g.sort()


def _separate(groups: list[list[int]], keys) -> None:
    """Swap ligands so that no simulation holds two with the same key."""
    for s, g in enumerate(groups):
        for k in range(len(g)):
            j = g[k]
            if keys[j] not in {keys[x] for x in g[:k]}:
                continue
            for t in [*range(s + 1, len(groups)), *range(s)]:
                h = groups[t]
                if keys[j] in {keys[x] for x in h}:
                    continue
                mine = {keys[x] for i, x in enumerate(g) if i != k}
                y = next((y for y in h if keys[y] not in mine), None)
                if y is not None:
                    h[h.index(y)] = j
                    g[k] = y
                    break
            else:
                raise ValueError(f"more ligands share the bond graph {keys[j]} than there "
                                 "are simulations")  # fmt: skip


def similarity_within(groups, fps) -> list[list[float]]:
    """For each ligand, its highest Tanimoto similarity to another in its simulation."""
    from rdkit import DataStructs

    out = []
    for g in groups:
        row = []
        for j in g:
            others = [fps[x] for x in g if x != j]
            row.append(max(DataStructs.BulkTanimotoSimilarity(fps[j], others), default=0.0))
        out.append(row)
    return out


# ---- the library ------------------------------------------------------------------


def _entries(lib: System) -> list[tuple[str, System]]:
    """(title, molecule) of each library entry: its cts, else its molecules
    (waters and single atoms left out)."""
    if lib.ncts > 1:
        return [(lib.ct(c).name or f"ligand {c + 1}", lib.clone(lib.ct_atoms(c)))
                for c in range(lib.ncts)]  # fmt: skip
    water = np.zeros(lib.natoms, bool)
    water[lib.select("water").ids] = True
    out = []
    for f in lib.fragments():
        ids = f.ids
        if len(ids) > 1 and not water[ids].all():
            out.append((f"ligand {len(out) + 1}", lib.clone(ids)))
    return out


def _bonds(s: System):
    return (np.column_stack([s.bonds["i"], s.bonds["j"]]).astype(np.int64),
            np.asarray(s.bonds["order"], np.int64))  # fmt: skip


def _rebuild(s: System, resnames, resids, names) -> System:
    """``s`` (no force field) with new residues and atom names."""
    out = System.from_arrays(s.positions, names=names, anum=s.atoms["anum"], resnames=resnames,
                             resids=resids, chains="L", charge=s.atoms["charge"],
                             formal_charge=s.atoms["formal_charge"],
                             mass=s.atoms["mass"])  # fmt: skip
    pairs, orders = _bonds(s)
    if len(pairs):
        out.add_bonds(pairs, order=orders)
    return out


def _one_residue(s: System, name: str) -> System:
    names = gaff._unique_names([str(x) for x in s.atoms["name"].tolist()],
                               s.atoms["anum"].tolist())  # fmt: skip
    return _rebuild(s, name, 1, names)


def split_peptide(s: System, forcefields) -> System | None:
    """``s`` cut into residues at its peptide bonds, named after the templates
    of ``forcefields`` they match (a peptide from an SDF file); None unless
    every piece matches one, so other molecules with amides stay whole."""
    from rdkit import Chem

    from ..chem import to_rdkit

    try:
        mol = to_rdkit(s, sanitize=True, implicit_hydrogens=False)
    except Exception:  # noqa: BLE001 - a molecule RDKit cannot read is not split
        return None
    pattern = Chem.MolFromSmarts("[CX4][CX3](=O)[NX3][CX4]")
    cut = {(min(c, n), max(c, n)) for _, c, _, n, _ in mol.GetSubstructMatches(pattern)}
    if not cut:
        return None
    parent = list(range(s.natoms))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs, _ = _bonds(s)
    for i, j in pairs.tolist():
        if (min(i, j), max(i, j)) not in cut:
            parent[root(i)] = root(j)
    roots = [root(a) for a in range(s.natoms)]
    first = {}
    for r in roots:
        first.setdefault(r, len(first))
    resids = [first[r] + 1 for r in roots]
    names = gaff._unique_names([str(x) for x in s.atoms["name"].tolist()],
                               s.atoms["anum"].tolist())  # fmt: skip
    pep = _rebuild(s, "UNK", resids, names)
    P = viparr._Parameterizer(pep, list(forcefields), False, False, True)
    resnames, atom_names = pep.residues["name"].tolist(), pep.atoms["name"].tolist()
    for r in range(pep.nresidues):
        atoms = pep.residue_atoms(r).tolist()
        for k, ff in enumerate(forcefields):
            tpl, pairs_ = P._find(ff, k, atoms)
            if tpl is not None:
                break
        else:
            return None
        resnames[r] = gaff._family(tpl.name)
        for ti, a in pairs_:
            atom_names[a] = tpl.names[ti]
    per_atom = [resnames[r] for r in pep.atoms["residue"].tolist()]
    out = _rebuild(pep, per_atom, resids, atom_names)
    out.reorder_atoms(np.argsort(out.atoms["residue"], kind="stable"))  # residues contiguous
    return out


def _rdkit(s: System):
    from rdkit import Chem

    from ..chem import assign_bond_orders, to_rdkit

    t = s.clone(structure_only=True)
    if gaff._needs_orders(t):
        assign_bond_orders(t)
    return Chem.RemoveHs(to_rdkit(t, implicit_hydrogens=False))


def load_library(path, forcefields) -> list[Ligand]:
    """The ligands of an SDF or DMS file, each named ``code(k)``."""
    from rdkit import Chem

    out = []
    for k, (title, mol) in enumerate(_entries(load(path))):
        c = code(k)
        has_ff = "nonbonded" in mol.table_names
        shape = mol.clone(structure_only=True)
        placed = None
        if not has_ff and shape.nresidues == 1:
            placed = split_peptide(shape, forcefields)
        if placed is None:  # residue LIG; the residue number tells ligands apart
            if has_ff or shape.nresidues == 1:
                placed = _one_residue(shape, "LIG")
            else:
                res = [str(x) or "LIG" for x in shape.residues["name"].tolist()]
                per_atom = [res[r] for r in shape.atoms["residue"].tolist()]
                placed = _rebuild(shape, per_atom, shape.atoms["residue"] + 1,
                                  [str(x) for x in shape.atoms["name"].tolist()])  # fmt: skip
        rd = _rdkit(placed)
        out.append(Ligand(c, title, placed, mol if has_ff else None,
                          Chem.MolToSmiles(rd, isomericSmiles=False), rd))  # fmt: skip
    return out


# ---- parameters -------------------------------------------------------------------


def _load_forcefields(spec):
    ffs = []
    for base, *patches in spec:
        ff = viparr.load_forcefield(base)
        for patch in patches:
            ff = viparr.merge_forcefields(ff, patch)
        ffs.append(ff)
    return ffs


def _gaff2_job(job) -> str:
    """GAFF2 templates for one ligand (a separate process)."""
    d, tag, spec, opts = job
    d = Path(d)
    patch = gaff.gaff2_patch(load(d / "ligand.dms"), _load_forcefields(spec),
                             workdir=d / "gaff2", tag=tag, **opts)  # fmt: skip
    viparr.write_forcefield(patch, d / "patch")
    return str(d / "patch")


def ligand_patches(ligands, args, forcefields, root: Path, jobs: int = 1, log=print):
    """The patch directory for each ligand (None when the force fields match it),
    made once and kept in ``root/<code>/``."""
    host = forcefields[gaff.host_index(forcefields)]
    out: list[Path | None] = [None] * len(ligands)
    todo = []
    for k, lig in enumerate(ligands):
        d = root / lig.code
        d.mkdir(parents=True, exist_ok=True)
        save(lig.system, d / "ligand.dms")
        patch = d / "patch"
        if (patch / "templates").is_file():
            out[k] = patch
        elif (d / "standard").is_file():
            continue
        elif lig.source is not None:  # its own force field, kept
            viparr.write_forcefield(viparr.patch_from_system(lig.source, lig.code, lig.code,
                                                             rules=host.rules), patch)  # fmt: skip
            out[k] = patch
        elif not gaff.find_unmatched(lig.system, forcefields):
            (d / "standard").write_text("every residue is matched by the force fields\n")
        else:
            todo.append(k)
    if todo:
        log(f"GAFF2 {args.ligandff} templates (AM1-BCC) for {len(todo)} ligands, "
            f"{jobs} at a time...")  # fmt: skip
        spec = [list(e) for e in args.forcefields]
        opts = dict(charges=args.ligand_charges, parents=args.parents,
                    protein_extent=args.protein_extent)  # fmt: skip
        work = [(str(root / ligands[k].code), ligands[k].code, spec, opts) for k in todo]
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                done = list(pool.map(_gaff2_job, work))
        else:
            done = [_gaff2_job(w) for w in work]
        for k, d in zip(todo, done, strict=True):
            out[k] = Path(d)
    return out


# ---- placing ----------------------------------------------------------------------


def _rotation(rng) -> np.ndarray:
    """A uniformly random rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def place(protein: System, ligands, copies: int, edge: float, rng, clearance: float = 3.0,
          tries: int = 20000, chain: str = "LIG") -> System:  # fmt: skip
    """The protein with ``copies`` copies of each ligand, each at a random
    position and orientation in the cube of edge ``edge`` (Å) around the
    protein, its heavy atoms at least ``clearance`` Å from any other."""
    out = protein.copy()
    center = protein.positions.mean(0)
    taken = [protein.positions[protein.atoms["anum"] > 1]]
    resid = 0
    for lig in ligands:
        xyz0 = lig.positions - lig.positions.mean(0)
        room = edge / 2 - float(np.linalg.norm(xyz0, axis=1).max())
        if room <= 0:
            raise ValueError(f"a ligand ({lig.residues['name'][0]}) does not fit in the box")
        heavy = lig.atoms["anum"] > 1
        for _ in range(copies):
            near = np.concatenate(taken)
            for _ in range(tries):
                xyz = xyz0 @ _rotation(rng).T + center + rng.uniform(-room, room, 3)
                h = xyz[heavy]
                if ((h[:, None, :] - near[None, :, :]) ** 2).sum(-1).min() >= clearance**2:
                    break
            else:
                raise ValueError("no room for another ligand copy: use fewer copies or a "
                                 "larger padding_nm")  # fmt: skip
            copy = lig.copy()
            copy.positions = xyz
            copy.chains["name"] = np.full(copy.nchains, chain)
            copy.residues["resid"] = np.arange(resid + 1, resid + 1 + copy.nresidues)
            resid += copy.nresidues
            out.append(copy)
            taken.append(h)
    return out


# ---- simulations ------------------------------------------------------------------


def prepare(args, library, types: int = 5, copies: int = 3, jobs: int = 1,
            clearance: float = 3.0, log=print, repel: bool = True) -> list[Path]:  # fmt: skip
    """Write one ``boonza md`` simulation per group of ligands into
    ``args.workdir``; returns their directories."""
    from .config import forcefield_kind, settings_of, write_settings
    from .prepare import forcefields, load_input

    if forcefield_kind(args.forcefields) == "xml":
        raise ValueError("boonza swim needs viparr force fields (-f), not OpenMM XML files")
    if args.input_structure is None:
        raise ValueError("give the protein structure")
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    protein = load_input(args.input_structure)
    _, ffs = forcefields(args)
    ligands = load_library(library, ffs)
    groups = deal([lig.mol for lig in ligands], types, keys=[lig.smiles for lig in ligands])
    sizes = [len(g) for g in groups]
    near = similarity_within(groups, fingerprints([lig.mol for lig in ligands]))
    top = [max(row, default=0.0) for row in near]
    log(f"{len(ligands)} ligands in {library}: {len(groups)} simulations of {sizes[0]}"
        + (f" ({sum(s != sizes[0] for s in sizes)} with {max(sizes)})" if max(sizes) != sizes[0]
           else "")
        + f"; highest similarity within a simulation: mean {np.mean(top):.2f}, "
        f"max {max(top):.2f}")  # fmt: skip
    patches = ligand_patches(ligands, args, ffs, root / "ligands", jobs, log)
    host = gaff.host_index(ffs)
    extent = float((protein.positions.max(0) - protein.positions.min(0)).max())
    used = set(protein.chains["name"].tolist())  # the ligands' own chain, for the repulsion
    chain = next(c for c in ("LIG", *(f"LIG{k}" for k in range(2, 1000))) if c not in used)
    edge = extent + 20.0 * args.padding_nm
    with (root / "assignment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["simulation", "ligand", "name", "smiles", "max_similarity_in_simulation"])
        for s, (g, row) in enumerate(zip(groups, near, strict=True)):
            for j, x in zip(g, row, strict=True):
                w.writerow([f"sim_{s:03d}", ligands[j].code, ligands[j].name, ligands[j].smiles,
                            f"{x:.3f}"])  # fmt: skip
    sims = []
    for s, members in enumerate(groups):
        d = root / f"sim_{s:03d}"
        sims.append(d)
        if (d / "md").is_dir() and any((d / "md").iterdir()):
            continue  # started: leave it be
        d.mkdir(exist_ok=True)
        rng = np.random.default_rng([int(args.seed), s])
        system = place(protein, [ligands[j].system for j in members], copies, edge, rng,
                       clearance, chain=chain)  # fmt: skip
        save(system, d / "input.dms")
        with (d / "ligands.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)  # which residues of chain LIG are which ligand
            w.writerow(["chain", "first_resid", "last_resid", "ligand", "copy", "name", "smiles"])
            resid = 0
            for j in members:
                for k in range(copies):
                    n = ligands[j].system.nresidues
                    w.writerow([chain, resid + 1, resid + n, ligands[j].code, k + 1,
                                ligands[j].name, ligands[j].smiles])  # fmt: skip
                    resid += n
        spec = [list(e) for e in args.forcefields]
        spec[host] += [str(patches[j].resolve()) for j in members if patches[j] is not None]
        settings = {**settings_of(args), "input_structure": str((d / "input.dms").resolve()),
                    "workdir": str((d / "md").resolve()), "forcefields": spec,
                    "box_nm": round(edge / 10.0, 4)}  # fmt: skip
        if repel and "repulsion_selection" not in args.specified:
            settings["repulsion_selection"] = f"chain {chain}"  # ligand copies apart
        if "dihedral_restraint" not in args.specified:
            settings["dihedral_restraint"] = "ss"  # the protein's helices and sheets hold
        if "dihedral_restraint_selection" not in args.specified:
            settings["dihedral_restraint_selection"] = f"not chain {chain}"  # ligands swim
        write_settings(d / "md.toml", settings)
    (root / "simulations.txt").write_text(
        "".join(f"boonza md --config {(d / 'md.toml').resolve()}\n" for d in sims)
    )
    log(f"{len(sims)} simulations in {root}: run each line of {root / 'simulations.txt'} "
        "(a job array), or pass --run")  # fmt: skip
    return sims


def main(argv=None) -> int:
    from .config import build_parser, parse_arguments
    from .run import run_workflow

    parser = build_parser("boonza swim")
    g = parser.add_argument_group("swim")
    g.add_argument("--ligands", help="SDF or DMS file of ligands (with or without a force field)")
    g.add_argument("--types", type=int, help="ligand types per simulation (default: 5)")
    g.add_argument("--copies", type=int, help="copies of each ligand type (default: 3)")
    g.add_argument("--jobs", type=int, help="ligands parameterized at once (default: 1)")
    g.add_argument(
        "--clearance",
        type=float,
        help="Å between a placed ligand's heavy atoms and any other (default: 3)",
    )
    g.add_argument("--run", action="store_true", help="run the simulations here, one by one")
    g.add_argument("--no-repulsion", dest="no_repulsion", action="store_true",
                   help="let ligand copies stick together (by default they repel)")  # fmt: skip
    args = parse_arguments(argv, parser)
    x = args.extra
    if "workdir" not in args.specified:
        args.workdir = "swim"
    try:
        if "ligands" not in x:
            raise ValueError("give the ligands with --ligands")
        sims = prepare(args, x["ligands"], x.get("types", 5), x.get("copies", 3),
                       x.get("jobs", 1), x.get("clearance", 3.0),
                       repel=not x.get("no_repulsion", False))  # fmt: skip
        if x.get("run"):
            for d in sims:
                print(f"== {d}")
                run_workflow(parse_arguments(["--config", str(d / "md.toml")]))
    except (ValueError, FileNotFoundError, RuntimeError, viparr.ViparrError) as e:
        print(f"boonza swim: error: {e}", file=sys.stderr)
        return 1
    return 0
