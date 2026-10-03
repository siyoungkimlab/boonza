"""The ``boonza`` command line.

    boonza info system.dms
    boonza convert system.mae system.dms [-s "protein or resname LIG"]
    boonza select system.dms "within 5 of resname LIG"
    boonza validate system.dms [--strict]
    boonza knots system.dms [--max-cycle 10] [--ignore-excluded]
    boonza diff a.dms b.dms [--rtol 1e-6] [--no-positions]
    boonza describe system.dms "resname LIG and name C1" [--terms all] [--pairs]
    boonza dssp protein.pdb [--simplified]
    boonza rmsd docked.pdb crystal.pdb [--ligandsel SEL] [--align order|sequence|none]
    boonza rmsd system.pdb crystal.pdb --traj md.xtc     (one RMSD per frame)
    boonza drmsd system.pdb --traj md.xtc [--reference crystal.pdb] [--cutoff 5]
    boonza poses system.dms --traj md.dcd [--cutoff 1.5] [-o poses/]   (representative frames)
    boonza sites system.dms --traj a.dcd b.dcd [-o sites/]   (where a ligand goes, pooled)
    boonza build --smiles 'CC(=O)Oc1ccccc1C(=O)O' -o aspirin.sdf
    boonza summarize complex.pdb [--focus 'resname LIG'] [--json]
    boonza build --sequence ACDEFGHIK --conformation helix -o peptide.pdb
    boonza parameterize in.dms out.dms -f aa.charmm.c36m -f water.tip3p_charmm

``validate``, ``knots`` and ``diff`` exit with status 1 when they find
something.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

#: what the site and pose commands align runs on, unless told otherwise
DEFAULT_ALIGN = "protein and name CA"


def _load(path, structure_only=False):
    import boonza

    if structure_only and str(path).lower().endswith((".dms", ".dms.gz", ".dms.bz2")):
        return boonza.load(path, structure_only=True)
    return boonza.load(path)


def _info(args) -> int:
    from .trajectory import is_trajectory

    if is_trajectory(args.file):
        return _trajectory_info(args.file)
    s = _load(args.file)
    print(f"{args.file}: {s.natoms} atoms, {s.nbonds} bonds, {s.nresidues} residues, "
          f"{s.nchains} chains, {s.ncts} cts, {s.nfragments} molecules")  # fmt: skip
    if s.cell.any():
        print("cell:", " ".join(f"[{' '.join(f'{x:g}' for x in row)}]" for row in s.cell))
    if s.atoms.props:
        print("extra atom columns:", ", ".join(s.atoms.props))
    info = s.nonbonded_info
    if info.vdw_funct:
        print(f"nonbonded: {info.vdw_funct} {info.vdw_rule}".rstrip())
    if s.tables:
        width = max(map(len, s.tables))
        for name in sorted(s.tables):
            t = s.tables[name]
            extra = f", {len(t.overrides)} overrides" if len(t.overrides) else ""
            print(f"  {name:<{width}}  {t.category:<10} {len(t):>9} terms "
                  f"{len(t.params):>7} params{extra}")  # fmt: skip
    if s.aux_tables:
        print("aux tables:", ", ".join(sorted(s.aux_tables)))
    return 0


def _trajectory_info(path) -> int:
    """What a trajectory holds: its frames, their atoms, the time they cover and
    the box they were under.

    Only the first and last frames are read, so this costs the same on a
    gigabyte as on a megabyte.
    """
    from pathlib import Path

    from .trajectory import open_trajectory

    traj = open_trajectory(path)
    size = Path(path).stat().st_size
    print(f"{path}: {traj.format}, {len(traj)} frames of {traj.natoms} atoms, "
          f"{size / 1e6:,.1f} MB")  # fmt: skip
    if not len(traj):
        return 0
    first, last = traj[0], traj[len(traj) - 1]
    if first.time is not None and last.time is not None:
        span = f"{first.time:g} to {last.time:g} ps"
        gap = (last.time - first.time) / max(len(traj) - 1, 1)
        every = f", {gap / 1000:g} ns apart" if len(traj) > 1 else ""
        steps = (f" (steps {first.step:,} to {last.step:,})"
                 if first.step is not None and last.step is not None else "")  # fmt: skip
        print(f"time: {span}{every}{steps}")
    for label, frame in (("first", first), ("last", last)):
        if frame.box is None or not np.any(frame.box):
            print(f"{label} frame: no box")
            continue
        box = np.asarray(frame.box, float)
        edges = " x ".join(f"{x:.2f}" for x in np.diag(box))
        skew = "" if np.allclose(box, np.diag(np.diag(box))) else " (not orthorhombic)"
        print(f"{label} frame: box {edges} A{skew}, volume "
              f"{abs(float(np.linalg.det(box))) / 1000:,.1f} nm^3")  # fmt: skip
        if len(traj) == 1:
            break
    return 0


def _convert(args) -> int:
    import boonza

    s = _load(args.input, args.structure_only)
    if args.selection:
        s = s.clone(s.select(args.selection).ids)
    boonza.save(s, args.output)
    print(f"wrote {s.natoms} atoms to {args.output}")
    return 0


def _select(args) -> int:
    ids = _load(args.file).select(args.selection).ids
    print(" ".join(map(str, ids.tolist())))
    return 0


def _validate(args) -> int:
    from .validate import validate

    problems = validate(_load(args.file), strict=args.strict, max_ring=args.max_ring)
    for p in problems:
        print(p)
    if not problems:
        print("no problems found")
    return int(bool(problems))


def _knots(args) -> int:
    from .validate import find_knots

    s = _load(args.file, structure_only=not args.ignore_excluded)
    knots = find_knots(s, args.max_cycle, args.selection, args.ignore_excluded)
    for ring, bond, _ in knots:
        print(f"bond {bond[0]}-{bond[1]} passes through ring {' '.join(map(str, ring))}")
    print(f"{len(knots)} knots")
    return int(bool(knots))


def _diff(args) -> int:
    from .diff import diff

    found = diff(
        _load(args.a),
        _load(args.b),
        rtol=args.rtol,
        positions=not args.no_positions,
        canonical=args.canonical,
    )
    for d in found:
        print(d)
    if not found:
        print("no differences")
    return int(bool(found))


def _describe(args) -> int:
    from .describe import describe

    report = describe(_load(args.file), args.selection, terms=args.terms, pairs=args.pairs or None)
    print(report)
    return 0


def _dssp(args) -> int:
    from .secondary import dssp

    s = _load(args.file)
    codes = np.asarray(dssp(s, simplified=args.simplified))[0]
    chains = s.residues["chain"]
    for c in np.unique(chains).tolist():
        line = "".join(x if len(x) == 1 else "-" for x in codes[chains == c].tolist())
        if line.strip("-"):
            print(f"{s.chains['name'][c] or '_'}: {line}")
    return 0


def _phipsi(args) -> int:
    from .secondary import backbone_dihedrals

    s = _load(args.file)
    phi, psi, omega = (a[0] for a in backbone_dihedrals(s))
    res = s.residues
    chains = s.chains["name"][res["chain"]]
    print(f"{'chain':>5} {'resid':>6} {'res':>4} {'phi':>8} {'psi':>8} {'omega':>8}")
    for i in np.flatnonzero(~(np.isnan(phi) & np.isnan(psi))).tolist():
        vals = " ".join(
            "       -" if np.isnan(v) else f"{v:8.2f}" for v in (phi[i], psi[i], omega[i])
        )
        print(f"{chains[i] or '_':>5} {res['resid'][i]:>6} {res['name'][i]:>4} {vals}")
    return 0


def _rmsd(args) -> int:
    from .symmetry import ligand_rmsd
    from .trajectory import open_trajectory

    mobile = _load(args.mobile)
    traj = open_trajectory(args.traj, mobile) if args.traj else None
    align = None if args.align == "none" else args.align
    r = ligand_rmsd(mobile, _load(args.reference), ligand=args.ligandsel,
                    reference_ligand=args.ref_ligandsel, fit=args.mobsel,
                    reference_fit=args.refsel, align=align, positions=traj)  # fmt: skip
    if traj is None:
        print(f"protein fit RMSD ({args.align}): {r.fit_rmsd:.3f} A")
        if r.plain_rmsd is not None:
            print(f"ligand RMSD, atoms in order: {r.plain_rmsd:.3f} A")
        print(f"ligand RMSD, symmetry-corrected: {r.rmsd:.3f} A "
              f"({len(r.reference_ligand)} heavy atoms)")  # fmt: skip
        return 0
    print(f"{'frame':>6} {'fit':>8} {'ligand':>8} {'in order':>9}")
    for k in range(len(r.rmsd)):
        plain = "-" if r.plain_rmsd is None else f"{r.plain_rmsd[k]:.3f}"
        print(f"{k:6d} {r.fit_rmsd[k]:8.3f} {r.rmsd[k]:8.3f} {plain:>9}")
    within = float((r.rmsd < 2.0).mean())
    print(f"ligand RMSD (symmetry-corrected) mean {r.rmsd.mean():.3f}, min {r.rmsd.min():.3f}, "
          f"max {r.rmsd.max():.3f} A; {within:.0%} of frames under 2 A")  # fmt: skip
    return 0


def _drmsd(args) -> int:
    from .symmetry import drmsd
    from .trajectory import open_trajectory

    system = _load(args.system)
    traj = open_trajectory(args.traj, system) if args.traj else None
    reference = _load(args.reference) if args.reference else None
    r = drmsd(system, reference, ligand=args.ligandsel, protein=args.proteinsel,
              cutoff=args.cutoff, positions=traj, reference_ligand=args.ref_ligandsel,
              symmetry=not args.no_symmetry, periodic=not args.no_pbc)  # fmt: skip
    print(f"pocket: {len(r.pocket)} atoms within {args.cutoff:g} A of "
          f"{len(r.reference_ligand)} ligand atoms")  # fmt: skip
    if traj is None:
        print(f"dRMSD, symmetry-corrected: {r.drmsd:.3f} A")
        if r.plain_drmsd is not None:
            print(f"dRMSD, atoms in order: {r.plain_drmsd:.3f} A")
        return 0
    print(f"{'frame':>6} {'dRMSD':>8} {'in order':>9}")
    for k in range(len(r.drmsd)):
        plain = "-" if r.plain_drmsd is None else f"{r.plain_drmsd[k]:.3f}"
        print(f"{k:6d} {r.drmsd[k]:8.3f} {plain:>9}")
    print(f"dRMSD mean {r.drmsd.mean():.3f}, min {r.drmsd.min():.3f}, "
          f"max {r.drmsd.max():.3f} A")  # fmt: skip
    return 0


def _separate_sites(system, prot, share, cutoff: float = 8.0) -> int:
    """How many spatially separate patches the contacted atoms fall into."""
    from .graph import connected_components

    touched = prot[share > 0.1]
    if len(touched) < 2:
        return len(touched)
    xyz = system.positions[touched]
    d = np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1))
    i, j = np.nonzero(np.triu(d <= cutoff, 1))
    _, groups = connected_components(len(touched), i, j)
    return groups


def _reference_frame(system, traj, args):
    """``(a typical bound frame, protein atoms touched per frame)``, in one pass.

    The first frame may have the ligand elsewhere -- out in bulk, or not yet
    settled -- and the *most* contacting frame is an outlier by construction,
    so neither should decide what the pocket is.
    """
    from .poses import bound_frame, pocket_contacts

    counts, share = pocket_contacts(system, traj, ligand=args.ligandsel,
                                    protein=args.proteinsel, cutoff=args.pocket_cutoff,
                                    periodic=not args.no_pbc)  # fmt: skip
    bound = int((counts > 0).sum())
    print(f"the ligand touches {args.proteinsel!r} in {bound} of {len(counts)} frames "
          f"({100 * bound / len(counts):.0f}%)")  # fmt: skip
    best = bound_frame(counts)
    print(f"pocket taken from frame {best}, a median one of those ({counts[best]} atoms within "
          f"{args.pocket_cutoff:g} A)")  # fmt: skip
    patches = _separate_sites(system, system.select(args.proteinsel).ids, share)
    if patches > 1:
        print(f"Warning: the ligand visits {patches} separate patches of the protein. This "
              "groups poses against one of them; use 'boonza sites' to separate them first, "
              "or --pocketsel to say which.")  # fmt: skip
    out = system.clone()
    frame = traj[best]  # a trajectory gives a Frame, an array of structures gives coordinates
    out.positions = getattr(frame, "positions", frame)
    return out, counts  # the counts settle the run too, from this one pass


def _structures(paths, log=print):
    """Frames from separate structure files: heavy atoms, matched by name.

    Docking and structure prediction hand back a folder rather than a
    trajectory, and those files rarely agree on how many hydrogens they carry
    or what order the atoms come in -- so the heavy atoms of each are put in
    the order of the first, by chain, residue and atom name.
    """
    from pathlib import Path

    import boonza

    def named(system, ids):
        chains, residues = system.chains["name"], system.residues
        of_residue = system.atoms["residue"]
        return [
            (str(chains[residues["chain"][r]]), int(residues["resid"][r]), str(n))
            for r, n in zip(of_residue[ids], system.atoms["name"][ids], strict=True)
        ]

    first = boonza.load(str(paths[0]))
    keep = first.select("noh").ids
    system = first.clone(keep)
    order = {k: i for i, k in enumerate(named(first, keep))}
    if len(order) != len(keep):
        raise ValueError(f"{paths[0]} names two atoms of a residue the same: they cannot be paired")
    frames, kept = [], []
    for path in paths:
        one = boonza.load(str(path))
        heavy = one.select("noh").ids
        here = named(one, heavy)
        if set(here) != set(order):
            log(f"  {Path(path).name} holds different atoms: left out")
            continue
        at = np.empty(len(order), np.int64)
        for i, key in enumerate(here):
            at[order[key]] = heavy[i]
        frames.append(one.positions[at])
        kept.append(str(path))
    if len(frames) < 2:
        raise ValueError(f"{len(frames)} of {len(paths)} structures could be compared")
    return system, np.array(frames), kept


def _poses(args) -> int:
    import json
    import shutil
    from pathlib import Path

    import boonza

    if not args.structures and not (args.system and args.traj):
        raise ValueError("give SYSTEM with --traj, or --structures for separate files")

    from .analysis import settled
    from .poses import pocket_contacts
    from .trajectory import open_trajectory

    if args.structures:  # a folder of structures rather than a trajectory
        system, traj, files = _structures(args.structures)
        print(f"comparing {len(traj)} of {len(args.structures)} structures")
        args.no_settle = True  # they are not a time series: there is no drift to drop
    else:
        system, files = _load(args.system), None
        traj = open_trajectory(args.traj, system)
    # probes.json sits beside the run (or a directory up, as swim writes it), and
    # names the probes the pocket must not be built from
    beside = [Path(args.system).parent] if getattr(args, "system", None) else []
    _cg_poses_selections(args, system, beside)
    counts = None
    if args.reference:
        reference = _load(args.reference)
    else:
        reference, counts = _reference_frame(system, traj, args)
    kept = np.arange(len(traj))
    if not args.no_settle:
        if counts is None:  # a reference was given, so nothing has counted yet
            counts, _ = pocket_contacts(system, traj, ligand=args.ligandsel,
                                        protein=args.proteinsel, cutoff=args.pocket_cutoff,
                                        periodic=not args.no_pbc)  # fmt: skip
        start = settled(counts.astype(float))
        # settling may only drop an approach, never a state. A series is not stationary
        # while the ligand arrives, and it is not stationary when the ligand moves from
        # one place to another either; what tells them apart is that during an approach
        # the ligand is not yet touching much.
        if start and np.median(counts[:start]) < 0.5 * np.median(counts[start:]):
            print(f"arrived by frame {start} of {len(traj)}: the {start} frames before it "
                  "touch little of the protein and are left out")  # fmt: skip
            kept = kept[start:]
        elif start:
            print(
                f"not settling: the {start} frames before frame {start} touch as much "
                "protein as the rest, so they are another state and not an approach"
            )
    kept = kept[:: args.stride]
    if len(kept) < 2:
        raise ValueError(f"{len(kept)} frames left to group: lower --stride")
    p = boonza.poses(system, traj[kept[0] : kept[-1] + 1 : args.stride], reference,
                     cutoff=args.cutoff,
                     min_population=args.min_population, ligand=args.ligandsel,
                     protein=args.proteinsel, pocket_cutoff=args.pocket_cutoff,
                     pocket=args.pocketsel,
                     symmetry=not args.no_symmetry, periodic=not args.no_pbc)  # fmt: skip
    what = "structures" if files else "frames"
    print(f"{len(p)} poses of {len(kept)} {what}, cut at {args.cutoff:g} A dRMSD")
    head = "representative" if files else "frame"
    print(f"{'pose':>4} {head:>14} {'share':>7} {'spread':>7} {what:>10}")
    for k, pose in enumerate(p):
        centre = Path(files[kept[pose.center]]).name if files else str(kept[pose.center])
        print(f"{k:4d} {centre:>14} {100 * pose.population:6.1f}% "
              f"{pose.spread:6.2f} A {len(pose):10d}")  # fmt: skip
    cutoffs, share, count = p.sweep()
    print("\ncutoff (A) " + " ".join(f"{c:5.2f}" for c in cutoffs))
    print("largest    " + " ".join(f"{100 * s:4.0f}%" for s in share))
    print("poses      " + " ".join(f"{c:5d}" for c in count))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        doc = {"frames": len(kept), "cutoff": args.cutoff, "pocket_cutoff": args.pocket_cutoff,
               "ligand": args.ligandsel, "stride": args.stride, "poses": []}  # fmt: skip
        for k, pose in enumerate(p):
            frame = int(kept[pose.center])
            if files:  # the representative is one of the files: copy it, hydrogens and all
                source = Path(files[frame])
                name = f"pose_{k:03d}_{source.name}"
                shutil.copy2(source, out / name)
            else:
                written = system.clone()
                written.positions = traj[frame].positions
                name = f"pose_{k:03d}.{args.format}"
                boonza.save(written, out / name)
            record = {"file": name, "population": pose.population, "spread": pose.spread,
                      "frames": len(pose)}  # fmt: skip
            if files:  # say which structures, not which frame numbers
                record["representative"] = files[frame]
                record["members"] = [files[int(i)] for i in kept[pose.frames]]
            else:
                record["frame"] = frame
                record["members"] = kept[pose.frames].tolist()
            doc["poses"].append(record)
        doc["sweep"] = {"cutoff": cutoffs.tolist(), "largest": share.tolist(),
                        "poses": count.tolist()}  # fmt: skip
        (out / "poses.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {len(p)} structures and poses.json to {out}")
    return 0


def _cg_poses_selections(args, system, workdirs=()) -> None:
    """A coarse-grained pocket is made of backbone beads, and its probes have to
    be named: the default ligand selection would take the protein itself."""
    from .symmetry import DEFAULT_FIT, DEFAULT_LIGAND

    if not _coarse_grained(system):
        return
    if getattr(args, "proteinsel", None) in (None, DEFAULT_ALIGN, DEFAULT_FIT):
        args.proteinsel = _backbone_selection(system, _probes_of(workdirs))
        print(f"the pocket is made of {args.proteinsel!r} "
              "(a coarse-grained system has no CA atoms)")  # fmt: skip
    if getattr(args, "ligandsel", None) in (None, DEFAULT_LIGAND):
        raise ValueError("say which probe to pose with --ligandsel, e.g. \"resname EK\": in a "
                         "coarse-grained system the default ligand selection would take the "
                         "protein itself.  'boonza probes' maps every probe at once")  # fmt: skip


def _probes_of(workdirs) -> list[str]:
    """The probe residue names of coarse-grained swim runs, from their probes.json."""
    import json
    from pathlib import Path

    out: list[str] = []
    for d in workdirs or []:
        for candidate in (Path(d) / "probes.json", Path(d).parent / "probes.json"):
            if candidate.is_file():
                out += json.loads(candidate.read_text())["probes"]
                break
    return list(dict.fromkeys(out))


def _backbone_selection(system, without=()) -> str:
    """The beads a coarse-grained backbone is made of, of those this system has:
    BB under Martini, GN, GC and GO under SIRAH.

    ``without`` are residue names to leave out -- the probes of a swim, which
    carry backbone beads of their own.  A fit on every BB bead in the box is a
    fit on 420 diffusing probes and 85 of protein, which puts every frame in a
    different frame of reference and leaves a parked probe looking like bulk.
    """
    from .md.monitor import CG_BACKBONE

    here = [n for n in CG_BACKBONE if len(system.select(f"name {n}").ids)]
    beads = "name " + " ".join(here or CG_BACKBONE)
    return f"({beads}) and not resname {' '.join(without)}" if without else beads


def _coarse_grained(system) -> bool:
    """A coarse-grained system: backbone beads rather than alpha carbons (BB
    under Martini, GN, GC and GO under SIRAH)."""
    from .md.monitor import CG_BACKBONE

    if len(system.select("name CA").ids):
        return False
    return len(system.select("name " + " ".join(CG_BACKBONE)).ids) >= 3


def _cg_selections(args, system, workdirs) -> None:
    """Fill in what a coarse-grained run needs: the probes of a coarse-grained
    `boonza swim` simulation, and its backbone beads to align on."""
    from .symmetry import DEFAULT_LIGAND

    if not _coarse_grained(system):
        return
    probes = _probes_of(workdirs)
    if getattr(args, "alignsel", None) in (None, DEFAULT_ALIGN):
        args.alignsel = _backbone_selection(system, probes)
        why = ", and a fit on the probes too is a fit on what moves" if probes else ""
        print(f"aligning on {args.alignsel!r} (a coarse-grained system has no CA atoms{why})")
    if getattr(args, "ligandsel", None) in (None, DEFAULT_LIGAND) and probes:
        # A probe is named after its sequence, and a one-residue probe of a
        # tryptophan is called W -- which is what Martini calls its water.  Asked
        # for the probes by name alone, a run of the single-residue library hands
        # back every water bead in the box as a ligand, and then nothing is a
        # site because the whole box is.  The probes of a swim are a chain of
        # their own, so name that too; a run from before they were given one has
        # the solvent named out instead.
        import re

        from .md.cgswim import PROBE_CHAIN
        from .probemap import SOLVENT_NAMES

        names = "resname " + " ".join(probes)
        theirs = sorted(c for c in {str(x).strip() for x in system.chains["name"]}
                        if re.fullmatch(PROBE_CHAIN + r"\d*", c))  # fmt: skip
        if theirs:
            args.ligandsel = f"({names}) and chain {' '.join(theirs)}"
        else:
            args.ligandsel = f"({names}) and not resname {' '.join(SOLVENT_NAMES)}"
        print(f"the probes of the run(s): {args.ligandsel}")


def _probes(args) -> int:
    """What each probe touches, residue by residue."""
    import json
    from pathlib import Path

    import boonza

    from .probemap import probe_contacts
    from .trajectory import open_trajectory

    runs, probes = [], []
    for d in args.workdir:
        found = list(args.probes) if args.probes else None
        if found is None:
            for candidate in (Path(d) / "probes.json", Path(d).parent / "probes.json"):
                if candidate.is_file():
                    found = json.loads(candidate.read_text())["probes"]
                    break
        if found is None:
            raise ValueError(f"{d} has no probes.json, so name the probes with --probes: it "
                             "is not a run of a coarse-grained 'boonza swim'")  # fmt: skip
        probes += [p for p in found if p not in probes]
        own = boonza.load(str(Path(d) / "solvated.dms"), without_tables=True)
        runs.append((own, open_trajectory(str(Path(d) / "trajectory.dcd"), own)))
    m = probe_contacts(runs[0][0], runs, probes, cutoff=args.cutoff, stride=args.stride,
                       periodic=not args.no_pbc)  # fmt: skip
    print(f"{len(runs)} runs, {m.frames} frames, {len(m.probes)} probes over "
          f"{len(m.residues)} residues; contact within {args.cutoff:g} A")  # fmt: skip
    by = "side chain" if args.by == "side-chain" else "probe"
    labels, matrix = (m.probes, m.contacts) if by == "probe" else m.side_chains()
    print(f"\n{by:10s} residues it touches most (share of frames)")
    import numpy as np

    for k, label in enumerate(labels):
        order = np.argsort(-matrix[:, k])[: args.top]
        best = ", ".join(f"{m.residues[r][2]}{m.residues[r][1]} {100 * matrix[r, k]:.0f}%"
                         for r in order if matrix[r, k] > 0)  # fmt: skip
        print(f"  {label:10s} {best or 'nothing it touched'}")
    if args.out:
        m.to_csv(args.out)
        print(f"\nwrote {args.out}: a residue by probe table")
    return 0


def _sites(args) -> int:
    import json

    if not args.workdir and not (args.system and args.traj):
        raise ValueError("give SYSTEM with --traj, or --workdir for runs of their own")
    from pathlib import Path

    import boonza

    from .trajectory import open_trajectory

    reference = _load(args.reference) if args.reference else None
    if args.workdir:  # each brings its own system: only the protein must match
        runs = []
        for d in args.workdir:
            own = _run_system(Path(d) / "solvated.dms", typed=args.features)
            runs.append((own, open_trajectory(str(Path(d) / "trajectory.dcd"), own)))
        system = runs[0][0]
    else:
        system = _load(args.system)
        runs = [open_trajectory(path, system) for path in args.traj]
    _cg_selections(args, system, args.workdir)
    if args.interval_ns is None and args.workdir:
        args.interval_ns = _interval_of(args.workdir)
    from .probemap import SOLVENT_NAMES

    pocket_protein = f"not ({args.ligandsel}) and not resname {' '.join(SOLVENT_NAMES)}"
    found = boonza.sites(system, runs, reference, ligand=args.ligandsel, align=args.alignsel,
                         spacing=args.spacing, enrichment=args.enrichment,
                         min_occupancy=args.min_occupancy, periodic=not args.no_pbc,
                         pocket_protein=pocket_protein, rank=args.rank,
                         radius=None if args.radius == "point" else args.radius,
                         **({} if args.buried is None else {"buried": args.buried}),
                         **({} if args.min_volume is None
                            else {"min_volume": args.min_volume}))  # fmt: skip
    frames = len(found.centroids)
    bulk = int((found.labels < 0).sum())
    topologies = len({own.natoms for own in found.systems}) if found.systems else 1
    extra = f" over {topologies} topologies" if topologies > 1 else ""
    print(f"{len(runs)} runs, {frames} pooled frames{extra}; {100 * bulk / frames:.1f}% in bulk")
    if found.drift is not None and len(found.drift):
        # every site and every pocket is measured in the reference's frame, so a
        # protein that changes shape measures them against a shape it no longer has
        moved = np.asarray(found.drift, float)
        worst = float(moved.max())
        where = "the structure the sites are measured in"
        if worst >= 2.0:
            print(f"  the protein moved {np.median(moved):.1f} A from {where} "
                  f"({moved.min():.1f} to {worst:.1f}):\n  its pockets are measured against a "
                  "shape the run no longer has.  Hold the fold -- Martini's elastic\n  network, "
                  "--dihedral-restraint ss for SIRAH -- or analyse the frames before it "
                  "moved")  # fmt: skip
        else:
            print(f"  the protein stayed within {worst:.1f} A of {where}")
    if args.interval_ns:
        # what the gate comes to in time, which is what makes a site a site: a
        # probe pauses anywhere for a nanosecond, so a run short enough turns
        # those pauses into sites, wherever they happened to be
        sampled = len(np.unique(found.where[:, [0, 2]], axis=0)) * args.interval_ns
        dwell = args.min_occupancy * sampled
        if dwell < 2.0:
            print(f"  {sampled:g} ns sampled, so --min-occupancy {args.min_occupancy:g} asks "
                  f"for only {dwell:.2f} ns in a place:\n  a probe pauses that long in bulk, "
                  "so expect sites that are nothing but a pause")  # fmt: skip
    maps, spots, scored = {}, [], {}
    if args.features:  # before the table, so one table carries the scores too
        print("typing each probe's atoms for the feature maps (a second pass over the frames)",
              flush=True)  # fmt: skip
        maps = boonza.feature_maps(system, runs, reference, ligand=args.ligandsel,
                                   align=args.alignsel, spacing=args.spacing,
                                   periodic=not args.no_pbc,
                                   backbone=args.feature_backbone)  # fmt: skip
        spots = boonza.hotspots(maps, enrichment=2.0 * args.enrichment)
        from boonza.pharmacophore import DSCORE

        for k, site in enumerate(found):
            score, philic = boonza.site_score(site, maps)
            if not np.isnan(score):  # a site with no pocket has nothing to score
                scored[k] = (philic, score, boonza.site_score(site, maps, DSCORE)[0])
    rates = {}
    if args.interval_ns:  # the dwell times come from the frames already in hand
        for k in range(len(found)):
            try:
                rates[k] = boonza.kinetics(system, found, k, interval_ns=args.interval_ns,
                                           temperature=args.temperature,
                                           hysteresis=args.hysteresis)  # fmt: skip
            except ValueError as e:
                print(f"site {k}: no rates ({e})")
    pooled = any(site.runs > 1 for site in found)

    def header() -> str:
        """The table's columns; ``row`` fills them in the same order.

        A column only appears where there is something in it: no scores without
        ``--features``, no rates without an interval between frames, no ``runs``
        until more than one run is pooled.
        """
        out = [f"{'site':>4}", f"{'occupied':>8}", f"{'pocket':>9}", f"{'burial':>6}"]
        if scored:
            out += [f"{'philic':>6}", f"{'score':>6}", f"{'Dscore':>6}"]
        if pooled:
            out.append(f"{'runs':>5}")
        out += [f"{'copies':>6}", f"{'arrivals':>8}"]
        if rates:
            out += [f"{'exits':>5}", f"{'stay_ns':>7}", f"{'dG':>6}", f"{'KD_mM':>9}"]
        return " ".join([*out, "centre"])

    def row(k: int, site) -> str:
        """Everything measured about one site, in one line: how often it was held,
        the pocket it sits in, what that pocket asks for, and how fast it fills
        and empties."""
        out = [f"{k:4d}", f"{100 * site.occupancy:7.1f}%",
               f"{(f'{site.volume:.0f} A^3' if site.volume else '-'):>9}",
               _cell(site.burial if site.volume else None, "6.2f")]  # fmt: skip
        if scored:
            philic, score, drug = scored.get(k, (None, None, None))
            out += [_cell(philic, "6.2f"), _cell(score, "6.2f"), _cell(drug, "6.2f")]
        if pooled:
            out.append(f"{site.runs:5d}")
        out += [f"{site.copies:6d}", f"{site.arrivals:8d}"]
        if rates:
            r = rates.get(k)
            out += [_cell(r and r.events, "5d"), _cell(r and r.residence_ns, "7.1f"),
                    _cell(r and r.dG, "+6.2f"), _cell(r and 1e3 * r.KD, "9.1e")]  # fmt: skip
        return " ".join([*out, " ".join(f"{x:7.1f}" for x in site.center)])

    print(header())
    for k, site in enumerate(found):
        print(row(k, site))
    if not len(found):
        print("no site is visited more than bulk solvent would explain")
    elif scored:
        print(f"  score and Dscore are SiteMap's shape -- 0.0733 sqrt(n) + 0.6688 burial"
              f" - 0.20 philic,\n  and 0.094, 0.60, -0.324 for Dscore -- over the pocket's n"
              f" cells of {args.spacing:g} A.\n  They rank these pockets against each other, not"
              f" against SiteMap's own 0.8 and 1.0:\n  its n counts site points of its own grid,"
              f" ours cells a probe's atoms reached.")  # fmt: skip
    elif not args.features and any(s.volume for s in found):
        print("  (--features adds the polar share of each pocket and a SiteMap-shaped score)")
    if rates:  # the table has the rate a site is read by; this is the rest of it
        print()
        for rate in rates.values():
            print(rate.summary())
    if args.features:
        print(f"\n{len(spots)} hotspots: what a pocket asks for, and how many molecules agree")
        placed = 0
        for k, site in enumerate(found):
            near = boonza.wanted(spots, site.center)
            if not near:
                continue
            placed += 1
            print(f"  site {k}:")
            for h in near[:5]:
                print(f"    {h.family:12s} {h.ligands:3d} ligands, {h.enrichment:6.0f}x bulk, "
                      f"radius {h.radius:.1f} A")  # fmt: skip
        if spots and not placed:  # hotspots of their own, where no site settled
            print("  none of them sits in a site; the strongest, wherever they are:")
            for h in sorted(spots, key=lambda h: -h.enrichment)[:10]:
                centre = " ".join(f"{x:7.1f}" for x in h.center)
                print(f"    {h.family:12s} {h.ligands:3d} ligands, {h.enrichment:6.0f}x bulk, "
                      f"radius {h.radius:.1f} A  {centre}")  # fmt: skip
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if spots:
            boonza.write_hotspots(out / "hotspots.pdb", spots)
            for fam, grid in maps.items():
                if len(getattr(grid, "places", ())):
                    grid.write_dx(out / f"{fam.lower()}.dx")
        doc = {"runs": [str(x) for x in args.traj], "frames": frames, "bulk": bulk,
               "spacing": args.spacing, "enrichment": args.enrichment,
               "interval_ns": args.interval_ns, "ligand": args.ligandsel, "sites": []}  # fmt: skip
        for k, site in enumerate(found):
            doc["sites"].append({"center": site.center.tolist(), "occupancy": site.occupancy,
                                 "copy_frames": site.copy_frames, "volume": site.volume,
                                 "burial": site.burial,
                                 "runs": site.runs, "copies": site.copies,
                                 "arrivals": site.arrivals, "spread": site.spread,
                                 "frames": found.frames(k).tolist()})  # fmt: skip
            if k in scored:
                philic, score, drug = scored[k]
                doc["sites"][k].update({"philic": philic, "score": score, "dscore": drug})
        for rate in rates.values():
            doc["sites"][rate.site].update(
                {"dG": rate.dG, "dG_interval": list(rate.dG_interval), "KD": rate.KD,
                 "k_on": rate.k_on, "k_off": rate.k_off, "residence_ns": rate.residence_ns,
                 "departures": rate.events, "arrivals_measured": rate.arrivals,
                 "bound_ns": rate.bound_ns, "unbound_ns": rate.unbound_ns,
                 "concentration_M": rate.concentration,
                 "occupancy_from_frames": rate.occupancy,
                 "occupancy_from_rates": rate.occupancy_from_rates,
                 "consistent": rate.consistent})  # fmt: skip
        (out / "sites.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        _write_sites_csv(out / "sites.csv", doc["sites"], args.interval_ns)
        if found.drift is not None and len(found.drift):
            _write_rmsd_csv(out / "rmsd.csv", found, args.interval_ns)
        found.density.write_dx(out / "density.dx")
        peak = float(found.density.enrichment.max())
        shape = None
        if found.occupancy is not None:  # where the atoms reach: the pockets' shape
            shape = found.occupancy.density(found.volume)
            shape.write_dx(out / "occupancy.dx")
            pockets = 0
            for k, site in enumerate(found):  # one map per site: its pocket alone
                if site.cells is None or not len(site.cells):
                    continue
                # a mask, not the enrichment: the question a pocket map answers is
                # where the pocket is, and an isosurface of enrichment inside it
                # shows the cells the probes visited most, which looks scattered.
                # occupancy.dx is there for how hot each cell is.
                mask = np.zeros(found.occupancy.counts.size)
                mask[site.cells] = 1.0
                shape.write_dx(out / f"pocket{k}.dx", mask.reshape(found.occupancy.dims))
                pockets += 1
        extra = " and the feature maps" if spots else ""
        where = ("sites.json, sites.csv, rmsd.csv, density.dx"
                 + (", occupancy.dx" if shape else ""))  # fmt: skip
        print(f"\nwrote {where}{extra} to {out} (centroids peak {peak:.0f}x bulk"
              + (f", atoms {float(shape.enrichment.max()):.0f}x)" if shape else ")"))  # fmt: skip
        if shape is not None and pockets:
            from boonza.sites import POCKET_LEVEL

            print(f"  pocket0.dx..pocket{pockets - 1}.dx are masks of the pockets, one value "
                  f"inside each: draw them at {POCKET_LEVEL:g}\n  for the whole volume that was "
                  "reported, which is what sites.pml and sites.tcl do")  # fmt: skip
        boonza.write_viewer_scripts(out, found.sites)
        _viewer_hint(args, out, system)
    return 0


#: The columns of sites.csv, and the key each one takes from sites.json.  Every
#: number the three printed tables hold is here, with the units in the name
#: where the printed line carries them in its text.
_SITE_COLUMNS = (("site", "site"), ("occupied", "occupancy"), ("of_pool", "copy_frames"),
                 ("pocket_A3", "volume"), ("burial", "burial"), ("philic", "philic"),
                 ("score", "score"), ("Dscore", "dscore"), ("runs", "runs"),
                 ("copies", "copies"), ("arrivals", "arrivals"), ("spread", "spread"),
                 ("x", "x"), ("y", "y"), ("z", "z"), ("interval_ns", "interval_ns"),
                 ("dG_kcal", "dG"), ("dG_low", "dG_low"), ("dG_high", "dG_high"),
                 ("KD_mM", "KD_mM"), ("k_on_per_M_per_ns", "k_on"), ("k_off_per_ns", "k_off"),
                 ("residence_ns", "residence_ns"), ("departures", "departures"),
                 ("arrivals_measured", "arrivals_measured"), ("bound_ns", "bound_ns"),
                 ("unbound_ns", "unbound_ns"), ("concentration_mM", "concentration_mM"),
                 ("bound_frac_frames", "occupancy_from_frames"),
                 ("bound_frac_rates", "occupancy_from_rates"),
                 ("rates_agree", "consistent"))  # fmt: skip


def _write_sites_csv(path, sites, interval_ns=None) -> None:
    """Every number sites.json holds about a site, one row each.

    The three things boonza sites prints -- how often a site was held, its rates,
    its score -- are one row here, which is what a spreadsheet or a dataframe
    wants, and the occupancies are fractions rather than the percentages a table
    reads better in.  A column a run has nothing for -- no score without
    ``--features``, no rates without an interval between frames -- is left empty
    rather than filled with a nought that would average into the next plot.
    """
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _ in _SITE_COLUMNS])
        for k, site in enumerate(sites):
            row = {**site, "site": k, "interval_ns": interval_ns}
            row.update(zip("xyz", site["center"], strict=True))
            low, high = site.get("dG_interval") or (None, None)
            row.update(dG_low=low, dG_high=high)
            # the rates are printed in mM, where they are read in M: say which
            for name, key in (("KD_mM", "KD"), ("concentration_mM", "concentration_M")):
                if site.get(key) is not None:
                    row[name] = 1e3 * site[key]
            writer.writerow([_csv_number(row.get(key)) for _, key in _SITE_COLUMNS])


def _cell(value, spec: str) -> str:
    """A number in its column, or a dash as wide where there is nothing to say.

    A site with no pocket has no burial, and a site nothing left has no rate: a
    nought in either column would be read as a measurement.
    """
    if value is None or value is False or (isinstance(value, float) and not np.isfinite(value)):
        return f"{'-':>{len(f'{0:{spec}}')}}"  # as wide as the number would have been
    return f"{value:{spec}}"


def _write_rmsd_csv(path, found, interval_ns=None) -> None:
    """How far the protein is from the structure the sites are measured in, frame
    by frame.

    Every site and every pocket lives in the reference's frame of reference, so
    this is the trace that says how much to believe them: a run that wanders
    measures its pockets against a shape it has left behind.  It is the backbone
    the frames were fitted on -- alpha carbons, or the beads a coarse-grained
    model puts there -- after the fit, so a rigid protein reads near zero however
    far it has travelled or turned.
    """
    import csv

    runs = (found.drift_runs if found.drift_runs is not None
            else np.zeros(len(found.drift), np.int64))  # fmt: skip
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "frame", "time_ns", "rmsd_A"])
        counts: dict[int, int] = {}
        for run, rmsd in zip(runs.tolist(), np.asarray(found.drift, float).tolist(), strict=True):
            frame = counts.get(run, 0)
            counts[run] = frame + 1
            when = f"{(frame + 1) * interval_ns:.6g}" if interval_ns else ""
            writer.writerow([run, frame, when, f"{rmsd:.3f}"])


def _csv_number(value) -> str:
    """A number as short as it can be said without losing it, and nothing at all
    for what was not measured."""
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    if not np.isfinite(value):
        return ""  # a rate with no departures to measure it from: not nan, nothing
    return f"{value:.6g}"


def _run_system(path, typed: bool = False):
    """A run's system for an analysis: its structure, and its force field only
    where something needs it.

    The tables are the slow part of a .dms and an analysis reads the structure,
    so they are skipped -- but not ``structure_only``, which would drop the
    pseudo particles (Martini's virtual sites, CHARMM's lone pairs) the
    trajectory still carries.

    ``typed`` is for the feature maps: Martini says what a bead stands for in
    the bead's name, where SIRAH says it in the force field's type, so a SIRAH
    run has to bring its nonbonded table along to be typed at all.
    """
    import boonza

    s = boonza.load(str(path), without_tables=True)
    if typed and len(s.select("name GN GC GO").ids):
        return boonza.load(str(path))  # SIRAH: its chemistry is in its types
    return s


def _interval_of(workdirs):
    """The ns between frames, from the runs' own settings; None if they disagree.

    Every time a rate is reported in -- residence, k_on, k_off, and so dG -- is
    frames times this, so taking it from the run beats typing it and being out
    by a factor of ten.
    """
    import tomllib
    from pathlib import Path

    found = set()
    for d in workdirs:
        for name in (Path(d) / "final.toml", Path(d) / "md.toml", Path(d).parent / "md.toml"):
            if name.is_file():
                try:
                    settings = tomllib.loads(name.read_text())
                except (OSError, tomllib.TOMLDecodeError):
                    continue
                if "production_report_interval_ns" in settings:
                    found.add(float(settings["production_report_interval_ns"]))
                    break
    if len(found) == 1:
        interval = found.pop()
        print(f"frames are {interval:g} ns apart (the runs' own "
              f"production_report_interval_ns); --interval-ns overrides it")  # fmt: skip
        return interval
    if len(found) > 1:
        print(f"the runs report frames at different intervals ({sorted(found)} ns): no rates "
              "without --interval-ns")  # fmt: skip
    return None


#: What a box holds that a viewer should throw away: Martini's water and ions,
#: then SIRAH's, then an all-atom box's.  Bulk only -- a magnesium or a zinc is
#: usually part of the structure rather than part of the solvent, and a run that
#: holds one wants to see it.
_VIEWER_SOLVENT = ("W", "WF", "WN", "WT4", "WLS", "ION", "NaW", "KW", "ClW",
                   "HOH", "WAT", "TIP3", "SOL", "SPC", "NA", "CL", "K", "SOD",
                   "CLA", "POT")  # fmt: skip

#: The backbone beads that say a coarse-grained file still calls them what the
#: run does -- a view file written before boonza stopped renaming them does not.
_VIEWER_BEADS = ("BB", "GC", "GN", "GO")


def _still_named(dms, beads) -> bool:
    """Whether a view file calls its beads what the run's system calls them.

    One written before boonza stopped renaming the backbone to ``CA`` holds
    beads a viewer mis-bonds, which is the one thing the file exists to avoid,
    so such a run is pointed back at its ``solvated.dms``.  The names come
    straight out of the .dms, which is a SQLite database, rather than by loading
    it: this is a hint, not a pass over an all-atom box.
    """
    import sqlite3

    if not beads:
        return True  # an all-atom run renames nothing
    try:
        with sqlite3.connect(f"file:{dms}?mode=ro", uri=True) as db:
            held = {str(row[0]) for row in db.execute("select distinct name from particle")}
    except sqlite3.Error:  # not readable as one: let the viewer say so
        return True
    return bool(set(beads) & held)


def _ligand_flag(system, selection: str) -> tuple[str, str]:
    """``--ligand`` for vizard and for pizard, as short as the system allows.

    A chain of their own says it in two words, which is what a swim gives the
    probes; a list of 105 residue names is a line no one can read, so it is
    left to probes.json rather than printed.
    """
    if not selection:
        return "", ""
    try:
        ids = system.select(selection).ids
        chains = {str(system.chains["name"][c]) for c in
                  {int(system.residues["chain"][r]) for r in
                   {int(system.atoms["residue"][a]) for a in ids.tolist()}}}  # fmt: skip
    except Exception:  # noqa: BLE001 - a selection the system cannot answer is not a hint
        chains = set()
    if len(chains) == 1:
        only = chains.pop()
        if only and len(system.select(f"chain {only}").ids) == len(ids):
            return f'--ligand "chain {only}"', f'--ligand "chain {only}"'
    if len(selection) <= 50:
        names = selection.split()
        pml = f'--ligand "resn {"+".join(names[1:])}"' if names[:1] == ["resname"] else \
              f'--ligand "{selection}"'  # fmt: skip
        return f'--ligand "{selection}"', pml
    return "", ""  # the names are in probes.json; a truncated flag would be worse


def _viewer_hint(args, out, system=None) -> None:
    """Say how to look at what was just written, with the commands to type.

    Neither VMD nor PyMOL reads a bead file of its own, so the session comes
    from viswizard -- vizard for VMD, pizard for PyMOL -- and the maps are
    sourced into it.  ``view.dms`` is what a run writes for this, whatever the
    model; a run built before boonza stopped renaming beads for a viewer is
    pointed back at its ``solvated.dms``, which was always right.
    """
    from pathlib import Path

    beads = [b for b in _VIEWER_BEADS
             if system is not None and len(system.select(f"name {b}").ids)]  # fmt: skip
    structure = trajectory = None
    if getattr(args, "workdir", None):
        here = Path(args.workdir[0])  # as it was given: a relative path stays relative
        structure, trajectory = here / "solvated.dms", here / "trajectory.dcd"
        view = here / "view.dms"
        if view.is_file() and _still_named(view, beads):
            structure = view  # what a run writes to be looked at
    elif getattr(args, "system", None):
        structure = Path(args.system)
        trajectory = Path(args.traj[0]) if getattr(args, "traj", None) else None
    # the scripts name their files in full, so neither viewer has to be in that
    # directory and nothing has to be changed into it
    session = f"source {out / 'sites.tcl'}", f"@{out / 'sites.pml'}"
    print("\nto look at them (viswizard sets the session up; neither viewer reads beads alone):")
    if structure is None:
        print(f"  in the session:  {session[0]}     (PyMOL: {session[1]})")
        return
    vmd_lig, pml_lig = _ligand_flag(system, args.ligandsel) if system is not None else ("", "")
    where = f"{structure}{f' {trajectory}' if trajectory else ''}"
    strip = _VIEWER_SOLVENT
    both = (("vizard", vmd_lig, f'resname {" ".join(strip)}', session[0]),
            ("pizard", pml_lig, f'resn {"+".join(strip)}', session[1]))  # fmt: skip
    for viewer, lig, names, how in both:
        flags = " ".join(x for x in (lig, f'--strip "{names}"') if x)
        print(f"  {viewer} {where} {flags}  # then: {how}")


def _summarize(args) -> int:
    import boonza

    s = _load(args.file)
    summary = boonza.summarize(s, focus=args.focus, cutoff=args.cutoff, max_items=args.max_items)
    print(summary.to_json(indent=1) if args.json else summary, end="" if not args.json else "\n")
    return 0


def _build(args) -> int:
    import boonza

    if args.smiles:
        s = boonza.from_smiles(args.smiles, name=args.name, seed=args.seed,
                               optimize=not args.no_optimize,
                               conformers=args.conformers)  # fmt: skip
    else:
        s = boonza.peptide(args.sequence, conformation=args.conformation, seed=args.seed,
                           optimize=not args.no_optimize)  # fmt: skip
    boonza.save(s, args.output)
    print(f"wrote {args.output}: {s.natoms} atoms, {s.nbonds} bonds, {s.nresidues} residues")
    return 0


def _draw(args) -> int:
    from pathlib import Path

    from .draw import draw, read_smiles

    smiles, names = list(args.smiles or []), None
    if args.input:
        pairs = read_smiles(Path(args.input).read_text())
        if not pairs:
            raise ValueError(f"{args.input} holds no SMILES")
        smiles += [s for s, _ in pairs]
        names = [""] * len(args.smiles or []) + [n for _, n in pairs]
    if not smiles:
        raise ValueError("give SMILES to draw, or --input a file of them")
    d = draw(smiles, args.output, names=names, size=tuple(args.size), columns=args.columns,
             rows=args.rows, highlight=args.highlight, mcs=args.mcs, align=args.align,
             labels=not args.no_labels)  # fmt: skip
    where = ", ".join(str(f) for f in d.files)
    print(f"wrote {where}: {d.drawn} molecule{'s' if d.drawn != 1 else ''}")
    if d.failures:
        print(f"  {len(d.failures)} could not be read: {', '.join(d.failures[:5])}")
    if args.mcs:
        print(f"  common core: {d.core}" if d.core else "  no common core to mark")
    if args.align:
        print(f"  {d.aligned} of {d.drawn} laid out on the core")
    return 0


def _where(path) -> str:
    """What to say about a parameter file: there it is, or you have to supply it."""
    from pathlib import Path

    path = Path(path)
    if path.is_file():
        return f"{path.name}, which it carries" if path.parent.name == "params" else str(path)
    return f"{path.name}, which is not written"


def _martini_files(args):
    """The parameter files to use: the ones given, else the ones boonza carries
    for the version of Martini asked for."""

    from .martini import LIPIDS_FOR, NONBONDED_FOR, parameters

    version = 2 if getattr(args, "model", "martini3") == "martini2" else 3
    itp = args.martini_itp or str(parameters(NONBONDED_FOR[version])[0])
    lipids = getattr(args, "lipid_itp", None)
    if lipids is None:
        lipids = [str(p) for p in parameters(*LIPIDS_FOR[version])]
    return itp, lipids


def _martinize(args) -> int:
    import boonza

    from .martini import FORCEFIELD_FOR

    cys = args.cys if args.cys in ("auto", "none") else float(args.cys)
    version = 2 if args.model == "martini2" else 3
    m = boonza.martinize(_load(args.input), args.selection, ss=args.ss, elastic=args.elastic,
                         elastic_fc=args.elastic_fc, elastic_lower=args.elastic_lower,
                         elastic_upper=args.elastic_upper, elastic_decay=args.elastic_decay,
                         elastic_power=args.elastic_power, elastic_min_fc=args.elastic_min_fc,
                         res_min_dist=args.res_min_dist, cys=cys,
                         neutral_termini=args.neutral_termini, scfix=not args.no_scfix,
                         extdih=args.extdih, forcefield=FORCEFIELD_FOR[version])  # fmt: skip
    if args.solvate:
        from .martini import solvate

        m = solvate(m, padding=args.padding, salt=args.salt, seed=args.seed)
    itp, _ = _martini_files(args)
    top = m.save(args.output, martini_itp=itp)
    water = ", ".join(f"{c} {n}" for n, c in m.solvent)
    print(
        f"wrote {top} ({len(m.molecules)} molecules, {m.nbeads} beads"
        f"{'; ' + water if water else ''}) and cg.gro; it includes {_where(itp)}"
    )
    return 0


def _parameterize(args) -> int:
    import boonza
    from boonza import viparr

    if args.xml:
        if args.forcefields:
            raise SystemExit(
                "give viparr force fields (-f/-m/-a) or OpenMM XML files (-x), not both"
            )
        cons = None if args.without_constraints else "hbonds"
        s = boonza.parameterize_openmm(_load(args.input), args.xml, constraints=cons)
        boonza.save(s, args.output)
        tables = ", ".join(f"{n} {len(s.table(n))}" for n in s.table_names)
        print(f"wrote {args.output}: {s.natoms} atoms; {tables}")
        return 0
    ffs = []
    for kind, name in args.forcefields:
        if kind == "f":
            ffs.append(viparr.load_forcefield(name, args.ffpath))
        elif not ffs:
            raise SystemExit(f"-{kind} {name}: give a force field with -f before patching it")
        else:
            ffs[-1] = viparr.merge_forcefields(ffs[-1], name, append_only=kind == "a",
                                               path=args.ffpath)  # fmt: skip
    system = _load(args.input)
    if args.gaff2:
        from boonza import gaff

        charges = {}
        for item in args.charge:
            res, _, q = item.partition("=")
            if not res or not q.lstrip("+-").isdigit():
                raise SystemExit(f"--charge {item}: give RESNAME=CHARGE, e.g. LIG=-1")
            charges[res] = int(q)
        parents = {}
        for item in args.parent:
            res, _, parent = item.partition("=")
            if not res or not parent:
                raise SystemExit(f"--parent {item}: give RESNAME=PARENT, e.g. MSE=MET")
            parents[res] = parent
        patch = gaff.gaff2_patch(system, ffs, charges=charges, parents=parents,
                                 protein_extent=args.protein_extent, draw=args.draw)  # fmt: skip
        if args.save_patch:
            viparr.write_forcefield(patch, args.save_patch)
        names = ", ".join(t.name for t in patch.templates) or "none needed"
        print(f"GAFF2 templates: {names}")
        for p in patch.parents:
            print(f"  {p['residue']} ({p['chain']}): parent {p['parent']} ({p['source']}); "
                  f"{p['protein_heavy_atoms']} of {p['heavy_atoms']} heavy atoms keep "
                  "protein types")  # fmt: skip
        if ffs:  # onto the protein force field, wherever it is listed
            host = gaff.host_index(ffs)
            ffs[host] = viparr.merge_forcefields(ffs[host], patch)
        else:
            ffs = [patch]
    elif args.charge or args.parent or args.save_patch or args.draw:
        raise SystemExit("--charge, --parent, --save-patch and --draw go with --gaff2")
    s = viparr.parameterize(system, ffs, rename_atoms=args.rename_atoms,
                            rename_residues=args.rename_residues,
                            fix_masses=not args.without_fix_masses, fatal=not args.non_fatal,
                            cmap_chirality=not args.viparr_cmap,
                            reorder_ids=not args.keep_ids,
                            constraints=not args.without_constraints)  # fmt: skip
    boonza.save(s, args.output)
    tables = ", ".join(f"{n} {len(s.table(n))}" for n in s.table_names)
    print(f"wrote {args.output}: {s.natoms} atoms; {tables}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boonza", description="Molecular system tools.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("md", help="prepare and run OpenMM MD, restartable (boonza md --help)")
    sub.add_parser("swim", help="ligands swimming around a protein, many simulations "
                   "(boonza swim --help)")  # fmt: skip

    q = sub.add_parser("info", help="summarize a structure or trajectory file")
    q.add_argument("file")
    q.set_defaults(run=_info)

    q = sub.add_parser("convert", help="convert between formats, optionally a selection")
    q.add_argument("input")
    q.add_argument("output")
    q.add_argument("-s", "--selection", help="atoms to keep")
    q.add_argument("--structure-only", action="store_true", help="skip force-field tables")
    q.set_defaults(run=_convert)

    q = sub.add_parser("select", help="print the atom indices of a selection")
    q.add_argument("file")
    q.add_argument("selection")
    q.set_defaults(run=_select)

    q = sub.add_parser("validate", help="check for likely errors")
    q.add_argument("file")
    q.add_argument("--strict", action="store_true", help="also run simulation-readiness checks")
    q.add_argument("--max-ring", type=int, default=10, help="largest ring checked for knots")
    q.set_defaults(run=_validate)

    q = sub.add_parser("knots", help="find bonds passing through rings")
    q.add_argument("file")
    q.add_argument("--max-cycle", type=int, default=10)
    q.add_argument("-s", "--selection", default="all")
    q.add_argument("--ignore-excluded", action="store_true",
                   help="skip knots whose atoms are excluded from the ring")  # fmt: skip
    q.set_defaults(run=_knots)

    q = sub.add_parser("diff", help="compare two systems, force fields included")
    q.add_argument("a")
    q.add_argument("b")
    q.add_argument("--rtol", type=float, default=1e-6)
    q.add_argument("--no-positions", action="store_true", help="ignore coordinates and cell")
    q.add_argument(
        "--canonical",
        action="store_true",
        help="compare force fields made by different programs (energy-equivalent forms)",
    )
    q.set_defaults(run=_diff)

    q = sub.add_parser("describe", help="force-field parameters of selected atoms")
    q.add_argument("file")
    q.add_argument("selection")
    q.add_argument("--terms", choices=("any", "all"), default="any")
    q.add_argument("--pairs", action="store_true", help="list pairs even for large selections")
    q.set_defaults(run=_describe)

    q = sub.add_parser("dssp", help="secondary structure per chain")
    q.add_argument("file")
    q.add_argument("--simplified", action="store_true", help="H/E/C alphabet")
    q.set_defaults(run=_dssp)

    q = sub.add_parser("phipsi", help="backbone phi/psi/omega per residue (degrees)")
    q.add_argument("file")
    q.set_defaults(run=_phipsi)

    from .symmetry import DEFAULT_FIT, DEFAULT_LIGAND

    q = sub.add_parser("rmsd", help="symmetry-corrected ligand RMSD after fitting proteins")
    q.add_argument("mobile", help="docked or simulated structure")
    q.add_argument("reference", help="reference (e.g. crystal) structure")
    q.add_argument("--mobsel", default=DEFAULT_FIT, help="atoms fitted in mobile")
    q.add_argument("--refsel", default=None,
                   help="atoms fitted in reference (default: --mobsel)")  # fmt: skip
    q.add_argument("--ligandsel", default=DEFAULT_LIGAND, help="ligand atoms")
    q.add_argument("--ref-ligandsel", default=None,
                   help="reference ligand (default: --ligandsel)")  # fmt: skip
    q.add_argument("--align", choices=("order", "sequence", "none"), default="order",
                   help="pair fit atoms in order, by sequence alignment, or no fit")  # fmt: skip
    q.add_argument("--traj", help="trajectory of the mobile system (DCD/XTC): one RMSD per frame")
    q.set_defaults(run=_rmsd)

    q = sub.add_parser("summarize", help="plain-text summary of a structure (for people and AI)")
    q.add_argument("file")
    q.add_argument("--focus", help="selection to describe in detail")
    q.add_argument("--cutoff", type=float, default=4.0, help="contact distance (A)")
    q.add_argument("--max-items", type=int, default=20, help="longest list printed")
    q.add_argument("--json", action="store_true", help="print the summary as JSON")
    q.set_defaults(run=_summarize)

    q = sub.add_parser("build", help="3D structure from a SMILES string or a peptide sequence")
    what = q.add_mutually_exclusive_group(required=True)
    what.add_argument("--smiles", help="molecule as SMILES")
    what.add_argument("--sequence", help="peptide as one-letter sequence")
    q.add_argument("-o", "--output", required=True, help="output file (.sdf, .pdb, .dms, ...)")
    q.add_argument("--conformation", default="helix",
                   help="peptide: helix, sheet, extended or polyproline")  # fmt: skip
    q.add_argument("--name", default="LIG", help="SMILES: residue name")
    q.add_argument("--conformers", type=int, default=1, help="SMILES: conformers to try")
    q.add_argument("--seed", type=int, default=42, help="random seed for the embedding")
    q.add_argument("--no-optimize", action="store_true", help="skip the MMFF minimization")
    q.set_defaults(run=_build)

    q = sub.add_parser("draw", help="a picture of molecules from SMILES (PNG or SVG)")
    q.add_argument("smiles", nargs="*", help="SMILES to draw")
    q.add_argument("-o", "--output", required=True, help="output file (.png or .svg)")
    q.add_argument("--input", help="file of SMILES, one per line, each with an optional name")
    q.add_argument("--columns", type=int, default=4, help="molecules across (default: 4)")
    q.add_argument(
        "--rows",
        type=int,
        help="molecules down; more than one page's worth go to NAME-2, NAME-3, ...",
    )
    q.add_argument("--size", type=int, nargs=2, default=[300, 250], metavar=("W", "H"),
                   help="pixels per molecule (default: 300 250)")  # fmt: skip
    q.add_argument("--mcs", action="store_true",
                   help="find the largest scaffold they share and mark it")  # fmt: skip
    q.add_argument("--highlight", metavar="SMARTS", help="mark this instead of the MCS")
    q.add_argument("--align", action="store_true",
                   help="draw every molecule with the core the same way up")  # fmt: skip
    q.add_argument("--no-labels", action="store_true", help="no names under the molecules")
    q.set_defaults(run=_draw)

    q = sub.add_parser("martinize", help="Martini beads and topology for proteins, as martinize2")
    q.add_argument("input", help="all-atom structure (hydrogens, if present, set protonation)")
    q.add_argument("output", help="directory for topol.top, molecule_N.itp and cg.gro")
    q.add_argument("--model", default="martini3", choices=("martini3", "martini2"),
                   help="which Martini to build (default martini3)")  # fmt: skip
    q.add_argument("--selection", default="protein", help="atoms to coarse-grain")
    q.add_argument("--ss", help="DSSP codes, one per residue (default: boonza's DSSP)")
    q.add_argument("--elastic", action="store_true", help="add an elastic network (-elastic)")
    q.add_argument("--elastic-fc", type=float, default=700.0, help="kJ/mol/nm^2 (-ef)")
    q.add_argument("--elastic-lower", type=float, default=0.0, help="A (-el, in nm there)")
    q.add_argument("--elastic-upper", type=float, default=9.0, help="A (-eu, in nm there)")
    q.add_argument("--elastic-decay", type=float, default=0.0, help="-ea")
    q.add_argument("--elastic-power", type=float, default=0.0, help="-ep")
    q.add_argument("--elastic-min-fc", type=float, default=0.0, help="-em")
    q.add_argument("--res-min-dist", type=int, help="-ermd (default 2)")
    q.add_argument("--cys", default="auto", help="auto, none, or an S-S distance in A")
    q.add_argument("--neutral-termini", action="store_true")
    q.add_argument("--no-scfix", action="store_true", help="no side-chain corrections")
    q.add_argument("--extdih", action="store_true", help="dihedrals for extended regions (-ed)")
    q.add_argument("--solvate", action="store_true", help="add Martini water and NaCl")
    q.add_argument("--padding", type=float, default=10.0, help="water beyond the protein (A)")
    q.add_argument("--salt", type=float, default=0.15, help="NaCl (mol/L), after neutralizing")
    q.add_argument("--seed", type=int, default=0, help="which waters become ions")
    q.add_argument("--martini-itp", default=None,
                   help="the Martini parameter file topol.top includes "
                        "(default: the Martini 3 file boonza carries)")  # fmt: skip
    q.set_defaults(run=_martinize)

    q = sub.add_parser("parameterize", help="apply viparr force fields (first match wins)")
    q.add_argument("input", help="structure with bonds and hydrogens")
    q.add_argument("output", help="output file (.dms keeps the force field)")
    q.add_argument(
        "-f",
        "--ff",
        dest="forcefields",
        action="append",
        default=[],
        type=lambda v: ("f", v),
        help="force field name or directory, in priority order",
    )
    q.add_argument(
        "-m",
        "--merge",
        dest="forcefields",
        action="append",
        type=lambda v: ("m", v),
        help="patch the previous -f force field (replaces templates and types)",
    )
    q.add_argument(
        "-a",
        "--append",
        dest="forcefields",
        action="append",
        type=lambda v: ("a", v),
        help="like -m, but refuse to replace anything",
    )
    q.add_argument(
        "-x",
        "--xml",
        action="append",
        default=[],
        help="OpenMM force field XML file (e.g. amber19-all.xml), instead of -f",
    )
    q.add_argument("--ffpath", help="directories of named force fields (default $VIPARR_FFPATH)")
    q.add_argument("--rename-atoms", action="store_true", help="copy atom names from templates")
    q.add_argument(
        "--rename-residues", action="store_true", help="copy residue names from templates"
    )
    q.add_argument(
        "--without-fix-masses",
        action="store_true",
        help="keep per-type masses (default: median mass per element, as viparr)",
    )
    q.add_argument("--non-fatal", action="store_true", help="warn about missing parameters")
    q.add_argument(
        "--viparr-cmap", action="store_true", help="give D residues the L CMAP grid, as viparr does"
    )
    q.add_argument(
        "--keep-ids",
        action="store_true",
        help="append pseudos after all atoms (viparr's default order)",
    )
    q.add_argument(
        "--without-constraints",
        action="store_true",
        help="no constraint_ahN/constraint_hoh tables (viparr adds them)",
    )
    q.add_argument(
        "--gaff2",
        action="store_true",
        help="GAFF2/AM1-BCC templates (AmberTools) for what the force fields cannot "
        "match, ligands and covalent adducts included, merged onto the first -f",
    )
    q.add_argument(
        "--charge",
        action="append",
        default=[],
        metavar="RES=Q",
        help="formal charge of a residue for --gaff2, where the input has none",
    )
    q.add_argument(
        "--parent",
        action="append",
        default=[],
        metavar="RES=PARENT",
        help="the standard residue a modified residue comes from, e.g. MSE=MET (--gaff2)",
    )
    q.add_argument("--save-patch", metavar="DIR", help="write the --gaff2 templates as a viparr ff")
    q.add_argument(
        "--protein-extent",
        choices=("matched", "cb"),
        default="matched",
        help="amino acids with GAFF2 atoms keep protein types as far as they match "
        "(matched) or on the backbone and CB only (cb)",
    )
    q.add_argument("--draw", metavar="DIR", help="2D drawings of covalent adducts (PNG)")
    q.set_defaults(run=_parameterize)  # fmt: skip

    q = sub.add_parser("drmsd", help="pocket-ligand distance RMSD, symmetry-corrected")
    q.add_argument("system", help="structure (topology of --traj)")
    q.add_argument("--traj", help="trajectory (DCD/XTC): one dRMSD per frame")
    q.add_argument("--reference", help="reference structure (default: first frame)")
    q.add_argument("--ligandsel", default=DEFAULT_LIGAND, help="ligand atoms")
    q.add_argument("--ref-ligandsel", default=None,
                   help="reference ligand (default: --ligandsel)")  # fmt: skip
    q.add_argument("--proteinsel", default="protein and name CA", help="pocket candidates")
    q.add_argument("--cutoff", type=float, default=5.0, help="pocket cutoff (A)")
    q.add_argument("--no-symmetry", action="store_true", help="pair ligand atoms in order")
    q.add_argument("--no-pbc", action="store_true", help="ignore periodic boxes")
    q.set_defaults(run=_drmsd)

    q = sub.add_parser("poses", help="representative frames: which pose, and how much of the run")
    q.add_argument("system", nargs="?", help="structure (topology of --traj)")
    q.add_argument("--traj", help="trajectory (DCD/XTC)")
    q.add_argument("--structures", nargs="+", default=[],
                   help="separate structure files instead: what docking and structure "
                        "prediction write. Their heavy atoms are paired by name, so files "
                        "that differ in hydrogens or atom order still compare")  # fmt: skip
    q.add_argument("--reference", help="structure the pocket comes from "
                   "(default: the frame whose ligand touches the most protein)")  # fmt: skip
    q.add_argument("--ligandsel", default=DEFAULT_LIGAND, help="ligand atoms")
    q.add_argument("--proteinsel", default="protein and name CA", help="pocket candidates")
    q.add_argument("--pocket-cutoff", type=float, default=5.0, help="pocket cutoff (A)")
    q.add_argument("--pocketsel", default=None,
                   help="the pocket atoms themselves; --proteinsel, --pocket-cutoff and the "
                        "reference then decide nothing")  # fmt: skip
    q.add_argument("--cutoff", type=float, default=1.5, help="one pose, in dRMSD (A)")
    q.add_argument("--min-population", type=float, default=0.02,
                   help="share of frames a pose must hold to be listed")  # fmt: skip
    q.add_argument("--stride", type=int, default=1, help="use every nth frame")
    q.add_argument("--no-settle", action="store_true",
                   help="group the whole run, including the frames before the ligand arrived. "
                        "Settling trims those by how much protein the ligand touches, which "
                        "says when it arrived and not which pose it took")  # fmt: skip
    q.add_argument("--no-symmetry", action="store_true", help="pair ligand atoms in order")
    q.add_argument("--no-pbc", action="store_true", help="ignore periodic boxes")
    q.add_argument("-o", "--out", help="write the representative structures here")
    q.add_argument("--format", default="dms", help="structure format to write (default: dms)")
    q.set_defaults(run=_poses)

    q = sub.add_parser("probes",
                       help="what each probe of a coarse-grained swim touches, "
                            "residue by residue")  # fmt: skip
    q.add_argument(
        "--workdir",
        nargs="+",
        required=True,
        help="runs of a coarse-grained 'boonza swim', each with its probes.json",
    )
    q.add_argument("--probes", nargs="+", metavar="RESNAME",
                   help="the probes' residue names, for a run without probes.json (any "
                        "coarse-grained model, or molecules of your own)")  # fmt: skip
    q.add_argument("--cutoff", type=float, default=6.0, help="contact distance (A)")
    q.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    q.add_argument("--by", choices=("probe", "side-chain"), default="side-chain",
                   help="report per probe, or pooled by side chain (default)")  # fmt: skip
    q.add_argument("--top", type=int, default=6, help="residues listed per probe")
    q.add_argument("--no-pbc", action="store_true", help="ignore the periodic box")
    q.add_argument("-o", "--out", help="write the whole table as CSV")
    q.set_defaults(run=_probes)

    q = sub.add_parser("sites", help="where a ligand goes, pooled over runs and copies")
    q.add_argument("system", nargs="?", help="structure (topology of --traj)")
    q.add_argument("--traj", nargs="+", default=[], help="one or more trajectories of SYSTEM")
    q.add_argument("--workdir", nargs="+", default=[],
                   help="work directories, each read with its own solvated.dms: runs whose "
                        "ligands differ pool as long as the protein does not")  # fmt: skip
    q.add_argument("--reference", help="structure the runs are superposed on (default: the first)")
    q.add_argument("--ligandsel", default=DEFAULT_LIGAND, help="ligand atoms, every copy")
    q.add_argument("--alignsel", default=DEFAULT_ALIGN, help="atoms the runs align on")
    q.add_argument("--spacing", type=float, default=1.0, help="grid spacing (A)")
    q.add_argument("--enrichment", type=float, default=20.0,
                   help="how many times more visited than bulk a site must be")  # fmt: skip
    q.add_argument("--min-occupancy", type=float, default=0.05,
                   help="share of the frames a site must hold something in, whichever "
                        "ligand it is; not a share of the pooled copy-frames, which the "
                        "copy count dilutes")  # fmt: skip
    q.add_argument("--radius", choices=("sigma", "rmin", "point"), default="sigma",
                   help="how much room a probe particle takes on the occupancy grid: half "
                        "the sigma of its own nonbonded term (default), half of 2^(1/6) "
                        "sigma, or point: only the cell its centre fell in")  # fmt: skip
    q.add_argument("--rank", choices=("pocket", "occupied", "agreement", "burial"),
                   default="pocket",
                   help="order the sites by their pocket (how much room, how enclosed; the "
                        "default), by dwell, by how many copies chose them, or by enclosure "
                        "alone.  Dwell is the tempting one and the wrong one: a sticky patch "
                        "holds something all run without being anywhere a ligand fits")  # fmt: skip
    q.add_argument("--feature-backbone", action="store_true",
                   help="with --features on a Martini run, also type the BB beads (an amide: "
                        "donor and acceptor), which every probe carries")  # fmt: skip
    q.add_argument("--features", action="store_true",
                   help="also map what each pocket asks for -- donor, acceptor, aromatic, "
                        "greasy -- and where")  # fmt: skip
    q.add_argument("--interval-ns", type=float, default=None,
                   help="ns between frames; with it, rates, residence times and dG")  # fmt: skip
    q.add_argument("--temperature", type=float, default=310.0, help="K, for dG")
    q.add_argument("--hysteresis", type=float, default=2.0,
                   help="leave a site at this many times the distance it is entered at. "
                        "One boundary counts every recrossing as a departure")  # fmt: skip
    q.add_argument("--buried", type=float, default=None, metavar="SHARE",
                   help="how enclosed a pocket must be, 0 open water to 1 shut in "
                        "(default 0.4): what separates a pocket from a sticky patch")  # fmt: skip
    q.add_argument("--min-volume", dest="min_volume", type=float, default=None, metavar="A3",
                   help="the smallest pocket worth reporting, in cubic angstroms (default "
                        "20, where one a ligand sits in runs to hundreds)")  # fmt: skip
    q.add_argument("--no-pbc", action="store_true", help="ignore periodic boxes")
    q.add_argument("-o", "--out", help="write sites.json here")
    q.set_defaults(run=_sites, needs=("system and --traj", "or --workdir"))
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "md":  # its own parser: settings files, restarts
        from .md.cli import main as md_main

        return md_main(argv[1:])
    if argv and argv[0] == "swim":
        from .md.swim import main as swim_main

        return swim_main(argv[1:])
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        print(f"boonza {argv[0]}: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
