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


def _load(path, structure_only=False):
    import boonza

    if structure_only and str(path).lower().endswith((".dms", ".dms.gz", ".dms.bz2")):
        return boonza.load(path, structure_only=True)
    return boonza.load(path)


def _info(args) -> int:
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
            own = _load(str(Path(d) / "solvated.dms"), structure_only=True)
            runs.append((own, open_trajectory(str(Path(d) / "trajectory.dcd"), own)))
        system = runs[0][0]
    else:
        system = _load(args.system)
        runs = [open_trajectory(path, system) for path in args.traj]
    found = boonza.sites(system, runs, reference, ligand=args.ligandsel, align=args.alignsel,
                         spacing=args.spacing, enrichment=args.enrichment,
                         min_occupancy=args.min_occupancy, periodic=not args.no_pbc)  # fmt: skip
    frames = len(found.centroids)
    bulk = int((found.labels < 0).sum())
    topologies = len({own.natoms for own in found.systems}) if found.systems else 1
    extra = f" over {topologies} topologies" if topologies > 1 else ""
    print(f"{len(runs)} runs, {frames} pooled frames{extra}; {100 * bulk / frames:.1f}% in bulk")
    print(f"{'site':>4} {'occupied':>9} {'runs':>5} {'copies':>7} {'arrivals':>9} {'spread':>7}"
          f"  centre")  # fmt: skip
    for k, site in enumerate(found):
        centre = " ".join(f"{x:7.1f}" for x in site.center)
        print(f"{k:4d} {100 * site.occupancy:8.1f}% {site.runs:5d} {site.copies:7d} "
              f"{site.arrivals:9d} {site.spread:6.1f} A  {centre}")  # fmt: skip
    if not len(found):
        print("no site is visited more than bulk solvent would explain")
    rates = []
    if args.interval_ns:
        print()
        for k in range(len(found)):
            try:
                rate = boonza.kinetics(system, found, k, interval_ns=args.interval_ns,
                                       temperature=args.temperature,
                                       hysteresis=args.hysteresis)  # fmt: skip
            except ValueError as e:
                print(f"site {k}: no kinetics ({e})")
                continue
            print(rate.summary())
            rates.append(rate)
    spots = []
    if args.features:
        maps = boonza.feature_maps(system, runs, reference, ligand=args.ligandsel,
                                   align=args.alignsel, spacing=args.spacing,
                                   periodic=not args.no_pbc)  # fmt: skip
        spots = boonza.hotspots(maps, enrichment=2.0 * args.enrichment)
        print(f"\n{len(spots)} hotspots: what a pocket asks for, and how many molecules agree")
        for k, site in enumerate(found):
            near = boonza.wanted(spots, site.center)
            if not near:
                continue
            print(f"  site {k}:")
            for h in near[:5]:
                print(f"    {h.family:12s} {h.ligands:3d} ligands, {h.enrichment:6.0f}x bulk, "
                      f"radius {h.radius:.1f} A")  # fmt: skip
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
               "ligand": args.ligandsel, "sites": []}  # fmt: skip
        for k, site in enumerate(found):
            doc["sites"].append({"center": site.center.tolist(), "occupancy": site.occupancy,
                                 "runs": site.runs, "copies": site.copies,
                                 "arrivals": site.arrivals, "spread": site.spread,
                                 "frames": found.frames(k).tolist()})  # fmt: skip
        for rate in rates:
            doc["sites"][rate.site].update(
                {"dG": rate.dG, "dG_interval": list(rate.dG_interval), "KD": rate.KD,
                 "k_on": rate.k_on, "k_off": rate.k_off, "residence_ns": rate.residence_ns,
                 "departures": rate.events, "arrivals_measured": rate.arrivals,
                 "bound_ns": rate.bound_ns, "unbound_ns": rate.unbound_ns,
                 "concentration_M": rate.concentration,
                 "occupancy_from_rates": rate.occupancy_from_rates,
                 "consistent": rate.consistent})  # fmt: skip
        (out / "sites.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        found.density.write_dx(out / "density.dx")
        peak = float(found.density.enrichment.max())
        extra = " and the feature maps" if spots else ""
        print(f"\nwrote sites.json and density.dx{extra} to {out} (peak {peak:.0f}x bulk)")
    return 0


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

    version = int(getattr(args, "martini", 3))
    itp = args.martini_itp or str(parameters(NONBONDED_FOR[version])[0])
    lipids = getattr(args, "lipid_itp", None)
    if lipids is None:
        lipids = [str(p) for p in parameters(*LIPIDS_FOR[version])]
    return itp, lipids


def _martinize(args) -> int:
    import boonza

    cys = args.cys if args.cys in ("auto", "none") else float(args.cys)
    m = boonza.martinize(_load(args.input), args.selection, ss=args.ss, elastic=args.elastic,
                         elastic_fc=args.elastic_fc, elastic_lower=args.elastic_lower,
                         elastic_upper=args.elastic_upper, elastic_decay=args.elastic_decay,
                         elastic_power=args.elastic_power, elastic_min_fc=args.elastic_min_fc,
                         res_min_dist=args.res_min_dist, cys=cys,
                         neutral_termini=args.neutral_termini, scfix=not args.no_scfix,
                         extdih=args.extdih)  # fmt: skip
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


def _composition(text: str) -> dict:
    """``POPC:7,CHOL:3`` -> {"POPC": 7.0, "CHOL": 3.0}; a bare name counts 1."""
    out = {}
    for part in text.split(","):
        name, _, share = part.partition(":")
        out[name.strip()] = float(share) if share else 1.0
    return out


def _bilayer(args) -> int:
    import boonza
    from boonza.martini import bilayer

    protein = None
    if args.protein:
        protein = boonza.martinize(_load(args.protein), args.selection, elastic=args.elastic,
                                   neutral_termini=args.neutral_termini)  # fmt: skip
    size = args.size[0] if len(args.size) == 1 else tuple(args.size)
    itp, lipid_itp = _martini_files(args)
    m = bilayer(lipid_itp, _composition(args.upper),
                _composition(args.lower) if args.lower else None, size=size,
                area_per_lipid=args.apl, water=args.water, salt=args.salt, protein=protein,
                protein_origin=args.opm, protein_shift=args.shift, martini=args.martini,
                seed=args.seed)  # fmt: skip
    top = m.save(args.output, martini_itp=itp)
    lipids = ", ".join(f"{c} {n}" for n, c, _ in m.lipids)
    water = ", ".join(f"{c} {n}" for n, c in m.solvent)
    print(f"wrote {top} ({lipids}; {water}; box {' x '.join(f'{v:.1f}' for v in m.cell.diagonal())}"
          f" A) and cg.gro; it includes {_where(itp)}")  # fmt: skip
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

    q = sub.add_parser("info", help="summarize a structure file")
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

    q = sub.add_parser("martinize", help="Martini 3 beads and topology for proteins, as martinize2")
    q.add_argument("input", help="all-atom structure (hydrogens, if present, set protonation)")
    q.add_argument("output", help="directory for topol.top, molecule_N.itp and cg.gro")
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

    q = sub.add_parser("bilayer", help="a Martini bilayer in water, optionally around a protein")
    q.add_argument("output", help="directory for topol.top, the .itp files and cg.gro")
    q.add_argument("--lipid-itp", nargs="+", default=None,
                   help="Martini lipid and sterol topologies (default: the phospholipids "
                        "and sterols boonza carries)")  # fmt: skip
    q.add_argument("--upper", required=True, help="upper leaflet, e.g. POPC:7,CHOL:3")
    q.add_argument("--lower", help="lower leaflet (default: as the upper)")
    q.add_argument("--size", type=float, nargs="+", default=[100.0], help="x [y] edge (A)")
    q.add_argument("--apl", type=float, default=60.0, help="area per lipid (A^2)")
    q.add_argument("--water", type=float, default=25.0, help="water beyond the lipids (A)")
    q.add_argument("--salt", type=float, default=0.15, help="NaCl (mol/L), after neutralizing")
    q.add_argument("--protein", help="all-atom protein, martinized and embedded")
    q.add_argument("--selection", default="protein", help="its atoms to coarse-grain")
    q.add_argument("--elastic", action="store_true", help="the protein's elastic network")
    q.add_argument("--neutral-termini", action="store_true")
    q.add_argument("--opm", action="store_true",
                   help="the protein's z = 0 is the midplane, as OPM orients it")  # fmt: skip
    q.add_argument("--shift", type=float, default=0.0, help="move the protein along z (A)")
    q.add_argument("--martini", type=int, choices=(2, 3), default=3,
                   help="which Martini the lipids, water and ions are (default: 3)")  # fmt: skip
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--martini-itp", default=None,
                   help="the Martini parameter file topol.top includes "
                        "(default: the Martini 3 file boonza carries)")  # fmt: skip
    q.set_defaults(run=_bilayer)

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

    q = sub.add_parser("sites", help="where a ligand goes, pooled over runs and copies")
    q.add_argument("system", nargs="?", help="structure (topology of --traj)")
    q.add_argument("--traj", nargs="+", default=[], help="one or more trajectories of SYSTEM")
    q.add_argument("--workdir", nargs="+", default=[],
                   help="work directories, each read with its own solvated.dms: runs whose "
                        "ligands differ pool as long as the protein does not")  # fmt: skip
    q.add_argument("--reference", help="structure the runs are superposed on (default: the first)")
    q.add_argument("--ligandsel", default=DEFAULT_LIGAND, help="ligand atoms, every copy")
    q.add_argument("--alignsel", default="protein and name CA", help="atoms the runs align on")
    q.add_argument("--spacing", type=float, default=1.0, help="grid spacing (A)")
    q.add_argument("--enrichment", type=float, default=20.0,
                   help="how many times more visited than bulk a site must be")  # fmt: skip
    q.add_argument("--min-occupancy", type=float, default=0.005,
                   help="share of pooled frames a site must hold")  # fmt: skip
    q.add_argument("--features", action="store_true",
                   help="also map what each pocket asks for -- donor, acceptor, aromatic, "
                        "greasy -- and where")  # fmt: skip
    q.add_argument("--interval-ns", type=float, default=None,
                   help="ns between frames; with it, rates, residence times and dG")  # fmt: skip
    q.add_argument("--temperature", type=float, default=310.0, help="K, for dG")
    q.add_argument("--hysteresis", type=float, default=2.0,
                   help="leave a site at this many times the distance it is entered at. "
                        "One boundary counts every recrossing as a departure")  # fmt: skip
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
