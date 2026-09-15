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
        patch = gaff.gaff2_patch(system, ffs, charges=charges,
                                 protein_extent=args.protein_extent, draw=args.draw)  # fmt: skip
        if args.save_patch:
            viparr.write_forcefield(patch, args.save_patch)
        names = ", ".join(t.name for t in patch.templates) or "none needed"
        print(f"GAFF2 templates: {names}")
        if ffs:
            ffs[0] = viparr.merge_forcefields(ffs[0], patch)
        else:
            ffs = [patch]
    elif args.charge or args.save_patch or args.draw:
        raise SystemExit("--charge, --save-patch and --draw go with --gaff2")
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
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "md":  # its own parser: settings files, restarts
        from .md.cli import main as md_main

        return md_main(argv[1:])
    args = _parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
