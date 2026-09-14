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
    boonza build --sequence ACDEFGHIK --conformation helix -o peptide.pdb

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

    found = diff(_load(args.a), _load(args.b), rtol=args.rtol, positions=not args.no_positions)
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boonza", description="Molecular system tools.")
    sub = p.add_subparsers(dest="command", required=True)

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
    args = _parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
