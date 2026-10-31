"""boonza fep prep | run | result: relative binding free energies.

    boonza fep prep receptor.pdb --ligands posed.sdf [boonza md's settings]
    boonza fep run boonza_fep/edges/A~B/complex --repeat 1
    boonza fep result boonza_fep

prep takes boonza md's settings (force fields, salt, padding, cutoff,
temperature, the secondary-structure dihedral restraints, ...) and its own:
the ligands, repeats and lambda windows.  Its defaults are openfe's where they
differ from md's: 1 ns equilibration, 5 ns production, hydrogen mass
repartitioning with a 4 fs step.  An edge that changes the net charge turns a
water far from the solute into Na+ or Cl- alongside, and runs openfe's longer
protocol (22 windows, 20 ns); the network takes one only to join ligands of
different charge.  Without a receptor, each edge is a relative hydration free
energy (a leg in water, one in vacuum), and the charge cannot change; with
--phase octanol or wet-octanol, a relative transfer free energy from water to
1-octanol (a leg in water, one in an octanol box built once in the work
directory), whose change of log P result reports beside it.  With a receptor,
--rem first relaxes the receptor around every ligand's pose, the ligands held
as posed: a soft minimization (REM), then one with the force field (EM), so
every edge still meets one and the same receptor (boonza.fep.relief).

    boonza fep prep receptor.pdb --ligands more.sdf --workdir boonza_fep --extend

adds ligands to a prep: the SDF holds its ligands as they were and the new
ones; only the edges that join the new ones are built (commands_new.txt), and
every edge already run is reused.  result writes the ligands' SDF back with
the results as fields (results_ligands.sdf), and beside each correlation plot
its data and plot_correlation.py, which redraws it without boonza.
"""

from __future__ import annotations

import argparse
import sys

#: openfe's defaults, where boonza md's differ
FEP_DEFAULTS = {"equilibration_ns": 1.0, "production_ns": 5.0, "hmr": True,
                "integration_fs": 4.0, "workdir": "boonza_fep"}  # fmt: skip


def _prep(argv) -> int:
    from .. import viparr
    from ..md.config import build_parser, parse_arguments
    from .network import MAX_CHANGE, MAX_DEGREE
    from .prep import CHARGE_PRODUCTION_NS, CHARGE_WINDOWS, LARGE_CHANGE_ATOMS, PHASES, prep

    parser = build_parser("boonza fep prep")
    g = parser.add_argument_group("fep")
    g.add_argument("--ligands", help="the ligands, posed in the receptor's frame (SDF, with "
                   "hydrogens and formal charges)")  # fmt: skip
    g.add_argument("--cofactor-sdf", dest="cofactor_sdf", nargs="+", metavar="SDF",
                   help="cofactors to add to the receptor, posed in its frame (SDF, with "
                        "hydrogens and formal charges): GAFF2 like the ligands")  # fmt: skip
    g.add_argument("--repeats", type=int, help="independent runs of each leg (default: 3)")
    g.add_argument("--windows", type=int, help="lambda windows, each a replica (default: 11)")
    g.add_argument("--iteration-ps", dest="iteration_ps", type=float,
                   help="ps of dynamics between replica exchanges (default: 2.5)")  # fmt: skip
    g.add_argument("--charge-windows", dest="charge_windows", type=int,
                   help="lambda windows of an edge that changes the net charge "
                        "(default: 22)")  # fmt: skip
    g.add_argument("--large-change-atoms", dest="large_change_atoms", type=int,
                   help="an edge where either ligand has this many heavy atoms of its own "
                        "(not mapped) runs 23 windows, dense from lambda 0.3 to 0.7 where their "
                        "Lennard-Jones is switched, instead of --windows; 0 never "
                        f"(default: {LARGE_CHANGE_ATOMS})")  # fmt: skip
    g.add_argument("--complexes", metavar="DIR",
                   help="a co-complex per ligand (an MD snapshot, a crystal structure), "
                        "named after it (NAME.mae, .dms, .cif, .pdb), instead of one "
                        "receptor: each ligand's pose comes from its complex, and an edge's "
                        "windows from lambda 0.5 on start from B's receptor; water and ions "
                        "are left out, and every protein must be the same atom for "
                        "atom")  # fmt: skip
    g.add_argument("--network", choices=("3d", "2d"),
                   help="score ligand pairs on their atom mapping in the poses (3d), or on "
                        "their graphs' maximum common substructure, leaving the poses out of "
                        "the choice (2d); either way an edge is built from its 3D mapping "
                        "(default: 3d, or the first prep's with --extend)")  # fmt: skip
    g.add_argument("--max-change", dest="max_change", type=int,
                   help="an edge that changes more heavy atoms is taken only where no route "
                        "through smaller ones exists; 0 no limit "
                        f"(default: {MAX_CHANGE})")  # fmt: skip
    g.add_argument("--max-degree", dest="max_degree", type=int,
                   help="edges a ligand may have, so no ligand becomes a hub, unless it can "
                        f"be joined no other way; 0 no limit (default: {MAX_DEGREE})")  # fmt: skip
    g.add_argument("--reference", metavar="NAME",
                   help="the ligand with a known (crystal) pose: its edges preferred and "
                        "free of --max-degree, so the network grows out from it")  # fmt: skip
    g.add_argument("--charge-production-ns", dest="charge_production_ns", type=float,
                   help="production of an edge that changes the net charge, per window "
                        "(default: 20)")  # fmt: skip
    g.add_argument("--phase", choices=PHASES,
                   help="without a receptor, the leg set against the water leg: vacuum (a "
                        "hydration free energy), or 1-octanol, dry or water-saturated as a "
                        "shake-flask log P is measured (a water-to-octanol transfer free "
                        "energy); neutral ligands only (default: vacuum)")  # fmt: skip
    g.add_argument("--octanol-box", dest="octanol_box", metavar="DMS",
                   help="an octanol box built before (WORKDIR/octanol/box.dms of another "
                        "prep), instead of packing and equilibrating one")  # fmt: skip
    g.add_argument("--extend", action="store_true",
                   help="add ligands to the prep already in --workdir: --ligands holds its "
                        "ligands (as they were) and the new ones; only edges that join the new "
                        "ones are built, and every edge already run is reused")  # fmt: skip
    args = parse_arguments(argv, parser)
    for k, v in FEP_DEFAULTS.items():
        if k not in args.specified and not (k == "integration_fs" and "hmr" in args.specified
                                            and not args.hmr):  # fmt: skip
            setattr(args, k, v)
    x = args.extra
    try:
        if args.model != "aa":
            raise ValueError("boonza fep is all-atom: --model aa")
        if not x.get("ligands"):
            raise ValueError("give the posed ligands with --ligands")
        prep(args, x["ligands"], repeats=x.get("repeats", 3), windows=x.get("windows", 11),
             iteration_ps=x.get("iteration_ps", 2.5),
             cofactors=x.get("cofactor_sdf") or (),
             charge_windows=x.get("charge_windows", CHARGE_WINDOWS),
             charge_production_ns=x.get("charge_production_ns", CHARGE_PRODUCTION_NS),
             phase=x.get("phase") or "vacuum", octanol_box=x.get("octanol_box"),
             extend=bool(x.get("extend")),
             large_change_atoms=x.get("large_change_atoms", LARGE_CHANGE_ATOMS),
             network_method=x.get("network"), max_change=x.get("max_change"),
             max_degree=x.get("max_degree"),
             reference=x.get("reference"), complexes=x.get("complexes"))  # fmt: skip
    except (ValueError, FileNotFoundError, RuntimeError, viparr.ViparrError) as e:
        print(f"boonza fep prep: error: {e}", file=sys.stderr)
        return 1
    return 0


def _run(argv) -> int:
    from .run import run_leg

    p = argparse.ArgumentParser(prog="boonza fep run",
                                description="run one leg of one edge (restartable)")  # fmt: skip
    p.add_argument("leg", help="a leg's directory (edges/A~B/complex or .../solvent)")
    p.add_argument("--repeat", type=int, default=1, help="which repeat (default: 1)")
    p.add_argument("--platform", help="OpenMM platform (default: the fastest)")
    a = p.parse_args(argv)
    try:
        run_leg(a.leg, a.repeat, a.platform)
    except (ValueError, FileNotFoundError) as e:
        print(f"boonza fep run: error: {e}", file=sys.stderr)
        return 1
    return 0


def _result(argv) -> int:
    from .result import results

    p = argparse.ArgumentParser(prog="boonza fep result",
                                description="every edge's ΔΔG and every ligand's ΔG")  # fmt: skip
    p.add_argument("workdir", nargs="?", default="boonza_fep")
    p.add_argument("--experimental", metavar="SDF",
                   help="measured values to compare with, by ligand name: one of dG "
                        "(kcal/mol), KD_nM, KD_uM or pIC50 per record (default: those the "
                        "ligands' SDF gave prep)")  # fmt: skip
    p.add_argument("--ligands", metavar="SDF",
                   help="the SDF to write back with the results as fields, for a work "
                        "directory prepared before prep kept one (default: "
                        "WORKDIR/ligands.sdf)")  # fmt: skip
    a = p.parse_args(argv)
    try:
        results(a.workdir, experimental=a.experimental, ligands=a.ligands)
    except (ValueError, FileNotFoundError) as e:
        print(f"boonza fep result: error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {"prep": _prep, "run": _run, "result": _result}
    if not argv or argv[0] not in commands:
        print("usage: boonza fep {prep,run,result} ...\n\n" + __doc__, file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])
