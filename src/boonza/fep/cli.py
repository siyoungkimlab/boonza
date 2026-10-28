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
energy (a leg in water, one in vacuum), and the charge cannot change.
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
    from .prep import CHARGE_PRODUCTION_NS, CHARGE_WINDOWS, prep

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
    g.add_argument("--charge-production-ns", dest="charge_production_ns", type=float,
                   help="production of an edge that changes the net charge, per window "
                        "(default: 20)")  # fmt: skip
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
             charge_production_ns=x.get("charge_production_ns", CHARGE_PRODUCTION_NS))  # fmt: skip
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
    a = p.parse_args(argv)
    try:
        results(a.workdir)
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
