"""``boonza ctmd``: run the biased replicas, and rank what came out of them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .meta import Settings


def _parser() -> argparse.ArgumentParser:
    cfg = Settings()
    p = argparse.ArgumentParser(
        prog="boonza ctmd",
        description="Rank docked poses by the reversible work c(t) it takes to break them "
        "(CTMD, J. Chem. Inf. Model. 2026, 66, 11929), biasing the pocket-ligand "
        "drmsd with well-tempered metadynamics.",  # fmt: skip
        epilog="examples:\n"
        "  boonza ctmd run complex.dms -r docked.mae -w out/ZINC123\n"
        "  boonza ctmd run complex.dms -w out/lig --replicas 5 --ns 5\n"
        "  boonza ctmd score out/* --hits hits.csv --fraction 0.3\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("run", help="one pose: the unbiased filter, then the biased replicas")
    q.add_argument("system", help="the prepared, equilibrated complex (boonza md's "
                   "equilibrated.dms); parameterized here if it has no force "
                   "field")  # fmt: skip
    q.add_argument("-w", "--workdir", required=True, help="where the COLVAR files go, one "
                   "directory per pose (its name is the pose's)")  # fmt: skip
    q.add_argument("-r", "--reference", help="the structure the pose is measured against, "
                   "which is the docked pose: drift during the unbiased filter then counts "
                   "against the ligand (default: the system's own coordinates)")  # fmt: skip
    q.add_argument("--ligand", help="the ligand's atoms (default: what is neither polymer, "
                   "water nor ion, heavy atoms only)")  # fmt: skip
    q.add_argument("--pocket", default=None, help="the atoms the ligand is measured against "
                   "(default: protein alpha carbons within 5 A of it)")  # fmt: skip
    q.add_argument("--pocket-cutoff", type=float, default=None, metavar="A",
                   help="how near the ligand a pocket atom must be (default 5)")  # fmt: skip
    q.add_argument("--replicas", type=int, default=cfg.replicas,
                   help=f"biased trajectories (default {cfg.replicas}; the published protocol's "
                        "results converge by 5)")  # fmt: skip
    q.add_argument("--ns", type=float, default=cfg.nanoseconds,
                   help=f"ns per replica (default {cfg.nanoseconds:g})")  # fmt: skip
    q.add_argument("--filter-ns", type=float, default=cfg.filter_nanoseconds,
                   help=f"ns of unbiased MD first; a ligand that leaves the site there scores "
                        f"0 and is not biased at all (default {cfg.filter_nanoseconds:g}, "
                        "0 to skip)")  # fmt: skip
    q.add_argument("--height", type=float, default=cfg.height, metavar="KJ",
                   help=f"hill height, kJ/mol (default {cfg.height:g})")  # fmt: skip
    q.add_argument("--sigma", type=float, default=cfg.sigma, metavar="NM",
                   help=f"hill width along the coordinate, nm (default {cfg.sigma:g}, the "
                        "published value for an aligned RMSD)")  # fmt: skip
    q.add_argument("--pace", type=float, default=cfg.pace, metavar="PS",
                   help=f"ps between hills (default {cfg.pace:g})")  # fmt: skip
    q.add_argument("--bias-factor", type=float, default=cfg.bias_factor,
                   help=f"well-tempered bias factor (default {cfg.bias_factor:g})")  # fmt: skip
    q.add_argument("--grid", type=float, nargs=2, default=list(cfg.grid), metavar=("MIN", "MAX"),
                   help=f"the coordinate's range in nm (default {cfg.grid[0]:g} "
                        f"{cfg.grid[1]:g})")  # fmt: skip
    q.add_argument("--bins", type=int, default=cfg.bins,
                   help=f"grid points across that range (default {cfg.bins})")  # fmt: skip
    q.add_argument("--cutoff", type=float, default=cfg.cutoff, metavar="NM",
                   help=f"past this the ligand has left (default {cfg.cutoff:g})")  # fmt: skip
    q.add_argument("--commit", type=float, default=cfg.commit, metavar="PS",
                   help=f"how long it must stay out to count as gone (default "
                        f"{cfg.commit:g})")  # fmt: skip
    q.add_argument("--stride", type=float, default=cfg.stride, metavar="PS",
                   help=f"ps between COLVAR lines (default {cfg.stride:g})")  # fmt: skip
    q.add_argument("--temperature", type=float, default=cfg.temperature, metavar="K",
                   help=f"K (default {cfg.temperature:g})")  # fmt: skip
    q.add_argument("--timestep", type=float, default=cfg.timestep, metavar="PS",
                   help=f"ps (default {cfg.timestep:g})")  # fmt: skip
    q.add_argument("--no-barostat", action="store_true", help="run at constant volume")
    q.add_argument("--seed", type=int, default=cfg.seed,
                   help="random seed; each replica takes this plus its number (default 0, "
                        "which is a new run every time)")  # fmt: skip
    q.add_argument("--platform", default=None, help="OpenMM platform (default: the fastest)")
    q.add_argument("-f", "--forcefield", action="append", default=None, metavar="NAME",
                   help="force field for a structure without one; repeat for several")  # fmt: skip

    q = sub.add_parser("score", help="rank poses from their COLVAR files")
    q.add_argument("folders", nargs="+", help="one directory per pose")
    q.add_argument("--hits", help="a CSV of pose name,1 or 0 saying which are binders; with it "
                   "the enrichment factor and BEDROC are reported too")  # fmt: skip
    q.add_argument("--replicas", type=int, default=cfg.replicas,
                   help=f"replicas a score is the lowest of; with more in a folder, draws of "
                        f"this many are bootstrapped (default {cfg.replicas})")  # fmt: skip
    q.add_argument("--cutoff", type=float, default=cfg.cutoff, metavar="NM",
                   help=f"past this the ligand has left (default {cfg.cutoff:g})")  # fmt: skip
    q.add_argument("--commit", type=float, default=cfg.commit, metavar="PS",
                   help=f"how long it must stay out (default {cfg.commit:g})")  # fmt: skip
    q.add_argument("--until", type=float, default=None, metavar="PS",
                   help="read only this much of each replica, to ask what a shorter run "
                        "would have said")  # fmt: skip
    q.add_argument(
        "--fraction",
        type=float,
        default=0.3,
        help="the top share the enrichment factor is taken over (default 0.3)",
    )
    q.add_argument(
        "--alpha", type=float, default=20.0, help="BEDROC's weighting of the top (default 20)"
    )
    q.add_argument(
        "--bootstrap",
        type=int,
        default=250,
        help="draws when there are more replicas than --replicas (default 250)",
    )
    q.add_argument("--seed", type=int, default=0, help="seed for those draws (default 0)")
    q.add_argument("-o", "--output", help="write the ranking to this CSV as well")
    return p


def _settings(args) -> Settings:
    cv = {}
    if args.ligand:
        cv["ligand"] = args.ligand
    if args.pocket:
        cv["pocket"] = args.pocket
    if args.pocket_cutoff is not None:
        cv["cutoff"] = args.pocket_cutoff
    return Settings(
        sigma=args.sigma, height=args.height, bias_factor=args.bias_factor, pace=args.pace,
        grid=(args.grid[0], args.grid[1]), bins=args.bins, temperature=args.temperature,
        timestep=args.timestep, replicas=args.replicas, nanoseconds=args.ns,
        filter_nanoseconds=args.filter_ns, stride=args.stride, cutoff=args.cutoff,
        commit=args.commit, seed=args.seed, pressure=None if args.no_barostat else 1.0,
        cv=cv,
    )  # fmt: skip


def _run(args) -> int:
    import boonza

    from .meta import run

    system = boonza.load(args.system)
    reference = boonza.load(args.reference) if args.reference else None
    pose = run(system, args.workdir, reference, settings=_settings(args),
               forcefields=args.forcefield, platform=args.platform, log=print)  # fmt: skip
    print(pose)
    return 0


#: what a hit list may call a binder, and what it may call anything else
YES = {"1", "true", "yes", "y", "binder", "active", "hit"}
NO = {"0", "false", "no", "n", "non-binder", "nonbinder", "decoy", "inactive"}


def _read_hits(path) -> dict:
    """``{pose: 1 or 0}`` from a list of which ligands bind.

    One ligand a line, its name and then whether it binds, separated by commas, tabs
    or spaces: ``ZINC000547933290,1`` or, as the CTMD authors' ``hit_data_mol2.spc``
    has it, ``ZINC000547933290.mol2 True -1``.  A trailing ``.mol2`` goes, so their
    file names the same poses their directories do.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.lower().startswith(("name,", "ligand,", "pose,")):
            continue
        parts = [x for x in line.replace("\t", " ").replace(",", " ").split() if x]
        if len(parts) < 2:
            raise ValueError(f"{path}: cannot read a name and a 0 or 1 from {line!r}")
        name = parts[0][:-5] if parts[0].lower().endswith(".mol2") else parts[0]
        says = parts[1].lower()
        if says not in YES and says not in NO:
            raise ValueError(f"{path}: {parts[1]!r} is not a yes or a no, for {name}")
        out[name] = int(says in YES)
    return out


def _score(args) -> int:
    import numpy as np

    from .score import bedroc, enrichment_factor, rank, score_pose

    poses, failed = [], []
    for folder in args.folders:
        try:
            poses.append(score_pose(folder, replicas=args.replicas, cutoff=args.cutoff,
                                    commit=args.commit, bootstrap=args.bootstrap,
                                    until=args.until, seed=args.seed))  # fmt: skip
        except ValueError as e:
            # a run that never wrote anything is not a pose that scored badly: it is
            # not a pose at all, and ranking it either way would be a guess
            failed.append((Path(folder).name, str(e)))
    if not poses:
        raise ValueError("none of these folders has a replica with any data in it")
    ordered = rank(poses)
    width = max(len(p.name) for p in ordered)
    print(f"{'pose':<{width}}  {'c(t)':>8}  {'+/-':>6}  {'out at':>9}  replicas")
    for p in ordered:
        print(f"{p.name:<{width}}  {p.score:8.1f}  {p.error:6.1f}  {p.exit_time:7.0f} ps  "
              f"{len(p.replicas)}{' (filtered)' if p.filtered else ''}")  # fmt: skip
    if args.output:
        with open(args.output, "w") as f:
            f.write("pose,score,error,exit_ps,replicas,filtered\n")
            for p in ordered:
                f.write(f"{p.name},{p.score:.4f},{p.error:.4f},{p.exit_time:.1f},"
                        f"{len(p.replicas)},{int(p.filtered)}\n")  # fmt: skip
        print(f"wrote {args.output}")
    if failed:
        print(f"\nnot scored, and left out of the ranking: "
              f"{', '.join(name for name, _ in failed)}")  # fmt: skip
    if args.hits:
        hits = _read_hits(args.hits)
        missing = [p.name for p in ordered if p.name not in hits]
        if missing:
            raise ValueError(f"{args.hits} says nothing about {', '.join(missing[:5])}")
        labels = np.array([hits[p.name] for p in ordered])
        scores = np.array([p.score for p in ordered])
        ef = enrichment_factor(labels, scores, args.fraction)
        print(f"\n{int(labels.sum())} binders of {len(labels)} ligands "
              f"({100 * labels.mean():.1f}% by chance)")  # fmt: skip
        print(f"EF{round(100 * args.fraction)} = {ef:.2f}  "
              f"(1.0 is chance)")  # fmt: skip
        if 0 < labels.sum() < len(labels):
            print(f"BEDROC(alpha={args.alpha:g}) = {bedroc(labels, scores, args.alpha):.3f}")
        else:
            print("BEDROC needs both binders and non-binders among the poses scored")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args) if args.command == "run" else _score(args)
    except (ValueError, FileNotFoundError, NotADirectoryError, RuntimeError) as e:
        print(f"boonza ctmd: error: {e}", file=sys.stderr)
        return 1
