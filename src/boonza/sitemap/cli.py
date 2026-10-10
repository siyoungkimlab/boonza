"""boonza sitemap: SiteMap-style sites of coarse-grained proteins.

    boonza sitemap traj --workdir RUN/md_solute --model martini3 -o OUT [--apo APO.mae]
    boonza sitemap structure APO.pdb --model martini3 -o OUT [--holo HOLO.pdb [--holo-ligand SEL]]

traj: the sites of every frame of a coarse-grained run under the model's preset
(dynamic200 unless --preset), grouped into pockets by the residues lining them
(traj.consensus), written as OUT/pockets.csv and OUT/view.pml (report.py), with
every frame's sites in OUT/sites.pkl so the grouping can be redone without them.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

MODELS = ("martini2", "martini3", "sirah")


def cmd_traj(args) -> int:
    import numpy as np

    from . import report
    from . import traj as T

    t0 = time.time()
    frames = slice(args.first, args.last, args.every)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, ids, coords, rmsd, pids, pcoords = T.load_run(args.workdir, frames=frames,
                                                               probes=True)  # fmt: skip
    every = args.every
    print(f"{len(coords)} frames (every {every}th) of {len(ids)} protein beads, made whole and "
          f"fitted on the backbone; backbone RMSD to the first frame up to {rmsd.max():.2f} A; "
          f"{len(pids)} probe beads kept for the view only", flush=True)  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "sites.pkl"
    from .presets import preset

    key = {
        "workdir": str(Path(args.workdir).resolve()),
        "model": args.model,
        "frames": (args.first, args.last, every),
        "preset": preset(args.model, args.preset),
    }  # sites made under other presets are not reused
    sites = None
    if cache.exists() and not args.recompute:
        saved = pickle.load(open(cache, "rb"))
        if saved.get("key") == key:
            sites = saved["sites"]
            print(f"sites of every frame from {cache}", flush=True)
    if sites is None:
        sites = T.frame_sites(system, ids, coords, args.model, jobs=args.jobs,
                              preset=args.preset)  # fmt: skip
        with open(cache.with_suffix(".partial"), "wb") as fh:
            pickle.dump({"key": key, "sites": sites, "fit_rmsd": rmsd}, fh)
        cache.with_suffix(".partial").replace(cache)
        print(f"{len(sites)} sites in {len(coords)} frames ({time.time() - t0:.0f} s)", flush=True)
    pockets = T.consensus(sites, args.similarity, args.cooccur, absorb=args.absorb)
    from .pharm import hotspots

    spots, all_frames = [], pcoords
    if len(pids):
        if every != 1:  # the probes of every frame: cheap to read, and five times the counts
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                all_frames = T.load_run(args.workdir, frames=slice(args.first, args.last),
                                        probes=True)[5]  # fmt: skip
        spots = hotspots(system, pids, all_frames, min_probes=args.min_probes)
    print(f"{len(spots)} pharmacophore hotspots of at least {args.min_probes} distinct probe "
          f"molecules over {len(all_frames)} frames (boonza.pharmacophore.hotspots)",
          flush=True)  # fmt: skip
    reference = system.clone(ids)  # the first fitted frame, which every frame is fitted on
    reference.positions = np.asarray(coords[0], float)
    apo = None
    if args.apo:
        from ..align import superpose
        from ..io import load

        bb = "BB" if "BB" in set(np.asarray(reference.atoms["name"]).tolist()) else "GC"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            apo = load(args.apo)
        fit = superpose(apo, reference, sel="protein and name CA", ref_sel=f"name {bb}")
        print(f"apo superposed on the first frame: {fit.n_used}/{fit.n_matched} alpha carbons, "
              f"RMSD {fit.rmsd:.2f} A", flush=True)  # fmt: skip
    view_system = None  # the run's system without its bands, if it wrote one
    view_dms = Path(args.workdir) / "view.dms"
    if view_dms.exists():
        from ..io import load

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            view_system = load(str(view_dms))
        if view_system.natoms != system.natoms:
            view_system = None
    holo = holo_ligand = ligand = None
    if args.holo:
        from ..pockets.cli import _holo_on

        ligand, info, holo = _holo_on(args.holo, reference, args.holo_ligand, args.holo_top,
                                      args.holo_fit)  # fmt: skip
        ligand, holo_ligand = np.asarray(ligand, float), info["ligand"]
    view = report.write(out, args.model, pockets, len(coords), every, system, ids, coords,
                        pids, pcoords, apo, args.top, args.rank, spots, view_system,
                        holo, holo_ligand, ligand)  # fmt: skip
    print(f"{len(pockets)} pockets, ranked by {args.rank} -> {out / 'pockets.csv'}, {view}")
    import csv

    rows = list(csv.DictReader(open(out / "pockets.csv")))
    print("the first five under each ranking (pocket numbers as in the view):")
    for name in T.RANKINGS:
        top5 = sorted(rows, key=lambda r: int(r[f"rank_{name}"]))[:5]
        print(f"  {name:12s} {' '.join(r['rank'] for r in top5)}")
    return 0


def cmd_structure(args) -> int:
    from .structure import run

    view = run(args.input, args.model, args.out, args.selection, args.ligand, args.top,
               args.holo, args.holo_ligand, args.holo_top, args.holo_fit,
               args.preset)  # fmt: skip
    print(f"-> {Path(args.out) / 'pockets.csv'}, {view}")
    return 0


def main(argv=None) -> int:
    from .presets import STRUCTURE, TRAJ
    from .traj import COOCCUR, RANKINGS, SIMILARITY

    ap = argparse.ArgumentParser(
        prog="boonza sitemap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("traj", help="pockets of a coarse-grained trajectory")
    t.add_argument("--workdir", required=True, help="the run's md_solute directory "
                   "(solvated.dms and trajectory.dcd)")  # fmt: skip
    t.add_argument("--model", required=True, choices=MODELS)
    t.add_argument("-o", "--out", required=True)
    t.add_argument("--apo", help="an all-atom apo structure to draw, superposed on the run")
    t.add_argument("--holo", help="a structure of the same protein with a ligand, superposed on "
                   "the run: each pocket is scored against that ligand in its best frame (PPc, "
                   "MOc, LVC, PVN) and by the share of its frames that are PPc-right")  # fmt: skip
    t.add_argument("--holo-ligand", default=None, metavar="SEL",
                   help="the ligand in --holo (default: its largest residue that is neither "
                        "protein, nucleic, solvent nor a buffer salt)")  # fmt: skip
    t.add_argument("--holo-top", help="topology of a coarse-grained --holo")
    t.add_argument("--holo-fit", choices=["whole", "chain"], default="whole",
                   help="how the holo ligand is carried onto the run: superposing the whole holo "
                        "protein (default) or the chain the ligand sits in")  # fmt: skip
    t.add_argument("--every", type=int, default=1, help="analyse every N-th frame")
    t.add_argument("--first", type=int, default=None, help="first frame index")
    t.add_argument("--last", type=int, default=None, help="stop before this frame index")
    t.add_argument(
        "--similarity",
        type=float,
        default=SIMILARITY,
        help="Jaccard similarity of lining residues to join a pocket",
    )
    t.add_argument(
        "--cooccur",
        type=float,
        default=COOCCUR,
        help="pockets open together in at most this share of frames may merge",
    )
    t.add_argument(
        "--no-absorb",
        dest="absorb",
        action="store_false",
        help="keep fragments (rarely open or small pockets) as pockets of their own "
        "instead of joining them to a neighbouring pocket",
    )
    t.add_argument(
        "--min-probes",
        type=int,
        default=3,
        help="distinct probe molecules a pharmacophore hotspot needs",
    )
    t.add_argument("--top", type=int, default=10, help="pockets drawn in the view")
    t.add_argument(
        "--rank",
        default="p_max",
        choices=RANKINGS,
        help="how pockets are ordered (all are in pockets.csv)",
    )
    t.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="worker processes (default: every core but two)",
    )
    t.add_argument("--recompute", action="store_true", help="ignore OUT/sites.pkl")
    t.add_argument("--preset", metavar="NAME|FILE", default=TRAJ,
                   help="presets to use: a named set kept in boonza or a presets file "
                        f"(default: {TRAJ}, tuned on MD frames; {STRUCTURE}: on crystal "
                        "structures)")  # fmt: skip
    s = sub.add_parser("structure", help="sites of one structure")
    s.add_argument("input", help="an all-atom structure (mapped to beads), or a coarse-grained "
                   "one that carries its bead types")  # fmt: skip
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--selection", help="what of the structure is the protein (default: protein "
                   "but the probe chain, and but --ligand)")  # fmt: skip
    s.add_argument("--ligand", metavar="SEL", help="a ligand in the structure itself: left out "
                   "of the protein, and each site scored against it (PPc, MOc, LVC)")  # fmt: skip
    s.add_argument("--holo", help="a structure of the same protein with a ligand: each site is "
                   "scored against that ligand (PPc, MOc, LVC)")  # fmt: skip
    s.add_argument("--holo-ligand", default=None, metavar="SEL",
                   help="the ligand in --holo (default: its largest residue that is neither "
                        "protein, nucleic, solvent nor a buffer salt)")  # fmt: skip
    s.add_argument("--holo-top", help="topology of a coarse-grained --holo")
    s.add_argument("--holo-fit", choices=["whole", "chain"], default="whole",
                   help="how the holo ligand is carried onto the structure: superposing the "
                        "whole holo protein (default) or the chain the ligand sits in")  # fmt: skip
    s.add_argument("--top", type=int, default=10, help="sites drawn in the view")
    s.add_argument("--preset", metavar="NAME|FILE", default=STRUCTURE,
                   help="presets to use: a named set kept in boonza or a presets file "
                        f"(default: {STRUCTURE}, tuned on crystal structures; {TRAJ}: on MD "
                        "frames)")  # fmt: skip
    args = ap.parse_args(argv)
    try:
        return {"traj": cmd_traj, "structure": cmd_structure}[args.cmd](args)
    except (ValueError, FileNotFoundError) as e:
        print(f"boonza sitemap {args.cmd}: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
