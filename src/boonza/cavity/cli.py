"""boonza cavity: cavities of coarse-grained proteins, found SiteMap-style.

    boonza cavity traj --workdir RUN/md_solute --model martini3 -o OUT [--apo APO.mae]
    boonza cavity structure APO.pdb --model martini3 -o OUT [--holo HOLO.pdb [--holo-ligand SEL]]

traj: the sites of every frame of a coarse-grained run under the model's preset
(volume200 unless --preset), grouped into pockets by place (traj.group)
(traj.consensus), written as OUT/pockets.csv and OUT/view.pml (report.py), with
every frame's sites in OUT/sites.pkl so the grouping can be redone without them, and
where the time went in OUT/timing.json and OUT/timing.txt.
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


def _finding(p: dict) -> dict:
    """What of a preset decides its sites, so that a sites.pkl from before the contact
    energies were dropped (its volume200 still named a probe and a contact test switched
    off, and score weights of 0) is reused."""
    gone = {"probe", "contact", "contact_form"}
    params = {k: v for k, v in p.get("params", {}).items() if k not in gone}
    score = {k: v for k, v in p.get("score", {}).items() if k in ("intercept", "sqrt_n")}
    return {"spacing": p.get("spacing"), "ray_step": p.get("ray_step", 1.0), "params": params,
            "score": score}  # fmt: skip


def _write_timing(out: Path, timing: dict, frames: dict, n_frames: int, n_beads: int) -> None:
    """OUT/timing.json and OUT/timing.txt: wall seconds per stage, and how a frame's site
    finding splits between building its grid and finding its sites (worker seconds, summed
    over frames)."""
    import json

    total = sum(timing.values())
    doc = {"stages_s": {k: round(v, 3) for k, v in timing.items()}, "total_s": round(total, 3),
           "frames": n_frames, "protein_beads": n_beads,
           "site_finding": {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in frames.items()}}  # fmt: skip
    (out / "timing.json").write_text(json.dumps(doc, indent=1) + "\n")
    lines = [f"{k:16s} {v:9.2f} s {100 * v / total:5.1f}%" for k, v in timing.items()]
    lines.append(f"{'total':16s} {total:9.2f} s")
    if frames.get("from_cache"):
        lines.append(f"site finding read from sites.pkl ({n_frames} frames, {n_beads} beads)")
    elif "wall" in frames:
        cpu = {k: frames[f"{k}_cpu"] for k in ("grid", "sites", "lining")}
        lines.append(f"site finding: {n_frames} frames of {n_beads} beads on {frames['workers']} "
                     f"workers, {frames['wall'] / max(n_frames, 1):.3f} s/frame wall, "
                     f"{frames['grid_points_per_frame']:.0f} grid points/frame")  # fmt: skip
        lines += [f"  {k:14s} {v:9.2f} s worker {100 * v / max(sum(cpu.values()), 1e-9):5.1f}%"
                  for k, v in cpu.items()]  # fmt: skip
    (out / "timing.txt").write_text("\n".join(lines) + "\n")


def _same_atoms(view, system) -> bool:
    """``view`` is ``system`` atom for atom: the same names in the same residues, in
    the same order.  A Go run's view.dms copied before the run gathered it like its
    system has the virtual sites after the protein instead, and frames drawn from it
    would put the wrong beads at the right places."""
    import numpy as np

    if view.natoms != system.natoms:
        return False
    names = [[str(n) for n in s.atoms["name"]] for s in (view, system)]
    res = [np.asarray(s.atoms["residue"]) for s in (view, system)]
    return names[0] == names[1] and bool(np.array_equal(*res))


def cmd_traj(args) -> int:
    import numpy as np

    from . import report
    from . import traj as T

    t0 = time.time()
    clock, timing = [t0], {}

    def lap(stage):  # the seconds since the last lap, under ``stage``
        clock.append(time.time())
        timing[stage] = clock[-1] - clock[-2]

    frames = slice(args.first, args.last, args.every)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, ids, coords, rmsd, pids, pcoords = T.load_run(args.workdir, frames=frames,
                                                               probes=True)  # fmt: skip
    lap("load_and_fit")
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
        "preset": _finding(preset(args.model, args.preset)),
    }  # sites made under other presets are not reused
    sites = None
    if cache.exists() and not args.recompute:
        saved = pickle.load(open(cache, "rb"))
        old = saved.get("key") or {}
        if {**old, "preset": _finding(old.get("preset") or {})} == key:
            sites = saved["sites"]
            print(f"sites of every frame from {cache}", flush=True)
    frames_timing = {"from_cache": sites is not None}
    if sites is None:
        sites = T.frame_sites(system, ids, coords, args.model, jobs=args.jobs,
                              preset=args.preset, timing=frames_timing)  # fmt: skip
        with open(cache.with_suffix(".partial"), "wb") as fh:
            pickle.dump({"key": key, "sites": sites, "fit_rmsd": rmsd}, fh)
        cache.with_suffix(".partial").replace(cache)
        print(f"{len(sites)} sites in {len(coords)} frames ({time.time() - t0:.0f} s)", flush=True)
    lap("site_finding")
    big = sum(1 for s in sites if args.max_volume and s.props["n"] > args.max_volume)
    if big:
        print(f"{big} of {len(sites)} frame sites over {args.max_volume:.0f} A^3 left out of "
              "the grouping (--max-volume)", flush=True)  # fmt: skip
    pockets = T.group(sites, args.grouping, args.cutoff, args.similarity, args.cooccur,
                      absorb=args.absorb, max_volume=args.max_volume)  # fmt: skip
    lap("grouping")
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
    lap("hotspots")
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
        if not _same_atoms(view_system, system):
            view_system = None
    lap("apo_and_view")
    holo = holo_ligand = ligand = None
    if args.holo:
        from ..pockets.cli import _holo_on

        ligand, info, holo = _holo_on(args.holo, reference, args.holo_ligand, args.holo_top,
                                      args.holo_fit)  # fmt: skip
        ligand, holo_ligand = np.asarray(ligand, float), info["ligand"]
    lap("holo_placement")
    from ..pockets.cli import structure_format

    inside = {}
    view = report.write(out, args.model, pockets, len(coords), every, system, ids, coords,
                        pids, pcoords, apo, args.top, args.rank, spots, view_system,
                        holo, holo_ligand, ligand,
                        apo_format=structure_format(args.apo or ""),
                        holo_format=structure_format(args.holo or ""),
                        max_median_volume=args.max_median_volume, timing=inside)  # fmt: skip
    lap("report")
    timing["report"] -= inside["ranking"] + inside["ligand"]
    timing.update(ranking=inside["ranking"], ligand_scoring=inside["ligand"])
    _write_timing(out, timing, frames_timing, len(coords), len(ids))
    print(f"{len(pockets)} pockets, ranked by {args.rank} -> {out / 'pockets.csv'}, {view}")
    huge = sum(1 for p in pockets if args.max_median_volume
               and p.summary(len(coords))["median_volume"] > args.max_median_volume)  # fmt: skip
    if huge:
        print(f"{huge} pockets of median volume over {args.max_median_volume:.0f} A^3 ranked "
              "after the rest (--max-median-volume)")  # fmt: skip
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
    from .presets import DEFAULT
    from .traj import (
        COOCCUR,
        CUTOFF,
        GROUPING,
        GROUPINGS,
        MAX_MEDIAN_VOLUME,
        MAX_VOLUME,
        RANK,
        RANKINGS,
        SIMILARITY,
    )

    ap = argparse.ArgumentParser(
        prog="boonza cavity",
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
    t.add_argument("--grouping", default=GROUPING, choices=GROUPINGS,
                   help="how frame sites become pockets: by their centres (centroid, the "
                        "default) or by the residues lining them (consensus)")  # fmt: skip
    t.add_argument(
        "--cutoff",
        type=float,
        default=CUTOFF,
        help="centroid grouping: a site joins a pocket whose centre is this close (A)",
    )
    t.add_argument(
        "--similarity",
        type=float,
        default=SIMILARITY,
        help="consensus grouping: Jaccard similarity of lining residues to join a pocket",
    )
    t.add_argument(
        "--cooccur",
        type=float,
        default=COOCCUR,
        help="consensus grouping: pockets open together in at most this share of frames may merge",
    )
    t.add_argument("--max-volume", type=float, default=MAX_VOLUME, metavar="A3",
                   help=f"leave out frame sites larger than this before grouping, so one "
                        f"ballooned frame cannot set a pocket's p_max (default {MAX_VOLUME:.0f}; "
                        "0 keeps all). Regroups from sites.pkl without recomputing")  # fmt: skip
    t.add_argument("--max-median-volume", type=float, default=MAX_MEDIAN_VOLUME, metavar="A3",
                   help=f"rank pockets whose median site is larger than this after all the "
                        f"rest, under every ranking, so a groove that is huge in every frame "
                        f"does not come first (default {MAX_MEDIAN_VOLUME:.0f}; 0 ranks by "
                        "value alone). Reorders from sites.pkl without recomputing")  # fmt: skip
    t.add_argument(
        "--no-absorb",
        dest="absorb",
        action="store_false",
        help="consensus grouping: keep fragments (rarely open or small pockets) as pockets "
        "of their own instead of joining them to a neighbouring pocket",
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
        default=RANK,
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
    t.add_argument("--preset", metavar="NAME|FILE", default=DEFAULT,
                   help="presets to use: a named set kept in boonza or a presets file "
                        f"(default: {DEFAULT})")  # fmt: skip
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
    s.add_argument("--preset", metavar="NAME|FILE", default=DEFAULT,
                   help="presets to use: a named set kept in boonza or a presets file "
                        f"(default: {DEFAULT})")  # fmt: skip
    args = ap.parse_args(argv)
    try:
        return {"traj": cmd_traj, "structure": cmd_structure}[args.cmd](args)
    except (ValueError, FileNotFoundError) as e:
        print(f"boonza cavity {args.cmd}: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
