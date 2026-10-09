"""``boonza pockets``: fpocket on coarse-grained proteins, one structure or a trajectory.

    boonza pockets run protein.mae --model martini3 -o out/
    boonza pockets run cg.gro --top topol.top --model martini3 -o out/ --apo protein.mae
    boonza pockets traj --workdir run/md --model sirah -o out/ --apo apo.mae --holo holo.mae
    boonza pockets traj cg.gro --top topol.top --traj md.xtc --model martini3 -o out/
    boonza pockets flags --model sirah

Flags after ``--`` go to fpocket (and mdpocket) and override the preset's.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .beads import MODELS, guess_model, write_fpocket_pdb
from .prepare import NOT_PROBES, coarse_grain, protein_ids, searchable
from .run import find_fpocket, merge_flags, preset

#: what is scored against a holo ligand, per pocket, after its own columns
LIGAND_COLUMNS = ("center_to_nearest_ligand_atom", "center_to_ligand_centroid",
                  "ligand_atoms_within_3A", "spheres_within_3A", "PPc", "MOc")  # fmt: skip
CORE_LIGAND_COLUMNS = ("share_of_open_frames_PPc", "core_center_to_nearest_ligand_atom",
                       "PPc_core", "core_ligand_volume_covered", "core_volume_near_ligand",
                       "core_volume_in_ligand", "core_DVO")  # fmt: skip


def _load(path, top=None):
    from ..io import load

    if top is not None:
        from ..io import load_top

        return load_top(top, coordinates=path, structure_only=True)
    return load(str(path))


def structure_format(path) -> str:
    """The format a structure was given in, to write it back in the same one: MAE stays
    MAE (bond orders and all), CIF stays CIF; DMS is written as MAE, which PyMOL opens;
    PDB for formats boonza cannot write."""
    from ..io import _WRITERS

    suffixes = [x.lower() for x in Path(str(path)).suffixes]
    if suffixes and suffixes[-1] in (".gz", ".bz2"):
        suffixes = suffixes[:-1]
    last = suffixes[-1] if suffixes else ""
    ext = {".maegz": ".mae", ".cmsgz": ".cms", ".dms": ".mae"}.get(last, last)
    return ext[1:] if ext in _WRITERS else "pdb"


def _backbone(system) -> str:
    names = set(np.asarray(system.atoms["name"]).tolist())
    found = next((n for n in ("CA", "BB", "GC") if n in names), None)
    if found is None:
        raise ValueError("no CA, BB or GC atoms to superpose on")
    return found


def _model_of(system, ids) -> str:
    """``aa``, ``martini`` or ``sirah``: what the atoms ``ids`` are, by their names and their
    residues' names (SIRAH has no BB bead; its residue names, sA, sK, ..., say it)."""
    names = np.asarray(system.atoms["name"])[ids]
    resnames = np.asarray(system.residues["name"])[np.asarray(system.atoms["residue"])[ids]]
    return guess_model(names, resnames)


def _apo_on(apo_path, reference, same_frame: bool):
    """The all-atom apo protein in ``reference``'s frame (superposed by sequence unless
    the beads were mapped from it)."""
    from ..align import superpose

    apo = _load(apo_path).select(f"protein and {NOT_PROBES}").clone()
    if not same_frame:
        fit = superpose(apo, reference, sel="name CA", ref_sel=f"name {_backbone(reference)}",
                        match="sequence", apply=True)  # fmt: skip
        print(f"all-atom apo superposed on the beads: {fit.n_used}/{fit.n_matched} alpha "
              f"carbons, RMSD {fit.rmsd:.2f} A")  # fmt: skip
    return apo


#: how --holo carries the ligand onto the apo
HOLO_FITS = ("whole", "chain")


def _holo_on(holo_path, reference, ligand, holo_top=None, fit: str = "whole"):
    """``(ligand heavy atoms, what was used, the holo structure moved)`` in ``reference``'s
    frame.

    ``fit="whole"``: the whole holo protein -- alpha carbons of every chain, paired by
    sequence with the reference's backbone and pruned at 2 A -- is superposed, and the
    ligand moves with it, as the benchmarks behind the presets placed it.
    ``fit="chain"``: :func:`boonza.sites.known_ligand`, which fits the one protein chain
    the ligand sits in or touches.  On a symmetric oligomer the two can put the ligand in
    different, symmetry-equivalent copies of its site: on the dimer 1MPU/8EA5, holo chain
    A pairs with apo chain B by sequence and the ligand lands across the twofold axis.
    ``ligand`` defaults to known_ligand's guess.
    """
    from ..align import superpose
    from ..sites import known_ligand

    if fit not in HOLO_FITS:
        raise ValueError(f"--holo-fit {fit!r}: one of {', '.join(HOLO_FITS)}")
    holo = _load(holo_path, holo_top)
    if ligand is None:
        ligand = known_ligand(holo, reference)[1]["ligand"]
    back = f"name {_backbone(reference)} and {NOT_PROBES}"
    moved = holo.clone()
    if fit == "chain":
        lig, info = known_ligand(holo, reference, ligand)
        # the same fit, applied to the whole holo structure for the view
        superpose(moved, reference, sel=f'chain "{info["chain"]}" and protein and name CA',
                  ref_sel=back, match="sequence", apply=True)  # fmt: skip
        info = {**info, "fit": "chain"}
        how = f"holo chain {info['chain']} superposed ({info['paired']} alpha carbons kept"
    else:
        sup = superpose(moved, reference, sel=f"protein and name CA and {NOT_PROBES}",
                        ref_sel=back, match="sequence", apply=True)  # fmt: skip
        lig = np.asarray(moved.positions)[moved.select(f"({ligand}) and not element H").ids]
        if not len(lig):
            raise ValueError(f"the holo structure has no {ligand}")
        info = {"fit": "whole", "ligand": ligand, "atoms": int(len(lig)),
                "paired": int(sup.n_used), "matched": int(sup.n_matched),
                "fit_rmsd": float(sup.rmsd)}  # fmt: skip
        how = f"whole holo protein superposed ({sup.n_used}/{sup.n_matched} alpha carbons kept"
    print(f"holo ligand {ligand}: {info['atoms']} heavy atoms; {how}, "
          f"RMSD {info['fit_rmsd']:.2f} A)")  # fmt: skip
    if info["paired"] < 20 or info["fit_rmsd"] > 3.0:
        print("warning: a poor superposition; the distances to the ligand are unreliable",
              file=sys.stderr)  # fmt: skip
    return lig, info, moved


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 4)
    return value


def _write(out: Path, rows: list[dict], settings: dict) -> None:
    """``pockets.csv`` and ``pockets.json``: the same rows, and in the JSON every setting
    the numbers depend on."""
    out.mkdir(parents=True, exist_ok=True)
    rows = [{k: _number(v) for k, v in r.items()} for r in rows]
    doc = {"settings": settings, "pockets": rows}
    (out / "pockets.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    head = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "pockets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows({k: "" if r.get(k) is None else r[k] for k in head} for r in rows)


def _settings(args, model: str, flags: list[str], **more) -> dict:
    from .. import __version__

    return {"boonza": __version__, "model": model, "fpocket_flags": flags,
            "n_polar": bool(preset(model).get("n_polar")),
            "holo": str(args.holo) if args.holo else None,
            "holo_ligand": args.holo_ligand if args.holo else None,
            "holo_fit": args.holo_fit if args.holo else None, **more}  # fmt: skip


def _ligand_measures(pocket, centre, lig) -> dict:
    from ..sites import dca, dcc
    from .overlap import ligand_atoms_within, moc, ppc, spheres_within

    return {"center_to_nearest_ligand_atom": dca(centre[None], lig),
            "center_to_ligand_centroid": dcc(centre[None], lig),
            "ligand_atoms_within_3A": ligand_atoms_within(pocket, lig),
            "spheres_within_3A": spheres_within(pocket, lig),
            "PPc": ppc(centre[None], lig), "MOc": moc(pocket, lig)}  # fmt: skip


def cmd_run(args, extra) -> int:
    from .run import run_fpocket

    model = args.model
    find_fpocket(args.fpocket)  # before any work: a missing or stock build fails at once
    system = _load(args.structure, args.top)
    beads, ids = searchable(system, model, args.selection)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pdb = out / f"{Path(args.structure).name.split('.')[0]}_{model}.pdb"
    write_fpocket_pdb(beads, pdb, ids, model, bool(preset(model).get("n_polar")))
    flags = merge_flags(list(preset(model)["flags"]), extra)
    print(f"fpocket on {pdb.name}: {len(ids)} {'atoms' if model == 'aa' else 'beads'}, "
          f"{' '.join(flags) or 'its own defaults'}")  # fmt: skip
    pockets = run_fpocket(pdb, flags, args.fpocket)
    all_atom = _model_of(system, protein_ids(system)) == "aa"
    apo_path = args.apo or (args.structure if all_atom else None)
    reference = _load(pdb)
    apo = _apo_on(apo_path, reference, same_frame=apo_path == args.structure) if apo_path else None
    lig = moved = None
    if args.holo:
        lig, held, moved = _holo_on(args.holo, apo if apo is not None else reference,
                                    args.holo_ligand, args.holo_top, args.holo_fit)  # fmt: skip
        args.holo_ligand = held["ligand"]
    rows, verdicts = [], {}
    for k, p in enumerate(pockets, 1):
        row = {
            "rank": k,
            "fpocket_score": p.score,
            "fpocket_volume": p.volume,
            "alpha_spheres": len(p.centres),
            **dict(zip(("center_x", "center_y", "center_z"), p.centre, strict=True)),
        }
        if lig is not None:
            row.update(_ligand_measures(p.centres, p.centre, lig))
            verdicts[k] = (row["PPc"], row["MOc"])
        rows.append(row)
    _write(out, rows, _settings(args, model, flags, structure=str(args.structure)))
    print(f"{len(pockets)} pockets -> {out / 'pockets.csv'}")
    if verdicts:
        first = {c: next((k for k, v in verdicts.items() if v[i]), None)
                 for i, c in enumerate(("PPc", "MOc"))}  # fmt: skip
        print("first right for the holo ligand: " + ", ".join(
            f"{c} rank {k}" if k else f"{c} none" for c, k in first.items()))  # fmt: skip
    if apo is not None:
        from .view import write_run_view

        fmt = structure_format(apo_path)
        view = write_run_view(out, pockets, apo, moved, args.holo_ligand
                              if moved is not None else None, verdicts, fmt)  # fmt: skip
        print(f"all-atom view: pymol {view}")
    else:
        print("no view: it shows the pockets on the all-atom apo structure; give it with --apo")
    return 0


def _frames_of(args):
    """``(system, trajectory window, interval_ns)`` of the run asked for."""
    from ..cli import _interval_of, _window_of
    from ..trajectory import open_trajectory

    if args.workdir:
        here = Path(args.workdir)
        system = _load(here / "solvated.dms")
        traj = open_trajectory(str(here / "trajectory.dcd"), system)
        interval = args.interval_ns or _interval_of([here])
    else:
        if not args.system or not args.traj:
            raise ValueError("give SYSTEM with --traj, or --workdir")
        system = _load(args.system, args.top)
        traj = open_trajectory(str(args.traj), system)
        interval = args.interval_ns
    window = _window_of(len(traj), interval, args.from_ns, args.until_ns, args.every_ns)
    return system, traj[window], interval


def _bead_radii(system, ids, model: str) -> np.ndarray:
    """Each bead's radius, sigma/2: from the force field the system carries, else from the
    model's table by bead type."""
    from ..sites import particle_radii
    from .core import bead_radii

    if "nonbonded" in system.table_names:
        return particle_radii(system, ids, "sigma")
    radii = bead_radii(np.asarray(system.atoms["type"])[ids], model)
    if np.isnan(radii).any():
        raise ValueError(f"{int(np.isnan(radii).sum())} beads have no type in the {model} "
                         "table: give the topology (--top) so their sizes can be read")  # fmt: skip
    return radii


def cmd_traj(args, extra) -> int:
    from ..glue import Glue
    from ..trajectory import open_writer
    from .consensus import consensus_pockets, frame_pockets, write_frame_pockets
    from .core import SPACING, enclosed_core
    from .overlap import grid_pocket, ppc, volume_overlap
    from .run import run_fpocket, run_mdpocket, write_isosurface
    from .view import write_pqr, write_traj_view

    model = args.model
    if model == "aa":
        raise ValueError(
            "traj is for coarse-grained runs: give --model martini2, martini3 or sirah"
        )
    find_fpocket(args.fpocket)  # before the frames are prepared: that takes a while
    system, traj, interval = _frames_of(args)
    ids = protein_ids(system, args.selection)
    names = np.asarray(system.atoms["name"])
    if _model_of(system, ids) == "aa":
        raise ValueError("traj needs the run's own coarse-grained structure")
    fit = ids[np.isin(names[ids], ["BB", "GC"])]
    whole = ids if system.nbonds else None
    glue = Glue(system, glue=[ids], center=ids, fit=fit, whole=whole)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_polar = bool(preset(model).get("n_polar"))
    coords = []
    with open_writer(str(out / "md.dcd"), len(ids)) as w:
        for block in glue.frames(traj):
            for k in range(len(block)):
                coords.append(np.asarray(block.positions[k][ids], np.float32))
                w.write(coords[-1], None)
    if not coords:
        raise ValueError("no frames in the window asked for")
    beads = system.clone(ids)
    beads.positions[:] = coords[0]
    write_fpocket_pdb(beads, out / "md.pdb", None, model, n_polar)
    print(f"{len(coords)} frames of {len(ids)} protein beads, made whole and fitted on "
          f"{len(fit)} backbone beads -> {out / 'md.pdb'}, {out / 'md.dcd'}")  # fmt: skip
    flags = merge_flags(list(preset(model)["flags"]), extra)
    reference = _load(out / "md.pdb")
    apo = _apo_on(args.apo, reference, same_frame=False) if args.apo else None
    lig = moved = None
    if args.holo:
        lig, held, moved = _holo_on(args.holo, apo if apo is not None else reference,
                                    args.holo_ligand, args.holo_top, args.holo_fit)  # fmt: skip
        args.holo_ligand = held["ligand"]
    iso = None
    if args.mdpocket:
        run_mdpocket(out / "md.pdb", out / "md.dcd", out / "mdpocket", flags, args.fpocket)
        iso = preset(model).get("mdpocket_density_iso")
        if iso is not None:
            write_isosurface(out / "mdpocket_dens.dx", iso)
        (out / "pocket_frequency.dx").write_bytes((out / "mdpocket_freq.dx").read_bytes())
        (out / "pocket_density.dx").write_bytes((out / "mdpocket_dens.dx").read_bytes())
    crystal = None
    if apo is not None:
        # what is open in the apo crystal structure already, to call the others cryptic
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            write_fpocket_pdb(coarse_grain(apo, model), Path(tmp) / "crystal.pdb", None, model,
                              n_polar)  # fmt: skip
            found = run_fpocket(Path(tmp) / "crystal.pdb", flags, args.fpocket, quiet=True)
        crystal = np.array([p.centre for p in found]).reshape(-1, 3)
    print(f"fpocket on every frame: {' '.join(flags)}", flush=True)
    per_frame = list(frame_pockets(out / "md.pdb", coords, flags, args.fpocket))
    ranked = consensus_pockets(per_frame, args.consensus_cutoff, crystal)
    radii = _bead_radii(system, ids, model)
    for q in ranked:
        b = q.best
        q.core = enclosed_core(b.pocket.centres, b.pocket.radii, coords[b.frame], radii)
    rows, verdicts = [], {}
    for q in ranked:
        b, c = q.best, q.centre
        row = {"rank_quality": q.rank_quality, "rank_persistence": q.rank_persistence,
               "rank_quality_burial": q.rank_quality_burial, "quality": q.quality,
               "persistence": q.persistence, "quality_burial": q.quality_burial,
               "occupancy": q.occupancy, "pocket_burial": q.burial, "cryptic": q.cryptic,
               "frames_open": len(q.open), "best_frame": b.frame, "fpocket_score": b.pocket.score,
               "fpocket_p": b.p, "fpocket_volume": b.pocket.volume,
               "alpha_spheres": len(b.pocket.centres),
               "center_x": c[0], "center_y": c[1], "center_z": c[2],
               "core_volume": len(q.core) * SPACING**3}  # fmt: skip
        cc = q.core.mean(0) if len(q.core) else np.full(3, np.nan)
        row.update(core_center_x=cc[0], core_center_y=cc[1], core_center_z=cc[2])
        if lig is not None:
            row.update(_ligand_measures(b.pocket.centres, c, lig))
            row["share_of_open_frames_PPc"] = np.mean([ppc(m.centre[None], lig) for m in q.open])
            if len(q.core):
                ov = volume_overlap(grid_pocket(q.core, SPACING), lig)
                row.update(
                    core_center_to_nearest_ligand_atom=float(
                        np.linalg.norm(lig - cc, axis=1).min()
                    ),
                    PPc_core=ppc(q.core, lig),
                    core_ligand_volume_covered=ov["ligand_volume_covered"],
                    core_volume_near_ligand=ov["pocket_volume_near_ligand"],
                    core_volume_in_ligand=ov["pocket_volume_in_ligand"],
                    core_DVO=ov["DVO"],
                )
            else:
                row.update({k: None for k in CORE_LIGAND_COLUMNS[1:]}, PPc_core=False)
            verdicts[q.rank_quality] = (row["PPc"], row["MOc"])
        rows.append(row)
    settings = _settings(args, model, flags, consensus_cutoff=args.consensus_cutoff,
                         frames=len(coords), interval_ns=interval, from_ns=args.from_ns,
                         until_ns=args.until_ns, every_ns=args.every_ns,
                         workdir=str(args.workdir) if args.workdir else None,
                         apo=str(args.apo) if args.apo else None,
                         core_spacing_A=SPACING, best_frames=args.best_frames)  # fmt: skip
    files = _write_best_frames(args, out, ranked, beads, coords)
    for row in rows:
        row["best_frame_file"] = files.get(row["rank_quality"])
    _write(out, rows, settings)
    write_frame_pockets(out / "frame_pockets.npz", ranked)
    write_pqr(out / "pockets.pqr", [(q.rank_quality, q.best.pocket.centres, q.best.pocket.radii)
                                    for q in ranked])  # fmt: skip
    cell = SPACING * (3 / (4 * np.pi)) ** (1 / 3)  # a sphere of one grid cell's volume
    write_pqr(out / "cores.pqr", [(q.rank_quality, q.core, cell) for q in ranked], "COR")
    with open(out / "fpocket_info.txt", "w") as fh:
        for q in ranked:
            b = q.best
            fh.write(
                f"Consensus pocket {q.rank_quality} (quality rank; persistence rank "
                f"{q.rank_persistence}, quality x burial rank {q.rank_quality_burial}): "
                f"best in frame {b.frame}, where it is fpocket's pocket {b.rank}\n"
            )
            fh.write(b.pocket.info.split("\n", 1)[1] if "\n" in b.pocket.info else "\n")
    with open(out / "frames.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("frame", "pockets"), *enumerate(map(len, per_frame))])
    print(f"{len(ranked)} consensus pockets from {sum(map(len, per_frame))} pockets in "
          f"{len(coords)} frames -> {out / 'pockets.csv'}")  # fmt: skip
    if verdicts:
        for key in ("quality", "quality_burial", "persistence"):
            hit = min((getattr(q, f"rank_{key}") for q in ranked if verdicts[q.rank_quality][0]),
                      default=None)  # fmt: skip
            print(f"  by {key}: first right by PPc at rank {hit or 'none'}")
    view = write_traj_view(out, ranked, apo, moved, args.holo_ligand if moved is not None else None,
                           verdicts, structure_format(args.apo or args.holo or "x.pdb"), iso,
                           files)  # fmt: skip
    print(
        f"view: pymol {view}"
        + ("" if apo is not None else "  (on the beads; --apo draws the all-atom structure)")
    )
    return 0


def _write_best_frames(args, out: Path, ranked, beads, coords) -> dict:
    """Each pocket's best frame -- the protein beads in the frame its p is highest -- for the
    pockets in the top ``--best-frames`` of any ranking, as ``best_frames/pocket<quality
    rank>_frame<frame>.<ext>``.  The format is the run's own: a .dms or .mae run is written
    as MAE, which keeps its bonds; others as they came, or PDB.  Returns the file of each
    pocket written, by quality rank."""
    from ..io import save

    if not args.best_frames:
        return {}
    source = Path(args.workdir) / "solvated.dms" if args.workdir else Path(args.system)
    fmt = structure_format(source)
    where = out / "best_frames"
    where.mkdir(exist_ok=True)
    files = {}
    for q in ranked:
        if min(q.rank_quality, q.rank_persistence, q.rank_quality_burial) > args.best_frames:
            continue
        frame = beads.clone()
        frame.positions = np.asarray(coords[q.best.frame], float)
        name = f"pocket{q.rank_quality}_frame{q.best.frame}.{fmt}"
        save(frame, where / name)
        files[q.rank_quality] = f"best_frames/{name}"
    print(f"best frames of {len(files)} pockets (top {args.best_frames} of any ranking) -> "
          f"{where}")  # fmt: skip
    return files


def cmd_flags(args, extra) -> int:
    print(" ".join(merge_flags(list(preset(args.model)["flags"]), extra)))
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boonza pockets", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, models=MODELS):
        q.add_argument("--model", required=True, choices=models,
                       help="what the protein is searched as: all-atom (aa, fpocket unchanged) "
                            "or one of the bead models, whose tuned preset is used")  # fmt: skip
        q.add_argument("--top", help="GROMACS topology of a coarse-grained structure, which "
                                     "gives the bead types")  # fmt: skip
        q.add_argument("--selection", help="atoms or beads to search (default: the protein; "
                                           "probes in chain LIG are always left out)")  # fmt: skip
        q.add_argument(
            "--apo",
            help="the all-atom apo structure, for a view on all-atom "
            "structures (superposed on the beads by sequence)",
        )
        q.add_argument(
            "--holo",
            help="a structure of the same protein with a ligand: each "
            "pocket is scored against that ligand (PPc, MOc)",
        )
        q.add_argument("--holo-ligand", default=None, metavar="SEL",
                       help="the ligand in --holo (default: its largest residue that is "
                            "neither protein, nucleic, solvent nor a buffer salt)")  # fmt: skip
        q.add_argument("--holo-top", help="topology of a coarse-grained --holo")
        q.add_argument(
            "--holo-fit",
            choices=HOLO_FITS,
            default="whole",
            help="how the holo ligand is carried onto the apo: superposing the "
            "whole holo protein, every chain (default, as the presets were "
            "benchmarked), or only the chain the ligand sits in or touches "
            "(boonza.sites.known_ligand).  On a symmetric oligomer the two can "
            "put the ligand in different, symmetry-equivalent copies of its site",
        )
        q.add_argument("--fpocket", help="the fpocket build to run (its directory or the "
                                         "program; default: $FPOCKET_HOME, then PATH)")  # fmt: skip
        q.add_argument("-o", "--out", required=True, help="the output directory")

    q = sub.add_parser("run", help="fpocket on one structure, all-atom or coarse-grained")
    q.add_argument("structure", help="all-atom (mapped onto beads first) or coarse-grained")
    common(q)
    q.set_defaults(run=cmd_run)

    q = sub.add_parser("traj", help="consensus pockets over a coarse-grained trajectory")
    q.add_argument("system", nargs="?", help="the run's structure (with --traj)")
    q.add_argument("--traj", help="its trajectory")
    q.add_argument("--workdir", help="a run directory, read as solvated.dms and trajectory.dcd")
    common(q, MODELS[1:])
    q.add_argument(
        "--consensus-cutoff",
        type=float,
        default=6.0,
        metavar="A",
        help="a frame's pocket joins a consensus pocket whose centroid is this "
        "close (default 6: on 80 apo trajectories it halves the splitting of "
        "one site that 4 gives; 7 to 10 start to merge neighbouring sites)",
    )
    q.add_argument("--mdpocket", action="store_true",
                   help="also run mdpocket for its pocket-frequency and density maps")  # fmt: skip
    q.add_argument("--interval-ns", type=float, default=None,
                   help="ns between frames (default: the run's own, with --workdir)")  # fmt: skip
    q.add_argument("--from-ns", type=float, default=None, metavar="NS",
                   help="read the run from this time on (default: its first frame)")  # fmt: skip
    q.add_argument("--until-ns", type=float, default=None, metavar="NS",
                   help="read the run up to this time (default: its last frame)")  # fmt: skip
    q.add_argument("--every-ns", type=float, default=None, metavar="NS",
                   help="read one frame this far apart (default: every frame; fpocket runs "
                        "once per frame read, about a second each)")  # fmt: skip
    q.add_argument("--best-frames", type=int, default=10, metavar="K",
                   help="write each pocket's best frame (its highest p) for the pockets in the "
                        "top K of any ranking, in the run's own format: MAE for a .dms or .mae "
                        "run, keeping its bonds (default 10; 0 writes none)")  # fmt: skip
    q.set_defaults(run=cmd_traj)

    q = sub.add_parser("flags", help="print a model's tuned fpocket flags")
    q.add_argument("--model", required=True, choices=MODELS)
    q.set_defaults(run=cmd_flags)
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1 :]
    args = _parser().parse_args(argv)
    from .run import PocketError

    try:
        return args.run(args, extra)
    except (ValueError, FileNotFoundError, PocketError) as e:
        print(f"boonza pockets {argv[0] if argv else ''}: error: {e}", file=sys.stderr)
        return 1
