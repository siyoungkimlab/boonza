"""What a trajectory's pockets are, as a table and a PyMOL view.

``pockets.csv``: one row per pocket, in ``rank`` order: occupancy (the share of
frames it is in), its value and place under every traj.RANKINGS, scores,
volumes, its best MD frame, the residues lining it (its core: those lining at
least half its sites), the probe molecules at it in its best frame, and the
pharmacophore hotspots in it (pharm.py: family, enrichment over bulk, distinct
probe molecules).

``view.pml`` (run from anywhere: it loads its files relative to itself), in the
run's fitted frame:

  apo             the all-atom apo structure, superposed on the run (if given)
  beads           the protein's beads in the first frame (off at the start)
  pocket_<rank>   a group per pocket, holding
    site_<rank>     its site in its best frame, coloured by rank, labelled with
                    rank, occupancy, p (its best SiteScore as a probability) and
                    best MD frame
    sites_<rank>    the site of every frame it is in, as small dots (off); the
                    file links each frame's points as site finding does, so
                    "show lines, sites_<rank>" draws how they connect
    frame_<rank>    the protein in that best frame: beads as spheres and its
                    bonds as lines, without the elastic network (off)
    probes_<rank>   the probe molecules (chain LIG) with a bead within PROBE_NEAR
                    of the site in that frame (off)
    pharm_<rank>    the pharmacophore hotspots of the pocket over the whole run
                    (each hotspot goes to the one pocket at it most often), a
                    sphere each: colour its family (donor sky blue, acceptor
                    salmon, cation blue, anion red, aromatic orange, hydrophobe
                    green), radius its extent (a ball of the hotspot region's
                    volume: where the feature may sit), and a label, hidden at
                    the start ("show labels, pharm_<rank>"): family, enrichment
                    over bulk and distinct probe molecules (off)

Scenes: ``overview`` (the view as it opens) and ``pocket_<rank>`` (that pocket
alone, zoomed), as buttons at the foot of the viewer, PgUp/PgDn or ``scene
<name>``; and the commands ``overview`` and ``only <rank>`` (that pocket alone):
turning a group on turns on everything in it, and a scene is the way back.

Probes and hotspots are shown only: site finding and ranking never use them.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

PROBE_NEAR = 3.0  # A
#: pharmacophore colours, by boonza.pharmacophore.CODES
PHARM_COLORS = {"DON": "skyblue", "ACC": "salmon", "CAT": "blue", "ANI": "red",
                "ARO": "orange", "HYD": "green"}  # fmt: skip
CODES = {"Donor": "DON", "Acceptor": "ACC", "Aromatic": "ARO", "Hydrophobe": "HYD",
         "PosIonizable": "CAT", "NegIonizable": "ANI"}  # fmt: skip
MAX_SERIAL = 99999  # PDB atom serials: a pocket's frames beyond it are drawn without lines


def _residue_names(system, residues) -> str:
    res = system.residues
    out = []
    for r in sorted(residues):
        chain = str(system.chains["name"][res["chain"][r]]).strip()
        out.append(f"{chain}:{str(res['name'][r]).strip()}{int(res['resid'][r])}"
                   f"{str(res['insertion'][r]).strip()}")  # fmt: skip
    return " ".join(out)


BOND_MAX = 5.5  # A: longer bonds of a frame are restraints (Martini 2.2's sheet bonds)


def _sites_pdb(path: Path, pocket, spacing: float) -> None:
    """Every frame's site of ``pocket``, one residue per frame, with a CONECT bond for
    each pair of its points site finding links (sites.LINK on the grid)."""
    from .sites import LINK, _pairs

    atoms, bonds, n = [], [], 0
    for s in sorted(pocket.sites, key=lambda s: s.frame):
        if n + len(s.xyz) > MAX_SERIAL:
            break
        ijk = np.rint(s.xyz / spacing).astype(np.int64)
        bi, bj = _pairs(ijk, LINK)
        for x, y, z in s.xyz:
            n += 1
            atoms.append(f"HETATM{n:5d}  C   FRM F{s.frame % 10000:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")  # fmt: skip
        first = n - len(s.xyz) + 1
        bonds += [(first + i, first + j) for i, j in zip(bi, bj, strict=True)]
    conect = [f"CONECT{a:5d}{b:5d}" for a, b in bonds]
    path.write_text("\n".join(atoms + conect + ["END"]) + "\n")


def _drawable(system, ids, xyz):
    """The beads ``ids`` at ``xyz``, without bonds longer than BOND_MAX: what a viewer
    should draw lines along (the elastic network is gone already from a view system)."""
    s = system.clone(ids)
    s.positions = np.asarray(xyz, float)
    if s.nbonds:
        i, j = np.asarray(s.bonds["i"]), np.asarray(s.bonds["j"])
        length = np.linalg.norm(s.positions[i] - s.positions[j], axis=1)
        if (length > BOND_MAX).any():
            s.delete_bonds(np.flatnonzero(length > BOND_MAX))
    return s


def _pharm_pdb(path: Path, spots) -> None:
    """Hotspots as one atom each: residue name the family's code, B-factor its
    enrichment, occupancy its distinct probe molecules."""
    path.write_text("".join(
        f"HETATM{n:5d}  X   {CODES[h.family]} H{n % 10000:4d}    "
        f"{h.center[0]:8.3f}{h.center[1]:8.3f}{h.center[2]:8.3f}"
        f"{min(h.ligands, 999):6.2f}{min(h.enrichment, 999.99):6.2f}           X\n"
        for n, h in enumerate(spots, 1)) + "END\n")  # fmt: skip


def write(out, model: str, pockets, n_frames: int, every: int, system, ids, coords,
          pids=None, pcoords=None, apo=None, top: int | None = None,
          rank: str = "p_max", spots=None, view_system=None, holo=None,
          holo_ligand: str | None = None, ligand=None) -> Path:  # fmt: skip
    """pockets.csv and view.pml for ``pockets`` (traj.consensus, any order) under ``out``.

    ``coords``/``pcoords``: protein/probe positions of the analysed frames (frame k is MD
    frame k * every).  ``apo``: an all-atom System already superposed on the run.
    ``top``: draw only the best ``top`` pockets (all are in the table).  ``rank``: the
    traj.RANKINGS value that orders them.  ``spots``: the run's pharmacophore hotspots
    (pharm.hotspots).  ``view_system``: the run's system without its bands (its
    view.dms, boonza's for_viewing), atom for atom as ``system``, for the frames' bonds.
    ``holo``, ``holo_ligand``, ``ligand``: a holo structure superposed on the run, its
    ligand's selection and that ligand's heavy atoms there: every pocket is measured
    against it in its best frame (PPc, MOc, LVC, PVN) and by the share of its frames whose
    site is PPc-right (frames_PPc)."""
    from ..io import save
    from ..pockets import view as BV
    from .pharm import assign
    from .presets import preset
    from .structure import _against
    from .traj import RANKINGS

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("frame_*.*"), *out.glob("frames_*.*"), *out.glob("sites_*.*"),
                *out.glob("probes_*.*"),
                *out.glob("pharm_*.*"), *out.glob("frames.pqr")]:  # fmt: skip
        old.unlink()  # from an earlier writing
    summaries = {id(p): p.summary(n_frames) for p in pockets}
    pockets = sorted(pockets, key=lambda p: (-summaries[id(p)][rank], -p.best.score))
    places = {}
    for name in RANKINGS:
        order = sorted(pockets, key=lambda p: (-summaries[id(p)][name], -p.best.score))
        places[name] = {id(p): k for k, p in enumerate(order, 1)}
    spacing = preset(model)["spacing"]
    fmt = "mae" if system.nbonds else "pdb"
    pids = np.zeros(0, int) if pids is None else pids
    pres = np.asarray(system.atoms["residue"])[pids]
    spots = spots or []
    owned = assign(spots, pockets)
    rows, drawn, objects = [], [], []
    for k, p in enumerate(pockets, 1):
        s = summaries[id(p)]
        f = p.best.frame
        near = np.zeros(len(pids), bool)
        if len(pids):
            d2 = ((pcoords[f][:, None] - p.best.xyz[None]) ** 2).sum(-1).min(1)
            near = np.isin(pres, np.unique(pres[d2 <= PROBE_NEAR**2]))
        probe_names = sorted({str(system.residues["name"][r]).strip() for r in pres[near]})
        mine = owned[k - 1]
        rows.append({"rank": k, "occupancy": round(s["occupancy"], 3), "frames": s["frames"],
                     **{name: round(s[name], 3) for name in RANKINGS if name != "occupancy"},
                     **{f"rank_{name}": places[name][id(p)] for name in RANKINGS},
                     "best_score": round(s["best_score"], 3),
                     "mean_score": round(s["mean_score"], 3),
                     "median_volume": s["median_volume"], "max_volume": s["max_volume"],
                     "best_md_frame": f * every, "core_residues": _residue_names(system, p.core),
                     "probes_at_best_frame": " ".join(probe_names),
                     "hotspots": "; ".join(f"{CODES[h.family]} x{h.enrichment:.0f} "
                                           f"({h.ligands} probes)" for h in mine)})  # fmt: skip
        verdict = ""
        if ligand is not None:
            from ..pockets.overlap import ppc

            m = _against(p.best, spacing, ligand)
            per_frame = p.per_frame_best().values()
            rows[-1].update(
                {
                    **m,
                    "frames_PPc": round(float(np.mean([ppc(x.xyz, ligand) for x in per_frame])), 3),
                }
            )
            verdict = f"{' PPc' if m['PPc'] else ''}{' MOc' if m['MOc'] else ''} LVC {m['LVC']:.2f}"
        if top is not None and k > top:
            continue
        label = f"{k} occ {s['occupancy']:.2f} p {s['p_max']:.2f} md frame {f * every}{verdict}"
        drawn.append((k, label, p.best.xyz.mean(0)))
        frame = _drawable(view_system or system, ids, coords[f])
        names = {"frame": f"frame_{k}_md{f * every}.{fmt}", "sites": f"sites_{k}.pdb"}
        _sites_pdb(out / names["sites"], p, spacing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save(frame, out / names["frame"])
            if near.any():
                mols = system.clone(pids[near])
                mols.positions = np.asarray(pcoords[f][near], float)
                names["probes"] = f"probes_{k}_md{f * every}.{fmt}"
                save(mols, out / names["probes"])
        if mine:
            names["pharm"] = f"pharm_{k}.pdb"
            _pharm_pdb(out / names["pharm"], mine)
            names["radii"] = [h.radius for h in mine]
            names["labels"] = [f"{CODES[h.family]} x{h.enrichment:.0f} {h.ligands}p" for h in mine]
        objects.append((k, names))
    with open(out / "pockets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["rank"])
        w.writeheader()
        w.writerows(rows)

    shown = {k for k, *_ in drawn}
    BV.write_pqr(out / "sites.pqr", [(k, pockets[k - 1].best.xyz, spacing / 2) for k in shown])
    beads = _drawable(view_system or system, ids, coords[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save(beads, out / f"beads.{fmt}")
        if apo is not None:
            save(apo.select("protein").clone(), out / "apo.pdb")
        if holo is not None:
            save(holo.select(f"protein and not ({holo_ligand})").clone(), out / "holo.pdb")
            save(holo.select(holo_ligand).clone(), out / "ligand.pdb")
    lines = [f"# boonza sitemap: pockets of a {model} run ({n_frames} frames, every {every}th MD "
             f"frame, {len(pockets)} pockets ranked by {rank}, the best {len(drawn)} drawn)",
             "# each pocket is a group, pocket_<rank>: site_<rank> (its site in its best",
             "# frame), sites_<rank> (every frame's site points), frame_<rank> (the",
             "# protein in the best frame, no elastic network), probes_<rank> (the probes there",
             "# then) and pharm_<rank> (the run's pharmacophore hotspots in the pocket:",
             "# donor sky blue, acceptor salmon, cation blue, anion red, aromatic orange,",
             "# hydrophobe green)", *BV.PREAMBLE]  # fmt: skip
    if apo is not None:
        lines += ["load apo.pdb, apo"]
    if holo is not None:
        lines += ["load holo.pdb, holo", "load ligand.pdb, ligand"]
    lines += [f"load beads.{fmt}, beads", "set connect_mode, 1", "load sites.pqr, sites_all",
              "set connect_mode, 0", "hide everything", "bg_color black",
              "set ray_opaque_background, 0"]  # fmt: skip
    if apo is not None:
        lines += ["show cartoon, apo", "color wheat, apo", "set cartoon_transparency, 0.45, apo"]
    if holo is not None:
        lines += ["show cartoon, holo", "color lightblue, holo",
                  "# click holo in the object panel to compare the folds", "disable holo",
                  "show sticks, ligand and not hydro", "color tv_blue, ligand and elem C",
                  "set stick_radius, 0.3, ligand"]  # fmt: skip
    lines += ["show spheres, beads", "set sphere_scale, 0.35, beads", "color grey70, beads",
              "set sphere_transparency, 0.6, beads"]  # fmt: skip
    lines += [] if apo is None else ["disable beads"]
    lines += BV.pocket_object_lines("sites_all", drawn)
    for k, names in objects:
        color = BV._color(k)
        lines += [f"set sphere_scale, 0.5, pocket_{k}", "set connect_mode, 1",
                  f"load {names['sites']}, sites_{k}", "set connect_mode, 0",
                  f"hide everything, sites_{k}",
                  f"show spheres, sites_{k}", f"set sphere_scale, 0.15, sites_{k}",
                  f"set sphere_transparency, 0.6, sites_{k}", f"color {color}, sites_{k}",
                  f"disable sites_{k}", f"load {names['frame']}, frame_{k}",
                  f"hide everything, frame_{k}", f"show spheres, frame_{k}",
                  f"show lines, frame_{k}", f"set line_width, 2, frame_{k}",
                  f"set sphere_scale, 0.35, frame_{k}", f"color grey70, frame_{k}",
                  f"set sphere_transparency, 0.5, frame_{k}", f"disable frame_{k}"]  # fmt: skip
        members = [f"site_{k}", f"sites_{k}", f"frame_{k}"]
        if "probes" in names:
            lines += [f"load {names['probes']}, probes_{k}", f"hide everything, probes_{k}",
                      f"show sticks, probes_{k}", f"show spheres, probes_{k}",
                      f"set sphere_scale, 0.3, probes_{k}", f"set stick_radius, 0.15, probes_{k}",
                      f"color white, probes_{k}", f"disable probes_{k}"]  # fmt: skip
            members.append(f"probes_{k}")
        if "pharm" in names:
            lines += [f"load {names['pharm']}, pharm_{k}", f"hide everything, pharm_{k}",
                      f"show spheres, pharm_{k}", f"set sphere_transparency, 0.3, pharm_{k}",
                      *[f"alter pharm_{k} and id {n}, vdw={r:.2f}"
                        for n, r in enumerate(names["radii"], 1)],
                      "rebuild",
                      *[f'label pharm_{k} and id {n}, "{t}"'
                        for n, t in enumerate(names["labels"], 1)],
                      f"hide labels, pharm_{k}",
                      *[f"color {c}, pharm_{k} and resn {code}"
                        for code, c in PHARM_COLORS.items()],
                      f"disable pharm_{k}"]  # fmt: skip
            members.append(f"pharm_{k}")
        lines += [f"set_name pocket_{k}, site_{k}", f"group pocket_{k}, {' '.join(members)}",
                  f"group pocket_{k}, action=close"]  # fmt: skip
    lines += ["# the whole protein in view",
              "zoom apo, 5" if apo is not None else "zoom beads, 5"]  # fmt: skip
    # scenes: turning a group on turns on all it holds, so the way back to a tidy view is
    # a stored one -- PgUp/PgDn step through them, "scene overview" returns to the start
    pocket_names = " ".join(f"pocket_{k}" for k, _ in objects)
    lines += ["# scenes: overview (this view) and pocket_<rank> (that pocket alone, zoomed),",
              "# stepped through with PgUp and PgDn, or 'scene <name>'",
              "scene overview, store"]  # fmt: skip
    for k, _ in objects:
        lines += [f"disable {pocket_names}", f"enable pocket_{k}",
                  f"disable sites_{k} frame_{k} probes_{k} pharm_{k}",
                  f"zoom site_{k}, 10", f"scene pocket_{k}, store"]  # fmt: skip
    ranks = [k for k, _ in objects]
    lines += ["scene overview, recall", "set scene_buttons, 1",
              "# commands: 'overview' (the view as it opened), 'only <rank>' (that pocket's",
              "# site alone, the other pockets off)",
              "python",
              "from pymol import cmd",
              f"_ranks = {ranks}",
              "def overview():",
              "    cmd.scene('overview', 'recall')",
              "def only(rank):",
              "    k = int(rank)",
              "    cmd.disable(' '.join(f'pocket_{j}' for j in _ranks))",
              "    cmd.enable(f'pocket_{k}')",
              "    cmd.enable(f'site_{k}')",
              "    extra = ('sites', 'frame', 'probes', 'pharm')",
              "    cmd.disable(' '.join(f'{o}_{k}' for o in extra))",
              "    cmd.zoom(f'site_{k}', 10)",
              "cmd.extend('overview', overview)",
              "cmd.extend('only', only)",
              "python end"]  # fmt: skip
    view = out / "view.pml"
    view.write_text("\n".join(lines) + "\n")
    return view
