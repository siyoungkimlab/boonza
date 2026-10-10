"""The sites of one structure, as a table and a PyMOL view.

An all-atom structure is mapped to the model's beads by boonza (beads.py), as
the benchmarks map theirs; a coarse-grained one that carries its bead types (a
run's .dms) is used as it is.  Its sites are found under the model's preset
(presets.find) and written, best first:

The ligand a site is judged against is either in the structure itself
(``ligand``, a selection) or in a holo structure of the same protein
(``holo``, its ligand ``holo_ligand`` or boonza's guess), carried onto the
structure as ``boonza pockets`` carries it (boonza.pockets.cli._holo_on:
the whole holo protein superposed by sequence, or with ``holo_fit="chain"``
the chain the ligand sits in).

``pockets.csv``: rank, SiteScore and p (its probability of being a ligand site,
1 / (1 + exp(-score))), volume, enclosure, exposure, the residues lining it
(a bead within its radius + traj.LINING of a site point) and, if a ligand is
given, how the site matches it: PPc (centre within 4 A of a ligand atom), MOc
(more than half the ligand's atoms within 3 A of a site point and more than a
fifth of the site's points within 3 A of the ligand), LVC (share of the
ligand's volume in the site) and PVN (share of the site within 2 A of it).
The ligand is only measured against: it is never part of the protein.

``view.pml`` (run from anywhere): the structure (cartoon), the holo structure
(cartoon, off at the start) if given, the ligand (sticks),
the beads (off at the start) and each site as ``pocket_<rank>``, labelled with
its rank, p, volume and, with a ligand, PPc/MOc/LVC; scene buttons
(``overview`` and each pocket alone) and the commands ``overview`` and
``only <rank>``.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

PPC, MOC_D, MOC_LIGAND, MOC_POCKET = 4.0, 3.0, 0.5, 0.2  # fpocket's criteria


def _cubes(points: np.ndarray, spacing: float) -> np.ndarray:
    """The 0.5 A voxels of each point's grid cube (boonza.pockets.overlap.grid_pocket)."""
    from ..pockets.overlap import SPACING

    half = spacing / 2
    n = int(np.ceil(half / SPACING))
    off = np.mgrid[-n : n + 1, -n : n + 1, -n : n + 1].reshape(3, -1).T
    cand = np.round(points / SPACING).astype(int)[:, None] + off[None]
    keep = (np.abs(cand * SPACING - points[:, None]) <= half + 1e-9).all(2)
    return np.unique(cand[keep], axis=0)


def _against(site, spacing: float, lig: np.ndarray) -> dict:
    from ..pockets.overlap import SHELL, _count, ligand_voxels

    near = ((lig[:, None] - site.xyz[None]) ** 2).sum(-1) < MOC_D**2
    lv = np.ascontiguousarray(ligand_voxels(lig))
    sh = np.ascontiguousarray(ligand_voxels(lig, SHELL))
    pocket = np.ascontiguousarray(_cubes(site.xyz, spacing))
    inter = _count(lv, pocket)
    return {
        "PPc": bool(np.sqrt(((lig - site.centre) ** 2).sum(1)).min() < PPC),
        "MOc": bool(near.any(1).mean() > MOC_LIGAND and near.any(0).mean() > MOC_POCKET),
        "LVC": round(inter / len(lv), 3),
        "PVN": round(_count(sh, pocket) / len(pocket), 3),
    }


def run(path, model: str, out, selection: str | None = None, ligand: str | None = None,
        top: int = 10, holo=None, holo_ligand: str | None = None, holo_top=None,
        holo_fit: str = "whole") -> Path:  # fmt: skip
    """Find and write the sites of the structure at ``path`` (see the module docstring)."""
    from ..io import load, save
    from ..pockets import view as BV
    from ..pockets.prepare import NOT_PROBES, protein_ids
    from ..spatial import min_dist2
    from . import beads as B
    from .presets import find, preset
    from .report import _residue_names
    from .traj import LINING

    if ligand and holo:
        raise ValueError("give the ligand either in the structure (--ligand) or in --holo")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for old in ("holo.pdb", "ligand.pdb"):
        (out / old).unlink(missing_ok=True)  # from an earlier writing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = load(str(path))
    lig = None
    if ligand:
        ids = system.select(f"({ligand}) and not hydrogen").ids
        if not len(ids):
            raise ValueError(f"--ligand {ligand!r} selects no heavy atoms")
        lig = np.asarray(system.positions, float)[ids]
    keep = f"protein and {NOT_PROBES}" + (f" and not ({ligand})" if ligand else "")
    if selection:
        keep = f"({selection})" + (f" and not ({ligand})" if ligand else "")
    all_atom = bool(len(system.select("name CA").ids))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if all_atom:
            beads, cg = B.coarse_grain(system, model, keep, with_system=True)
        else:
            ids = protein_ids(system, keep if selection or ligand else None)
            beads, cg = B.from_system(system, model, ids), system.clone(ids)
    moved = None
    if holo:
        from ..pockets.cli import _holo_on

        lig, info, moved = _holo_on(str(holo), system if all_atom else cg, holo_ligand,
                                    holo_top, holo_fit)  # fmt: skip
        lig = np.asarray(lig, float)
        holo_ligand = info["ligand"]
    found, grid = find(beads, model)
    spacing = preset(model)["spacing"]
    residue = np.asarray(cg.atoms["residue"])
    rows, drawn = [], []
    for k, (score, s) in enumerate(found, 1):
        reach = beads.radius + LINING
        d2 = min_dist2(beads.xyz, s.xyz, float(reach.max())).astype(float)
        lining = set(np.unique(residue[d2 <= reach**2]).tolist())
        p = 1 / (1 + np.exp(-score))
        row = {"rank": k, "score": round(score, 3), "p": round(float(p), 3),
               "volume": s.props["n"], "enclosure": round(s.props["enclosure"], 3),
               "exposure": round(s.props["exposure"], 3), "points": len(s.points),
               "residues": _residue_names(cg, lining)}  # fmt: skip
        label = f"{k} p {p:.2f} {s.props['n']:.0f} A3"
        if lig is not None:
            m = _against(s, spacing, lig)
            row.update(m)
            label += f"{' PPc' if m['PPc'] else ''}{' MOc' if m['MOc'] else ''} LVC {m['LVC']:.2f}"
        rows.append(row)
        if k <= top:
            drawn.append((k, label, s.centre))
    with open(out / "pockets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["rank"])
        w.writeheader()
        w.writerows(rows)

    fmt = "mae" if cg.nbonds else "pdb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save(cg, out / f"beads.{fmt}")
        if all_atom:
            shown = f"({keep})" + (f" or ({ligand})" if ligand else "")
            save(system.select(shown).clone(), out / "structure.pdb")
            if ligand:
                save(system.select(ligand).clone(), out / "ligand.pdb")
        if moved is not None:
            save(moved.select(f"protein and not ({holo_ligand})").clone(), out / "holo.pdb")
            save(moved.select(holo_ligand).clone(), out / "ligand.pdb")
    BV.write_pqr(out / "sites.pqr", [(k, found[k - 1][1].xyz, spacing / 2) for k, *_ in drawn])
    lines = [f"# boonza sitemap: sites of {Path(path).name} as {model} beads ({len(found)} "
             f"sites, the best {len(drawn)} drawn)", *BV.PREAMBLE]  # fmt: skip
    lines += ["load structure.pdb, structure"] if all_atom else []
    has_ligand = (out / "ligand.pdb").exists()
    lines += ["load ligand.pdb, ligand"] if has_ligand else []
    lines += ["load holo.pdb, holo"] if moved is not None else []
    lines += [f"load beads.{fmt}, beads", "set connect_mode, 1", "load sites.pqr, sites_all",
              "set connect_mode, 0", "hide everything", "bg_color black",
              "set ray_opaque_background, 0"]  # fmt: skip
    if all_atom:
        lines += ["show cartoon, structure", "color wheat, structure",
                  "set cartoon_transparency, 0.45, structure"]  # fmt: skip
    if moved is not None:
        lines += [
            "show cartoon, holo",
            "color lightblue, holo",
            "# click holo in the object panel to compare the folds",
            "disable holo",
        ]
    if has_ligand:
        lines += ["show sticks, ligand and not hydro", "color tv_blue, ligand and elem C",
                  "set stick_radius, 0.3, ligand"]  # fmt: skip
    lines += ["show spheres, beads", "set sphere_scale, 0.35, beads", "color grey70, beads",
              "set sphere_transparency, 0.6, beads", "show lines, beads"]  # fmt: skip
    lines += ["disable beads"] if all_atom else []
    lines += BV.pocket_object_lines("sites_all", drawn)
    lines += [f"set sphere_scale, 0.5, pocket_{k}" for k, *_ in drawn]
    names = " ".join(f"pocket_{k}" for k, *_ in drawn)
    focus = "structure" if all_atom else "beads"
    lines += [f"zoom {focus}, 5", "scene overview, store"]
    for k, *_ in drawn:
        lines += [f"disable {names}", f"enable pocket_{k}", f"zoom pocket_{k}, 10",
                  f"scene pocket_{k}, store"]  # fmt: skip
    lines += ["scene overview, recall", "set scene_buttons, 1",
              "# commands: 'overview' (the view as it opened), 'only <rank>' (that site alone)",
              "python", "from pymol import cmd", f"_ranks = {[k for k, *_ in drawn]}",
              "def overview():", "    cmd.scene('overview', 'recall')",
              "def only(rank):", "    cmd.disable(' '.join(f'pocket_{j}' for j in _ranks))",
              "    cmd.enable(f'pocket_{int(rank)}')", "    cmd.zoom(f'pocket_{int(rank)}', 10)",
              "cmd.extend('overview', overview)", "cmd.extend('only', only)",
              "python end"]  # fmt: skip
    view = out / "view.pml"
    view.write_text("\n".join(lines) + "\n")
    del grid
    return view
