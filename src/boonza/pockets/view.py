"""PyMOL scripts that show coarse-grained pockets on the all-atom structures.

The beads fpocket searched are hidden: the apo protein is drawn as a cartoon,
a holo ligand (when given) as sticks, and each pocket as its own object,
``pocket_<rank>``, holding its alpha spheres and a label, so one click in the
object panel shows or hides it.

As :func:`boonza.write_viewer_scripts` does, a script names its files relative
to itself and finds itself when it runs, so the directory can be copied off the
machine it was written on.  A ``.pml`` comment holds no semicolon: PyMOL splits
commands on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: pocket colours by rank; the others are grey
RANK_COLORS = ("red", "orange", "yellow", "green", "cyan")
#: a trajectory's view draws the consensus pockets in the top this-many of any ranking
VIEW_TOP = 5

#: run from any directory: PyMOL sets __script__ to the script it is running
PREAMBLE = [
    "python",
    "import os",
    "from pymol import cmd",
    'cmd.cd(os.path.dirname(os.path.abspath(__script__)) if "__script__" in globals() else ".")',
    "python end",
]


def _color(rank: int) -> str:
    return RANK_COLORS[rank - 1] if 1 <= rank <= len(RANK_COLORS) else "grey50"


def pocket_object_lines(source: str, drawn) -> list[str]:
    """PyMOL lines making each pocket one object, ``pocket_<rank>``.

    ``source``: the object the alpha spheres were loaded into, one residue number
    per pocket.  ``drawn``: ``[(rank, label, centre)]``.  Each object holds the
    pocket's alpha spheres (scaled by 0.3, as fpocket's own script draws them,
    coloured by rank) and, as one more atom named LBL at its centre, its label.
    """
    lines = []
    for k, label, (x, y, z) in drawn:
        obj = f"pocket_{k}"
        lines += [f"create {obj}, {source} and resi {k}",
                  f'pseudoatom {obj}, name=LBL, resi={k}, pos=[{x:.3f}, {y:.3f}, {z:.3f}], '
                  f'label="{label}"',
                  f"hide everything, {obj}",  # PyMOL bonds nearby points of a .pqr
                  f"set sphere_scale, 0.3, {obj}", f"set sphere_transparency, 0.2, {obj}",
                  f"show spheres, {obj} and not name LBL", f"color {_color(k)}, {obj}",
                  f"show labels, {obj} and name LBL"]  # fmt: skip
    lines += [f"delete {source}", "set label_size, 18", "set label_color, white",
              "set label_position, (0, 0, 8)"]  # fmt: skip
    return lines


def _structures(where: Path, apo, holo, ligand, fmt: str) -> list[str]:
    from ..io import save

    save(apo, where / f"apo.{fmt}")
    lines = [f"load apo.{fmt}, apo"]
    if holo is not None:
        save(holo.select(f"not ({ligand})").clone(), where / f"holo.{fmt}")
        save(holo.select(f"({ligand})").clone(), where / f"ligand.{fmt}")
        lines += [f"load holo.{fmt}, holo", f"load ligand.{fmt}, ligand"]
    return lines


def _style(holo) -> list[str]:
    lines = [
        "hide everything",
        "bg_color black",
        "set ray_opaque_background, 0",
        "show cartoon, apo",
        "color wheat, apo",
        "set cartoon_transparency, 0.45, apo",
    ]
    if holo is not None:
        lines += ["show cartoon, holo", "color lightblue, holo",
                  "# click holo in the object panel to compare the folds", "disable holo",
                  "show sticks, ligand and not hydro", "color tv_blue, ligand and elem C",
                  "set stick_radius, 0.3, ligand"]  # fmt: skip
    return lines


def write_pqr(path, groups, residue: str = "STP") -> None:
    """Spheres as fpocket writes them: ``groups`` is ``[(number, centres, radii)]``,
    each group one residue, the radius in the last column."""
    lines, k = [], 0
    for number, centres, radii in groups:
        for (x, y, z), r in zip(np.asarray(centres).reshape(-1, 3), np.broadcast_to(
                radii, (len(centres),)), strict=True):  # fmt: skip
            k += 1
            lines.append(f"ATOM  {k % 100000:5d}    C {residue}  {number % 10000:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  0.00 {float(r):6.2f}")  # fmt: skip
    Path(path).write_text("\n".join(lines + ["END"]) + "\n")


def write_run_view(where, pockets, apo, holo=None, ligand: str | None = None,
                   verdicts=None, fmt: str = "pdb") -> Path:  # fmt: skip
    """``view.pml`` for one structure's pockets (``pockets``, fpocket's ranking).

    ``verdicts``: ``{rank: (PPc, MOc)}`` against the holo ligand, shown in the labels.
    """
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    write_pqr(where / "pockets.pqr", [(k, p.centres, p.radii) for k, p in enumerate(pockets, 1)])
    lines = ["# fpocket pockets of a coarse-grained protein, on the all-atom structures",
             *PREAMBLE, *_structures(where, apo, holo, ligand, fmt),
             "load pockets.pqr, pockets_all", *_style(holo)]  # fmt: skip
    verdicts = verdicts or {}
    drawn = []
    for k, p in enumerate(pockets, 1):
        ppc, moc = verdicts.get(k, (False, False))
        drawn.append((k, f"{k}{' PPc' if ppc else ''}{' MOc' if moc else ''}", p.centre))
    lines += ["# each pocket is the object pocket_<rank>: click it in the object panel"]
    lines += pocket_object_lines("pockets_all", drawn)
    focus = next((k for k, v in sorted(verdicts.items()) if v[0]), 1)
    lines += [f"zoom pocket_{focus}, 12" if pockets else "orient apo"]
    view = where / "view.pml"
    view.write_text("\n".join(lines) + "\n")
    return view


def view_selection(ranked, verdicts=None) -> list:
    """The consensus pockets a trajectory's view draws: those in the top ``VIEW_TOP`` of
    any ranking, and the best-ranked (by quality) one correct by PPc and by MOc, so the
    site is never hidden."""
    verdicts = verdicts or {}
    top = {
        q.rank_quality
        for q in ranked
        if min(q.rank_quality, q.rank_persistence, q.rank_quality_burial) <= VIEW_TOP
    }
    for k in (0, 1):
        hits = [q.rank_quality for q in ranked if verdicts.get(q.rank_quality, (0, 0))[k]]
        if hits:
            top.add(min(hits))
    return [q for q in ranked if q.rank_quality in top]


def write_traj_view(
    where,
    ranked,
    apo,
    holo=None,
    ligand: str | None = None,
    verdicts=None,
    fmt: str = "pdb",
    maps: float | None = None,
) -> Path:
    """``view.pml`` for a trajectory's consensus pockets (``ranked``, quality order).

    Each drawn pocket is its alpha spheres in its best frame, labelled with its
    quality rank, ``*`` when cryptic, and PPc / MOc when right for the holo ligand
    (``verdicts``, ``{quality rank: (PPc, MOc)}``); its enclosed core is
    ``core_<rank>``.  ``maps``: the level mdpocket's density was calibrated at, when
    pocket_frequency.dx and pocket_density.dx are beside the view.
    """
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    verdicts = verdicts or {}
    lines = ["# consensus pockets of a coarse-grained trajectory, on the all-atom structures",
             *PREAMBLE, *_structures(where, apo, holo, ligand, fmt)]  # fmt: skip
    if maps is not None:
        lines += ["load pocket_frequency.dx, frequency", "load pocket_density.dx, density"]
    lines += _style(holo)
    if maps is not None:
        lines += ["# mdpocket maps, off by default: click pocket_frequency or pocket_density",
                  "isosurface pocket_frequency, frequency, 0.5",
                  "color red, pocket_frequency", "set transparency, 0.4, pocket_frequency",
                  f"isomesh pocket_density, density, {maps:g}", "color yellow, pocket_density",
                  "disable pocket_frequency", "disable pocket_density"]  # fmt: skip
    drawn = []
    for q in view_selection(ranked, verdicts):
        tag = f"{q.rank_quality}{'*' if q.cryptic else ''}"
        ppc, moc = verdicts.get(q.rank_quality, (False, False))
        drawn.append((q.rank_quality, tag + (" PPc" if ppc else "") + (" MOc" if moc else ""),
                      q.centre))  # fmt: skip
    lines += ["# consensus pockets in the top 5 of any ranking, and the best-ranked one right",
              "# for the holo ligand by PPc and by MOc, each in its best frame.  Label = quality",
              "# rank, * = cryptic (closed in the apo crystal structure).  Every pocket is in",
              "# pockets.csv",
              "load pockets.pqr, consensus_all"]  # fmt: skip
    lines += pocket_object_lines("consensus_all", drawn)
    lines += [
        "# each pocket's enclosed core, the object core_<rank>, one sphere per grid cell",
        "load cores.pqr, cores_all",
    ]
    for k, _, _ in drawn:
        lines += [f"create core_{k}, cores_all and resi {k}", f"hide everything, core_{k}",
                  f"show spheres, core_{k}", f"color {_color(k)}, core_{k}",
                  f"set sphere_transparency, 0.45, core_{k}"]  # fmt: skip
    lines += ["delete cores_all"]
    lines += ["zoom ligand, 10"] if holo is not None else ["orient apo"]
    view = where / "view.pml"
    view.write_text("\n".join(lines) + "\n")
    return view
