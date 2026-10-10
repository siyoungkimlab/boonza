"""Each model's tuned site finding: parameters, probe and SiteScore weights.

data/sitemap/presets.json holds, per model, the best trial of the search on
the static training set (the cgsitemap benchmark: 200 prepared holo complexes of
train263, less every protein of the validation sets and the complexes whose site
the file leaves incomplete; its export_presets.py writes this file): the
site-finding parameters, the contact probe by name, and the SiteScore terms
(sqrt n, enclosure, philic) with their coefficients fitted on all 200.

data/sitemap/presets/<name>.json keeps earlier presets under a name, frozen, so
that runs can use them again and compare them with the current ones
(``--preset NAME``); ``name`` can also be the path of a presets file of one's
own (a tuning candidate, say).  Named so far:

  static200   valine's side-chain bead in every model (Martini 2 AC2, Martini 3
              SC3, SIRAH Y4Cv), tuned on the 200 static holo complexes alone
              (2026-10-10), before any fine-tuning on MD frames
              (docs/guide/sitemap_static200.md: data, search, validation)
  dynamic200  static200 fine-tuned on MD frames of the same complexes' runs, their
              ligand removed (5 frames of each of 191 100-ns runs per model;
              docs/guide/sitemap_dynamic200.md)

Each command has its default: ``boonza sitemap traj`` finds a run's sites under
dynamic200 (TRAJ), tuned on frames like the ones it reads, and ``boonza sitemap
structure`` under static200 (STRUCTURE), tuned on crystal structures.
presets.json is what :func:`find` uses when given no name.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

PRESETS = Path(__file__).resolve().parents[1] / "data" / "sitemap" / "presets.json"
NAMED = PRESETS.parent / "presets"
TRAJ = "dynamic200"  #: boonza sitemap traj's default presets
STRUCTURE = "static200"  #: boonza sitemap structure's default presets


def names() -> list[str]:
    """The named presets kept beside the current ones."""
    return sorted(p.stem for p in NAMED.glob("*.json"))


def path(name: str | None = None) -> Path:
    """The file of presets ``name``: the current ones (None), a named set or a path."""
    if name is None:
        return PRESETS
    if (NAMED / f"{name}.json").exists():
        return NAMED / f"{name}.json"
    if Path(name).expanduser().is_file():
        return Path(name).expanduser().resolve()
    raise ValueError(f"no presets {name!r}: neither a file nor one of {', '.join(names())}")


@cache
def presets(name: str | None = None) -> dict:
    return json.loads(path(name).read_text())


def preset(model: str, name: str | None = None) -> dict:
    try:
        return presets(name)[model]
    except KeyError:
        raise ValueError(f"no preset for {model!r}: {', '.join(presets(name))}") from None


def score(props: dict, p: dict) -> float:
    """The preset's SiteScore of a site's properties (sites.properties)."""
    w = p["score"]
    return float(w["intercept"] + w["sqrt_n"] * np.sqrt(props["n"])
                 + w["enclosure"] * props["enclosure"] + w["philic"] * props["philic"])  # fmt: skip


def find(beads, model: str | None = None, name: str | None = None):
    """The sites of ``beads`` under the preset of their model (of presets ``name``), best
    first, as ``[(score, Site)]``; only the grid points the preset can use are computed."""
    from .grid import compute
    from .sites import find_sites

    p = preset(model or beads.model, name)
    params = {**p["params"], "probe": 0}
    g = compute(beads, p["spacing"], (p["probe"],), outside=params["outside"])
    found = [(score(s.props, p), s) for s in find_sites(g, params)]
    return sorted(found, key=lambda x: -x[0]), g
