"""Each model's tuned site finding: grid spacing, ray step, thresholds and score.

data/cavity/presets/<name>.json keeps presets under a name, frozen, so that runs
can use them again (``--preset NAME``); ``name`` can also be the path of a presets
file of one's own (a tuning candidate, say).  There is one so far:

  volume200   SiteMap's thresholds with only outside and enclosure tuned (on MD frames of
              all 200 training runs), rays sampled every 2 A out to 10 A, sites ranked by
              volume alone (score = intercept + sqrt_n sqrt(V), no weights fitted)
              (docs/guide/cavity_volume200.md)

A preset gives ``spacing`` (A), ``ray_step`` (A between ray samples; 1.0 when absent),
the site-finding ``params`` (sites.DEFAULTS for the rest) and ``score``.  Both
``boonza cavity traj`` and ``boonza cavity structure`` use DEFAULT unless told
otherwise, as does :func:`find` given no name.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

NAMED = Path(__file__).resolve().parents[1] / "data" / "cavity" / "presets"
DEFAULT = "volume200"  #: the presets boonza cavity uses unless told otherwise


def names() -> list[str]:
    """The named presets."""
    return sorted(p.stem for p in NAMED.glob("*.json"))


def path(name: str | None = None) -> Path:
    """The file of presets ``name``: DEFAULT's (None), a named set or a path."""
    name = name or DEFAULT
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
    """The preset's score of a site's properties (sites.properties): by volume alone."""
    w = p["score"]
    return float(w["intercept"] + w["sqrt_n"] * np.sqrt(props["n"]))


def find(beads, model: str | None = None, name: str | None = None, times: dict | None = None):
    """The sites of ``beads`` under the preset of their model (of presets ``name``), best
    first, as ``[(score, Site)]``, and the grid; only the grid points the preset can use
    are computed.  ``times``: gets the seconds of the grid and of site finding, and the
    grid's points."""
    from time import perf_counter

    from .grid import compute
    from .sites import find_sites

    p = preset(model or beads.model, name)
    t0 = perf_counter()
    g = compute(beads, p["spacing"], outside=p["params"]["outside"],
                ray_step=p.get("ray_step", 1.0))  # fmt: skip
    t1 = perf_counter()
    found = [(score(s.props, p), s) for s in find_sites(g, p["params"])]
    if times is not None:
        times.update(grid=t1 - t0, sites=perf_counter() - t1, points=len(g.ijk))
    return sorted(found, key=lambda x: -x[0]), g
