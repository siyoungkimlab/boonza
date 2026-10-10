"""Each model's tuned site finding: parameters, probe and SiteScore weights.

data/sitemap/presets.json holds, per model, the best trial of the search on
the static training set (the cgsitemap benchmark: 200 prepared holo complexes of
train263, less every protein of the validation sets and the complexes whose site
the file leaves incomplete; its export_presets.py writes this file): the
site-finding parameters, the contact probe by name, and the SiteScore terms
(sqrt n, enclosure, philic) with their coefficients fitted on all 200.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

PRESETS = Path(__file__).resolve().parents[1] / "data" / "sitemap" / "presets.json"


@cache
def presets() -> dict:
    return json.loads(PRESETS.read_text())


def preset(model: str) -> dict:
    try:
        return presets()[model]
    except KeyError:
        raise ValueError(f"no preset for {model!r}: {', '.join(presets())}") from None


def score(props: dict, p: dict) -> float:
    """The preset's SiteScore of a site's properties (sites.properties)."""
    w = p["score"]
    return float(w["intercept"] + w["sqrt_n"] * np.sqrt(props["n"])
                 + w["enclosure"] * props["enclosure"] + w["philic"] * props["philic"])  # fmt: skip


def find(beads, model: str | None = None):
    """The sites of ``beads`` under the preset of their model, best first, as
    ``[(score, Site)]``; only the grid points the preset can use are computed."""
    from .grid import compute
    from .sites import find_sites

    p = preset(model or beads.model)
    params = {**p["params"], "probe": 0}
    g = compute(beads, p["spacing"], (p["probe"],), outside=params["outside"])
    found = [(score(s.props, p), s) for s in find_sites(g, params)]
    return sorted(found, key=lambda x: -x[0]), g
