"""Lennard-Jones pair parameters of the coarse-grained force fields boonza parameterizes with.

SiteMap's contact test puts a probe particle on a grid point and sums its
Lennard-Jones energy with the protein.  A bead's own sigma and epsilon do not
describe that: Martini 2 and 3 write their nonbonded terms per *pair* of types
(NBFIX; every per-type sigma and epsilon in a parameterized system is 0), and
SIRAH writes per-type terms, combined arithmetically for sigma and
geometrically for epsilon, with pair overrides on top.  A probe type also need
not occur in the protein, whose parameterized system holds only the pairs of
the types it has.  So the whole pair table is read here from the force-field
files boonza ships and parameterizes from:

  martini2   data/martini/params/martini_v2.2.itp      [ nonbond_params ], C6 and C12
  martini3   data/martini/params/martini_v3.0.0.itp    [ nonbond_params ], sigma and epsilon
  sirah      data/sirah/sirah_x2.2.zip ffnonbonded.itp [ atomtypes ] + [ nonbond_params ]

in boonza's units: A and kcal/mol (the files are in nm and kJ/mol).
tests/test_forcefield.py checks the table against boonza's parameterized
systems pair by pair.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

KJ = 1 / 4.184  # kcal per kJ
NM = 10.0  # A per nm


def _boonza_data() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _sections(text: str, name: str) -> list[list[str]]:
    keep, out = False, []
    for line in text.splitlines():
        bare = line.split(";")[0].strip()
        if bare.startswith("["):
            keep = bare.strip("[] \t") == name
            continue
        if keep and bare:
            out.append(bare.split())
    return out


@dataclass
class PairTable:
    """sigma (A) and epsilon (kcal/mol) for every pair of bead types of one force field."""

    model: str
    types: list[str]
    sigma: np.ndarray  #: (ntypes, ntypes)
    epsilon: np.ndarray  #: (ntypes, ntypes)

    def index(self, names) -> np.ndarray:
        """The table index of each type name (KeyError for a type the force field lacks)."""
        where = {t: k for k, t in enumerate(self.types)}
        return np.array([where[str(n).strip()] for n in names], np.int64)

    def pair(self, a: str, b: str) -> tuple[float, float]:
        i, j = self.index([a, b])
        return float(self.sigma[i, j]), float(self.epsilon[i, j])

    def radius(self, names) -> np.ndarray:
        """Each type's own size: sigma/2 of its self-pair, A (the radius boonza's
        particle_radii and data/cg_radii use)."""
        k = self.index(names)
        return self.sigma[k, k] / 2


def _from_pairs(model: str, pairs: dict) -> PairTable:
    types = sorted({t for ab in pairs for t in ab})
    k = {t: i for i, t in enumerate(types)}
    sig = np.full((len(types),) * 2, np.nan)
    eps = np.full((len(types),) * 2, np.nan)
    for (a, b), (s, e) in pairs.items():
        sig[k[a], k[b]] = sig[k[b], k[a]] = s
        eps[k[a], k[b]] = eps[k[b], k[a]] = e
    return PairTable(model, types, sig, eps)


def _martini(path: Path, c6c12: bool) -> dict:
    pairs = {}
    for row in _sections(path.read_text(), "nonbond_params"):
        if len(row) < 5:
            continue
        a, b, x, y = row[0], row[1], float(row[3]), float(row[4])
        if c6c12:  # C6, C12 (kJ nm^6, kJ nm^12) -> sigma, epsilon
            s, e = ((y / x) ** (1 / 6), x * x / (4 * y)) if x > 0 and y > 0 else (0.0, 0.0)
        else:
            s, e = x, y
        pairs[(a, b)] = (s * NM, e * KJ)
    return pairs


def _sirah() -> dict:
    with zipfile.ZipFile(_boonza_data() / "sirah" / "sirah_x2.2.zip") as z:
        text = io.TextIOWrapper(z.open("ffnonbonded.itp"), "utf-8").read()
    own = {}
    for row in _sections(text, "atomtypes"):
        if len(row) >= 6:
            own[row[0]] = (float(row[4]), float(row[5]))
    pairs = {}
    names = sorted(own)
    for i, a in enumerate(names):  # combining rule 2: arithmetic sigma, geometric epsilon
        for b in names[i:]:
            pairs[(a, b)] = ((own[a][0] + own[b][0]) / 2 * NM,
                             float(np.sqrt(own[a][1] * own[b][1])) * KJ)  # fmt: skip
    for row in _sections(text, "nonbond_params"):
        if len(row) >= 5:
            pairs[(row[0], row[1])] = (float(row[3]) * NM, float(row[4]) * KJ)
    return pairs


@cache
def pair_table(model: str) -> PairTable:
    """The pair table of ``model`` (martini2, martini3 or sirah)."""
    params = _boonza_data() / "martini" / "params"
    if model == "martini2":
        return _from_pairs(model, _martini(params / "martini_v2.2.itp", c6c12=True))
    if model == "martini3":
        return _from_pairs(model, _martini(params / "martini_v3.0.0.itp", c6c12=False))
    if model == "sirah":
        return _from_pairs(model, _sirah())
    raise ValueError(f"unknown model {model!r}: martini2, martini3 or sirah")
