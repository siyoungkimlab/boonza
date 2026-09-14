"""Topological distances (vs networkx), bare pair energies (vs OpenMM) and DataFrames."""

from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file

import boonza

DATA = Path(__file__).parent / "data"


def _small_protein():
    s = boonza.load(msys_file("ww.dms")).clone("resid 1 to 3 and protein")
    s.cell = np.zeros((3, 3))
    return s


def test_topological_distances_match_networkx():
    nx = pytest.importorskip("networkx")
    s = boonza.load(msys_file("ww.dms")).clone("resid 1 to 6 and protein")
    s.append(boonza.load(msys_file("ww.dms")).clone("water and resid 100"))  # unconnected
    g = nx.Graph()
    g.add_nodes_from(range(s.natoms))
    g.add_edges_from(zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True))
    want = np.full((s.natoms, s.natoms), -1)
    for a, lengths in nx.all_pairs_shortest_path_length(g):
        for b, d in lengths.items():
            want[a, b] = d
    np.testing.assert_array_equal(boonza.topological_distances(s), want)
    sub, tgt = [5, 0, 17], [3, 3, 40, s.natoms - 1]  # any order, repeats
    np.testing.assert_array_equal(boonza.topological_distances(s, sub, tgt),
                                  want[np.ix_(sub, tgt)])  # fmt: skip
    capped = boonza.topological_distances(s, max_distance=3)
    np.testing.assert_array_equal(capped, np.where(want > 3, -1, want))
    assert (want[: s.natoms - 3, -3:] == -1).all()


def test_pair_energies_sum_to_openmm_nonbonded():
    pytest.importorskip("openmm")
    s = _small_protein()
    report = boonza.describe(s, pairs=True)
    assert len(report.pairs) == s.natoms * (s.natoms - 1) // 2
    total = sum(p["energy"] for p in report.pairs)
    e = boonza.openmm_energies(s, nonbonded_method="NoCutoff", constraints=False)
    # Coulomb and 1-4 exceptions in "nonbonded", geometric-rule LJ in "nonbonded_vdw"
    want = e["nonbonded"] + e.get("nonbonded_vdw", 0.0) + e.get("pair_12_6_es", 0.0)
    assert total == pytest.approx(want, rel=1e-9, abs=1e-6)
    one_four = [p for p in report.pairs if p["bonds"] == 3]
    assert one_four and all("aij" in p for p in one_four)  # 1-4 pairs are scaled
    assert all(p["e_vdw"] == 0 and p["e_es"] == 0 for p in report.pairs if p["bonds"] in (1, 2))


def test_report_to_pandas():
    pytest.importorskip("pandas")
    s = _small_protein()
    frames = boonza.describe(s, "resid 1", pairs=True).to_pandas()
    assert {"atoms", "pairs", "stretch_harm"} <= set(frames)
    pairs = frames["pairs"]
    assert {"i", "j", "bonds", "r", "energy"} <= set(pairs.columns)
    stretch = frames["stretch_harm"]
    assert {"atom0", "atom1", "r0", "fc"} <= set(stretch.columns)
    np.testing.assert_allclose(pairs["r"], np.linalg.norm(
        s.positions[pairs["j"]] - s.positions[pairs["i"]], axis=1))  # fmt: skip
