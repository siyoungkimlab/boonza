"""Structure summaries: the facts they state are checked against direct computations."""

import json
from pathlib import Path

import numpy as np
import pytest

import boonza

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def hemoglobin():
    return boonza.load(DATA / "1HHO.pdb")


def test_composition_and_chains(hemoglobin):
    s = hemoglobin
    summary = boonza.summarize(s)
    text = str(summary)
    comp = summary.to_dict()["composition"]
    assert comp["atoms"] == s.natoms and comp["polymer_chains"] == ["A", "B"]
    assert comp["other_molecules"]["HEM"] == 2
    chains = {c["chain"]: c for c in summary.to_dict()["chains"]}
    assert chains["A"]["sequence"] == boonza.sequence(s, chain="A")
    assert chains["A"]["helix"] > 0.6  # hemoglobin is all helix
    assert "## Chains" in text and "## Checks" in text
    json.dumps(summary.to_dict())  # plain data


def test_heme_site(hemoglobin):
    s = hemoglobin
    summary = boonza.summarize(s)
    # in 1HHO the alpha-chain heme is residue 143 (142 is a phosphate)
    heme = next(m for m in summary.to_dict()["molecules"] if m["molecule"] == "HEM A143")
    assert heme["formula"] == "C34FeN4O4"  # no hydrogens in this crystal structure
    closest = heme["neighbors"][0]
    ids = s.select("resname HEM and chain A and noh").ids
    protein = s.select("protein and noh").ids
    d = boonza.pbc.distances(s.positions[ids], s.positions[protein]).min()
    assert closest["distance"] == pytest.approx(d, abs=0.01)
    assert closest["residue"] == "HIS A87"  # the proximal histidine binds the iron
    assert any("HIS A87 NE2" in b for b in heme["bonds_out"])
    assert 0.5 < heme["buried_fraction"] < 1.0
    assert any(c["distance"] <= 3.5 for c in heme["polar_contacts"])


def test_disulfides_and_focus():
    s = boonza.load(DATA / "1LYZ.pdb")
    text = str(boonza.summarize(s))
    assert "4 disulfide bond(s)" in text and "CYS A6-CYS A127" in text
    focus = boonza.summarize(s, focus="resname CYS and resid 6").to_dict()["focus"]
    assert focus["atoms"] == len(s.select("resname CYS and resid 6"))
    assert any(b.startswith("SG to CYS A127 SG") for b in focus["bonds_out"])
    with pytest.raises(ValueError, match="no atoms"):
        boonza.summarize(s, focus="resname XYZ")


def test_clashes_are_reported():
    s = boonza.load(DATA / "1LYZ.pdb")
    water = s.select("water and noh").ids[0]
    target = s.select("protein and name CA and resid 50").ids[0]
    pos = s.positions
    pos[water] = pos[target] + [1.0, 0.0, 0.0]
    s.positions = pos
    summary = boonza.summarize(s)
    clashes = summary.to_dict()["clashes"]
    protein = s.select("protein and noh").ids
    closest = boonza.pbc.distances(pos[[water]], pos[protein]).min()  # the real worst clash
    assert clashes and clashes[0]["distance"] == pytest.approx(closest, abs=0.01)
    assert closest < 1.01 and "clashes" in str(summary)


def test_cli(tmp_path, capsys):
    from boonza.cli import main

    assert main(["summarize", str(DATA / "1HHO.pdb"), "--focus", "resname HEM and chain B"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ") and "## Focus: resname HEM and chain B" in out
    assert main(["summarize", str(DATA / "1HHO.pdb"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["composition"]["polymer_chains"] == ["A", "B"]
    assert np.isfinite(data["composition"]["atoms"])
