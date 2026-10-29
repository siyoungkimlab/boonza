"""boonza fep's octanol leg: the box (boonza.fep.octanol) and a water-to-octanol
transfer edge from prep through run to result."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from boonza.fep import cli, octanol
from boonza.fep import prep as fep_prep
from boonza.fep.run import run_leg

DATA = Path(__file__).parent / "data" / "fep"


def _tools() -> bool:
    from boonza.gaff import AmberToolsError, find_amber_tools

    try:
        find_amber_tools()
    except AmberToolsError:
        return False
    return shutil.which("packmol") is not None


needs_tools = pytest.mark.skipif(not _tools(), reason="AmberTools (GAFF2, packmol) not installed")


def test_water_count():
    assert octanol.n_waters(200, wet=False) == 0
    n = octanol.n_waters(200, wet=True)
    assert n == 74 and n / (n + 200) == pytest.approx(octanol.WET_WATER_FRACTION, abs=0.002)


def test_molecule():
    m = octanol.molecule()
    anum = np.asarray(m.atoms["anum"]).tolist()
    assert m.natoms == 27 and anum.count(6) == 8 and anum.count(8) == 1 and anum.count(1) == 18
    assert set(m.residues["name"].tolist()) == {octanol.RESNAME}


@needs_tools
def test_pack():
    box = octanol.pack(octanol.molecule(), octanol._water(), 20, 7)
    assert box.natoms == 20 * 27 + 7 * 3 and len(set(np.asarray(box.fragids).tolist())) == 27
    edge = box.cell[0, 0]
    mass = 20 * 130.23 + 7 * 18.015
    assert mass / 6.02214076e23 / (edge * 1e-8) ** 3 == pytest.approx(octanol.PACK_DENSITY,
                                                                       rel=0.01)  # fmt: skip
    pos = np.asarray(box.positions)
    i, j = np.asarray(box.bonds["i"]), np.asarray(box.bonds["j"])
    assert np.linalg.norm(pos[i] - pos[j], axis=1).max() < 2.0  # molecules whole


def _min_distance(pos, edge) -> float:
    d = pos[:, None, :] - pos[None, :, :]
    d -= edge * np.round(d / edge)
    r = np.sqrt((d**2).sum(-1))
    r[np.diag_indices(len(pos))] = np.inf
    return float(r.min())


@needs_tools
def test_transfer_prep_run_result(tmp_path, monkeypatch, capsys):
    # a small box, briefly equilibrated, and little padding: the plumbing, not the numbers
    monkeypatch.setattr(octanol, "N_OCTANOL", 40)
    monkeypatch.setattr(octanol, "EQUILIBRATION_NS", 0.002)
    monkeypatch.setattr(fep_prep, "SOLVENT_PADDING_NM", 0.75)  # the water box > two cutoffs
    sdf = tmp_path / "two.sdf"
    w = Chem.SDWriter(str(sdf))
    measured = {"benzene": -2.91, "toluene": -3.73}  # ΔG_transfer from log P (kcal/mol)
    for m in Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False):
        if m.GetProp("_Name") in measured:
            m.SetProp("dG", str(measured[m.GetProp("_Name")]))
            w.write(m)
    w.close()
    work = tmp_path / "work"
    assert cli.main(["prep", "--ligands", str(sdf), "--workdir", str(work), "--phase",
                     "wet-octanol", "--windows", "3", "--repeats", "1", "--iteration-ps", "0.02",
                     "--equilibration-ns", "0.00002", "--production-ns", "0.0001",
                     "--platform", "CPU"]) == 0  # fmt: skip
    built = json.loads((work / "octanol" / "box.json").read_text())
    assert built["wet"] and built["octanols"] == 40 and built["waters"] == 15
    edge = work / "edges" / "benzene~toluene"
    assert (edge / "octanol").is_dir() and (edge / "solvent").is_dir()
    assert not (edge / "vacuum").exists()
    meta = json.loads((edge / "octanol" / "leg.json").read_text())
    assert meta["leg"] == "octanol" and meta["phase"] == "wet-octanol"
    assert meta["alchemical_waters"] == []
    # the box once over, one copy of each molecule: none lost at the seam, none doubled
    state = np.load(edge / "octanol" / "state.npz")
    pos, cell = state["positions"], np.diag(state["box"])
    ligands = np.concatenate([state["core"], state["unique_old"], state["unique_new"]])
    environment = np.setdiff1d(np.arange(len(pos)), ligands)  # octanol and water
    assert any(27 * a + 3 * b == len(environment) for a in range(34, 41) for b in range(12, 16))
    # (the hybrid's own dummies sit on each other: only the box's atoms are compared)
    assert _min_distance(pos[environment], cell) > 0.08  # nm
    lines = (work / "commands.txt").read_text().splitlines()
    assert len(lines) == 2 and all("--repeat 1" in x for x in lines)
    for leg in ("solvent", "octanol"):
        run_leg(edge / leg, 1, platform="CPU", minimize=500, log=lambda *_: None)
    capsys.readouterr()
    assert cli.main(["result", str(work)]) == 0
    out = capsys.readouterr().out
    assert "benzene~toluene" in out and "log P" in out
    row = next(csv.DictReader((work / "results_edges.csv").open()))
    assert row["kind"] == "transfer"
    ddg = float(row["ddG"])
    assert ddg == pytest.approx(float(row["dG_octanol"]) - float(row["dG_solvent"]), abs=1e-3)
    rt_ln10 = 0.0019872041 * meta["temperature"] * np.log(10.0)
    assert float(row["dlogP"]) == pytest.approx(-ddg / rt_ln10, abs=1e-3)
    # both periodic legs carry their end states' dispersion difference (toluene's
    # methyl adds Lennard-Jones, so more tail: negative), and result adds it back
    dks = [json.loads((edge / leg / "leg.json").read_text())["dispersion_dK"]
           for leg in ("octanol", "solvent")]  # fmt: skip
    assert all(dk < 0 for dk in dks)
    assert -0.5 < float(row["dispersion"]) < 0.5 and float(row["dispersion"]) != 0.0
    # the measured values prep kept, compared by result: the edge against their difference
    kept = {r["name"]: r for r in csv.DictReader((work / "ligands.csv").open())}
    assert {n: float(kept[n]["exp_dG"]) for n in measured} == measured
    assert "measured: 2 of 2 ligands (dG)" in out
    assert float(row["exp_ddG"]) == pytest.approx(measured["toluene"] - measured["benzene"])
    assert all((work / "plots" / f).exists() for f in ("dG_white.pdf", "dG_black.pdf",
                                                      "ddG_white.pdf", "ddG_black.pdf"))
    # a binding constant is refused for a transfer set, from an SDF given to result as well
    other = tmp_path / "kd.sdf"
    with Chem.SDWriter(str(other)) as w:
        m = next(iter(Chem.SDMolSupplier(str(sdf), removeHs=False)))
        m.ClearProp("dG")
        m.SetProp("KD_nM", "50")
        w.write(m)
    assert cli.main(["result", str(work), "--experimental", str(other)]) == 1


@needs_tools
def test_phase_needs_no_receptor(tmp_path):
    sdf = tmp_path / "one.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(next(iter(Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"),
                                         removeHs=False))))  # fmt: skip
    w.close()
    pdb = tmp_path / "receptor.pdb"  # never read: --phase is refused first
    pdb.write_text("END\n")
    assert cli.main(["prep", str(pdb), "--ligands", str(sdf), "--workdir", str(tmp_path / "w"),
                     "--phase", "octanol"]) == 1  # fmt: skip
