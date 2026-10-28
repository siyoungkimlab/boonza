"""boonza fep prep -> run -> result, end to end, on relative hydration free
energies (no receptor) with tiny settings: the plumbing, not the physics."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from boonza.fep import cli
from boonza.fep.run import run_leg

DATA = Path(__file__).parent / "data" / "fep"


def _amber_tools() -> bool:
    from boonza.gaff import AmberToolsError, find_amber_tools

    try:
        find_amber_tools()
    except AmberToolsError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")


def test_prep_run_result(tmp_path, capsys):
    sdf = tmp_path / "two.sdf"
    w = Chem.SDWriter(str(sdf))
    for m in Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False):
        if m.GetProp("_Name") in ("benzene", "toluene"):
            w.write(m)
    w.close()
    work = tmp_path / "work"
    assert cli.main(["prep", "--ligands", str(sdf), "--workdir", str(work), "--windows", "3",
                     "--repeats", "2", "--iteration-ps", "0.02", "--equilibration-ns", "0.00002",
                     "--production-ns", "0.0001"]) == 0  # fmt: skip
    leg = work / "edges" / "benzene~toluene" / "solvent"
    meta = json.loads((leg / "leg.json").read_text())
    assert meta["lambdas"] == [0.0, 0.5, 1.0] and meta["repeats"] == 2
    assert meta["timestep_fs"] == 4.0 and meta["core"] == 11
    edge = work / "edges" / "benzene~toluene"
    assert not (edge / "complex").exists() and (edge / "vacuum").is_dir()  # hydration
    lines = (work / "commands.txt").read_text().splitlines()
    assert len(lines) == 4 and lines[0].startswith("boonza fep run ")
    # each repeat runs, and a run started again carries on (here: nothing left)
    for d in (leg, edge / "vacuum"):
        for r in (1, 2):
            run_leg(d, r, platform="CPU", minimize=20, log=lambda *_: None)
    before = np.load(leg / "run_1.npz")["iteration"]
    run_leg(leg, 1, platform="CPU", minimize=20, log=lambda *_: None)
    assert np.load(leg / "run_1.npz")["iteration"] == before == 1 + 5
    capsys.readouterr()
    assert cli.main(["result", str(work)]) == 0
    table = capsys.readouterr().out
    assert "benzene~toluene" in table and "ΔΔG" in table
    rows = list(csv.DictReader((work / "results_edges.csv").open()))
    assert len(rows) == 1 and np.isfinite(float(rows[0]["ddG"])) and rows[0]["repeats"] == "2"
    assert rows[0]["kind"] == "hydration"
    ligs = list(csv.DictReader((work / "results_ligands.csv").open()))
    assert {x["ligand"] for x in ligs} == {"benzene", "toluene"}
    assert sum(float(x["dG"]) for x in ligs) == pytest.approx(0, abs=1e-6)


def test_prep_refuses_what_it_cannot_run(tmp_path, capsys):
    assert cli.main(["prep", "--workdir", str(tmp_path / "w")]) == 1
    assert "--ligands" in capsys.readouterr().err
    assert cli.main(["prep", "--ligands", "x.sdf", "--model", "martini3"]) == 1
    assert "all-atom" in capsys.readouterr().err
    assert cli.main([]) == 2
