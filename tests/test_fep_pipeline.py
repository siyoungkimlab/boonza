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


def _sdf(path, names, props=None, shift=None):
    w = Chem.SDWriter(str(path))
    by_name = {m.GetProp("_Name"): m
               for m in Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"),
                                           removeHs=False)}  # fmt: skip
    for n in names:
        m = Chem.Mol(by_name[n])
        for k, v in (props or {}).get(n, {}).items():
            m.SetProp(k, str(v))
        if shift and n in shift:
            conf = m.GetConformer()
            for i in range(m.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (p.x + shift[n], p.y, p.z))
        w.write(m)
    w.close()
    return path


def _run_all(lines):
    for line in lines:
        leg, repeat = line.split()[3], int(line.split()[-1])
        run_leg(Path(leg), repeat, platform="CPU", minimize=20, log=lambda *_: None)


def test_ligands_added_later_reuse_every_edge_run(tmp_path, capsys):
    tiny = ["--windows", "3", "--repeats", "1", "--iteration-ps", "0.02",
            "--equilibration-ns", "0.00002", "--production-ns", "0.0001"]  # fmt: skip
    work = tmp_path / "work"
    measured = {"benzene": {"dG": -0.9}, "toluene": {"dG": -0.8}, "phenol": {"dG": -6.6}}
    first = _sdf(tmp_path / "first.sdf", ["benzene", "toluene", "phenol"], measured)
    assert cli.main(["prep", "--ligands", str(first), "--workdir", str(work), *tiny]) == 0
    assert (work / "ligands.sdf").exists()  # kept, every field with it
    old = (work / "commands.txt").read_text().splitlines()
    _run_all(old)
    ran = {p: p.read_bytes() for p in (work / "edges").glob("*/*/run_*.npz")}
    # the same ligands (any order) and a new one, without a measured value
    more = _sdf(tmp_path / "more.sdf", ["anisole", "phenol", "toluene", "benzene"], measured)
    capsys.readouterr()
    assert cli.main(["prep", "--ligands", str(more), "--workdir", str(work), "--extend",
                     *tiny]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "adding 1 to the 3 of the first prep: anisole" in out
    names = [r["name"] for r in csv.DictReader((work / "ligands.csv").open())]
    assert names == ["benzene", "toluene", "phenol", "anisole"]  # first indices kept
    new = (work / "commands_new.txt").read_text().splitlines()
    assert new and all("anisole" in line for line in new)
    assert (work / "commands.txt").read_text().splitlines() == old + new
    assert all(p.read_bytes() == b for p, b in ran.items())  # nothing run is touched
    _run_all(new)
    capsys.readouterr()
    assert cli.main(["result", str(work)]) == 0
    out = capsys.readouterr().out
    assert "1 without a measured value, predicted: anisole" in out
    ligs = list(csv.DictReader((work / "results_ligands.csv").open()))
    assert {x["ligand"] for x in ligs} == set(names) and all(x["dG_shifted"] for x in ligs)
    sdf = list(Chem.SDMolSupplier(str(work / "results_ligands.sdf"), removeHs=False))
    assert [m.GetProp("_Name") for m in sdf] == names
    assert all(m.HasProp("FEP_dG") and m.HasProp("FEP_dG_shifted") for m in sdf)
    assert sdf[0].GetProp("dG") == "-0.9"  # the fields it came with, kept
    plotted = (work / "plots" / "dG.csv").read_text().splitlines()
    assert plotted[-1].startswith("anisole,") and plotted[-1].endswith(",,,")
    # refused: other settings, or a first ligand in another pose
    capsys.readouterr()
    again = _sdf(tmp_path / "again.sdf", ["benzene", "toluene", "phenol", "anisole", "styrene"])
    assert cli.main(["prep", "--ligands", str(again), "--workdir", str(work), "--extend",
                     *tiny[:1], "5", *tiny[2:]]) == 1  # fmt: skip
    assert "windows 3 (now 5)" in capsys.readouterr().err
    moved = _sdf(tmp_path / "moved.sdf", ["benzene", "toluene", "phenol", "anisole", "styrene"],
                 shift={"toluene": 0.5})  # fmt: skip
    assert cli.main(["prep", "--ligands", str(moved), "--workdir", str(work), "--extend",
                     *tiny]) == 1  # fmt: skip
    assert "toluene is not in the pose" in capsys.readouterr().err
