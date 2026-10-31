"""boonza fep prep --complexes: a receptor for every ligand (boonza.fep.complexes)."""

import json

import numpy as np
import pytest
from rdkit import Chem
from test_fep_pipeline import DATA, _amber_tools

import boonza
from boonza.fep import cli
from boonza.fep.prep import load_ligands
from boonza.fep.run import run_leg

pytestmark = pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")


def _peptide(near):
    from boonza.viewer.edit import protonate

    s = boonza.peptide("GAG")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions)
    s.positions = pos - pos.mean(0) + near + np.array([8.0, 0.0, 0.0])
    return s


def _complexes(tmp_path):
    """benzene and toluene, each in its own copy of a tripeptide: toluene's turned
    and moved as a whole (superposition takes that out) and its alanine's side
    chain moved 0.4 Å (which it keeps), saved as MAE; and the SDF, its poses off."""
    supplier = Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False)
    by_name = {m.GetProp("_Name"): m for m in supplier}
    sdf = tmp_path / "ligands.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for name in ("benzene", "toluene"):
            m = Chem.Mol(by_name[name])
            conf = m.GetConformer()
            for k in range(m.GetNumAtoms()):  # the SDF's poses are not used
                p = conf.GetAtomPosition(k)
                conf.SetAtomPosition(k, (p.x + 5.0, p.y, p.z))
            w.write(m)
    _, ligs, _ = load_ligands(DATA / "benzene_modifications.sdf")
    names = [m.GetProp("_Name") for m in supplier]
    folder = tmp_path / "complexes"
    folder.mkdir()
    near = np.asarray(ligs[names.index("benzene")].positions).mean(0)
    theta = np.radians(30.0)
    turn = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])  # fmt: skip
    for name in ("benzene", "toluene"):
        prot = _peptide(near)
        if name == "toluene":
            xyz = np.asarray(prot.positions, float)
            name_ = np.asarray(prot.atoms["name"])
            side = np.flatnonzero(np.isin(name_, ["CB", "HB1", "HB2", "HB3"]))
            xyz[side] += [0.0, 0.4, 0.0]  # with its hydrogens: their bonds are constrained
            prot.positions = xyz
        both = prot.copy()
        both.append(ligs[names.index(name)])
        if name == "toluene":
            both.positions = np.asarray(both.positions, float) @ turn.T + [3.0, -2.0, 1.0]
        boonza.save(both, folder / f"{name}.mae")
    return sdf, folder


def test_every_ligand_in_its_own_receptor(tmp_path, capsys):
    sdf, folder = _complexes(tmp_path)
    work = tmp_path / "work"
    assert cli.main(["prep", "--ligands", str(sdf), "--complexes", str(folder),
                     "--workdir", str(work), "--windows", "3", "--repeats", "1",
                     "--iteration-ps", "0.02", "--equilibration-ns", "0.00002",
                     "--production-ns", "0.0001", "--platform", "CPU"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "a receptor for every ligand" in out and "C-alpha RMSD" in out
    assert (work / "receptors" / "benzene.mae").exists()
    benzene = boonza.load(work / "receptors" / "benzene.mae")
    toluene = boonza.load(work / "receptors" / "toluene.mae")
    ca = np.asarray(benzene.atoms["name"]) == "CA"
    diff = np.linalg.norm(np.asarray(toluene.positions) - np.asarray(benzene.positions), axis=1)
    assert diff[ca].max() < 1e-3  # superposed: the turn is gone
    assert diff.max() > 0.3  # the side chain's move is not
    # the poses come from the complexes (superposed), not the SDF's, 5 Å off
    kept = Chem.SDMolSupplier(str(work / "ligands.sdf"), removeHs=False)[0]
    sdf_pose = Chem.SDMolSupplier(str(sdf), removeHs=False)[0].GetConformer().GetPositions()
    assert np.abs(kept.GetConformer().GetPositions() - sdf_pose).max() > 4.0
    leg = next((work / "edges").glob("*/complex"))
    meta = json.loads((leg / "leg.json").read_text())
    state = np.load(leg / "state.npz")
    assert meta["start_b_from"] == 0.5 and "positions_b" in state.files
    a, b = meta["edge"]
    moved = np.linalg.norm(state["positions_b"] - state["positions"], axis=1)
    assert 0.03 < moved.max() < 0.06  # nm: B's alanine, 0.4 Å over, in the same box
    run_leg(leg, 1, platform="CPU", minimize=20, log=lambda *_: None)
    assert (leg / "run_1.npz").exists()


def test_complexes_that_do_not_agree_are_refused(tmp_path, capsys):
    sdf, folder = _complexes(tmp_path)
    other = boonza.load(folder / "toluene.mae")
    keep = np.flatnonzero(np.asarray(other.atoms["name"]) != "HA2")[: other.natoms - 1]
    boonza.save(other.clone(keep), folder / "toluene.mae")  # one protein hydrogen short
    assert cli.main(["prep", "--ligands", str(sdf), "--complexes", str(folder),
                     "--workdir", str(tmp_path / "w"), "--windows", "3"]) == 1  # fmt: skip
    assert "atom for atom" in capsys.readouterr().err
