"""boonza fep prep --rem: a ligand pose that clashes with the receptor, relieved
by a soft minimization of the ligand alone, its stereochemistry held and the
receptor left exactly as it was (boonza.fep.relief)."""

import csv

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import boonza
from boonza.fep import cli, relief

openmm = pytest.importorskip("openmm")
from test_fep_pipeline import DATA, _amber_tools  # noqa: E402

needs_amber = pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")


def _mol(smiles, name, seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    m.SetProp("_Name", name)
    return m


def _cip(m) -> dict:
    m = Chem.Mol(m)
    Chem.AssignStereochemistryFrom3D(m)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}


def test_held_torsions():
    m = _mol("C/C=C/[C@H](F)Cl", "x")
    torsions = relief.held_torsions(m)
    centre = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 4
                  and any(n.GetSymbol() == "F" for n in a.GetNeighbors()))  # fmt: skip
    double = next(b for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    assert len(torsions) == 2
    assert any(t[1] == centre for t in torsions)  # about the stereocentre
    pair = {double.GetBeginAtomIdx(), double.GetEndAtomIdx()}
    assert any({t[1], t[2]} == pair for t in torsions)  # across the double bond
    assert relief.held_torsions(_mol("c1ccccc1", "benzene")) == []  # nothing to hold


def _clashing(tmp_path, mols, depth=1.2):
    """A charged-termini tripeptide slid against the first ligand until their closest
    heavy atoms are ``depth`` Å apart (a substituent into a side chain), the
    ligands' SDF, and the peptide's PDB."""
    from boonza.viewer.edit import protonate

    sdf = tmp_path / "ligands.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for m in mols:
            w.write(m)
    s = boonza.peptide("GAG")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions) - np.asarray(s.positions).mean(0)
    lig = mols[0].GetConformer().GetPositions()
    lig = lig[[a.GetAtomicNum() > 1 for a in mols[0].GetAtoms()]]
    heavy = np.asarray(s.atoms["anum"]) > 1
    far, near = 20.0, 0.0  # along x from the ligand's centre: closest-distance bisection
    for _ in range(40):
        mid = 0.5 * (far + near)
        at = pos + lig.mean(0) + np.array([mid, 0.0, 0.0])
        d = np.sqrt(((at[heavy][:, None] - lig[None]) ** 2).sum(-1)).min()
        far, near = (mid, near) if d > depth else (far, mid)
    s.positions = pos + lig.mean(0) + np.array([far, 0.0, 0.0])
    boonza.save(s, tmp_path / "receptor.pdb")
    return sdf, tmp_path / "receptor.pdb"


@needs_amber
def test_a_clash_is_relieved_and_the_receptor_kept(tmp_path):
    from boonza import gaff, viparr
    from boonza.fep import prep as fep_prep
    from boonza.fep.cli import FEP_DEFAULTS
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields, load_input

    mol = _mol("C[C@H](O)c1ccccc1", "phenylethanol")
    sdf, pdb = _clashing(tmp_path, [mol])
    args = parse_arguments([str(pdb), "--workdir", str(tmp_path / "w"), "--platform", "CPU"],
                           build_parser("boonza fep prep"))  # fmt: skip
    for k, v in FEP_DEFAULTS.items():
        setattr(args, k, v)
    mols, ligs, names = fep_prep.load_ligands(sdf)
    receptor = load_input(str(pdb), lambda *_: None)
    before = np.asarray(receptor.positions, float).copy()
    assert relief.clashes(receptor, ligs, names)[0][1] < 1.5  # heavy atoms all but on top
    _, ff = forcefields(args)
    both = receptor.copy()
    both.append(ligs[0])
    patch = gaff.gaff2_patch(both, ff, charge_geometry="reference")
    host = gaff.host_index(ff)
    ff[host] = viparr.merge_forcefields(ff[host], patch)
    stereo = _cip(mols[0])
    rows = relief.relieve(receptor, ligs, mols, names, ff, args, log=lambda *_: None)
    (row,) = rows
    assert row["closest_before"] < 1.5 and row["closest_after"] > 2.0
    assert row["moved_rmsd"] < 0.6  # the clash moved out of; the pose, tethered, kept
    assert relief.clashes(receptor, ligs, names) == []
    assert np.array_equal(np.asarray(receptor.positions, float), before)  # the receptor untouched
    assert _cip(mols[0]) == stereo and stereo  # the stereocentre held, R stays R
    assert np.allclose(mols[0].GetConformer().GetPositions(), ligs[0].positions)


@needs_amber
def test_prep_rem_writes_what_it_did(tmp_path, capsys):
    # posed on top of each other, as a series is (the mapping reads their poses)
    supplier = Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False)
    by_name = {m.GetProp("_Name"): m for m in supplier}
    mols = [by_name["benzene"], by_name["toluene"]]
    sdf, pdb = _clashing(tmp_path, mols)
    work = tmp_path / "work"
    common = ["--windows", "3", "--repeats", "1", "--platform", "CPU"]
    assert cli.main(["prep", str(pdb), "--ligands", str(sdf), "--workdir", str(tmp_path / "plain"),
                     *common]) == 0  # fmt: skip
    assert "--rem relieves them first" in capsys.readouterr().out  # without it: named only
    assert cli.main(["prep", str(pdb), "--ligands", str(sdf), "--workdir", str(work), "--rem",
                     *common]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "REM: relieving each ligand's pose" in out and "benzene: closest heavy atom" in out
    rows = list(csv.DictReader((work / "rem.csv").open()))
    assert [r["ligand"] for r in rows] == ["benzene", "toluene"]
    assert all(float(r["closest_after"]) > float(r["closest_before"]) for r in rows)
    relieved = list(Chem.SDMolSupplier(str(work / "ligands_rem.sdf"), removeHs=False))
    assert [m.GetProp("_Name") for m in relieved] == ["benzene", "toluene"]
    assert (work / "edges").is_dir() and (work / "commands.txt").exists()
