"""3D structures from text: SMILES (RDKit round trip, geometry) and peptide sequences
(sequence, backbone dihedrals and DSSP read back with boonza's own tools)."""

import numpy as np
import pytest

import boonza

Chem = pytest.importorskip("rdkit.Chem")


def _lengths(s, z1, z2):
    i, j = s.bonds["i"], s.bonds["j"]
    anum = s.atoms["anum"]
    pick = ((anum[i] == z1) & (anum[j] == z2)) | ((anum[i] == z2) & (anum[j] == z1))
    return np.linalg.norm(s.positions[i[pick]] - s.positions[j[pick]], axis=1)


def test_from_smiles_geometry_and_round_trip():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    s = boonza.from_smiles(smiles, seed=7)
    assert s.natoms == 21 and s.nbonds == 21 and s.nresidues == 1
    assert s.residues["name"][0] == "LIG"
    assert len(set(s.atoms["name"].tolist())) == s.natoms  # C1, C2, ..., H1, ...
    np.testing.assert_allclose(_lengths(s, 6, 1), 1.09, atol=0.02)
    assert 1.36 < _lengths(s, 6, 6).min() and _lengths(s, 6, 6).max() < 1.55
    back = Chem.MolToSmiles(Chem.RemoveHs(s.to_rdkit()))
    assert back == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    again = boonza.from_smiles(smiles, seed=7)
    np.testing.assert_allclose(again.positions, s.positions)  # reproducible
    ion = boonza.from_smiles("C[NH3+]", name="MAM")
    assert ion.atoms["formal_charge"].sum() == 1 and ion.residues["name"][0] == "MAM"
    with pytest.raises(ValueError, match="SMILES"):
        boonza.from_smiles("C1CC")


def test_best_of_several_conformers_is_no_worse():
    from rdkit.Chem import AllChem

    smiles = "CCCCCCCCO"

    def energy(s):
        mol = s.to_rdkit()
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return ff.CalcEnergy()

    one = boonza.from_smiles(smiles, seed=3, conformers=1)
    many = boonza.from_smiles(smiles, seed=3, conformers=10)
    assert energy(many) <= energy(one) + 1e-6


@pytest.mark.parametrize(
    ("conformation", "target"), [("helix", (-57, -47)), ("sheet", (-120, 130))]
)
def test_peptide_backbone(conformation, target):
    seq = "ACDEFGHIKLMNQRSTVWY"
    s = boonza.peptide(seq, conformation)
    assert boonza.sequence(s) == seq
    assert s.residues["name"][:3].tolist() == ["ALA", "CYS", "ASP"]
    phi, psi, omega = (x[0] for x in boonza.backbone_dihedrals(s))
    inner = slice(1, len(seq) - 1)
    assert np.abs(((phi[inner] - target[0]) + 180) % 360 - 180).max() < 3.0
    assert np.abs(((psi[inner] - target[1]) + 180) % 360 - 180).max() < 3.0
    assert np.nanmin(np.abs(omega)) > 160  # every peptide bond trans
    codes = boonza.dssp(s)[0]
    if conformation == "helix":
        assert (codes[2:-2] == "H").mean() > 0.8
    else:
        assert "H" not in codes.tolist()
    clashes = boonza.pbc.self_distances(s.positions)
    assert clashes.min() > 0.9  # no overlapping atoms after the MMFF relaxation


def test_peptide_with_proline_and_angles_per_residue():
    s = boonza.peptide("GPPG", [(-75, 145)] * 4)
    phi, psi, _ = (x[0] for x in boonza.backbone_dihedrals(s))
    assert np.abs(((psi[:3] - 145) + 180) % 360 - 180).max() < 3.0
    assert -90 < phi[1] < -50  # proline's ring holds phi near -65
    with pytest.raises(ValueError, match="pairs"):
        boonza.peptide("GPPG", [(-75, 145)] * 3)


def test_cli_build(tmp_path, capsys):
    from boonza.cli import main

    assert main(["build", "--smiles", "c1ccccc1O", "-o", str(tmp_path / "phenol.sdf")]) == 0
    assert boonza.load(tmp_path / "phenol.sdf").natoms == 13
    assert main(["build", "--sequence", "AAAA", "-o", str(tmp_path / "ala.pdb")]) == 0
    assert "residues" in capsys.readouterr().out
