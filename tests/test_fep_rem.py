"""boonza fep prep --rem: a receptor relaxed around ligand poses that clash with
it, the poses held exactly, a soft minimization (REM) and then the force
field's (EM), the pocket's residues free, backbone too, and their stereocentres held
(boonza.fep.relief)."""

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


def _improper(xyz, t) -> float:
    a, b, c, d = (np.asarray(xyz[i], float) for i in t)
    n1, n2 = np.cross(b - a, c - b), np.cross(c - b, d - c)
    return float(np.degrees(np.arctan2(np.dot(np.cross(n1, n2), (c - b) / np.linalg.norm(c - b)),
                                       np.dot(n1, n2))))  # fmt: skip


def test_held_torsions_and_the_pocket():
    s = boonza.peptide("GIT")
    torsions = relief.held_torsions(s, [0, 1, 2])
    name = np.asarray(s.atoms["name"])
    assert len(torsions) == 4  # Gly none; Ile and Thr each CA and CB
    assert sorted(str(name[t[1]]) for t in torsions) == ["CA", "CA", "CB", "CB"]
    free = name[relief.free_atoms(s, [1])]
    assert {"N", "CA", "C", "O", "CB", "CD1"} <= set(free)  # the whole residue
    assert len(free) == np.sum(np.asarray(s.atoms["residue"]) == 1)


def _clashing(tmp_path, mols, depth=1.2):
    """A charged-termini Ala-Lys-Ala slid, Lys's NZ first, against the first ligand
    until their closest heavy atoms are ``depth`` Å apart (a side chain where a
    ligand was placed), the ligands' SDF, and the peptide's PDB."""
    from boonza.viewer.edit import protonate

    sdf = tmp_path / "ligands.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for m in mols:
            w.write(m)
    s = boonza.peptide("AKA")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions) - np.asarray(s.positions).mean(0)
    tip = pos[int(np.flatnonzero(np.asarray(s.atoms["name"]) == "NZ")[0])]
    lead = -tip / np.linalg.norm(tip)  # from NZ back to the peptide's centre
    lig = mols[0].GetConformer().GetPositions()
    lig = lig[[a.GetAtomicNum() > 1 for a in mols[0].GetAtoms()]]
    heavy = np.asarray(s.atoms["anum"]) > 1
    far, near = 20.0, 0.0  # NZ toward the ligand's centre: closest-distance bisection
    for _ in range(40):
        mid = 0.5 * (far + near)
        at = pos + lig.mean(0) + mid * lead
        d = np.sqrt(((at[heavy][:, None] - lig[None]) ** 2).sum(-1)).min()
        far, near = (mid, near) if d > depth else (far, mid)
    s.positions = pos + lig.mean(0) + far * lead
    boonza.save(s, tmp_path / "receptor.pdb")
    return sdf, tmp_path / "receptor.pdb"


@needs_amber
def test_the_receptor_makes_room_and_the_pose_is_kept(tmp_path):
    from boonza import gaff, viparr
    from boonza.fep import prep as fep_prep
    from boonza.fep.cli import FEP_DEFAULTS
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields, load_input

    mols = [_mol("C[C@H](O)c1ccccc1", "phenylethanol"), _mol("Cc1ccccc1", "toluene")]
    sdf, pdb = _clashing(tmp_path, mols)
    args = parse_arguments([str(pdb), "--workdir", str(tmp_path / "w"), "--platform", "CPU"],
                           build_parser("boonza fep prep"))  # fmt: skip
    for k, v in FEP_DEFAULTS.items():
        setattr(args, k, v)
    mols, ligs, names = fep_prep.load_ligands(sdf)
    receptor = load_input(str(pdb), lambda *_: None)
    before = np.asarray(receptor.positions, float).copy()
    poses = [np.asarray(g.positions, float).copy() for g in ligs]
    assert relief.clashes(receptor, ligs, names)[0][1] < 1.5  # heavy atoms all but on top
    _, ff = forcefields(args)
    both = receptor.copy()
    for g in ligs:
        both.append(g)
    patch = gaff.gaff2_patch(both, ff, charge_geometry="reference")
    host = gaff.host_index(ff)
    ff[host] = viparr.merge_forcefields(ff[host], patch)
    residues = relief.pocket(receptor, ligs)
    torsions = relief.held_torsions(receptor, residues)
    lig_rows, res_rows = relief.relieve(receptor, ligs, names, ff, args, log=lambda *_: None)
    after = np.asarray(receptor.positions, float)
    assert lig_rows[0]["closest_before"] < 1.5 and lig_rows[0]["closest_after"] > 2.0
    for g, p in zip(ligs, poses, strict=True):  # the poses exactly as they were
        assert np.array_equal(np.asarray(g.positions, float), p)
    name = np.asarray(receptor.atoms["name"])
    backbone = np.isin(name, ("N", "CA", "C", "O"))
    shift = np.linalg.norm(after[backbone] - before[backbone], axis=1).max()
    assert 0.1 < shift < 1.5  # free within 5 Å: the backbone gives too, a little
    assert res_rows[0]["resname"] == "LYS" and res_rows[0]["moved"] > 0.5  # room made
    for t in torsions:  # every stereocentre's handedness kept
        assert np.sign(_improper(after, t)) == np.sign(_improper(before, t))


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
    assert "REM, then EM" in out and "benzene: closest receptor heavy atom" in out
    rows = list(csv.DictReader((work / "rem.csv").open()))
    assert [r["ligand"] for r in rows] == ["benzene", "toluene"]
    assert all(float(r["closest_after"]) > float(r["closest_before"]) for r in rows)
    moved = list(csv.DictReader((work / "rem_residues.csv").open()))
    assert moved and moved[0]["resname"] == "LYS"
    relaxed = boonza.load(work / "receptor_rem.pdb")  # the input's format
    assert relaxed.natoms == boonza.load(pdb).natoms
    assert np.allclose(boonza.load(work / "receptor.pdb").positions, relaxed.positions,
                       atol=1e-3)  # what the legs were built in, for boonza view  # fmt: skip
    from boonza.fep import prep as fep_prep

    assert fep_prep.receptor_format("x.dms") == ".dms"
    assert fep_prep.receptor_format("x.maegz") == fep_prep.receptor_format("x.gro") == ".mae"
    assert (work / "edges").is_dir() and (work / "commands.txt").exists()
