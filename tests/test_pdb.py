from pathlib import Path

import numpy as np
import pytest
from canonical import assert_same, canon
from conftest import msys_file, run_msys

import boonza
from boonza import System
from boonza.elements import element_for_abbreviation
from boonza.spatial import pairs_within

FILES = ["1DUF.pdb", "3RYZ.pdb", "alanin.pdb", "arginine.pdb", "h2o.pdb"]


@pytest.mark.parametrize("name", FILES)
def test_load_matches_msys(name):
    path = msys_file(name)  # msys ignores CONECT, SSBOND and LINK records
    ours = boonza.load(path, conect=False, ssbond=False, link=False)
    assert_same(canon(ours), run_msys("load", path))


@pytest.mark.parametrize("name", FILES)
def test_write_roundtrip(name, tmp_path):
    src = boonza.load(msys_file(name), conect=False, ssbond=False, link=False)
    out = tmp_path / "out.pdb"
    boonza.save(src, out, models=True)
    back = boonza.load(out)
    assert_same(canon(back), run_msys("load", out))  # msys reads our file the same way
    ours, ref = canon(back), canon(src)
    np.testing.assert_allclose(ours.pop("cell"), ref.pop("cell"), atol=1e-3)
    assert_same(ours, ref)


def test_ter_splits_reused_chain_ids(tmp_path):
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        "TER",
        "ATOM      2  N   ALA A   1      10.000   0.000   0.000  1.00  0.00           N",
        "ATOM      3  C   ALA B   2      20.000   0.000   0.000  1.00  0.00           C",
        "ATOM      4  C   ALA A   1      30.000   0.000   0.000  1.00  0.00           C",
    ]
    path = tmp_path / "ter.pdb"
    path.write_text("\n".join(lines) + "\n")
    s = boonza.load(path)
    assert s.chains["name"].tolist() == ["A", "A", "B"]
    assert s.atoms["residue"].tolist() == [0, 1, 2, 1]
    out = tmp_path / "out.pdb"
    boonza.save(s, out)
    assert_same(canon(boonza.load(out)), canon(s))


def test_element_lookup():
    assert element_for_abbreviation("C") == 6
    assert element_for_abbreviation("CL") == 17
    assert element_for_abbreviation("cl") == 17
    assert element_for_abbreviation("C1") == 6
    assert element_for_abbreviation("X") == 0
    assert element_for_abbreviation("") == 0


def test_pairs_within_matches_brute_force():
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 20, (500, 3))
    cell = np.diag([20.0, 20.0, 20.0])
    for box in (None, cell):
        i, j, d2 = pairs_within(pos, 3.0, box)
        d = pos[None, :, :] - pos[:, None, :]
        if box is not None:
            d -= np.round(d / 20.0) * 20.0
        full = (d**2).sum(-1)
        bi, bj = np.nonzero(np.triu(full <= 9.0, 1))
        assert (i.tolist(), j.tolist()) == (bi.tolist(), bj.tolist())
        np.testing.assert_allclose(d2, full[bi, bj])


def test_guess_bonds_periodic():
    s = System()
    s.cell = np.diag([10.0, 10.0, 10.0])
    s.add_atoms(2, name=["C1", "C2"], anum=6, pos=[[0.2, 5, 5], [9.6, 5, 5]])
    s.guess_bonds()
    assert s.nbonds == 0
    s.guess_bonds(periodic=True)
    assert s.nbonds == 1


def _bond_set(s):
    return {(int(i), int(j)) for i, j in zip(s.bonds["i"], s.bonds["j"], strict=True)}


def _label(s, a):
    atom = s.atom(int(a))
    return f"{atom.residue.name}{atom.residue.resid}:{atom.name}"


def test_ssbond_and_conect_records():
    data = Path(__file__).parent / "data"
    s = boonza.load(data / "1LYZ.pdb")  # an old structure: 3 of 4 S-S too long to guess
    sg = s.select("name SG").ids
    ss = {(_label(s, i), _label(s, j)) for i, j in _bond_set(s) if i in sg and j in sg}
    assert ss == {("CYS6:SG", "CYS127:SG"), ("CYS30:SG", "CYS115:SG"),
                  ("CYS64:SG", "CYS80:SG"), ("CYS76:SG", "CYS94:SG")}  # fmt: skip
    plain = boonza.load(data / "1LYZ.pdb", conect=False, ssbond=False, link=False)
    assert len(_bond_set(s) - _bond_set(plain)) == 3
    assert len(boonza.load(data / "1LYZ.pdb", conect=False).bonds) == len(s.bonds)  # SSBOND alone

    # CONECT is authoritative among atoms that have records: 1MBN's heme geometry makes
    # the distance rule bond C3D-CBD, which the heme's CONECT records do not list
    m = boonza.load(data / "1MBN.pdb")
    labels = {(_label(m, i), _label(m, j)) for i, j in _bond_set(m)}
    assert ("HEM155:C3D", "HEM155:CBD") not in labels
    assert ("HIS93:NE2", "HEM155:FE") in labels  # the His-Fe link from CONECT


def test_conect_orders_hex_serials_and_symmetry_copies(tmp_path):
    rows = [("C1", 0.0), ("C2", 1.34), ("O3", 2.6), ("S4", 20.0), ("S5", 40.0)]
    ssbond = "SSBOND   {} CYS A    3    CYS A    4                          1555   {}  2.04"
    lines = [ssbond.format(1, "1555"), ssbond.format(2, "3655")]
    for k, (nm, x) in enumerate(rows):
        serial = f"{k + 1:5d}" if k < 3 else f"{0xA0000 + k:05x}"
        res = "LIG" if k < 3 else "CYS"
        resid = 1 if k < 3 else k
        el = nm[0]
        name = "SG" if k >= 3 else nm
        lines.append(f"HETATM{serial} {name:<4s} {res} A{resid:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                     f"  1.00  0.00          {el:>2s}")  # fmt: skip
    lines += ["CONECT    1    2    2", "CONECT    2    1    1    3", "CONECT    3    2", "END"]
    path = tmp_path / "records.pdb"
    path.write_text("\n".join(lines) + "\n")
    s = boonza.load(path)
    got = {
        (_label(s, i), _label(s, j), int(o))
        for i, j, o in zip(s.bonds["i"], s.bonds["j"], s.bonds["order"], strict=True)
    }
    assert got == {("LIG1:C1", "LIG1:C2", 2), ("LIG1:C2", "LIG1:O3", 1),
                   ("CYS3:SG", "CYS4:SG", 1)}  # fmt: skip
    # the second SSBOND (to a symmetry copy) alone would not bond
    path.write_text("\n".join(lines[1:]) + "\n")
    assert len(boonza.load(path).bonds) == 2


def test_writer_one_model_ensembles_and_conect_round_trip(tmp_path):
    data = Path(__file__).parent / "data"
    protein = boonza.load(data / "1MBN.pdb")
    lig = boonza.load(data / "1HHO.pdb").clone("resname HEM and chain A")
    s = protein.copy()
    s.append(lig)  # a second ct
    out = tmp_path / "complex.pdb"
    boonza.save(s, out)
    text = out.read_text()
    assert "MODEL" not in text and "CONECT" in text
    back = boonza.load(out)
    assert back.natoms == s.natoms and back.ncts == 1
    assert _bond_set(back) == _bond_set(s)  # guessed plus CONECT reproduce every bond
    boonza.save(s, out, models=True)
    assert out.read_text().count("MODEL") == 2 and boonza.load(out).ncts == 2
    boonza.save(s, out, conect=False)
    assert "CONECT" not in out.read_text()

    # an ensemble (the same atoms in every ct) keeps its MODELs
    ens = lig.copy()
    moved = lig.copy()
    moved.positions = moved.positions + 1.0
    ens.append(moved)
    boonza.save(ens, out)
    assert out.read_text().count("MODEL") == 2
    again = boonza.load(out)
    assert again.ncts == 2 and _bond_set(again) == _bond_set(ens)
