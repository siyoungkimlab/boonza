"""Batch building (add_chains, add_residues, from_arrays) and pandas export."""

from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file

import boonza

DATA = Path(__file__).parent / "data"


def test_add_chains_and_residues_in_batches():
    s = boonza.System()
    ch = s.add_chains(2, name=["A", "B"], segid="PROT")
    res = s.add_residues(4, chain=[ch[0], ch[0], ch[1], ch[1]], name=["ALA", "GLY", "SER", "THR"],
                         resid=[1, 2, 1, 2])  # fmt: skip
    s.add_atoms(8, residue=np.repeat(res, 2), name=["N", "CA"] * 4, anum=[7, 6] * 4)
    assert s.chains["name"].tolist() == ["A", "B"] and s.chains["segid"].tolist() == ["PROT"] * 2
    assert s.residues["chain"].tolist() == [0, 0, 1, 1]
    assert s.residues["resid"].tolist() == [1, 2, 1, 2]
    assert [a.residue.chain.name for a in s.atoms] == ["A"] * 4 + ["B"] * 4
    more = s.add_residues(2, name="HOH")  # into the last chain
    assert more.tolist() == [4, 5] and s.residues["chain"].tolist()[-2:] == [1, 1]
    with pytest.raises(ValueError, match="values"):
        s.add_residues(3, name=["A", "B"])
    with pytest.raises(IndexError):
        s.add_residues(1, chain=[7])


def test_from_arrays_matches_a_loaded_pdb():
    p = boonza.load(DATA / "1HHO.pdb")
    res = p.atoms["residue"]
    chain = p.residues["chain"][res]
    q = boonza.System.from_arrays(
        p.positions, names=p.atoms["name"], anum=p.atoms["anum"],
        resnames=p.residues["name"][res], resids=p.residues["resid"][res],
        chains=p.chains["name"][chain], segids=p.chains["segid"][chain],
        insertions=p.residues["insertion"][res], cell=p.cell,
        bonds=np.column_stack([p.bonds["i"], p.bonds["j"]]), bfactor=p.atoms["bfactor"],
    )  # fmt: skip
    for col in ("name", "anum", "bfactor"):
        np.testing.assert_array_equal(q.atoms[col], p.atoms[col])
    np.testing.assert_array_equal(q.positions, p.positions)
    for col in ("name", "resid", "insertion"):
        np.testing.assert_array_equal(q.residues[col], p.residues[col])
    assert q.nbonds == p.nbonds and np.allclose(q.cell, p.cell)
    # elements from symbols, or guessed from names (NA in residue NA is sodium)
    r = boonza.System.from_arrays(np.zeros((3, 3)), names=["CA", "NA", "O1"],
                                  resnames=["ALA", "NA", "LIG"])  # fmt: skip
    assert r.atoms["anum"].tolist() == [6, 11, 8]
    e = boonza.System.from_arrays(np.zeros((2, 3)), elements=["Ca", "Cl"])
    assert e.atoms["anum"].tolist() == [20, 17] and e.nresidues == 1


def test_to_pandas():
    pd = pytest.importorskip("pandas")
    s = boonza.load(DATA / "1HHO.pdb")
    df = s.to_pandas()
    assert isinstance(df, pd.DataFrame) and len(df) == s.natoms
    assert {"atom", "name", "element", "x", "y", "z", "resid", "resname", "chain",
            "fragid", "bfactor"} <= set(df.columns)  # fmt: skip
    np.testing.assert_array_equal(df[["x", "y", "z"]].to_numpy(), s.positions)
    heme = s.to_pandas("resname HEM and chain A")
    assert set(heme["resname"]) == {"HEM"} and set(heme["chain"]) == {"A"}
    raw = s.atoms.to_pandas()
    assert {"pos_x", "pos_y", "pos_z", "residue"} <= set(raw.columns)
    assert len(s.residues.to_pandas()) == s.nresidues

    ww = boonza.load(msys_file("ww.dms"))
    t = ww.table("stretch_harm")
    terms = t.to_pandas()
    assert len(terms) == t.nterms
    np.testing.assert_array_equal(terms["atom0"], t.atoms[:, 0])
    np.testing.assert_allclose(terms["r0"], t.params["r0"][t.param_ids])
