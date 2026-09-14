import numpy as np
import pytest
from canonical import assert_same, canon
from conftest import msys_file

import boonza
from boonza import StaleHandleError, System


def water_box(n=3):
    s = System("water")
    s.cell = np.diag([10.0, 10.0, 10.0])
    nb = s.add_nonbonded_from_schema("vdw_12_6", "geometric")
    ow = nb.params.add_param(sigma=3.15, epsilon=0.15)
    hw = nb.params.add_param(sigma=0.0, epsilon=0.0)
    st = s.add_table_from_schema("stretch_harm")
    oh = st.params.add_param(r0=0.9572, fc=450.0)
    chain = s.add_chain(name="W")
    for k in range(n):
        r = chain.add_residue("HOH", k + 1)
        o = r.add_atom(name="OW", anum=8, mass=15.999, charge=-0.834, pos=(3.0 * k, 0, 0))
        h1 = r.add_atom(name="HW1", anum=1, mass=1.008, charge=0.417, pos=(3.0 * k + 0.96, 0, 0))
        h2 = r.add_atom(name="HW2", anum=1, mass=1.008, charge=0.417, pos=(3.0 * k - 0.24, 0.93, 0))
        s.add_bonds([[o, h1], [o, h2]])
        nb.add_terms([[o], [h1], [h2]], [ow, hw, hw])
        st.add_terms([[o, h1], [o, h2]], oh)
    return s


def test_hierarchy_and_fragments():
    s = water_box()
    assert (s.natoms, s.nbonds, s.nresidues, s.nchains, s.ncts) == (9, 6, 3, 1, 1)
    assert s.fragids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    a = s.atom(4)
    assert a.residue.resid == 2 and a.residue.name == "HOH"
    assert a.chain.name == "W" and a.ct.id == 0
    assert a.fragment.ids.tolist() == [3, 4, 5]
    assert a.residue.atoms.ids.tolist() == [3, 4, 5]
    assert s.atom(3).bonded_atoms.ids.tolist() == [4, 5]
    assert s.chain(0).residues[2].resid == 3
    assert s.select([0, 7]).expand("fragment").ids.tolist() == [0, 1, 2, 6, 7, 8]
    assert a.element == "H"


def test_add_bond_is_idempotent():
    s = water_box()
    b = s.find_bond(0, 1)
    assert s.add_bond(1, 0) == b
    ids = s.add_bonds([[2, 0], [0, 3], [3, 0], [1, 0]])
    assert ids[0] == s.find_bond(0, 2).id
    assert ids[1] == ids[2] == s.nbonds - 1
    assert ids[3] == b.id
    assert s.nfragments == 2
    with pytest.raises(ValueError):
        s.add_bond(1, 1)


def test_custom_columns_roundtrip(tmp_path):
    s = water_box()
    s.atoms["temperature"] = np.arange(9.0)
    s.atoms["tag"] = np.array(["x"] * 9)
    s.bonds.add_prop("kind", str)
    s.bonds["kind"] = ["oh"] * s.nbonds
    s.atom(2)["temperature"] = 42.0
    assert s.atom(2)["temperature"] == 42.0
    with pytest.raises(ValueError):
        s.atoms["residue"] = 0
    out = tmp_path / "w.dms"
    boonza.save(s, out)
    t = boonza.load(out)
    assert_same(canon(t), canon(s))
    assert t.bonds["kind"].tolist() == ["oh"] * s.nbonds


def test_delete_atoms_updates_everything():
    s = water_box()
    old = s.atom(4)
    amap = s.delete_atoms([1, 3, 4, 5])
    assert amap.tolist() == [0, -1, 1, -1, -1, -1, 2, 3, 4]
    assert (s.natoms, s.nbonds, s.nresidues) == (5, 3, 2)
    assert s.table("nonbonded").atoms[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert s.table("stretch_harm").atoms.tolist() == [[0, 1], [2, 3], [2, 4]]
    assert s.fragids.tolist() == [0, 0, 1, 1, 1]
    with pytest.raises(StaleHandleError):
        _ = old.name
    assert s.atom(1).name == "HW2"


def test_delete_matches_clone_of_rest():
    s = water_box()
    keep = [0, 2, 6, 7, 8]
    clone = s.clone(keep)
    s.delete_atoms([1, 3, 4, 5])
    assert_same(canon(s), canon(clone))


def test_clone_compacts_and_shares_params():
    s = water_box()
    shared = s.add_table("pair", 2, "bond", params=s.table("stretch_harm").params)
    shared.add_term([0, 3], 0)
    c = s.clone([6, 7, 8, 3])
    assert c.natoms == 4
    assert c.table("stretch_harm").atoms.tolist() == [[0, 1], [0, 2]]
    assert c.table("pair").atoms.tolist() == [[3, 0]] or len(c.table("pair")) == 0
    assert c.table("pair").params is c.table("stretch_harm").params
    assert len(c.table("nonbonded").params) == 2
    assert c.residues["resid"].tolist() == [3, 2]


def test_append_offsets_and_merges_params():
    a, b = water_box(2), water_box(3)
    new = a.append(b)
    assert new.tolist() == list(range(6, 15))
    assert (a.natoms, a.nbonds, a.nchains, a.ncts) == (15, 10, 2, 2)
    assert a.nfragments == 5
    st = a.table("stretch_harm")
    assert len(st) == 10 and len(st.params) == 2
    assert st.atoms[4].tolist() == [6, 7]
    assert a.table("nonbonded").param_ids.tolist() == [0, 1, 1] * 2 + [2, 3, 3] * 3


def test_term_edit_copies_shared_param():
    s = water_box(1)
    st = s.table("stretch_harm")
    t0, t1 = st[0], st[1]
    t0["fc"] = 500.0
    assert t0["fc"] == 500.0 and t1["fc"] == 450.0
    assert len(st.params) == 2


def test_reorder_atoms():
    s = water_box(2)
    pos = s.positions.copy()
    s.reorder_atoms(np.arange(6)[::-1])
    assert np.array_equal(s.positions, pos[::-1])
    assert s.table("stretch_harm").atoms.tolist() == [[5, 4], [5, 3], [2, 1], [2, 0]]
    assert s.find_bond(5, 4) is not None
    assert s.residues["resid"].tolist() == [1, 2]
    assert s.atom(0).residue.resid == 2


def test_add_atoms_vectorized():
    s = System()
    r = s.add_residue(name="LIG")
    sel = s.add_atoms(4, r, name=["C1", "C2", "C3", "C4"], anum=6)
    s.add_bonds(np.array([[0, 1], [1, 2], [2, 3]]))
    assert sel.ids.tolist() == [0, 1, 2, 3]
    assert s.nfragments == 1
    assert s.atoms["name"].tolist() == ["C1", "C2", "C3", "C4"]


def test_selection_strings_choose_atoms():
    s = boonza.load(msys_file("ww.dms"))
    want = s.select("protein").ids
    sub = s.clone("protein")
    assert sub.natoms == len(want)
    assert sub.atoms["name"].tolist() == s.atoms["name"][want].tolist()
    c = s.copy()
    amap = c.delete_atoms("water")
    assert c.natoms == s.natoms - len(s.select("water"))
    assert (amap[s.select("water").ids] == -1).all()
    with pytest.raises(TypeError):
        s.delete_residues("protein")
