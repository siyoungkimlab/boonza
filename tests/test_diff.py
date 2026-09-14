import numpy as np
import pytest
from conftest import msys_file

import boonza
from boonza.diff import diff


@pytest.fixture(scope="module")
def ww():
    return boonza.load(msys_file("ww.dms"))


def _kinds(diffs):
    return {d.kind for d in diffs}


def test_identical_and_roundtrip(ww, tmp_path):
    assert diff(ww, ww.copy()) == []
    path = tmp_path / "ww.dms"
    boonza.save(ww, path)
    assert diff(ww, boonza.load(path)) == []


def test_reordered_atoms_with_map(ww):
    other = ww.copy()
    order = np.random.default_rng(0).permutation(other.natoms)
    amap = other.reorder_atoms(order)
    assert diff(ww, other, atom_map=amap) == []
    unmapped = diff(ww, other)
    assert "atoms" in _kinds(unmapped) and "terms" in _kinds(unmapped)
    with pytest.raises(ValueError):
        diff(ww, other, atom_map=np.zeros(ww.natoms, np.int64))


def test_term_order_and_param_sharing_do_not_matter(ww):
    other = ww.copy()
    old = other.tables["stretch_harm"]
    atoms, r0, fc = old.atoms[::-1, ::-1].copy(), old.values("r0")[::-1], old.values("fc")[::-1]
    constrained = old.values("constrained")[::-1]
    other.del_table("stretch_harm")
    new = other.add_table_from_schema("stretch_harm")
    pids = new.params.add_params(len(atoms), r0=r0, fc=fc)
    new.add_terms(atoms, pids, constrained=constrained)
    assert diff(ww, other) == []


def test_reports_changes(ww):
    other = ww.copy()
    other.tables["stretch_harm"][5]["fc"] = other.tables["stretch_harm"][5]["fc"] * 1.1
    other.tables["angle_harm"].delete_terms([0])
    other.atoms["charge"] = other.atoms["charge"] + np.where(np.arange(other.natoms) == 7, 0.1, 0)
    other.bond(3).order = 2
    found = diff(ww, other)
    text = "\n".join(map(str, found))
    assert "stretch_harm: 1 of" in text and "fc=" in text
    assert "angle_harm: 1 terms only in the first system" in text
    assert "charge differs for 1 atoms: 7:" in text
    assert "order differs for 1 bonds" in text
    # sharing: editing one term copied its param, other terms keep theirs
    assert sum(d.message.startswith("stretch_harm") for d in found) == 1


def test_positions_flag_and_counts(ww):
    other = ww.copy()
    other.positions = other.positions + 0.5
    assert [d.kind for d in diff(ww, other)] == ["atoms"]
    assert diff(ww, other, positions=False) == []
    small = ww.clone(ww.select("protein").ids)
    kinds = _kinds(diff(ww, small))
    assert {"atoms", "residues"} <= kinds
