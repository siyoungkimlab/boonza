"""Several structures in one MAE file: written by save_mae, read by iter_mae."""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file

import boonza
from boonza.io import iter_mae, load_mae, save_mae

HERE = Path(__file__).parent


def _two_cts(shift=0.0):
    """A structure of two cts, each with a title and properties of its own."""
    s = boonza.load(HERE / "data" / "1LYZ.pdb")
    lig = boonza.load(msys_file("ww.dms")).clone(structure_only=True)
    s.append(lig)
    assert s.ncts == 2
    s.positions = s.positions + shift
    s.ct(0).name, s.ct(1).name = "receptor", "ligand"
    s._ct_props[0].update({"source": "1LYZ", "shared": 1})
    s._ct_props[1].update({"docking_score": -7.25 - shift, "shared": 2, "note": "a b"})
    return s


def _same_cts(a, b):
    assert a.ncts == b.ncts
    assert a._ct_names == b._ct_names
    for k in range(a.ncts):
        np.testing.assert_array_equal(a.ct_atoms(k), b.ct_atoms(k))
        for key, value in b._ct_props[k].items():
            assert a._ct_props[k][key] == value, key
        assert not {"boonza_ct_index", "boonza_ct_count"} & set(a._ct_props[k])
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-5)
    assert a.atoms["name"].tolist() == b.atoms["name"].tolist()
    assert a.nbonds == b.nbonds


@pytest.mark.parametrize("merge", [True, False])
def test_structures_round_trip(tmp_path, merge):
    """Three structures of two cts each come back as three, cts, titles and
    properties intact, whether each was written as one ct or as two."""
    src = [_two_cts(shift) for shift in (0.0, 1.0, 2.0)]
    out = tmp_path / "poses.maegz"
    save_mae(src, out, merge_cts=merge)
    back = list(iter_mae(out))
    assert len(back) == 3
    for a, b in zip(back, src, strict=True):
        _same_cts(a, b)
    # load_mae still puts everything into one System
    whole = load_mae(out)
    assert whole.ncts == 6 and whole.natoms == sum(s.natoms for s in src)


def test_merged_is_one_ct_per_structure(tmp_path):
    """Merged, the file has one f_m_ct block per structure, which carries the
    first ct's title and the properties of both cts."""
    out = tmp_path / "poses.mae"
    save_mae([_two_cts(), _two_cts(1.0)], out)
    text = out.read_text()
    assert text.count("f_m_ct {") == 2
    from boonza.io._maeparse import parse_mae

    block = parse_mae(text)[0]
    assert block["m_title"] == "receptor"
    assert block["source"] == "1LYZ" and block["docking_score"] == -7.25
    assert block["shared"] == 1  # the first ct's, where the two disagree


def test_property_added_to_a_merged_ct(tmp_path):
    """A property another program puts on the merged ct lands on the first ct."""
    out = tmp_path / "one.mae"
    save_mae([_two_cts()], out)
    text = out.read_text().replace("  s_m_title\n", "  s_m_title\n  r_i_glide_gscore\n", 1)
    text = text.replace("  receptor\n", "  receptor\n  -9.5\n", 1)
    out.write_text(text)
    (back,) = iter_mae(out)
    assert back._ct_props[0]["i_glide_gscore"] == -9.5
    assert "i_glide_gscore" not in back._ct_props[1]


def test_one_system_keeps_its_cts(tmp_path):
    """One System saved the plain way is still one block per ct, and iter_mae
    gives it back as one structure."""
    src = _two_cts()
    out = tmp_path / "complex.mae"
    boonza.save(src, out)
    assert out.read_text().count("f_m_ct {") == 2
    (back,) = iter_mae(out)
    _same_cts(back, src)
    _same_cts(boonza.load(out), src)


def test_unmarked_cts_are_structures(tmp_path):
    """A file from elsewhere has no marks: every ct is a structure."""
    out = tmp_path / "library.mae"
    save_mae([_two_cts().clone(_two_cts().ct_atoms(k)) for k in (0, 1)], out)
    names = [s.ct(0).name for s in iter_mae(out)]
    assert names == ["receptor", "ligand"]


def test_broken_mark_warns(tmp_path):
    """Half of a marked structure (the file was filtered) is read on its own."""
    src = _two_cts()
    out = tmp_path / "half.mae"
    save_mae(src.clone(src.ct_atoms(0)), out)
    text = out.read_text().replace("  s_m_title\n", "  s_m_title\n  i_boonza_ct_index\n"
                                   "  i_boonza_ct_count\n", 1)  # fmt: skip
    out.write_text(text.replace("  receptor\n", "  receptor\n  0\n  2\n", 1))
    with pytest.warns(UserWarning, match="structure of its own"):
        (back,) = iter_mae(out)
    assert back.ncts == 1 and back.natoms == len(src.ct_atoms(0))


def test_force_field_survives_a_merge(tmp_path):
    """Two parameterized cts merged into one keep their terms and their cts."""
    one = boonza.load(msys_file("cdk2-ligand-Amber14EHT.dms"))
    src = one.clone()
    src.append(one)
    assert src.ncts == 2 * one.ncts
    out = tmp_path / "ff.mae"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_mae([src], out)
        (back,) = iter_mae(out)
    assert back.ncts == src.ncts
    for name in ("stretch_harm", "angle_harm", "nonbonded"):
        assert len(back.table(name)) == len(src.table(name)), name
    for k in range(src.ncts):
        assert len(back.ct_atoms(k)) == len(src.ct_atoms(k))


def test_cms_components_stay_together():
    """A CMS is one structure: the full_system ct and its components."""
    path = msys_file("t2-original.cms.gz")
    whole = load_mae(path, structure_only=True)
    (back,) = iter_mae(path, structure_only=True)
    assert back.ncts == whole.ncts > 1 and back.natoms == whole.natoms


@pytest.mark.parametrize("name", ["out.maegz", "out.mae.gz", "out.cmsgz"])
def test_maegz_is_gzip(tmp_path, name):
    """.maegz was written as plain text under a compressed file's name."""
    src = boonza.load(HERE / "data" / "1LYZ.pdb")
    out = tmp_path / name
    boonza.save(src, out)
    assert out.read_bytes()[:2] == b"\x1f\x8b"
    with gzip.open(out, "rt") as f:
        assert "f_m_ct" in f.read()
    assert boonza.load(out).natoms == src.natoms


def test_view_opens_a_model_per_structure(tmp_path):
    """boonza view: an MAE of several structures is a model for each, grouped
    under the file; one structure of two cts is one model."""
    from boonza import viewer

    poses = tmp_path / "poses.maegz"
    save_mae([_two_cts(shift) for shift in (0.0, 1.0, 2.0)], poses)
    v = viewer.from_paths([poses])
    assert [m.name for m in v.models] == ["receptor"] * 3
    assert {m.group for m in v.models} == {"poses.maegz"}
    one = _two_cts()
    for k in range(3):
        assert v.session(k).system.natoms == one.natoms and v.session(k).system.ncts == 2
    assert v.session(2).system.positions[0, 0] == pytest.approx(one.positions[0, 0] + 2.0)

    single = tmp_path / "complex.mae"
    boonza.save(one, single)
    v = viewer.from_paths([single])
    assert len(v.models) == 1 and v.models[0].group == "files"
    assert v.session(0).system.natoms == one.natoms
    # opened from the page, into a view that is already up
    assert len(v.open(poses)) == 3 and len(v.models) == 4


@pytest.mark.parametrize("merge", [True, False])
def test_each_structure_keeps_its_box(tmp_path, merge):
    """Simulation boxes, a different (triclinic) one on each structure."""
    src = []
    for k in range(3):
        s = _two_cts(float(k))
        s.cell[:] = [[40.0 + k, 0, 0], [3.0 * k, 50.0 + k, 0], [1.5, -2.0 * k, 60.0 + k]]
        src.append(s)
    out = tmp_path / "frames.maegz"
    save_mae(src, out, merge_cts=merge)
    back = list(iter_mae(out))
    assert len(back) == 3
    for a, b in zip(back, src, strict=True):
        np.testing.assert_allclose(a.cell, b.cell, atol=1e-6)
        assert a.ncts == 2


def test_crystal_cell_of_a_maestro_file(tmp_path):
    """Maestro keeps a PDB's CRYST1 cell in properties of its own, not as a box:
    they are read as the cell, and written back as they were, without a box."""
    src = boonza.load(HERE / "data" / "1LYZ.pdb")
    cell = src.cell.copy()
    assert cell.any()
    src.cell[:] = 0
    crystal = {"a": 79.1, "b": 79.1, "c": 37.9, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    src._ct_props[0].update({f"pdb_PDB_CRYST1_{k}": v for k, v in crystal.items()})
    out = tmp_path / "maestro.mae"
    save_mae([src, src], out)
    assert "chorus_box" not in out.read_text()
    back = list(iter_mae(out))
    assert len(back) == 2
    np.testing.assert_allclose(back[0].cell, cell, atol=1e-3)
    again = tmp_path / "again.mae"
    save_mae(back, again)
    text = again.read_text()
    assert "chorus_box" not in text and "r_pdb_PDB_CRYST1_gamma" in text
    np.testing.assert_allclose(next(iter_mae(again)).cell, cell, atol=1e-3)
    # a cell that is no longer the crystal's is a box, and is what is read
    back[0].cell[:] = np.diag([50.0, 60.0, 70.0])
    save_mae(back[0], again)
    assert "chorus_box" in again.read_text()
    np.testing.assert_allclose(boonza.load(again).cell, np.diag([50.0, 60.0, 70.0]))
