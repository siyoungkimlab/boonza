import json

import numpy as np
from conftest import msys_file, run_msys
from test_system import water_box

import boonza
from boonza.glue import Glue, make_whole, optimal_shifts
from boonza.io.pdb import cell_from_lengths_angles


def _broken(system, seed):
    """Scatter every atom by random lattice vectors."""
    rng = np.random.default_rng(seed)
    shifts = rng.integers(-2, 3, (system.natoms, 3))
    return system.positions + shifts @ system.cell


def _bond_lengths(system, pos):
    i, j = system.bonds["i"], system.bonds["j"]
    return np.linalg.norm(pos[i] - pos[j], axis=1)


def _box_system(cell):
    s = water_box(4)
    s.cell = cell
    pos = s.positions.copy()
    pos[:, 1:] += 4.0
    s.positions = pos
    return s


def test_optimal_shifts_cluster_points():
    x = np.array([0.95, 0.05, 0.02, 0.9])
    y = x + optimal_shifts(x)
    assert y.max() - y.min() < 0.2


def test_make_whole_orthorhombic_and_triclinic():
    for cell in (np.diag([12.0, 12.0, 12.0]), cell_from_lengths_angles(12, 13, 14, 75, 85, 105)):
        s = _box_system(cell)
        ref = _bond_lengths(s, s.positions)
        pos = make_whole(s, _broken(s, 1), cell)
        np.testing.assert_allclose(_bond_lengths(s, pos), ref, atol=1e-9)


def test_wrap_into_primary_cell_and_around_center():
    cell = cell_from_lengths_angles(12, 13, 14, 75, 85, 105)
    s = _box_system(cell)
    inv = np.linalg.inv(cell)
    pos, _ = Glue(s)(_broken(s, 2), cell)
    frac = np.array([pos[f.ids].mean(0) for f in s.fragments()]) @ inv
    assert (frac >= 0).all() and (frac < 1).all()
    pos, _ = Glue(s, center="resid 1")(_broken(s, 3), cell)
    c = pos[s.select("resid 1").ids].mean(0) @ inv
    frac = np.array([pos[f.ids].mean(0) for f in s.fragments()]) @ inv
    assert (np.abs(frac - c) <= 0.5 + 1e-9).all()


def test_glue_keeps_group_together():
    cell = np.diag([12.0, 12.0, 12.0])
    s = _box_system(cell)
    pos = s.positions.copy()
    pos[s.select("resid 2").ids] += cell[0] * 0.5 + 0.2  # far apart, across the boundary
    fixed, _ = Glue(s, glue="resid 1 2")(pos, cell)
    c1, c2 = (fixed[s.select(f"resid {r}").ids].mean(0) for r in (1, 2))
    assert np.linalg.norm(c1 - c2) < 0.5 * 12


def test_fit_onto_first_frame():
    s = water_box(4)
    fix = Glue(s, fit="all", whole=None, wrap=False)
    p0, _ = fix(s.positions)
    q, r = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    rot = q if np.linalg.det(q) > 0 else -q
    moved, _ = fix(s.positions @ rot.T + [5.0, 1.0, -3.0])
    np.testing.assert_allclose(moved, p0, atol=1e-9)
    assert fix.rmsd < 1e-9


def test_matches_msys_wrapper(tmp_path):
    path = msys_file("3.dms")
    s = boonza.load(path)
    pos = _broken(s, 5)
    np.save(tmp_path / "pos.npy", pos)
    ref = np.array(run_msys("wrap", path, tmp_path / "pos.npy", "protein", "[]"))
    ours, _ = Glue(s, center="protein")(pos, s.cell)
    ours -= ours[s.select("protein").ids].mean(0)  # msys moves the center to the origin
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_matches_msys_wrapper_with_glue(tmp_path):
    path = msys_file("3.dms")
    s = boonza.load(path)
    pos = _broken(s, 6)
    np.save(tmp_path / "pos.npy", pos)
    glue = ["protein or index 20000 to 20002"]
    ref = np.array(run_msys("wrap", path, tmp_path / "pos.npy", "protein", json.dumps(glue)))
    ours, _ = Glue(s, glue=glue, center="protein")(pos, s.cell)
    ours -= ours[s.select("protein").ids].mean(0)
    np.testing.assert_allclose(ours, ref, atol=1e-6)
