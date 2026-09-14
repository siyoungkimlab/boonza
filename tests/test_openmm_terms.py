"""Energies of the OpenMM translations that the Amber system does not exercise.

Each system is a chain of atoms with exactly known dihedral angles, so the
expected energy follows from the DMS functional form alone.  Phases other
than 0/180 degrees and an asymmetric CMAP grid catch sign, origin and axis
mistakes that symmetric parameters would hide.
"""

import numpy as np
import pytest
from conftest import run_msys

import boonza
from boonza.omm import from_openmm, openmm_energies, to_openmm

pytest.importorskip("openmm")

SIZE = 24  # CMAP grid points per axis (15 degree spacing)


def _chain(dihedrals, bond=1.5, angle=110.0):
    """Positions of a chain whose successive dihedrals are ``dihedrals`` (degrees)."""
    th = np.radians(angle)
    pos = [np.zeros(3), np.array([bond, 0.0, 0.0])]
    pos.append(pos[1] + bond * np.array([-np.cos(th), np.sin(th), 0.0]))
    for phi in dihedrals:
        a, b, c = pos[-3], pos[-2], pos[-1]
        bc = (c - b) / np.linalg.norm(c - b)
        n = np.cross(b - a, bc)
        n /= np.linalg.norm(n)
        m = np.cross(n, bc)
        p = np.radians(phi)
        pos.append(c - bond * np.cos(th) * bc + bond * np.sin(th) * (np.cos(p) * m + np.sin(p) * n))
    return np.array(pos)


def _wrap(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def _system(pos):
    s = boonza.System()
    s.add_atoms(len(pos), name=[f"C{i}" for i in range(len(pos))], anum=6, mass=12.0, pos=pos)
    return s


def _dihedral(pos, i):
    return float(np.degrees(boonza.pbc.dihedrals(*pos[i : i + 4])[0]))


def test_chain_builder_gives_iupac_dihedrals():
    pos = _chain([-60.0, 135.0, 10.0])
    for k, want in enumerate([-60.0, 135.0, 10.0]):
        assert _wrap(_dihedral(pos, k) - want) == pytest.approx(0.0, abs=1e-9)


def _grid():
    ang = -180.0 + 360.0 / SIZE * np.arange(SIZE)
    phi, psi = np.meshgrid(ang, ang, indexing="ij")  # psi varies fastest, as DMS requires
    p, q = np.radians(phi), np.radians(psi)
    energy = np.cos(p) + 0.5 * np.sin(2 * q) + 0.3 * np.sin(p) * np.cos(q) + 0.2 * np.sin(p - 2 * q)
    return ang, phi.ravel(), psi.ravel(), energy


def _cmap_system(phi, psi):
    s = _system(_chain([phi, psi]))
    ang, gphi, gpsi, energy = _grid()
    aux = boonza.ParamTable()
    for prop in ("phi", "psi", "energy"):
        aux.add_prop(prop, float)
    aux.add_params(len(gphi), phi=gphi, psi=gpsi, energy=energy.ravel())
    s.aux_tables["cmap1"] = aux
    t = s.add_table_from_schema("torsiontorsion_cmap")
    t.add_term([0, 1, 2, 3, 1, 2, 3, 4], t.params.add_param(cmapid="cmap1"))
    return s, energy, ang


@pytest.mark.parametrize("phi,psi", [(-180, -180), (-60, -45), (-120, 135), (60, 30),
                                     (165, -75), (0, 90)])  # fmt: skip
def test_cmap_equals_grid_at_nodes(phi, psi):
    s, energy, ang = _cmap_system(phi, psi)
    assert _wrap(_dihedral(s.positions, 0) - phi) == pytest.approx(0.0, abs=1e-9)
    assert _wrap(_dihedral(s.positions, 1) - psi) == pytest.approx(0.0, abs=1e-9)
    i = int(np.argmin(np.abs(_wrap(ang - phi))))
    j = int(np.argmin(np.abs(_wrap(ang - psi))))
    got = openmm_energies(s)["torsiontorsion_cmap"]
    assert got == pytest.approx(energy[i, j], abs=1e-6)


def test_grid_catches_axis_and_sign_mistakes():
    energy = _grid()[3]
    flipped = np.roll(energy[::-1, ::-1], 1, axis=(0, 1))  # E(-phi, -psi) on the same nodes
    assert np.abs(energy - energy.T).max() > 0.5
    assert np.abs(energy - flipped).max() > 0.5
    assert np.abs(energy - np.roll(energy, SIZE // 2, axis=(0, 1))).max() > 0.5  # 180° origin


def test_cmap_roundtrips(tmp_path):
    s, _, _ = _cmap_system(-63.0, -41.0)  # between grid nodes
    ref = openmm_energies(s)["torsiontorsion_cmap"]
    path = tmp_path / "cmap.dms"
    boonza.save(s, path)
    assert openmm_energies(boonza.load(path))["torsiontorsion_cmap"] == pytest.approx(ref, rel=1e-9)
    canon = run_msys("load", path)
    assert "torsiontorsion_cmap" in canon["tables"] and "cmap1" in canon["aux"]
    top, omm, pos = to_openmm(s)
    back = from_openmm(top, omm, pos)
    assert openmm_energies(back)["torsiontorsion_cmap"] == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("phi,phi0", [(-20.0, 30.0), (-175.0, 170.0), (100.0, -90.0),
                                      (45.0, 45.0)])  # fmt: skip
def test_improper_harm(phi, phi0):
    s = _system(_chain([phi]))
    t = s.add_table_from_schema("improper_harm")
    t.add_term([0, 1, 2, 3], t.params.add_param(phi0=phi0, fc=12.5))
    want = 12.5 * np.radians(_wrap(phi - phi0)) ** 2
    assert openmm_energies(s)["improper_harm"] == pytest.approx(want, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("phi", [-150.0, -30.0, 20.0, 110.0])
def test_dihedral_trig_phase_and_constant(phi):
    s = _system(_chain([phi]))
    t = s.add_table_from_schema("dihedral_trig")
    fc = {"fc0": 0.7, "fc1": 1.1, "fc2": 0.0, "fc3": 0.45, "fc4": 0.0, "fc5": 0.0, "fc6": 0.2}
    t.add_term([0, 1, 2, 3], t.params.add_param(phi0=35.0, **fc))
    p, p0 = np.radians(phi), np.radians(35.0)
    want = fc["fc0"] + sum(fc[f"fc{n}"] * np.cos(n * p - p0) for n in range(1, 7))
    e = openmm_energies(s)
    got = e["dihedral_trig"] + e.get("dihedral_trig_constant", 0.0)
    assert got == pytest.approx(want, rel=1e-6)


def test_posre_harm():
    s = _system(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    t = s.add_table_from_schema("posre_harm")
    t.add_term([0], t.params.add_param(fcx=2.0, fcy=3.0, fcz=4.0), x0=0.5, y0=2.5, z0=2.0)
    t.add_term([1], t.params.add_param(fcx=1.0, fcy=1.0, fcz=1.0), x0=4.0, y0=5.0, z0=6.0)
    want = 0.5 * (2.0 * 0.5**2 + 3.0 * 0.5**2 + 4.0 * 1.0**2)
    assert openmm_energies(s)["posre_harm"] == pytest.approx(want, rel=1e-9)
