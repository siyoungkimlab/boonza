"""OpenMM systems built by CHARMM, GROMACS-style and custom-force setups, converted both ways."""

import io
import math

import numpy as np
import pytest
from conftest import run_msys

import boonza
from boonza.omm import KCAL, from_openmm, openmm_energies, to_openmm

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

CATEGORY = {
    "HarmonicBondForce": "bond", "stretch_harm": "bond",
    "HarmonicAngleForce": "angle", "angle_harm": "angle",
    "PeriodicTorsionForce": "torsion", "RBTorsionForce": "torsion",
    "dihedral_trig": "torsion", "dihedral_trig_constant": "torsion",
    "CustomTorsionForce": "improper", "improper_harm": "improper",
    "CMAPTorsionForce": "cmap", "torsiontorsion_cmap": "cmap",
    "CustomExternalForce": "restraint", "posre_harm": "restraint",
    "NonbondedForce": "nonbonded", "CustomNonbondedForce": "nonbonded",
    "CustomBondForce": "nonbonded", "nonbonded": "nonbonded", "nonbonded_vdw": "nonbonded",
    "pair_12_6_es": "nonbonded",
}  # fmt: skip


def _energies(system, positions):
    """kcal/mol per category (and total) with every force in its own group."""
    forces = [f for f in system.getForces() if type(f).__name__ != "CMMotionRemover"]
    for k, f in enumerate(forces):
        f.setForceGroup(k)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx.setPositions(positions)
    ctx.computeVirtualSites()
    out = {"total": 0.0}
    for k, f in enumerate(forces):
        e = ctx.getState(getEnergy=True, groups={k}).getPotentialEnergy()
        e = e.value_in_unit(unit.kilojoule_per_mole) / KCAL
        cat = CATEGORY.get(f.getName(), CATEGORY[type(f).__name__])
        out[cat] = out.get(cat, 0.0) + e
        out["total"] += e
    return out


def _assert_close(ours, ref, rel=1e-7):
    assert ours.keys() == ref.keys()
    for k in ref:
        assert ours[k] == pytest.approx(ref[k], rel=rel, abs=1e-6), k


def _roundtrip(top, system, positions, rel=1e-7):
    ref = _energies(system, positions)
    s = from_openmm(top, system, positions)
    _, ours, pos = to_openmm(s, constraints=False)
    _assert_close(_energies(ours, pos), ref, rel)
    return s, ref


@pytest.fixture(scope="module")
def charmm_peptide():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSequence("ACDKPFWG"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    struct = app.PDBFile(io.StringIO(Chem.MolToPDBBlock(Chem.RemoveHs(mol))))
    ff = app.ForceField("charmm36.xml")
    model = app.Modeller(struct.topology, struct.positions)
    model.addHydrogens(ff)
    system = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                             rigidWater=False, removeCMMotion=False)  # fmt: skip
    return model.topology, system, model.positions


def test_charmm36_roundtrip(charmm_peptide, tmp_path):
    top, system, positions = charmm_peptide
    s, ref = _roundtrip(top, system, positions)
    assert {"improper_harm", "torsiontorsion_cmap", "pair_12_6_es", "stretch_harm"} <= set(s.tables)
    assert len(s.aux_tables) >= 2  # separate maps for glycine/proline and the rest
    ub = s.tables["stretch_harm"]
    assert len(ub) > s.nbonds  # Urey-Bradley 1-3 terms join the stretches
    path = tmp_path / "peptide.dms"
    boonza.save(s, path)
    _, ours, pos = to_openmm(boonza.load(path), constraints=False)
    _assert_close(_energies(ours, pos), ref)
    canon = run_msys("load", path)
    assert "improper_harm" in canon["tables"] and len(canon["aux"]) == len(s.aux_tables)
    # the peptide has no cell, so a "box" problem is expected
    problems = {p.check for p in boonza.validate(boonza.load(path), strict=True)}
    assert problems.isdisjoint({"nonbonded", "mass", "virtual", "knot", "stretch", "charge",
                                "term_topology", "extra_exclusions"})  # fmt: skip


def test_topology_only(charmm_peptide):
    top, _, positions = charmm_peptide
    s = from_openmm(top, positions=positions)
    assert s.natoms == top.getNumAtoms() and s.nbonds == top.getNumBonds() and not s.tables
    assert s.atoms["mass"][s.atoms["anum"] == 6] == pytest.approx(12.011, abs=0.01)
    assert s.residues["name"].tolist()[:2] == ["ALA", "CYS"]


def _chain_topology(n, box):
    top = app.Topology()
    res = top.addResidue("MOL", top.addChain())
    for i in range(n):
        top.addAtom(f"C{i}", app.Element.getBySymbol("C"), res)
    top.setPeriodicBoxVectors([openmm.Vec3(*row) for row in box])
    return top


def _chain_positions(n):
    rng = np.random.default_rng(5)
    zig = np.array([[0.15 * i, 0.12 * (i % 2), 0.0] for i in range(n)])
    return zig + rng.normal(0, 0.03, (n, 3)) + 1.0


def test_gromacs_style_forces():
    n = 6
    box = np.eye(3) * 3.0
    pos = _chain_positions(n)
    top = _chain_topology(n, box)
    system = openmm.System()
    for _ in range(n):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box])

    rb = openmm.RBTorsionForce()
    rb.addTorsion(0, 1, 2, 3, 1.2, -0.4, 0.8, 2.1, -0.3, 0.5)
    rb.addTorsion(1, 2, 3, 4, -0.7, 0.9, 0.0, 1.3, 0.25, -0.1)
    system.addForce(rb)
    imp = openmm.CustomTorsionForce(
        "0.5*k*(thetap-theta0)^2; thetap = step(-(theta-theta0+pi))*2*pi+theta"
        f"+step(theta-theta0-pi)*(-2*pi); pi = {math.pi:.15g}"
    )
    imp.addPerTorsionParameter("theta0")
    imp.addPerTorsionParameter("k")
    imp.addTorsion(2, 3, 4, 5, [0.3, 40.0])
    system.addForce(imp)
    restraint = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    for p in ("k", "x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    restraint.addParticle(0, [500.0, *(pos[0] + 0.02)])
    restraint.addParticle(5, [120.0, *(pos[5] - 0.03)])
    system.addForce(restraint)
    aniso = openmm.CustomExternalForce("0.5*(kx*(x-x0)^2+ky*(y-y0)^2+kz*(z-z0)^2)")
    for p in ("x0", "y0", "z0", "kx", "ky", "kz"):
        aniso.addPerParticleParameter(p)
    aniso.addParticle(2, [*(pos[2] + [0.01, -0.02, 0.03]), 100.0, 200.0, 300.0])
    system.addForce(aniso)

    charges = [0.3, -0.2, 0.1, -0.3, 0.2, -0.1]
    sigmas = [0.30, 0.32, 0.35, 0.30, 0.28, 0.33]
    epsilons = [0.4, 0.3, 0.5, 0.4, 0.2, 0.6]
    nb = openmm.NonbondedForce()
    lj = openmm.CustomNonbondedForce("A1*A2/r^12-C1*C2/r^6")
    lj.addPerParticleParameter("A")
    lj.addPerParticleParameter("C")
    for q, sg, ep in zip(charges, sigmas, epsilons, strict=True):
        nb.addParticle(q, 1.0, 0.0)
        lj.addParticle([math.sqrt(4 * ep * sg**12), math.sqrt(4 * ep * sg**6)])
    for i, j in [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (1, 3), (2, 4), (3, 5)]:
        nb.addException(i, j, 0.0, 1.0, 0.0)
        lj.addExclusion(i, j)
    for i, j in [(0, 3), (1, 4), (2, 5)]:
        nb.addException(i, j, 0.5 * charges[i] * charges[j], 1.0, 0.0)
        lj.addExclusion(i, j)
    system.addForce(nb)
    system.addForce(lj)
    pairs = openmm.CustomBondForce("-C/r^6+A/r^12")
    pairs.addPerBondParameter("C")
    pairs.addPerBondParameter("A")
    for i, j in [(0, 3), (1, 4), (2, 5)]:
        sg, ep = 0.5 * (sigmas[i] + sigmas[j]), 0.5 * math.sqrt(epsilons[i] * epsilons[j])
        pairs.addBond(i, j, [4 * ep * sg**6, 4 * ep * sg**12])
    system.addForce(pairs)

    s, _ = _roundtrip(top, system, pos * unit.nanometer, rel=1e-8)
    assert s.nonbonded_info.vdw_rule == "geometric"
    assert len(s.tables["posre_harm"]) == 3 and len(s.tables["improper_harm"]) == 1
    fc = s.tables["posre_harm"].values("fcx")
    assert fc[0] == pytest.approx(2 * 500.0 * 0.01 / KCAL)


@pytest.mark.parametrize("squared", [False, True])
def test_tabulated_nbfix(squared):
    n = 6
    box = np.eye(3) * 3.0
    pos = _chain_positions(n)
    top = _chain_topology(n, box)
    system = openmm.System()
    for _ in range(n):
        system.addParticle(12.0)
    sig = np.array([0.30, 0.34, 0.28, 0.36])
    eps = np.array([0.40, 0.25, 0.55, 0.30])
    s_ij = 0.5 * (sig[:, None] + sig[None, :])
    e_ij = np.sqrt(eps[:, None] * eps[None, :])
    s_ij[0, 2] = s_ij[2, 0] = 0.31  # NBFIX between types 0 and 2
    e_ij[0, 2] = e_ij[2, 0] = 0.9
    a, b = 4 * e_ij * s_ij**12, 4 * e_ij * s_ij**6
    if squared:  # CHARMM PSF style: sqrt(A) tabulated
        lj = openmm.CustomNonbondedForce("(a/r6)^2-b/r6; r6=r^6; a=acoef(type1, type2); "
                                         "b=bcoef(type1, type2)")  # fmt: skip
        a = np.sqrt(a)
    else:  # OpenMM LennardJonesForce style
        lj = openmm.CustomNonbondedForce("acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;")
    lj.addTabulatedFunction("acoef", openmm.Discrete2DFunction(4, 4, a.T.ravel().tolist()))
    lj.addTabulatedFunction("bcoef", openmm.Discrete2DFunction(4, 4, b.T.ravel().tolist()))
    lj.addPerParticleParameter("type")
    nb = openmm.NonbondedForce()
    for k, t in enumerate([0, 1, 2, 3, 0, 2]):
        lj.addParticle([float(t)])
        nb.addParticle([0.2, -0.3, 0.1, 0.25, -0.15, -0.1][k], 1.0, 0.0)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        nb.addException(i, j, 0.0, 1.0, 0.0)
        lj.addExclusion(i, j)
    system.addForce(nb)
    system.addForce(lj)
    s, _ = _roundtrip(top, system, pos * unit.nanometer, rel=1e-8)
    nbt = s.tables["nonbonded"]
    assert len(nbt.overrides) == 1
    (((p1, p2), values),) = nbt.overrides.items()
    assert values["sigma"] == pytest.approx(3.1) and values["epsilon"] == pytest.approx(0.9 / KCAL)


def test_local_coordinates_lone_pair(tmp_path):
    """A CHARMM-style lone pair (OpenMM LocalCoordinatesSite) becomes DMS virtual_fdat3."""
    top = app.Topology()
    res = top.addResidue("LIG", top.addChain())
    for name, symbol in (("I1", "I"), ("C1", "C"), ("C2", "C"), ("C3", "C")):
        top.addAtom(name, app.Element.getBySymbol(symbol), res)
    top.addAtom("LP1", None, res)
    pos = np.array([[1.0, 1.0, 1.0], [1.2, 1.05, 0.98], [1.3, 1.2, 1.02], [0.8, 1.3, 1.1],
                    [0.0, 0.0, 0.0]])  # fmt: skip
    system = openmm.System()
    for m in (126.9, 12.0, 12.0, 12.0, 0.0):
        system.addParticle(m)
    system.setVirtualSite(4, openmm.LocalCoordinatesSite(
        [0, 1, 2], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, -1.0, 1.0],
        openmm.Vec3(0.15, 0.04, -0.03)))  # fmt: skip
    nb = openmm.NonbondedForce()
    for q in (-0.1, 0.05, 0.05, -0.05, 0.05):
        nb.addParticle(q, 0.3, 0.2)
    for i, j in [(0, 4), (0, 1), (1, 4), (1, 2), (2, 4)]:
        nb.addException(i, j, 0.0, 1.0, 0.0)
    system.addForce(nb)
    s, _ = _roundtrip(top, system, pos * unit.nanometer, rel=1e-9)
    assert "virtual_fdat3" in s.tables
    d, theta, phi = (s.tables["virtual_fdat3"].values(p)[0] for p in ("c1", "c2", "c3"))
    assert d == pytest.approx(10 * math.sqrt(0.15**2 + 0.04**2 + 0.03**2))
    # the site rebuilt from DMS parameters sits where OpenMM put it
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx.setPositions(pos * unit.nanometer)
    ctx.computeVirtualSites()
    want = ctx.getState(getPositions=True).getPositions(asNumpy=True)[4]
    _, back, positions = to_openmm(s)
    ctx2 = openmm.Context(back, openmm.VerletIntegrator(0.001),
                          openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx2.setPositions(positions)
    ctx2.computeVirtualSites()
    got = ctx2.getState(getPositions=True).getPositions(asNumpy=True)[4]
    np.testing.assert_allclose(got.value_in_unit(unit.nanometer),
                               want.value_in_unit(unit.nanometer), atol=1e-12)  # fmt: skip
    path = tmp_path / "lp.dms"
    boonza.save(s, path)
    assert "virtual_fdat3" in run_msys("load", path)["tables"]


def test_boonza_forces_roundtrip():
    """to_openmm output, including the dihedral constant, converts back unchanged."""
    s = boonza.System()
    s.add_atoms(5, name=[f"C{i}" for i in range(5)], anum=6, mass=12.0,
                pos=_chain_positions(5) * 10)  # fmt: skip
    t = s.add_table_from_schema("dihedral_trig")
    t.add_term([0, 1, 2, 3], t.params.add_param(phi0=35.0, fc0=0.7, fc1=1.1, fc2=0.0, fc3=0.45,
                                                fc4=0.0, fc5=0.0, fc6=0.2))  # fmt: skip
    t = s.add_table_from_schema("improper_harm")
    t.add_term([1, 2, 3, 4], t.params.add_param(phi0=20.0, fc=15.0))
    t = s.add_table_from_schema("posre_harm")
    t.add_term([4], t.params.add_param(fcx=1.0, fcy=2.0, fcz=3.0), x0=10.0, y0=11.0, z0=12.0)
    ref = openmm_energies(s)["total"]
    top, system, positions = to_openmm(s)
    back = from_openmm(top, system, positions)
    assert openmm_energies(back)["total"] == pytest.approx(ref, rel=1e-9)


def _charmm_gui_cases():
    import os
    from pathlib import Path

    default = "~/Downloads/charmm-gui-6639204059:~/Downloads/charmm-gui-6874574739"
    cases = []
    for d in os.environ.get("BOONZA_CHARMM_GUI", default).split(":"):
        d = Path(d).expanduser()
        for stem in ("step1_pdbreader", "ligandrm"):
            if (d / f"{stem}.psf").exists() and (d / "toppar.str").exists():
                cases.append(pytest.param(d, stem, id=f"{d.name}-{stem}"))
                break
    return cases or [pytest.param(None, None, marks=pytest.mark.skip("no CHARMM-GUI inputs"))]


@pytest.mark.parametrize("folder,stem", _charmm_gui_cases())
def test_charmm_psf_roundtrip(folder, stem, tmp_path):
    import re

    names = re.findall(r"name\s+(\S+)", (folder / "toppar.str").read_text())
    files = [str(folder / f) for f in names if (folder / f).exists()]
    # ligand topologies and parameters from the CHARMM-GUI ligand reader are not in toppar.str
    files += sorted(str(p) for ext in ("rtf", "prm") for p in folder.glob(f"toppar/lig*.{ext}"))
    params = app.CharmmParameterSet(*files)
    psf = app.CharmmPsfFile(str(folder / f"{stem}.psf"))
    positions = app.CharmmCrdFile(str(folder / f"{stem}.crd")).positions
    system = psf.createSystem(params, nonbondedMethod=app.NoCutoff, constraints=None,
                              rigidWater=False)  # fmt: skip
    s, ref = _roundtrip(psf.topology, system, positions)
    if any(isinstance(f, openmm.CustomNonbondedForce) for f in system.getForces()):
        assert len(s.tables["nonbonded"].overrides) > 0  # NBFIX pairs
    path = tmp_path / f"{stem}.dms"
    boonza.save(s, path)
    _, ours, pos = to_openmm(boonza.load(path), constraints=False)
    _assert_close(_energies(ours, pos), ref)
    canon = run_msys("load", path)
    assert set(canon["tables"]) == set(s.tables)


def test_unknown_custom_force():
    top = _chain_topology(2, np.eye(3) * 3.0)
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    quartic = openmm.CustomBondForce("k*(r-r0)^4")
    quartic.addPerBondParameter("k")
    quartic.addPerBondParameter("r0")
    quartic.addBond(0, 1, [10.0, 0.15])
    system.addForce(quartic)
    with pytest.raises(NotImplementedError, match="12-6/Coulomb"):
        from_openmm(top, system)
    assert "pair_12_6_es" not in from_openmm(top, system, ignore_unknown=True).tables
