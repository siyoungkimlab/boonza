"""OpenMM XML force fields read by boonza vs OpenMM's own ForceField.createSystem.

Each system is parameterized twice, natively and by OpenMM (converted with
from_openmm), and every table must agree.
"""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza import ffxml

app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")
DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _have(xml):
    return (Path(app.__file__).parent / "data" / xml).exists()


def _openmm(xmls, pdb="1TEN.pdb", water=None, **kw):
    """Structure prepared by OpenMM, and OpenMM's own parameterization of it."""
    for x in xmls:
        if not _have(x):
            pytest.skip(f"OpenMM has no {x}")
    p = app.PDBFile(str(DATA / pdb))
    model = app.Modeller(p.topology, p.positions)
    model.deleteWater()
    model.delete([r for r in model.topology.residues()
                  if not {"N", "CA", "C"} <= {a.name for a in r.atoms()}])  # fmt: skip
    ff = app.ForceField(*xmls)
    model.addHydrogens(ff)
    if water:
        model.addSolvent(
            ff, model=water, padding=0.3 * unit.nanometer, ionicStrength=0.15 * unit.molar
        )
    args = {"nonbondedMethod": app.NoCutoff, "constraints": None, "rigidWater": False,
            "removeCMMotion": False, **kw}  # fmt: skip
    omm = ff.createSystem(model.topology, **args)
    return boonza.from_openmm(model.topology, omm, model.positions)


@pytest.mark.parametrize(("xmls", "pdb"), [
    (["amber14-all.xml"], "1TEN.pdb"),
    (["amber19-all.xml"], "1TEN.pdb"),  # residue-specific CMAP; lipid21's 1-4 scale script
    (["charmm36_2024.xml"], "1LYZ.pdb"),  # patches, disulfides, script impropers, NBFIX, UB
])  # fmt: skip
def test_proteins_match_createsystem(xmls, pdb):
    ref = _openmm(xmls, pdb)
    ours = ffxml.parameterize_openmm(ref, xmls, rigid_water=False)
    assert not boonza.diff(ours, ref)


@pytest.mark.parametrize(("xmls", "box"), [
    (["amber14-all.xml", "amber14/tip4pew.xml"], "tip4pew"),  # a virtual site per water
    (["amber19-all.xml", "amber19/opc.xml"], "tip4pew"),
])  # fmt: skip
def test_waters_ions_and_virtual_sites(xmls, box):
    ref = _openmm(xmls, water=box)
    real = np.flatnonzero(ref.atoms["anum"] > 0)
    ours = ffxml.parameterize_openmm(ref.clone(real), xmls, rigid_water=False)
    assert ours.natoms == ref.natoms  # the missing extra particles were added
    # the sites are placed from the force field's weights; OpenMM's come from its water box
    assert not boonza.diff(ours, ref, positions=False)
    # with H-bond constraints and rigid water, the constraints must agree too
    ref = _openmm(xmls, water=box, constraints=app.HBonds, rigidWater=True)
    ours = ffxml.parameterize_openmm(ref.clone(np.flatnonzero(ref.atoms["anum"] > 0)), xmls,
                                     constraints="hbonds")  # fmt: skip
    tables = [n for n in ref.table_names if n.startswith(("constraint", "virtual"))]
    assert "constraint_hoh" in tables
    assert not boonza.diff(
        ours, ref, positions=False, tables=[*tables, "nonbonded", "pair_12_6_es"]
    )
    a, b = boonza.openmm_energies(ref), boonza.openmm_energies(ours)
    assert b["total"] == pytest.approx(a["total"], abs=1e-3)


def test_load_and_errors(tmp_path):
    ff = ffxml.load_openmm_forcefield("amber14-all.xml")
    assert "templates" in repr(ff) and ff.torsions.improper
    s = boonza.from_openmm(app.PDBFile(str(DATA / "1TEN.pdb")).topology)
    with pytest.raises(ffxml.FFXMLError, match="No template found for residue"):
        ffxml.parameterize_openmm(s, ff)  # the crystal structure has no hydrogens
    bad = tmp_path / "script.xml"
    bad.write_text("<ForceField><Script>print('hello')</Script></ForceField>")
    with pytest.raises(ffxml.FFXMLError, match="Script"):
        ffxml.load_openmm_forcefield(str(bad))
    with pytest.raises(ffxml.FFXMLError, match="Could not locate"):
        ffxml.load_openmm_forcefield("no-such-file.xml")


def _chain4():
    s = boonza.System("t")
    r = s.add_residue(s.add_chain())
    for k in range(4):
        s.add_atom(r, name=f"C{k}", anum=6, pos=(k, 0.3 * k * k, 0.0))
    for i in range(3):
        s.add_bond(i, i + 1)
    return s


def test_canonical_forms_are_energy_equivalent():
    a, b = _chain4(), _chain4()
    t = a.add_table_from_schema("dihedral_trig")  # 2 + 2 cos(2 phi - 180)
    t.add_term([0, 1, 2, 3], t.params.add_param(phi0=180.0, fc0=2.0, fc2=2.0))
    t = b.add_table_from_schema("dihedral_trig")  # the same, reversed, as 2 - 2 cos(2 phi)
    t.add_term([3, 2, 1, 0], t.params.add_param(phi0=0.0, fc0=2.0, fc2=-2.0))
    t = a.add_table_from_schema("pair_12_6_es")  # 1-4 electrostatics and LJ as two terms ...
    t.add_term([0, 3], t.params.add_param(qij=0.1))
    t.add_term([0, 3], t.params.add_param(aij=1.0, bij=2.0))
    t = b.add_table_from_schema("pair_12_6_es")  # ... or one
    t.add_term([3, 0], t.params.add_param(aij=1.0, bij=2.0, qij=0.1))
    assert boonza.diff(a, b)
    assert not boonza.diff(a, b, canonical=True)
    c = _chain4()
    t = c.add_table_from_schema("dihedral_trig")
    t.add_term([0, 1, 2, 3], t.params.add_param(phi0=0.0, fc0=2.0, fc2=2.0))  # a real difference
    assert boonza.diff(a, c, canonical=True)


def test_viparr_and_openmm_compared_table_by_table():
    """Method 2: the same structure through viparr ff19SB and OpenMM's amber19 XML."""
    if not _have("amber19-all.xml"):
        pytest.skip("OpenMM has no amber19-all.xml")
    p = app.PDBFile(str(DATA / "1TEN.pdb"))
    model = app.Modeller(p.topology, p.positions)
    model.deleteWater()
    model.delete([r for r in model.topology.residues()
                  if not {"N", "CA", "C"} <= {a.name for a in r.atoms()}])  # fmt: skip
    model.addHydrogens(app.ForceField("amber19-all.xml"))
    s = boonza.from_openmm(model.topology, None, model.positions)
    ff = boonza.load_forcefield("aa.amber.ff19SB")
    ff.rules.es_scale = [0.0, 0.0, 1 / 1.2]  # viparr's file rounds 1/1.2 to 0.8333
    vp = boonza.parameterize(s, [ff], constraints=False)
    om = boonza.parameterize_openmm(s, ["amber19-all.xml"], rigid_water=False)
    kinds = {(d.kind, d.message.split(":")[0])
             for d in boonza.diff(vp, om, positions=False, canonical=True)}  # fmt: skip
    # what remains is real: two terminal charges rounded differently (and the 1-4 pairs of
    # those atoms), and impropers whose equivalent atoms the programs list in another order
    assert kinds == {("atoms", "charge differs for 2 atoms"), ("terms", "dihedral_fourier"),
                     ("terms", "pair_12_6_es")}  # fmt: skip
    cv, co = boonza.canonical_forcefield(vp), boonza.canonical_forcefield(om)
    odd = set(np.flatnonzero(~np.isclose(cv.atoms["charge"], co.atoms["charge"])).tolist())

    def pairs(x):
        t = x.table("pair_12_6_es")
        vals = np.column_stack([t.values(c) for c in ("aij", "bij", "qij")])
        return {tuple(sorted(a)): v for a, v in zip(t.atoms.tolist(), vals, strict=True)}

    pv, po = pairs(cv), pairs(co)
    assert pv.keys() == po.keys()
    changed = [k for k in pv if not np.allclose(pv[k], po[k], rtol=1e-6, atol=1e-9)]
    assert changed and all(k[0] in odd or k[1] in odd for k in changed)
    a, b = boonza.openmm_energies(vp), boonza.openmm_energies(om)
    assert abs(a["total"] - b["total"]) < 0.05


def test_charmm_script_impropers_ignore_the_structures_names():
    """CHARMM36's script keys impropers by the matched templates' atom names."""
    from boonza._columns import STR

    ref = _openmm(["charmm36_2024.xml"], "1LYZ.pdb")
    s = ref.clone(structure_only=True)
    renamed = s.copy()
    renamed.atoms["name"] = np.array([f"X{i}" for i in range(s.natoms)], dtype=STR)
    renamed.residues["name"] = np.array(["RES"] * s.nresidues, dtype=STR)
    a = ffxml.parameterize_openmm(s, ["charmm36_2024.xml"], rigid_water=False)
    b = ffxml.parameterize_openmm(renamed, ["charmm36_2024.xml"], rigid_water=False)
    assert len(b.table("improper_harm")) == len(a.table("improper_harm")) > 300
    names = ("atoms: name", "residues: residue name")
    assert not [d for d in boonza.diff(a, b) if not str(d).startswith(names)]
