"""GAFF2 templates for ligands and covalent adducts (AmberTools tests skip without it)."""

import os
import warnings

import numpy as np
import pytest

import boonza
from boonza import gaff, viparr

try:
    AMBERHOME = gaff.find_amber_tools()
except gaff.AmberToolsError:
    if os.environ.get("BOONZA_REQUIRE_AMBERTOOLS"):  # CI's AmberTools job must not skip
        raise
    AMBERHOME = None
needs_amber = pytest.mark.skipif(AMBERHOME is None, reason="AmberTools is not installed")
METADATA = ("atoms", "residues", "chains", "cts", "bonds")


def _cys_adduct() -> boonza.System:
    """ACE-ALA-CYS-ALA-NME with 3-amino-3-oxopropyl (an acrylamide adduct,
    residue LIG) on the Cys sulfur, all hydrogens, residues contiguous."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smi = ("[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][C@@H:10]"
           "([CH2:11][S:12][CH2:15][CH2:16][C:17](=[O:18])[NH2:19])[C:13](=[O:14])[NH:20]"
           "[C@@H:21]([CH3:22])[C:23](=[O:24])[NH:25][CH3:26]")  # fmt: skip
    spec = ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "ALA 2 N", "ALA 2 CA", "ALA 2 CB", "ALA 2 C",
            "ALA 2 O", "CYS 3 N", "CYS 3 CA", "CYS 3 CB", "CYS 3 SG", "CYS 3 C", "CYS 3 O",
            "LIG 4 C1", "LIG 4 C2", "LIG 4 C3", "LIG 4 O1", "LIG 4 N1", "ALA 5 N", "ALA 5 CA",
            "ALA 5 CB", "ALA 5 C", "ALA 5 O", "NME 6 N", "NME 6 CH3"]  # fmt: skip
    spec = [s.split() for s in spec]
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 13, 14, 20, 21, 22,
             23, 24, 25, 26]  # fmt: skip
    by_map = {m: spec[k] for k, m in enumerate(sorted(order))}
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) >= 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    count: dict = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            res, rid, nm = by_map[maps[a.GetIdx()]]
        else:
            res, rid, heavy = by_map[maps[a.GetNeighbors()[0].GetIdx()]]
            k = count[(rid, heavy)] = count.get((rid, heavy), 0) + 1
            nm = "H" if heavy == "N" and res != "LIG" else f"H{heavy}{k}"
        a.SetMonomerInfo(Chem.AtomPDBResidueInfo(nm, residueName=res, residueNumber=int(rid),
                                                 chainId="A"))  # fmt: skip
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    s = boonza.from_rdkit(mol, name="cys-adduct")
    s.reorder_atoms(np.argsort(s.atoms["residue"], kind="stable"))
    return s


@pytest.fixture(scope="module")
def adduct():
    return _cys_adduct()


def _terms(system, name, amap=None) -> dict:
    """{atom tuple (either direction): numeric parameters} of one table."""
    t = system.table(name)
    cols = sorted(c for c in t.params.props if t.params.prop_type(c) != "str")
    out = {}
    for atoms, pid in zip(t.atoms.tolist(), t.param_ids.tolist(), strict=True):
        if amap is not None:
            if any(a >= len(amap) for a in atoms):
                continue  # a hydrogen closing a cap
            atoms = [amap[a] for a in atoms]
        key = min(tuple(atoms), tuple(reversed(atoms)))
        out[key] = tuple(round(float(t.params[c][pid]), 5) for c in cols)
    return out


# ---- without AmberTools -----------------------------------------------------------


def test_pinned_external_elements_pick_the_template(adduct):
    ff = viparr.load_forcefield("aa.amber.ff14SB")
    cyx = ff.template("CYX")
    sg_ext = next(i for i in cyx._nbrs[cyx.names.index("SG")] if cyx.anum[i] == -1)

    def pinned(name, z):
        return viparr.Template(
            name, cyx.names, cyx.anum, cyx.charge, cyx.btype, cyx.nbtype, cyx.pset,
            cyx.bonds, cyx.impropers, external_anum={sg_ext: z}
        )  # fmt: skip

    cys = adduct.residue_atoms(2).tolist()
    for z, want in ((6, "CYC"), (16, "CYX")):  # bound to a carbon: the pinned copy wins
        both = viparr.ViparrForcefield("t", ff.rules, [*ff.templates, pinned("CYC", z)], ff.params)
        P = viparr._Parameterizer(adduct, [both], False, False, True)
        assert P._find(both, 0, cys)[0].name == want


def test_write_forcefield_round_trip(tmp_path):
    ff = viparr.load_forcefield("aa.amber.ff14SB")
    back = viparr.load_forcefield(viparr.write_forcefield(ff, tmp_path / "ff14SB"))
    assert back.rules == ff.rules
    assert [t.name for t in back.templates] == [t.name for t in ff.templates]
    for a, b in zip(ff.templates, back.templates, strict=True):
        assert (a.names, a.btype, a.nbtype, a.charge, a.bonds, a.impropers) == (
            b.names, b.btype, b.nbtype, b.charge, b.bonds, b.impropers)  # fmt: skip
    assert {k: v for k, v in back.params.items() if v} == {k: v for k, v in ff.params.items() if v}


def test_find_unmatched_groups_the_bound_residue(adduct):
    # the Cys matches CYX by graph, but it is bound to the unmatched ligand
    assert gaff.find_unmatched(adduct, ["aa.amber.ff14SB"]) == [[2, 3]]
    assert gaff.find_unmatched(adduct, []) == [list(range(6))]


def test_missing_amber_tools_is_reported(tmp_path):
    with pytest.raises(gaff.AmberToolsError, match="conda install"):
        gaff.find_amber_tools(tmp_path)


# ---- with AmberTools --------------------------------------------------------------


@needs_amber
def test_free_ligand_reproduces_tleap(tmp_path):
    s = boonza.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")
    b = gaff._Builder(s, [], run=dict(amberhome=AMBERHOME))
    patch = b.patch()
    ((frag, prmtop),) = b.runs
    assert frag == list(range(s.natoms))
    out = boonza.parameterize(s, [patch], constraints=False)
    assert [d for d in boonza.diff(out, prmtop, canonical=True) if d.kind not in METADATA] == []
    assert np.allclose(out.atoms["charge"], prmtop.atoms["charge"], atol=1e-6)
    assert np.allclose(out.atoms["mass"], prmtop.atoms["mass"])
    # written and read back, the patch gives the same force field
    ff = viparr.load_forcefield(viparr.write_forcefield(patch, tmp_path / "ain"))
    again = boonza.parameterize(s, [ff], constraints=False)
    assert [d for d in boonza.diff(out, again, canonical=True) if d.kind not in METADATA] == []


@needs_amber
def test_covalent_cys_keeps_protein_types(adduct):
    host = viparr.load_forcefield("aa.amber.ff14SB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", viparr.ViparrWarning)  # the Cys SG takes the correction
        b = gaff._Builder(adduct, [host], run=dict(amberhome=AMBERHOME))
        patch = b.patch()
    cys = next(t for t in patch.templates if t.name.endswith("_CYS"))
    types = dict(zip(cys.names, cys.btype, strict=False))
    assert [types[n] for n in ("N", "CA", "CB", "C", "O", "H")] == ["N", "CX", "2C", "C", "O", "H"]
    assert types["SG"] == "ss~1"
    assert sorted(cys.external_anum.values()) == [6, 6, 7]
    # protein atoms keep CYS charges; each residue has its formal charge
    parent = host.template("CYS")
    for n in ("N", "CA", "CB", "C", "O"):
        assert cys.charge[cys.names.index(n)] == parent.charge[parent.names.index(n)]
    out = boonza.parameterize(adduct, [viparr.merge_forcefields(host, patch)])
    q, res = out.atoms["charge"], out.atoms["residue"]
    assert np.allclose([q[res == r].sum() for r in range(out.nresidues)], 0, atol=1e-6)
    # every term that touches a GAFF2 atom has tleap's parameters
    ((frag, prmtop),) = b.runs
    gaff_atoms = {a for a, t in enumerate(out.table("nonbonded").params["type"][
        out.table("nonbonded").param_ids]) if "~" in t}  # fmt: skip
    a, c = boonza.canonical_forcefield(out), boonza.canonical_forcefield(prmtop)
    checked = 0
    for name in ("stretch_harm", "angle_harm", "dihedral_fourier"):
        mine, ref = _terms(a, name), _terms(c, name, frag)
        for key, val in ref.items():
            if gaff_atoms & set(key):
                assert mine[key] == val, (name, key)
                checked += 1
    assert checked > 50


@needs_amber
def test_covalent_cys_keeps_cmap_and_runs():
    openmm = pytest.importorskip("openmm")
    from openmm import unit

    s = _cys_adduct()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", viparr.ViparrWarning)
        patch = gaff.gaff2_patch(s, ["aa.amber.ff19SB"], amberhome=AMBERHOME)
    out = boonza.parameterize(s, [viparr.merge_forcefields("aa.amber.ff19SB", patch)])
    cmap = out.table("torsiontorsion_cmap")
    assert sorted(out.atoms["residue"][cmap.atoms[:, 2]].tolist()) == [1, 2, 4]
    top, system, pos = boonza.to_openmm(out)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                 0.002 * unit.picoseconds)  # fmt: skip
    ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10, 200)
    integrator.step(100)
    state = ctx.getState(getEnergy=True, getPositions=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole))
    assert np.isfinite(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)).all()
