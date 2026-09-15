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


def _mapped(smi: str, spec: list[str], chain: str) -> boonza.System:
    """A molecule from a SMILES whose atom map k names atom ``spec[k - 1]``
    ("RES resid NAME"); hydrogens are named after their atom."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    by_map = {k + 1: s.split() for k, s in enumerate(spec)}
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) >= 0
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
                                                 chainId=chain))  # fmt: skip
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    s = boonza.from_rdkit(mol)
    s.reorder_atoms(np.argsort(s.atoms["residue"], kind="stable"))
    return s


@pytest.fixture(scope="module")
def mixed():
    """ACE-ALA-CYS-ALA-NME whose Cys is bound to methanethiol (residue LIG, a
    mixed disulfide), next to two ACE-CYS-NME joined by a real disulfide
    (residues 7 and 8)."""
    adduct = _mapped(
        "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][C@@H:10]([CH2:11][S:12]"
        "[S:15][CH3:16])[C:13](=[O:14])[NH:17][C@@H:18]([CH3:19])[C:20](=[O:21])[NH:22][CH3:23]",
        ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "ALA 2 N", "ALA 2 CA", "ALA 2 CB", "ALA 2 C",
         "ALA 2 O", "CYS 3 N", "CYS 3 CA", "CYS 3 CB", "CYS 3 SG", "CYS 3 C", "CYS 3 O",
         "LIG 4 S1", "LIG 4 C1", "ALA 5 N", "ALA 5 CA", "ALA 5 CB", "ALA 5 C", "ALA 5 O",
         "NME 6 N", "NME 6 CH3"], "A")  # fmt: skip
    dimer = _mapped(
        "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH2:6][S:7][S:18][CH2:17][C@H:16]([NH:15][C:13]"
        "(=[O:14])[CH3:12])[C:19](=[O:20])[NH:21][CH3:22])[C:8](=[O:9])[NH:10][CH3:11]",
        ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "CYS 2 N", "CYS 2 CA", "CYS 2 CB", "CYS 2 SG",
         "CYS 2 C", "CYS 2 O", "NME 3 N", "NME 3 CH3", "ACE 4 CH3", "ACE 4 C", "ACE 4 O",
         "CYS 5 N", "CYS 5 CA", "CYS 5 CB", "CYS 5 SG", "CYS 5 C", "CYS 5 O", "NME 6 N",
         "NME 6 CH3"], "B")  # fmt: skip
    adduct.append(dimer)
    names = adduct.residues["name"].tolist()
    assert [names[r] for r in (2, 3, 7, 8)] == ["CYS", "LIG", "CYS", "CYS"]
    return adduct


def test_pinned_residue_formula_tells_ligand_sulfur_from_disulfide(mixed):
    ff = viparr.load_forcefield("aa.amber.ff14SB")
    cyx = ff.template("CYX")
    sg_ext = next(i for i in cyx._nbrs[cyx.names.index("SG")] if cyx.anum[i] == -1)
    lig = viparr._formula(mixed.atoms["anum"][mixed.residue_atoms(3)].tolist())

    def found(formulas, residue):
        t = viparr.Template("CYL", cyx.names, cyx.anum, cyx.charge, cyx.btype, cyx.nbtype,
                            cyx.pset, cyx.bonds, cyx.impropers, external_anum={sg_ext: 16},
                            external_formula=formulas)  # fmt: skip
        both = viparr.ViparrForcefield("t", ff.rules, [*ff.templates, t], ff.params)
        P = viparr._Parameterizer(mixed, [both], False, False, True)
        return P._find(both, 0, mixed.residue_atoms(residue).tolist())[0].name

    assert found({sg_ext: lig}, 2) == "CYL"  # the Cys bound to the ligand's sulfur
    assert found({sg_ext: lig}, 7) == "CYX"  # a real disulfide
    assert found({}, 7) == "CYL"  # the element alone cannot tell them apart


MSE_PEPTIDE = (
    "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][C@@H:10]([CH2:11][CH2:12]"
    "[Se:13][CH3:14])[C:15](=[O:16])[NH:17][C@@H:18]([CH3:19])[C:20](=[O:21])[NH:22][CH3:23]",
    ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "ALA 2 N", "ALA 2 CA", "ALA 2 CB", "ALA 2 C", "ALA 2 O",
     "{r} 3 N", "{r} 3 CA", "{r} 3 CB", "{r} 3 CG", "{r} 3 SE", "{r} 3 CE", "{r} 3 C", "{r} 3 O",
     "ALA 4 N", "ALA 4 CA", "ALA 4 CB", "ALA 4 C", "ALA 4 O", "NME 5 N", "NME 5 CH3"],
)  # fmt: skip


def _mse(resname="MSE"):
    """ACE-ALA-<selenomethionine>-ALA-NME, the modified residue named ``resname``."""
    smi, spec = MSE_PEPTIDE
    return _mapped(smi, [x.format(r=resname) for x in spec], "A")


def _parent_of(s, parents=None):
    """(parent template, source, heavy atoms keeping protein types) of residue 3."""
    b = gaff._Builder(s, [viparr.load_forcefield("aa.amber.ff19SB")], parents=parents)
    r = int(np.flatnonzero(s.residues["resid"] == 3)[0])
    t, _, ok, source = b._parent(r)
    heavy = sorted(int(a) for a in ok if s.atoms["anum"][a] > 1)
    return (None if t is None else t.name), source, heavy


def test_parent_of_a_modified_residue():
    s = _mse()
    names = s.atoms["name"]
    t, source, heavy = _parent_of(s)
    assert (t, source) == ("MET", "known")  # from the PDB's chemical component dictionary
    assert sorted(str(names[a]) for a in heavy) == ["C", "CA", "CB", "N", "O"]  # CG is bonded to Se
    x = _mse("XSE")  # unknown: MET, LEU, GLU, ... all match only the backbone and CB
    with pytest.raises(viparr.ViparrError, match="cannot tell the parent residue of XSE3"):
        _parent_of(x)
    x.residues.add_prop("parent", str)
    x.residues["parent"] = np.array(["", "", "MET", "", ""])
    assert _parent_of(x)[:2] == ("MET", "file")
    t, source, _ = _parent_of(x, parents={"XSE": "CYS"})  # given beats the file
    assert (gaff._family(t), source) == ("CYS", "given")  # the Cys template matching most
    # atom names are not needed: the backbone is found by structure
    ren = _mse("XSE")
    r3 = np.flatnonzero(ren.atoms["residue"] == 2)
    nm = ren.atoms["name"].copy()
    nm[r3] = [f"X{k}" for k in range(len(r3))]
    ren.atoms["name"] = nm
    assert gaff.find_unmatched(ren, ["aa.amber.ff19SB"]) == [[2]]  # the neighbours stay out
    assert _parent_of(ren, parents={"XSE": "MET"}) == ("MET", "given", heavy)


def test_free_molecules_and_their_hydrogens_stay_gaff2():
    free = boonza.from_smiles("NCC(=O)O", name="XGL")  # glycine, not in a chain
    free.atoms["name"] = np.array(["N", "CA", "C", "O", "OXT"]
                                  + [f"H{k}" for k in range(free.natoms - 5)])  # fmt: skip
    b = gaff._Builder(free, [viparr.load_forcefield("aa.amber.ff19SB")])
    assert b._parent(0)[0] is None
    # acetylated Lys: NZ is GAFF2, and so is the hydrogen on it
    s = _mapped(*LYS_ACETYL, "A")
    b = gaff._Builder(s, [viparr.load_forcefield("aa.amber.ff19SB")])
    r = int(np.flatnonzero(s.residues["resid"] == 3)[0])
    t, _, ok, source = b._parent(r)
    nz = next(a for a in s.residue_atoms(r).tolist() if s.atoms["name"][a] == "NZ")
    hz = [a for a in s.bonded_atoms(nz).tolist() if s.atoms["anum"][a] == 1]
    assert (source, nz in ok, any(h in ok for h in hz)) == ("guessed", False, False)
    assert t.name.endswith("LYS") or t.name == "LYN"


def test_modified_residue_records(tmp_path):
    s = _mse()
    pdb = tmp_path / "mse.pdb"
    boonza.save(s, pdb)
    pdb.write_text("MODRES 1ABC MSE A    3  MET  SELENOMETHIONINE\n" + pdb.read_text())
    assert boonza.load(pdb).residues["parent"].tolist() == ["", "", "MET", "", ""]
    cif = tmp_path / "mse.cif"
    boonza.save(s, cif)
    cif.write_text(cif.read_text().rstrip("\n") + "\nloop_\n"
                   + "".join(f"_pdbx_struct_mod_residue.{k}\n" for k in (
                       "id", "auth_asym_id", "auth_seq_id", "PDB_ins_code", "auth_comp_id",
                       "parent_comp_id"))
                   + "1 A 3 ? MSE MET\n")  # fmt: skip
    assert boonza.load(cif).residues["parent"].tolist() == ["", "", "MET", "", ""]


# ---- with AmberTools --------------------------------------------------------------


LYS_ACETYL = (
    "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][C@@H:10]([CH2:11][CH2:12]"
    "[CH2:13][CH2:14][NH:15][C:20](=[O:21])[CH3:22])[C:16](=[O:17])[NH:18][CH3:19]",
    ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "ALA 2 N", "ALA 2 CA", "ALA 2 CB", "ALA 2 C", "ALA 2 O",
     "LYS 3 N", "LYS 3 CA", "LYS 3 CB", "LYS 3 CG", "LYS 3 CD", "LYS 3 CE", "LYS 3 NZ",
     "LYS 3 C", "LYS 3 O", "NME 5 N", "NME 5 CH3", "LIG 4 C1", "LIG 4 O1", "LIG 4 C2"],
)  # fmt: skip


@needs_amber
def test_protein_extent_and_drawing(tmp_path):
    # an acetyl on Lys NZ: CG, CD and CE still look like lysine
    s = _mapped(*LYS_ACETYL, "A")
    got = {}
    for extent in ("matched", "cb"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", viparr.ViparrWarning)
            patch = gaff.gaff2_patch(s, ["aa.amber.ff19SB"], amberhome=AMBERHOME,
                                     protein_extent=extent, draw=tmp_path / extent)  # fmt: skip
        lys = next(t for t in patch.templates if t.name.endswith("_LYS"))
        got[extent] = sorted(n for n, b, z in zip(lys.names, lys.btype, lys.anum, strict=True)
                             if z > 1 and "~" in b)  # fmt: skip
        out = boonza.parameterize(s, [viparr.merge_forcefields("aa.amber.ff19SB", patch)])
        q, res = out.atoms["charge"], out.atoms["residue"]
        assert np.allclose([q[res == r].sum() for r in range(out.nresidues)], 0, atol=1e-6)
        png = tmp_path / extent / "covalent_LYS3+LIG4.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert got == {"matched": ["NZ"], "cb": ["CD", "CE", "CG", "NZ"]}
    with pytest.raises(ValueError, match="protein_extent"):
        gaff.gaff2_patch(s, ["aa.amber.ff19SB"], protein_extent="all")


@needs_amber
def test_sulfur_bound_ligand_leaves_disulfides_alone(mixed, tmp_path):
    host = viparr.load_forcefield("aa.amber.ff14SB")
    assert gaff.find_unmatched(mixed, [host]) == [[2, 3]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", viparr.ViparrWarning)
        patch = gaff.gaff2_patch(mixed, [host], amberhome=AMBERHOME)
    patch = viparr.load_forcefield(viparr.write_forcefield(patch, tmp_path / "p"),
                                   require_rules=False)  # fmt: skip
    cys = next(t for t in patch.templates if t.name.endswith("_CYS"))
    assert sorted(cys.external_formula.values()) == [viparr._formula([1, 1, 1, 6, 16])]
    out = boonza.parameterize(mixed, [viparr.merge_forcefields(host, patch)])
    nb = out.table("nonbonded")
    nbtype = nb.params["type"][nb.param_ids[np.argsort(nb.atoms[:, 0])]]
    sg = {int(out.atoms["residue"][a]): nbtype[a] for a in range(out.natoms)
          if out.atoms["name"][a] == "SG"}  # fmt: skip
    assert sg[2].endswith("~1")
    assert sg[7] == sg[8] == host.template("CYX").nbtype[host.template("CYX").names.index("SG")]


@needs_amber
def test_free_ligand_reproduces_tleap(tmp_path):
    s = boonza.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")
    b = gaff._Builder(s, [], run=dict(amberhome=AMBERHOME), draw=tmp_path / "draw")
    patch = b.patch()
    ((frag, prmtop),) = b.runs
    assert frag == list(range(s.natoms))
    assert b.drawings == [] and not (tmp_path / "draw").exists()  # not a covalent adduct
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


def test_gaff_parameters_by_version(tmp_path):
    parm = tmp_path / "dat" / "leap" / "parm"
    parm.mkdir(parents=True)
    title = "AMBER General Force Field for organic molecules (Version {}, {})\n"
    (parm / "gaff2.dat").write_text(title.format("2.11", "May 2016"))  # older AmberTools
    assert gaff.gaff_parameters(tmp_path, "2.11") == parm / "gaff2.dat"
    with pytest.raises(gaff.AmberToolsError, match="gaff2.dat is version 2.11"):
        gaff.gaff_parameters(tmp_path, "2.2")
    (parm / "gaff2.dat").write_text(title.format("2.2.30", "Oct 2025"))  # newer ones
    (parm / "gaff211.dat").write_text(title.format("2.11", "May 2016"))
    assert gaff.gaff_parameters(tmp_path, "2.11") == parm / "gaff211.dat"
    assert gaff.gaff_parameters(tmp_path, "2.2") == parm / "gaff2.dat"
    (parm / "gaff211.dat").unlink()
    with pytest.raises(gaff.AmberToolsError, match="gaff2.dat is version 2.2.30"):
        gaff.gaff_parameters(tmp_path, "2.11")
    if AMBERHOME is not None:  # the installed AmberTools has 2.11 under one name or the other
        assert gaff.gaff_parameters(AMBERHOME, "2.11").name in ("gaff211.dat", "gaff2.dat")
