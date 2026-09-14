"""Symmetry-corrected RMSD against networkx enumeration (the ~/rmsd.py method) and RDKit."""

import math
import time
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.symmetry import ligand_rmsd, symmetry_rmsd

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem, rdMolAlign  # noqa: E402

nx = pytest.importorskip("networkx")

DATA = Path(__file__).parent / "data"
SMILES = {
    "tBu-benzoic acid": "CC(C)(C)c1ccc(cc1)C(=O)O",
    "benzenesulfonic acid": "OS(=O)(=O)c1ccccc1",
    "cubane": "C12C3C4C1C5C2C3C45",
    "di-tBu-methane": "CC(C)(C)CC(C)(C)C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "glucose": "OCC1OC(O)C(O)C(O)C1O",
}
# molecules whose symmetry does not depend on bond orders, for RDKit's CalcRMS/GetBestRMS
RDKIT_SAFE = ["CC(C)(C)c1ccccc1", "c1ccc2ccccc2c1", "C12C3C4C1C5C2C3C45",
              "CC1(C)CCCC(C)(C)C1", "ClC(Cl)(Cl)c1ccccc1"]  # fmt: skip


def _conformers(smiles, n=4, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=seed)
    return mol


def _system(mol, conf_id, order=None):
    s = boonza.from_rdkit(mol, conf_id=conf_id)
    if order is not None:
        s.reorder_atoms(order)
    return s


def _networkx_min(mob, ref, superpose=False):
    """The ~/rmsd.py method: min RMSD over all element-preserving graph isomorphisms."""

    def graph(s):
        ids = np.flatnonzero(s.atoms["anum"] > 1)
        local = {int(a): k for k, a in enumerate(ids)}
        g = nx.Graph()
        for k, a in enumerate(ids):
            g.add_node(k, atomic_number=int(s.atoms["anum"][a]))
        for i, j in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True):
            if i in local and j in local:
                g.add_edge(local[i], local[j])
        return g, s.positions[ids]

    gr, pr = graph(ref)
    gm, pm = graph(mob)
    same = lambda a, b: a["atomic_number"] == b["atomic_number"]  # noqa: E731
    best = math.inf
    for mapping in nx.isomorphism.GraphMatcher(gr, gm, node_match=same).isomorphisms_iter():
        r, m = pr[list(mapping)], pm[list(mapping.values())]
        if superpose:
            best = min(best, boonza.rmsd(m, r, superpose=True))
        else:
            best = min(best, float(np.sqrt(((r - m) ** 2).sum() / len(r))))
    return best


@pytest.mark.parametrize("name", list(SMILES))
def test_matches_networkx_enumeration(name):
    mol = _conformers(SMILES[name])
    rng = np.random.default_rng(3)
    ref = _system(mol, 0)
    for conf in range(1, mol.GetNumConformers()):
        mob = _system(mol, conf, order=rng.permutation(mol.GetNumAtoms()))  # shuffled atoms
        got = symmetry_rmsd(mob, ref)
        assert got.rmsd == pytest.approx(_networkx_min(mob, ref), abs=1e-9)
        fitted = symmetry_rmsd(mob, ref, superpose=True)
        assert fitted.rmsd == pytest.approx(_networkx_min(mob, ref, superpose=True), abs=1e-9)
        assert fitted.rmsd <= got.rmsd + 1e-12
        # the reported mapping reproduces the RMSD
        m, r = mob.positions[got.mapping], ref.positions[got.reference_atoms]
        assert np.sqrt(((m - r) ** 2).sum(1).mean()) == pytest.approx(got.rmsd, abs=1e-12)


@pytest.mark.parametrize("smiles", RDKIT_SAFE)
def test_matches_rdkit(smiles):
    mol = _conformers(smiles, n=5)
    heavy = Chem.RemoveHs(mol)
    ref = _system(mol, 0)
    for conf in range(1, mol.GetNumConformers()):
        mob = _system(mol, conf)
        probe, target = Chem.Mol(heavy, confId=conf), Chem.Mol(heavy, confId=0)
        assert symmetry_rmsd(mob, ref).rmsd == pytest.approx(rdMolAlign.CalcRMS(probe, target),
                                                             abs=1e-6)  # fmt: skip
        want = rdMolAlign.GetBestRMS(Chem.Mol(probe), Chem.Mol(target))
        assert symmetry_rmsd(mob, ref, superpose=True).rmsd == pytest.approx(want, abs=1e-6)


def test_renamed_equivalent_atoms_do_not_count():
    """Same pose, t-butyl methyls and carboxyl oxygens named the other way round."""
    mol = _conformers("CC(C)(C)c1ccc(cc1)C(=O)O", n=1)
    ref = _system(mol, 0)
    mob = ref.copy()
    anum, pos = mob.atoms["anum"], mob.positions

    def heavy(a):
        return [int(b) for b in mob.bonded_atoms(a) if anum[b] > 1]

    quaternary = next(a for a in range(mob.natoms) if anum[a] == 6 and len(heavy(a)) == 4)
    methyls = [b for b in heavy(quaternary) if len(heavy(b)) == 1]
    carboxyl_c = next(a for a in range(mob.natoms)
                      if anum[a] == 6 and sum(anum[b] == 8 for b in heavy(a)) == 2)  # fmt: skip
    oxygens = [b for b in heavy(carboxyl_c) if anum[b] == 8]
    pos[methyls] = pos[np.roll(methyls, 1)]  # rotate the methyl names
    pos[oxygens] = pos[oxygens[::-1]]  # swap the oxygen names
    mob.positions = pos
    r = symmetry_rmsd(mob, ref)
    assert r.plain_rmsd > 1.0 and r.rmsd == pytest.approx(0.0, abs=1e-9)


def test_frames_bond_orders_and_errors():
    mol = _conformers("CC(C)(C)c1ccc(cc1)C(=O)O", n=4)
    ref, mob = _system(mol, 0), _system(mol, 1)
    frames = np.stack([_system(mol, k).positions for k in range(4)])
    per = symmetry_rmsd(mob, ref, positions=frames)
    assert per.rmsd.shape == (4,) and per.rmsd[0] == pytest.approx(0.0, abs=1e-12)
    for k in range(4):
        assert per.rmsd[k] == pytest.approx(symmetry_rmsd(_system(mol, k), ref).rmsd)
    # by connectivity: 3! methyl orders x ring flip x carboxyl oxygens = 24 mappings;
    # Kekule bond orders forbid the ring flip and the oxygen swap
    loose = symmetry_rmsd(mob, ref, superpose=True)
    strict = symmetry_rmsd(mob, ref, superpose=True, bond_orders=True)
    assert loose.isomorphisms == 24 and strict.isomorphisms == 6
    other = _system(_conformers("CC(C)(C)c1ccccc1", n=1), 0)
    with pytest.raises(ValueError, match="not the same molecule"):
        symmetry_rmsd(other, ref)


def _rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_ligand_rmsd_superposes_protein_and_moves_ligand():
    """Docked-pose workflow: protein + ligand, rigidly moved, ligand atoms shuffled."""
    protein = boonza.load(DATA / "1LYZ.pdb").clone("protein")
    lig_mol = _conformers("CC(C)(C)c1ccc(cc1)C(=O)O", n=2)
    crystal_lig = _system(lig_mol, 0)
    crystal_lig.residues["name"] = ["LIG"] * crystal_lig.nresidues
    pocket = protein.positions.mean(0) + [15.0, 0.0, 0.0]
    crystal_lig.positions = crystal_lig.positions + pocket
    crystal = protein.copy()
    crystal.append(crystal_lig)

    pose = _system(lig_mol, 1)  # another conformer, placed on the crystal ligand
    pose.residues["name"] = ["LIG"] * pose.nresidues
    rot, t = boonza.kabsch(pose.positions, crystal_lig.positions)
    pose.positions = pose.positions @ rot.T + t
    docked = protein.copy()
    docked.append(pose)
    want = symmetry_rmsd(docked, crystal, "resname LIG").rmsd  # already in one frame

    moved = docked.copy()
    moved.positions = moved.positions @ _rotation(1).T + [30.0, -12.0, 7.0]
    n = protein.natoms
    rng = np.random.default_rng(1)
    moved.reorder_atoms(np.concatenate([np.arange(n), n + rng.permutation(pose.natoms)]))

    got = ligand_rmsd(moved, crystal, ligand="resname LIG")
    assert got.fit_rmsd == pytest.approx(0.0, abs=1e-6)
    assert got.rmsd == pytest.approx(want, abs=1e-6)
    seq = ligand_rmsd(moved, crystal, ligand="resname LIG", align="sequence")
    assert seq.rmsd == pytest.approx(want, abs=1e-4)
    before = moved.positions.copy()
    ligand_rmsd(moved, crystal, ligand="resname LIG")
    np.testing.assert_array_equal(moved.positions, before)  # untouched unless apply=True
    ligand_rmsd(moved, crystal, ligand="resname LIG", apply=True)
    assert symmetry_rmsd(moved, crystal, "resname LIG").rmsd == pytest.approx(want, abs=1e-6)
    # the default ligand selection finds it too
    assert ligand_rmsd(moved, crystal).rmsd == pytest.approx(want, abs=1e-6)
    assert len(crystal.select("polymer")) == len(crystal.select("protein or nucleic"))


def test_cli(tmp_path, capsys):
    from boonza.cli import main

    s = boonza.load(DATA / "1HHO.pdb")
    moved = s.copy()
    moved.positions = moved.positions @ _rotation(2).T + [5.0, 0.0, 0.0]
    boonza.save(moved, tmp_path / "moved.pdb")
    assert main(["rmsd", str(tmp_path / "moved.pdb"), str(DATA / "1HHO.pdb"),
                 "--ligandsel", "resname HEM and chain A"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "symmetry-corrected" in out and "0.000" in out


def test_many_equivalent_groups_are_fast():
    """Four t-butyl groups: thousands of isomorphisms, pruned by branch and bound."""
    mol = _conformers("CC(C)(C)c1cc(C(C)(C)C)c(C(C)(C)C)cc1C(C)(C)C", n=2)
    ref, mob = _system(mol, 0), _system(mol, 1)
    t0 = time.perf_counter()
    r = symmetry_rmsd(mob, ref)
    assert time.perf_counter() - t0 < 2.0
    assert r.rmsd == pytest.approx(_networkx_min(mob, ref), abs=1e-9)


def _complex():
    """A protein with a ligand ("crystal") and the same protein with another ligand pose."""
    protein = boonza.load(DATA / "1LYZ.pdb").clone("protein")
    lig_mol = _conformers("CC(C)(C)c1ccc(cc1)C(=O)O", n=2)
    crystal_lig = _system(lig_mol, 0)
    crystal_lig.residues["name"] = ["LIG"] * crystal_lig.nresidues
    crystal_lig.positions = crystal_lig.positions + protein.positions.mean(0) + [15.0, 0.0, 0.0]
    crystal = protein.copy()
    crystal.append(crystal_lig)
    pose = _system(lig_mol, 1)
    pose.residues["name"] = ["LIG"] * pose.nresidues
    rot, t = boonza.kabsch(pose.positions, crystal_lig.positions)
    pose.positions = pose.positions @ rot.T + t
    docked = protein.copy()
    docked.append(pose)
    return crystal, docked


def test_ligand_rmsd_over_a_trajectory(tmp_path, capsys):
    crystal, docked = _complex()
    rng = np.random.default_rng(5)
    xyz = docked.positions
    frames = np.stack([xyz @ _rotation(k).T + rng.normal(0, 5, 3)
                       + rng.normal(0, 0.05 * k, xyz.shape) for k in range(6)])  # fmt: skip
    per = ligand_rmsd(docked, crystal, ligand="resname LIG", positions=frames)
    assert per.rmsd.shape == per.fit_rmsd.shape == (6,) and per.rotation.shape == (6, 3, 3)
    for k in range(6):  # each frame fitted on its own, exactly as a single structure
        one = docked.copy()
        one.positions = frames[k]
        single = ligand_rmsd(one, crystal, ligand="resname LIG")
        assert per.rmsd[k] == pytest.approx(single.rmsd, abs=1e-9)
        assert per.fit_rmsd[k] == pytest.approx(single.fit_rmsd, abs=1e-9)
        assert per.rmsd[k] <= per.plain_rmsd[k] + 1e-9
    seq = ligand_rmsd(docked, crystal, ligand="resname LIG", positions=frames, align="sequence")
    np.testing.assert_allclose(seq.rmsd, per.rmsd, atol=0.05)

    # the same through a trajectory file, read chunk by chunk (float32 coordinates)
    path = tmp_path / "md.dcd"
    with boonza.open_writer(path, docked.natoms) as w:
        for f in frames:
            w.write(f)
    traj = boonza.open_trajectory(path, docked)
    from_file = ligand_rmsd(docked, crystal, ligand="resname LIG", positions=traj)
    np.testing.assert_allclose(from_file.rmsd, per.rmsd, atol=1e-3)
    direct = symmetry_rmsd(docked, crystal, "resname LIG", positions=traj)
    assert direct.rmsd.shape == (6,)
    with pytest.raises(ValueError, match="single structure"):
        ligand_rmsd(docked, crystal, ligand="resname LIG", positions=frames, apply=True)

    from boonza.cli import main

    boonza.save(docked, tmp_path / "docked.pdb")
    boonza.save(crystal, tmp_path / "crystal.pdb")
    assert main(["rmsd", str(tmp_path / "docked.pdb"), str(tmp_path / "crystal.pdb"),
                 "--traj", str(path), "--ligandsel", "resname LIG"]) == 0  # fmt: skip
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split() == ["frame", "fit", "ligand", "in", "order"]
    assert len(lines) == 6 + 2 and "of frames under 2 A" in lines[-1]


def _user_drmsd(u, cutoff=5.0, proteinsel="protein and name CA",
                ligandsel="resname LIG and not name H*"):  # fmt: skip
    """The user's MDAnalysis script (drmsd_trj.py), verbatim apart from the name."""
    from MDAnalysis.lib.distances import distance_array

    u.trajectory[0]
    pocket = u.select_atoms(f"({proteinsel}) and (around {cutoff} ({ligandsel}))")
    ligs = u.select_atoms(f"({ligandsel}) and not name H*")
    ref = distance_array(pocket.positions, ligs.positions, box=u.dimensions)
    out = np.empty(len(u.trajectory))
    for i, ts in enumerate(u.trajectory):
        d = distance_array(pocket.positions, ligs.positions, box=ts.dimensions)
        out[i] = np.sqrt(np.mean((d - ref) ** 2))
    return out, pocket.indices


def _pocket_complex():
    """Lysozyme with a t-butylbenzoic acid in the middle, in a 60 A box."""
    s = boonza.load(DATA / "1LYZ.pdb").clone("protein")
    lig = _system(_conformers("CC(C)(C)c1ccc(cc1)C(=O)O", n=1), 0)
    lig.residues["name"] = ["LIG"] * lig.nresidues
    lig.positions = lig.positions - lig.positions.mean(0) + s.positions.mean(0)
    s.append(lig)
    s.positions = s.positions - s.positions.mean(0) + 30.0
    s.cell = np.diag([60.0, 60.0, 60.0])
    return s


def _drmsd_brute(s, xyz, box, pocket, ref_xyz, ref_box, lig):
    """min over networkx isomorphisms of the ligand graph (hydrogens dropped)."""
    ids = lig[s.atoms["anum"][lig] > 1]
    local = {int(a): k for k, a in enumerate(ids)}
    g = nx.Graph()
    for k, a in enumerate(ids):
        g.add_node(k, z=int(s.atoms["anum"][a]))
    for i, j in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True):
        if i in local and j in local:
            g.add_edge(local[i], local[j])
    dref = boonza.pbc.distances(ref_xyz[pocket], ref_xyz[ids], ref_box)
    d = boonza.pbc.distances(xyz[pocket], xyz[ids], box)
    same = lambda a, b: a["z"] == b["z"]  # noqa: E731
    maps = nx.isomorphism.GraphMatcher(g, g, node_match=same).isomorphisms_iter()
    return min(float(np.sqrt(((d[:, list(m.values())] - dref[:, list(m)]) ** 2).mean()))
               for m in maps)  # fmt: skip


def test_drmsd_matches_mdanalysis_script_and_brute_force(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    from boonza.symmetry import drmsd

    s = _pocket_complex()
    lig = s.select("resname LIG").ids
    heavy = lig[s.atoms["anum"][lig] > 1]
    rng = np.random.default_rng(3)
    frames, boxes = [], []
    for k in range(8):
        x = s.positions + rng.normal(0, 0.15 * k, s.positions.shape)
        if k % 2:  # swap the carboxylate oxygens and rotate the t-butyl group: same pose
            o = heavy[s.atoms["anum"][heavy] == 8]
            x[o] = x[o[::-1]]
        box = np.diag([60.0 + k, 60.0, 61.0 - k])
        x = x + [k * 7.0, -k * 3.0, 20.0]  # shift across the boundary, then wrap atom by atom
        x = x - np.floor(x / np.diag(box)) * np.diag(box)
        frames.append(x)
        boxes.append(box)
    frames, boxes = np.array(frames), np.array(boxes)
    path = tmp_path / "md.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x, b in zip(frames, boxes, strict=True):
            w.write(x, box=b)
    s.cell = boxes[0]
    boonza.save(s, tmp_path / "top.pdb")  # one model, although the ligand is a second ct

    u = mda.Universe(str(tmp_path / "top.pdb"), str(path))
    expected, pocket_ids = _user_drmsd(u)
    traj = boonza.open_trajectory(path, s)
    r = drmsd(s, ligand="resname LIG", positions=traj)
    np.testing.assert_array_equal(np.sort(r.pocket), np.sort(pocket_ids))
    assert len(r.pocket) >= 5
    np.testing.assert_allclose(r.plain_drmsd, expected, atol=2e-4)  # float32 in MDAnalysis
    assert r.drmsd[0] == pytest.approx(0, abs=1e-4)
    assert np.all(r.drmsd <= r.plain_drmsd + 1e-9)
    assert r.plain_drmsd[1] > 2 * r.drmsd[1]  # the oxygen swap counts only without symmetry

    ref_xyz = np.asarray(traj[0].positions, np.float64)
    for k in (1, 4, 7):
        brute = _drmsd_brute(s, np.asarray(traj[k].positions, np.float64), boxes[k], r.pocket,
                             ref_xyz, boxes[0], lig)  # fmt: skip
        assert r.drmsd[k] == pytest.approx(brute, abs=1e-9)

    # a separate reference system with the ligand atoms in another order: same answer
    ref = s.copy()
    ref.positions = ref_xyz
    order = np.arange(ref.natoms)
    order[lig] = rng.permutation(lig)
    ref.reorder_atoms(order)
    other = drmsd(s, ref, ligand="resname LIG", positions=frames[:, :, :], periodic=True)
    s_frames = boonza.trajectory.Frames(np.arange(8), frames, boxes, np.zeros(8), np.zeros(8))
    with_boxes = drmsd(s, ref, ligand="resname LIG", positions=s_frames)
    np.testing.assert_allclose(with_boxes.drmsd, r.drmsd, atol=1e-4)
    assert other.drmsd.shape == (8,)
    with pytest.raises(ValueError, match="no protein atoms"):
        drmsd(s, ligand="resname LIG", cutoff=0.1)

    from boonza.cli import main

    assert main(["drmsd", str(tmp_path / "top.pdb"), "--traj", str(path),
                 "--ligandsel", "resname LIG"]) == 0  # fmt: skip


def test_drmsd_single_structures_and_rigid_motion():
    from boonza.symmetry import drmsd

    s = _pocket_complex()
    s.cell = np.zeros((3, 3))
    moved = s.copy()
    moved.positions = s.positions @ _rotation(2).T + [5.0, -3.0, 8.0]
    one = drmsd(moved, s, ligand="resname LIG")
    assert isinstance(one.drmsd, float) and one.drmsd == pytest.approx(0, abs=1e-9)
    lig = s.select("resname LIG and noh").ids
    shifted = s.copy()
    pos = shifted.positions
    pos[lig] += [1.0, 0.0, 0.0]
    shifted.positions = pos
    r = drmsd(shifted, s, ligand="resname LIG")
    assert 0 < r.drmsd <= r.plain_drmsd + 1e-12 and r.reference_distances.shape == (
        len(r.pocket),
        len(lig),
    )
