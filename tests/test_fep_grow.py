"""boonza fep grow: the steps a ligand is grown from a reference in, and a growth."""

import json
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from test_fep_pipeline import _amber_tools

import boonza
from boonza.fep import grow


def _mol(smiles, name="x", seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    m.SetProp("_Name", name)
    return m


def _symbols(m, step):
    return "".join(sorted(m.GetAtomWithIdx(i).GetSymbol() for i in step))


def test_groups_and_rings_are_grown_whole():
    ref = _mol("c1ccccc1")
    for smiles, whole in [("OC(=O)Cc1ccccc1", None),  # a carboxylic acid: C, O, O together
                          ("[O-]C(=O)c1ccccc1", "COO"),
                          ("NC(=[NH2+])NCc1ccccc1", None),  # guanidinium: C and its 3 N
                          ("c1ccccc1-c1ncccn1", "CCCCNN")]:  # a pyrimidine, whole  # fmt: skip
        target = _mol(smiles)
        core = grow.core_map(ref, target)
        plan = grow.steps(target, core)
        rest = [a.GetIdx() for a in target.GetAtoms()
                if a.GetAtomicNum() > 1 and a.GetIdx() not in core]  # fmt: skip
        assert sorted(i for s in plan for i in s) == sorted(rest)
        if whole:
            assert any(_symbols(target, s) == whole for s in plan), (smiles, plan)
        if "C(=O)" in smiles or "(=O)" in smiles:  # the C=O never apart from its O-/OH
            for s in plan:
                syms = _symbols(target, s)
                assert not (syms.count("O") == 1 and "C" in syms and len(s) == 2)
        if "NH2+" in smiles:
            guan = next(s for s in plan if _symbols(target, s).count("N") >= 3)
            assert len(guan) >= 4
    # an unbranched chain: one step, however long (it reaches nothing until its end)
    target = _mol("CCCCCCc1ccccc1")
    assert [len(s) for s in grow.steps(target, grow.core_map(ref, target))] == [6]
    # a chain to a group: the chain, then the group whole
    target = _mol("[O-]C(=O)CCCCc1ccccc1")
    plan = grow.steps(target, grow.core_map(ref, target))
    assert [_symbols(target, s) for s in plan] == ["CCCC", "COO"]
    # branched, so not a chain: at most 3 heavy atoms a step
    target = _mol("CC(C)CC(C)Cc1ccccc1")
    assert max(len(s) for s in grow.steps(target, grow.core_map(ref, target))) <= 3


def test_every_intermediate_is_a_molecule():
    ref = _mol("c1ccccc1")
    target = _mol("[O-]C(=O)CCc1ccc(-c2ncccn2)cc1")
    core = grow.core_map(ref, target)
    placed = set(core)
    for s in grow.steps(target, core):
        placed |= set(s)
        mol, index = grow.intermediate(target, placed)
        assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        assert all(i in index for i in placed)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.RemoveHs(target))


def test_a_donor_alone_does_not_accept_a_hydrogen_bond():
    """A nitrogen carrying a hydrogen -- a guanidinium's, Trp's NE1, a backbone N --
    donates and cannot also accept, so two of them near each other make no hydrogen
    bond however their hydrogens point."""
    #            guanidinium N-H, pyridine N, carbonyl O, hydroxyl O, Trp NE1
    anum = np.array([7, 7, 8, 8, 7])
    hydrogens = {0: [10, 11], 3: [12], 4: [13]}
    assert grow._acceptors(anum, hydrogens, range(5)) == {0: False, 1: True, 2: True,
                                                          3: True, 4: False}  # fmt: skip


def test_the_interaction_energy_is_openmm_s_own():
    """The score's energy, read with numpy, against the force it stands for: OpenMM's
    NonbondedForce in a reaction field, with every pair inside each group excluded."""
    import openmm as mm
    import openmm.unit as u

    rng = np.random.default_rng(3)
    n = 8
    q = rng.normal(0, 0.8, n)
    q -= q.mean()
    sigma, epsilon = rng.uniform(2.5, 3.6, n), rng.uniform(0.1, 0.9, n)  # Å, kJ/mol
    xyz = rng.uniform(0, 9, (n, 3))  # Å
    a, b = [0, 1, 2], [3, 4, 5, 6, 7]
    system = mm.System()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(1.2 * u.nanometer)
    nb.setReactionFieldDielectric(grow.EPSILON_RF)
    for i in range(n):
        system.addParticle(12.0)
        nb.addParticle(q[i], sigma[i] / 10.0, epsilon[i])
    for group in (a, b):  # leave only the a-b pairs, which is what the energy is
        for p in range(len(group)):
            for r in range(p + 1, len(group)):
                nb.addException(group[p], group[r], 0.0, 1.0, 0.0)
    system.addForce(nb)
    ctx = mm.Context(system, mm.VerletIntegrator(1.0 * u.femtosecond),
                     mm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx.setPositions(xyz / 10.0)
    want = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        u.kilojoule_per_mole)  # fmt: skip
    got = grow._interaction(xyz, a, b, *grow._nonbonded(system))
    assert abs(got - want) < 1e-6 * abs(want)
    assert grow._interaction(xyz, [], b, *grow._nonbonded(system)) == 0.0


def test_a_ring_s_normal_is_perpendicular_to_it():
    ring = np.array([[np.cos(t), np.sin(t), 0.0] for t in np.linspace(0, 2 * np.pi, 6)])
    n = grow._normal(ring + [1.0, 2.0, 3.0])
    assert abs(abs(n @ [0, 0, 1]) - 1.0) < 1e-6


def test_placements_are_spread_not_merely_the_least_clashing():
    """Taking the n least-clashing conformers left them near-duplicates in one corner,
    so more conformers made the sample less diverse: 120 did worse than 60.  spread()
    takes the least-clashing first, then whatever is furthest from what it has."""
    rng = np.random.default_rng(0)
    new_atoms = [0, 1]
    # the protein sits over the far group, so the ranking (fewest clashes, then
    # furthest from it) prefers the crowd: exactly the case that went wrong
    protein = np.array([[10.0, 0.0, 20.0]])
    # ten conformers around x=0 and two around x=10: the crowd must not win every slot
    crowd = [np.array([[rng.normal(0, 0.05), 0.0, 0.0], [1.0, 0.0, 0.0]]) for _ in range(10)]
    far = [np.array([[10.0 + rng.normal(0, 0.05), 0.0, 0.0], [11.0, 0.0, 0.0]])
           for _ in range(2)]  # fmt: skip
    confs = crowd + far

    def apart(picked):
        xs = [c[0][0] for c in picked]
        return max(xs) - min(xs)

    got = grow.spread(confs, new_atoms, protein, 3)
    assert len(got) == 3
    assert apart(got) > 9.0  # it reached the far group
    # the old rule: the first 3 by clashes alone, all from the crowd
    ranked = sorted(confs, key=lambda c: (grow.clashes(c, new_atoms, protein)[0],
                                          -grow.clashes(c, new_atoms, protein)[1]))  # fmt: skip
    assert apart(ranked[:3]) < 1.0
    # asking for more than there are returns them all, and a step that grew nothing
    # falls back to the ranking
    assert len(grow.spread(confs, new_atoms, protein, 99)) == len(confs)
    assert len(grow.spread(confs, [], protein, 3)) == 3
    # a clashing conformer is still not preferred to a clean one
    clash = [np.array([[10.0, 0.0, 20.0], [10.0, 0.0, 20.5]])]
    assert grow.clashes(clash[0], new_atoms, protein)[0] > 0
    assert all(c[0][2] != 20.0 for c in grow.spread(confs + clash, new_atoms, protein, 3))


def test_runs_are_grouped_by_the_pose_they_agree_on():
    """--repeats keeps what the runs agree on, not what scored highest: across 24
    growths the score told good poses from bad only weakly, so the best score returned
    a 2.3 A pose while a 0.6 A one scored mid-pack."""
    a = np.zeros((5, 3))
    b = a + np.array([10.0, 0.0, 0.0])
    poses = [a, a + 0.1, b, a + 0.2, b + 0.1, a + 0.3]  # four together, two apart
    groups = grow.agree(poses, cut=1.0)
    assert [len(g) for g in groups] == [4, 2]  # the largest first
    assert groups[0] == [0, 1, 3, 5] and groups[1] == [2, 4]
    # everything apart: one group each, and everything together: one group
    assert len(grow.agree([a, b, b + 20.0], cut=1.0)) == 3
    assert grow.agree([a, a + 0.1, a + 0.2], cut=1.0) == [[0, 1, 2]]
    # single linkage: a chain of near poses is one group even end to end
    chain = [a + np.array([0.9 * k, 0.0, 0.0]) for k in range(4)]
    assert grow.agree(chain, cut=1.0) == [[0, 1, 2, 3]]


def test_repeats_and_the_sampling_flags_reach_the_command_line():
    """The MD is not reproducible on a GPU, so the same search takes different
    branches: growing 9B54's ligand from 6N4B five times reached the crystal's salt
    bridge once.  --repeats buys that back, and the rest of the sampling is already
    there to turn up."""
    out = subprocess.run([sys.executable, "-m", "boonza.cli", "fep", "grow", "--help"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    for flag in ("--repeats", "--md-ps", "--keep", "--conformers", "--beam", "--max-heavy",
                 "--hold", "--water", "--seed"):  # fmt: skip
        assert flag in out, flag


@pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")
def test_growing_each_ligand_several_times_keeps_the_best(tmp_path):
    """--repeats grows it again with different seeds and keeps the best-scoring run;
    every run's placements stay in the candidates file, told apart by name."""
    from boonza.fep.cli import FEP_DEFAULTS
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields
    from boonza.viewer.edit import protonate

    ref = _mol("c1ccccc1", "benzene")
    target = _mol("Cc1ccccc1", "toluene")
    s = boonza.peptide("GAG")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions)
    s.positions = pos - pos.mean(0) + ref.GetConformer().GetPositions().mean(0) + [7.0, 0, 0]
    args = parse_arguments(["--workdir", str(tmp_path), "--platform", "CPU"],
                           build_parser("boonza fep grow"))  # fmt: skip
    for k, v in FEP_DEFAULTS.items():
        setattr(args, k, v)
    _, ff = forcefields(args)
    grow.grow(s, ref, target, "toluene", ff, args, tmp_path, beam=1, keep=1,
              n_conformers=5, md_ps=2.0, repeats=3, log=lambda *_: None)  # fmt: skip
    record = json.loads((tmp_path / "grow" / "toluene.json").read_text())
    assert record["repeats"] == 3 and len(record["run_scores"]) == 3
    # the runs are ranked by one common measurement of the finished pose, not by the
    # score each reached in its own trajectory: over 48 poses the two agreed most of
    # the time, and of the times they did not the common one picked the closer pose in
    # 92%.  The runs' grouping is reported beside it, never used to choose
    assert len(record["rescored"]) == 3
    assert record["kept_run"] == record["rescored"].index(min(record["rescored"]))
    assert sum(len(g) for g in record["agreement"]) == 3
    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(
        str(tmp_path / "grow" / "toluene_candidates.sdf"), removeHs=False)]  # fmt: skip
    assert len(names) == len(set(names)) == 3  # one per run here, each named for its run
    assert names[0] == "step1_from0_c0" and "run2_" in names[1] and "run3_" in names[2]
    kept = np.load(str(tmp_path / "grow" / "toluene_candidates.sdf.npz"))
    assert sorted(kept.files) == sorted(names)


@pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")
def test_growing_toluene_from_benzene_in_a_peptide(tmp_path):
    from boonza.fep.cli import FEP_DEFAULTS
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields
    from boonza.viewer.edit import protonate

    ref = _mol("c1ccccc1", "benzene")
    target = _mol("Cc1ccccc1", "toluene")
    s = boonza.peptide("GAG")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions)
    s.positions = pos - pos.mean(0) + ref.GetConformer().GetPositions().mean(0) + [7.0, 0, 0]
    args = parse_arguments(["--workdir", str(tmp_path), "--platform", "CPU"],
                           build_parser("boonza fep grow"))  # fmt: skip
    for k, v in FEP_DEFAULTS.items():
        setattr(args, k, v)
    _, ff = forcefields(args)
    path = grow.grow(s, ref, target, "toluene", ff, args, tmp_path, beam=1, keep=1,
                     n_conformers=5, md_ps=2.0, log=lambda *_: None)  # fmt: skip
    out = boonza.load(path)
    assert out.natoms == s.natoms + target.GetNumAtoms()
    record = json.loads((tmp_path / "grow" / "toluene.json").read_text())
    assert record["core"] == 6 and record["steps"] and "score" in record["steps"][0]
    # the score is the interaction energy; hydrogen bonds and cation-pi are reported
    row = record["steps"][0]
    # a methyl 7 Å from the peptide feels it weakly, and there is no cation to stack
    # (OpenMM's CPU platform is not bit-reproducible across threads, so this is a
    # bound on the size of the term, not its value)
    assert abs(row["interaction"]) < 30.0 and row["cation_pi"] == 0.0 and "hbonds" in row
    assert abs(row["score"] - (-row["interaction"] / grow.INTERACTION_KJ
                               - 0.5 * row["drift"] + 0.01 * row["contacts"])) < 1e-6  # fmt: skip
    # every candidate's receptor is kept, so its complex can be rebuilt
    kept = np.load(tmp_path / "grow" / "toluene_candidates.sdf.npz")
    assert kept["step1_from0_c0"].shape == (s.natoms, 3)
    # the core stayed near where the reference was
    n = target.GetNumAtoms()
    lig = np.asarray(out.positions)[-n:]
    core = grow.core_map(ref, target)
    ref_xyz = ref.GetConformer().GetPositions()
    d = [np.linalg.norm(lig[t] - ref_xyz[r]) for t, r in core.items()]
    assert max(d) < grow.LOOSE_A + 0.5  # rings held, then the whole relaxed within LOOSE_A
