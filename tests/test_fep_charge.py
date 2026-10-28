"""A change of net charge: phenol -> phenolate, balanced by a water that becomes
Na+ in B.  The box stays neutral at both ends, the hybrid is still exact at
its ends, and prep plans and runs such an edge only with a receptor."""

import csv
import gzip
import json

import numpy as np
import pytest
from rdkit import Chem

import boonza
from boonza.fep import cli, lambdas
from boonza.fep.run import run_leg

openmm = pytest.importorskip("openmm")
from test_fep_pipeline import DATA, _amber_tools  # noqa: E402

pytestmark = pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")


def _ligands(path):
    """Phenol and phenolate (its hydroxyl hydrogen taken off), in one pose."""
    supplier = Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False)
    phenol = next(m for m in supplier if m.GetProp("_Name") == "phenol")
    rw = Chem.RWMol(phenol)
    o = next(a for a in rw.GetAtoms() if a.GetSymbol() == "O")
    h = next(n.GetIdx() for n in o.GetNeighbors() if n.GetSymbol() == "H")
    o.SetFormalCharge(-1)
    rw.RemoveAtom(h)
    phenolate = rw.GetMol()
    Chem.SanitizeMol(phenolate)
    phenolate.SetProp("_Name", "phenolate")
    w = Chem.SDWriter(str(path))
    w.write(phenol)
    w.write(phenolate)
    w.close()
    return phenol


def _receptor(path, near):
    """A charged-termini tripeptide a little way from the ligand."""
    from boonza.viewer.edit import protonate

    s = boonza.peptide("GAG")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions)
    s.positions = pos - pos.mean(0) + near + np.array([8.0, 0.0, 0.0])
    boonza.save(s, path)


def _net_charge(system, x):
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    values = lambdas.values(x)
    q = np.array([nb.getParticleParameters(i)[0]._value for i in range(nb.getNumParticles())])
    for k in range(nb.getNumParticleParameterOffsets()):
        p, i, dq, _, _ = nb.getParticleParameterOffset(k)
        q[i] += values[p] * dq
    return q


def test_charge_change_with_a_receptor(tmp_path, capsys):
    sdf = tmp_path / "two.sdf"
    phenol = _ligands(sdf)
    _receptor(tmp_path / "receptor.pdb", phenol.GetConformer().GetPositions().mean(0))
    work = tmp_path / "work"
    assert cli.main(["prep", str(tmp_path / "receptor.pdb"), "--ligands", str(sdf),
                     "--workdir", str(work), "--windows", "3", "--charge-windows", "5",
                     "--repeats", "1", "--iteration-ps", "0.02",
                     "--equilibration-ns", "0.00002", "--production-ns", "0.0001",
                     "--charge-production-ns", "0.0002"]) == 0  # fmt: skip
    rows = list(csv.DictReader((work / "edges.csv").open()))
    assert [r["charge_change"] for r in rows] == ["-1"]
    for leg in ("complex", "solvent"):
        d = work / "edges" / "phenol~phenolate" / leg
        meta = json.loads((d / "leg.json").read_text())
        assert meta["charge_change"] == -1 and len(meta["lambdas"]) == 5
        assert meta["production_ns"] == 0.0002 and len(meta["alchemical_waters"]) == 1
        with gzip.open(d / "system.xml.gz", "rt") as fh:
            system = openmm.XmlSerializer.deserialize(fh.read())
        for x in (0.0, 1.0):
            assert _net_charge(system, x).sum() == pytest.approx(0, abs=1e-4)
        # the water's oxygen ends as Na+, far from the solute
        o = meta["alchemical_waters"][0]
        assert _net_charge(system, 1.0)[o] == pytest.approx(1.0)
        assert _net_charge(system, 1.0)[o + 1] == pytest.approx(0.0)
        state = np.load(d / "state.npz")
        water = np.arange(o, o + 3)
        lig = np.concatenate([state["unique_old"], state["unique_new"],
                              np.setdiff1d(state["core"], water)]).astype(int)  # fmt: skip
        box = np.diag(state["box"])
        dx = state["positions"][lig] - state["positions"][o]
        dx -= box * np.round(dx / box)
        assert np.linalg.norm(dx, axis=1).min() >= 0.8
    leg = work / "edges" / "phenol~phenolate" / "solvent"
    run_leg(leg, 1, platform="CPU", minimize=20, log=lambda *_: None)
    u = np.load(leg / "run_1.npz")["u_kln"]
    assert u.shape[1:] == (5, 5) and np.all(np.isfinite(u))


def test_hydration_refuses_a_change_of_charge(tmp_path, capsys):
    sdf = tmp_path / "two.sdf"
    _ligands(sdf)
    assert cli.main(["prep", "--ligands", str(sdf), "--workdir", str(tmp_path / "w")]) == 1
    assert "net charges differ" in capsys.readouterr().err


def test_ends_are_exact(tmp_path):
    """The hybrid at 0 is A plus B's dummies' valence, at 1 B plus A's; B's
    end has the water as Na+."""
    from test_fep_hybrid import _valence

    from boonza import gaff, viparr
    from boonza.fep.mapping import map_atoms
    from boonza.fep.prep import build_leg, load_ligands
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields

    sdf = tmp_path / "two.sdf"
    _ligands(sdf)
    mols, ligs, _ = load_ligands(sdf)
    args = parse_arguments(["--model", "aa", "--hmr"], build_parser("t"))
    args.cutoff_nm = args.cutoff_nm or 0.9
    _, ff = forcefields(args)
    both = ligs[0].copy()
    both.append(ligs[1])
    patch = gaff.gaff2_patch(both, ff, charge_geometry="reference")
    host = gaff.host_index(ff)
    ff[host] = viparr.merge_forcefields(ff[host], patch)
    leg = build_leg(None, ligs[0], ligs[1], map_atoms(mols[0], mols[1]), ff, args, 1.0,
                    log=lambda *_: None, charge_change=-1)  # fmt: skip
    hyb = leg.hybrid
    n_old = leg.old_system.getNumParticles()
    x_old = hyb.positions[:n_old]
    x_new = hyb.positions[hyb.new_to_hybrid]

    def energy(system, xyz, lam=None):
        c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                           openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
        c.setPeriodicBoxVectors(*leg.old_system.getDefaultPeriodicBoxVectors())
        c.setPositions(xyz * openmm.unit.nanometer)
        if lam is not None:
            hyb.set_lambda(c, lam)
        return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule_per_mole)  # fmt: skip

    un = [int(np.flatnonzero(hyb.new_to_hybrid == h)[0]) for h in hyb.unique_new]
    dummies_new = _valence(leg.new_system, un, x_new)
    dummies_old = _valence(leg.old_system, hyb.unique_old, x_old)
    assert energy(hyb.system, hyb.positions, 0.0) == pytest.approx(
        energy(leg.old_system, x_old) + dummies_new, abs=1e-3)  # fmt: skip
    assert energy(hyb.system, hyb.positions, 1.0) == pytest.approx(
        energy(leg.new_system, x_new) + dummies_old, abs=1e-3)  # fmt: skip
