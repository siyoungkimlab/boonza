"""Replica exchange and its analysis on a particle in harmonic wells (exact ΔF)."""

import numpy as np
import pytest

from boonza.fep.analysis import KB_KCAL, analyze
from boonza.fep.sampler import ReplicaExchange, Replicas

openmm = pytest.importorskip("openmm")
K0, K1, T = 100.0, 1000.0, 298.15  # kJ/mol/nm^2, K


def _well():
    s = openmm.System()
    for _ in range(4):  # four particles, each in its own well
        s.addParticle(12.0)
    f = openmm.CustomExternalForce("0.5*k*(x^2+y^2+z^2); k = k0 + lam*(k1 - k0)")
    for name, v in (("k0", K0), ("k1", K1), ("lam", 0.0)):
        f.addGlobalParameter(name, v)
    for i in range(4):
        f.addParticle(i, [])
    s.addForce(f)
    return s


def test_replica_exchange_recovers_a_known_free_energy(tmp_path):
    states = [{"lam": x} for x in np.linspace(0, 1, 6)]
    rex = ReplicaExchange(_well(), states, temperature=T, timestep=0.002, steps=50,
                          platform="Reference", seed=7)  # fmt: skip
    r = rex.start(np.zeros((4, 3)) + 0.01, minimize=0)
    rex.iterate(r, 20, keep=False)
    path = tmp_path / "run.npz"
    rex.iterate(r, 200, checkpoint=path, every=50)
    # saved and read back, the run goes on where it stopped
    again = Replicas.load(path)
    assert again.iteration == 220 and again.equilibration == 20 and len(again.u_kln) == 200
    rex.iterate(again, 200)
    out = analyze(again, temperature=T)
    exact = 4 * 1.5 * KB_KCAL * T * np.log(K1 / K0)  # four particles, three dimensions
    assert out["dG"] == pytest.approx(exact, abs=4 * out["dG_error"] + 0.05)
    assert out["min_neighbour_overlap"] > 0.03 and out["min_neighbour_mixing"] > 0
    np.testing.assert_allclose(out["overlap"].sum(axis=1), 1, atol=1e-8)
    assert out["forward"].shape == (10, 2)
    assert out["forward"][-1, 0] == pytest.approx(out["dG"])


def test_states_must_name_the_systems_parameters():
    with pytest.raises(ValueError, match="no global parameters"):
        ReplicaExchange(_well(), [{"lam": 0}, {"lambda": 1}], platform="Reference")
    with pytest.raises(ValueError, match="two states"):
        ReplicaExchange(_well(), [{"lam": 0}], platform="Reference")
