"""Network planning and the maximum likelihood estimate of every ligand's ΔG."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from boonza.fep import network

DATA = Path(__file__).parent / "data" / "fep"


def test_benzenes_are_joined_and_every_one_sits_in_a_cycle():
    ligs = list(Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False))
    edges = network.edges(ligs)
    import networkx as nx

    g = nx.Graph([(e.a, e.b) for e in edges])
    assert nx.is_connected(g) and g.number_of_nodes() == len(ligs)
    assert min(dict(g.degree).values()) >= 2  # every ligand in a cycle
    assert all(e.score >= np.exp(-0.25) and e.mapping for e in edges)  # small changes only
    assert len(edges) < len(ligs) * (len(ligs) - 1) / 2


def test_the_estimate_recovers_true_values_and_cycles_close():
    rng = np.random.default_rng(0)
    true = np.array([0.0, 1.0, -0.5, 2.0, 0.3])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (0, 4)]
    results = [(a, b, true[b] - true[a] + rng.normal(0, 0.05), 0.05) for a, b in pairs]
    g, err = network.estimate(5, results, reference=0)
    np.testing.assert_allclose(g, true, atol=0.15)
    assert err[0] == pytest.approx(0, abs=1e-9) and np.all(err[1:] > 0)
    centred, _ = network.estimate(5, results)
    assert centred.sum() == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(centred - centred[0], g, atol=1e-9)
    closures = network.cycle_closures(5, results)
    assert len(closures) == len(pairs) - 5 + 1
    assert all(abs(total) < 0.3 for _, total in closures)
    exact = [(a, b, true[b] - true[a], 0.1) for a, b in pairs]
    assert all(abs(t) < 1e-12 for _, t in network.cycle_closures(5, exact))


def test_unconnected_ligands_are_refused():
    with pytest.raises(ValueError, match="connect"):
        network.estimate(3, [(0, 1, 1.0, 0.1)])
