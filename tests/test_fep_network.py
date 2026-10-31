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


def test_ligands_added_later_join_the_network_without_touching_it():
    import networkx as nx

    ligs = list(Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False))
    first = network.edges(ligs[:4])
    existing = [(e.a, e.b) for e in first]
    added = network.extend(ligs, existing, [4, 5, 6])
    assert added and all({e.a, e.b} & {4, 5, 6} for e in added)  # only pairs with a new one
    assert not {(e.a, e.b) for e in added} & set(existing)
    g = nx.Graph(existing + [(e.a, e.b) for e in added])
    assert nx.is_connected(g) and g.number_of_nodes() == len(ligs)
    assert all(g.degree[k] >= 2 for k in (4, 5, 6))  # every new ligand in a cycle
    with pytest.raises(ValueError, match="no new"):
        network.extend(ligs, existing, [])


def test_a_large_change_gets_the_dense_schedule():
    from boonza.fep import lambdas
    from boonza.fep.prep import schedule

    d = lambdas.dense()
    assert len(d) == 23 and d[0] == 0.0 and d[-1] == 1.0 and np.all(np.diff(d) > 0)
    assert np.allclose(np.diff(d[(d >= 0.3) & (d <= 0.7)]), 0.025)  # where sterics switch
    assert len(schedule(2, 5, False, 11, 22)) == 11
    assert len(schedule(7, 5, False, 11, 22)) == 11  # a benzyl ether's 7: evenly spaced
    assert len(schedule(12, 5, False, 11, 22)) == 23  # 12 heavy atoms of its own
    assert len(schedule(12, 5, True, 11, 22)) == 22  # a change of charge: its own
    assert len(schedule(12, 5, False, 11, 22, large_change_atoms=0)) == 11
    assert len(schedule(8, 5, False, 11, 22, large_change_atoms=8)) == 23


def _benzenes():
    return list(Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False))


def _graph(edges):
    import networkx as nx

    return nx.Graph([(e.a, e.b) for e in edges])


def test_no_hubs_small_edges_first_and_a_reference():
    import networkx as nx

    ligs = _benzenes()
    names = [m.GetProp("_Name") for m in ligs]
    free = network.edges(ligs, max_degree=0, max_change=0)  # LOMAP's tree, unconstrained
    hub = max(dict(_graph(free).degree).values())
    assert hub > 2  # benzene, the parent, joins most of the others
    capped = network.edges(ligs, max_degree=2, max_change=0)
    g = _graph(capped)
    assert nx.is_connected(g) and g.number_of_nodes() == len(ligs)
    assert max(dict(g.degree).values()) <= 2  # a chain, no hub
    # an edge that changes more than one heavy atom only where no smaller route joins
    small = network.edges(ligs, max_degree=0, max_change=1, redundant=False)
    big = [e for e in small if e.changed > 1]
    g = _graph([e for e in small if e.changed <= 1])
    g.add_nodes_from(range(len(ligs)))
    for e in big:
        assert not nx.has_path(g, e.a, e.b)
    # a redundant edge may be large: every ligand still sits in a cycle
    checked = _graph(network.edges(ligs, max_degree=0, max_change=1))
    assert min(dict(checked.degree).values()) >= 2
    # a reference: preferred, and free of the cap
    ref = names.index("toluene")
    anchored = network.edges(ligs, max_degree=2, reference=ref)
    deg = dict(_graph(anchored).degree)
    assert deg[ref] == max(deg.values()) and deg[ref] > 2


def test_a_network_from_the_graphs_alone():
    ligs = _benzenes()
    two_d = network.edges(ligs, method="2d")
    g = _graph(two_d)
    assert g.number_of_nodes() == len(ligs) and min(dict(g.degree).values()) >= 2
    assert all(e.mapping and e.score > 0 for e in two_d)  # built from the 3D mapping
    with pytest.raises(ValueError, match="3d or 2d"):
        network.edges(ligs, method="4d")
