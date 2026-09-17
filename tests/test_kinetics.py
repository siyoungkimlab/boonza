"""Kinetics: do the rates come back, and do the boundaries and censoring behave."""

import numpy as np
import pytest

import boonza
from boonza.kinetics import GAS, LITRES
from boonza.sites import Site, SiteSet

DT = 0.1  # ns per frame
BOX = 40.0
CONC = LITRES / BOX**3  # M, for one copy in the box
K_OFF = 0.05  # 1/ns: a residence time of 20 ns
K_ON_PSEUDO = 0.02  # 1/ns at that concentration
K_ON = K_ON_PSEUDO / CONC
DG = GAS * 310.0 * np.log(K_OFF / K_ON)


def _process(nruns=12, nframes=3000, seed=0, wobble=0.5, away=(14.0, 18.0), stop=None):
    """A ligand that binds and unbinds as a two-state Markov process, with the
    rates known: the kinetics have a right answer to be checked against."""
    rng = np.random.default_rng(seed)
    p_off, p_on = 1 - np.exp(-K_OFF * DT), 1 - np.exp(-K_ON_PSEUDO * DT)
    centre = np.array([5.0, 0.0, 0.0])
    xyz, where, flags = [], [], []
    for r in range(nruns):
        state, last = False, nframes if stop is None else stop[r]
        for f in range(last):
            if state and rng.random() < p_off:
                state = False
            elif not state and rng.random() < p_on:
                state = True
            if state:
                pos = centre + rng.normal(scale=wobble, size=3)
            else:
                v = rng.normal(size=3)
                pos = centre + v / np.linalg.norm(v) * rng.uniform(*away)
            xyz.append(pos)
            where.append((r, 0, f))
            flags.append(state)
    return np.array(xyz), np.array(where), np.array(flags)


def _as_site(xyz, where, flags, nruns):
    points = np.flatnonzero(flags)
    site = Site(center=xyz[points].mean(0), points=points, occupancy=len(points) / len(xyz),
                runs=nruns, copies=1, arrivals=0, spread=float(flags.mean()))  # fmt: skip
    return SiteSet(sites=[site], labels=np.where(flags, 0, -1), where=where, centroids=xyz,
                   spacing=1.0, enrichment=20.0)  # fmt: skip


@pytest.fixture(scope="module")
def box():
    s = boonza.peptide("AA")
    s.cell = np.diag([BOX, BOX, BOX])
    return s


def test_the_rates_come_back(box):
    xyz, where, flags = _process()
    found = _as_site(xyz, where, flags, 12)
    r = boonza.kinetics(box, found, 0, interval_ns=DT, bootstrap=200)
    assert r.k_off == pytest.approx(K_OFF, rel=0.2)
    assert r.k_on == pytest.approx(K_ON, rel=0.2)
    assert r.residence_ns == pytest.approx(1 / K_OFF, rel=0.2)
    assert r.dG == pytest.approx(DG, abs=0.2)
    assert r.occupancy == pytest.approx(flags.mean(), abs=0.01)
    assert r.dG_interval[0] <= DG <= r.dG_interval[1]  # and the interval covers the truth
    assert r.consistent and r.events > 30


def test_hysteresis_is_what_stops_invented_events(box):
    """A ligand rattling near the boundary crosses it constantly. One boundary
    counts every crossing as a departure; two do not."""
    xyz, where, flags = _process(wobble=2.5, away=(9.0, 16.0), seed=1)
    found = _as_site(xyz, where, flags, 12)
    real = int((flags[:-1] & ~flags[1:]).sum())
    none = boonza.kinetics(box, found, 0, interval_ns=DT, hysteresis=1.0, bootstrap=0)
    kept = boonza.kinetics(box, found, 0, interval_ns=DT, hysteresis=2.0, bootstrap=0)
    assert none.events > 8 * real  # crossings, not departures
    assert none.residence_ns < 0.2 / K_OFF  # and a residence time far too short
    assert kept.events == pytest.approx(real, rel=0.3)
    assert kept.residence_ns == pytest.approx(1 / K_OFF, rel=0.4)
    # the ratio survives either way: dG is robust to the boundary, kinetics is not
    assert none.dG == pytest.approx(kept.dG, abs=0.15)


def test_a_cut_run_costs_its_ending_not_its_time(box):
    """Runs stopped early -- the wall clock, or --early-stop -- must not bias the rates."""
    rng = np.random.default_rng(7)
    runs = 40  # enough departures for the rate to mean something: 26 events is +-20% by itself
    stop = rng.integers(400, 3000, runs)
    xyz, where, flags = _process(nruns=runs, stop=stop, seed=7)
    found = _as_site(xyz, where, flags, runs)
    r = boonza.kinetics(box, found, 0, interval_ns=DT, bootstrap=0)
    per_run = [sum(d.run == run for d in r.dwells) for run in range(runs)]
    # the first and last stretch of every run, which are the same one when a run holds only one
    assert r.censored == sum(min(2, n) for n in per_run)
    assert min(per_run) == 1  # a short run can stay unbound throughout
    assert sum(d.frames for d in r.dwells if d.bound and d.censored) > 0  # cut mid-dwell
    assert r.k_off == pytest.approx(K_OFF, rel=0.25)  # the time still counts, the ending does not
    assert r.bound_ns + r.unbound_ns == pytest.approx(stop.sum() * DT)  # every frame counted


def test_dwells_alternate_and_cover_every_frame(box):
    xyz, where, flags = _process(nruns=3, nframes=800, seed=4)
    found = _as_site(xyz, where, flags, 3)
    items = boonza.dwells(found, 0)
    for run in range(3):
        mine = [d for d in items if d.run == run]
        assert [d.bound for d in mine] == [mine[0].bound ^ (k % 2 == 1) for k in range(len(mine))]
        assert sum(d.frames for d in mine) == 800
        assert mine[0].censored and mine[-1].censored
        assert not any(d.censored for d in mine[1:-1])


def test_it_refuses_what_it_cannot_measure(box):
    xyz, where, flags = _process(nruns=2, nframes=500, seed=5)
    found = _as_site(xyz, where, flags, 2)
    naked = boonza.peptide("AA")  # no cell, so no concentration
    with pytest.raises(ValueError, match="no periodic cell"):
        boonza.kinetics(naked, found, 0, interval_ns=DT)
    everywhere = _as_site(xyz, where, flags, 2)
    everywhere.sites[0].points = np.arange(len(xyz))  # a boundary so loose nothing is outside
    with pytest.raises(ValueError, match="never entered or never left"):
        boonza.kinetics(box, everywhere, 0, interval_ns=DT, hysteresis=1e6)
