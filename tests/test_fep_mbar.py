"""MBAR and its timeseries analysis, against harmonic oscillators (exact free energies)."""

import numpy as np
import pytest

from boonza.fep.mbar import detect_equilibration, mbar, statistical_inefficiency, subsample

K_SPRING = np.array([1.0, 2.0, 4.0, 8.0, 16.0])  # in kT/length^2
CENTRE = np.array([0.0, 0.3, 0.6, 0.9, 1.2])


def _oscillators(n_per_state, rng, k=K_SPRING, x0=CENTRE):
    """Samples drawn from each state, and every sample's reduced potential at
    every state: u_k(x) = k_k (x - x0_k)^2 / 2."""
    x = np.concatenate([rng.normal(x0[i], 1 / np.sqrt(k[i]), n)
                        for i, n in enumerate(n_per_state)])  # fmt: skip
    u = 0.5 * k[:, None] * (x[None, :] - x0[:, None]) ** 2
    return u, np.asarray(n_per_state)


def _exact(k=K_SPRING):
    f = -np.log(np.sqrt(2 * np.pi / k))
    return f - f[0]


def test_mbar_recovers_exact_free_energies_within_its_errors():
    rng = np.random.default_rng(1)
    u, N = _oscillators([2000] * 5, rng)
    out = mbar(u, N)
    exact = _exact()
    err = np.sqrt(np.diag(out.covariance))
    assert np.all(np.abs(out.f - exact) < 4 * err + 1e-12)
    d, dd = out.delta(0, -1)
    assert d == pytest.approx(exact[-1], abs=4 * dd)
    # the overlap matrix: rows sum to 1, the diagonal largest
    np.testing.assert_allclose(out.overlap.sum(axis=1), 1, atol=1e-8)
    assert np.all(np.diag(out.overlap) >= out.overlap.max(axis=1) - 1e-12)


def test_the_stated_error_is_the_spread_of_repeats():
    rng = np.random.default_rng(2)
    estimates, stated = [], []
    for _ in range(60):
        u, N = _oscillators([300] * 5, rng)
        d, dd = mbar(u, N).delta(0, -1)
        estimates.append(d)
        stated.append(dd)
    spread = np.std(estimates, ddof=1)
    assert np.mean(stated) == pytest.approx(spread, rel=0.3)


def test_an_unsampled_state_is_estimated_too():
    rng = np.random.default_rng(3)
    u, N = _oscillators([3000, 3000, 0, 3000, 3000], rng)
    out = mbar(u, N)
    exact = _exact()
    err = np.sqrt(np.diag(out.covariance))
    assert abs(out.f[2] - exact[2]) < 4 * err[2] + 1e-3


def test_mbar_refuses_bad_input():
    with pytest.raises(ValueError, match="N_k"):
        mbar(np.zeros((2, 10)), [5, 4])
    u = np.zeros((2, 4))
    u[0, 1] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        mbar(u, [2, 2])


def test_equilibration_is_detected_and_the_rest_thinned():
    rng = np.random.default_rng(4)
    T, phi = 4000, 0.9  # AR(1): g = (1 + phi) / (1 - phi) = 19
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal()
    g = statistical_inefficiency(x)
    assert 10 < g < 30
    drift = x + 30 * np.exp(-np.arange(T) / 100)  # a start far from equilibrium
    t0, g0, neff = detect_equilibration(drift)
    assert 200 < t0 < 1500 and neff < T
    idx = subsample(T - t0, g0)
    assert len(idx) == pytest.approx((T - t0) / g0, rel=0.1)
    assert np.all(np.diff(idx) > 0)
