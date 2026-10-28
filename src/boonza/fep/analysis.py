"""From a replica-exchange run to a free energy, as openfe reports one.

The start that has not equilibrated is found on the replicas' summed reduced
potential in their own states (Chodera 2016) and dropped; the rest is thinned
to uncorrelated samples; MBAR gives the free energy between the first and last
states with its error, the overlap matrix (neighbours should share at least
~0.03 of their samples), the replica mixing matrix, and forward and reverse
estimates over growing fractions of the data (they should meet).
"""

from __future__ import annotations

import numpy as np

from .mbar import detect_equilibration, mbar, subsample

KB_KCAL = 0.0019872043  # kcal/mol/K


def analyze(replicas, temperature: float = 298.15, fractions: int = 10) -> dict:
    """ΔG (kcal/mol) from the first to the last state of a run (a
    :class:`boonza.fep.sampler.Replicas`), with what tells whether to trust it."""
    u = np.asarray(replicas.u_kln, dtype=np.float64)  # T x replicas x states
    states = np.asarray(replicas.history, dtype=np.int64)  # T x replicas
    T, K, _ = u.shape
    if T < 4:
        raise ValueError(f"{T} iterations: too few to analyze")
    kT = KB_KCAL * temperature
    own = np.take_along_axis(u, states[:, :, None], axis=2)[:, :, 0].sum(axis=1)
    t0, g, neff = detect_equilibration(own)
    kept = t0 + subsample(T - t0, g)

    def estimate(rows):
        res = mbar(u[rows].reshape(-1, K).T, np.full(K, len(rows)))
        d, dd = res.delta(0, -1)
        return d * kT, dd * kT, res

    dg, ddg, res = estimate(kept)
    forward, reverse = [], []
    for k in range(1, fractions + 1):
        n = max(2, int(round(len(kept) * k / fractions)))
        forward.append(estimate(kept[:n])[:2])
        reverse.append(estimate(kept[-n:])[:2])
    mixing = np.zeros((K, K))
    for a, b in zip(states[:-1].ravel(), states[1:].ravel(), strict=True):
        mixing[a, b] += 1
    mixing /= np.maximum(mixing.sum(axis=1, keepdims=True), 1)
    overlap = res.overlap
    return {
        "dG": float(dg), "dG_error": float(ddg),
        "iterations": int(T), "equilibration": int(t0), "statistical_inefficiency": float(g),
        "samples": int(len(kept)),
        "overlap": overlap, "min_neighbour_overlap": float(np.diag(overlap, 1).min()),
        "mixing": mixing, "min_neighbour_mixing": float(np.diag(mixing, 1).min()),
        "forward": np.asarray(forward), "reverse": np.asarray(reverse),
        "f_states": res.f * kT,
    }  # fmt: skip
