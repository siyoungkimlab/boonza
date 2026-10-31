"""MBAR, and the timeseries analysis that comes before it.

The multistate Bennett acceptance ratio (Shirts and Chodera 2008) estimates the
free energy of every state from samples drawn from all of them::

    f_i = -ln sum_n exp(-u_i(x_n)) / sum_k N_k exp(f_k - u_k(x_n))

``u_kn`` holds the reduced potential (energy / kT) of every sample n at every
state k; ``N_k`` how many samples state k contributed.  The f solve the
minimum of a convex function, done here with Newton steps on its exact
gradient and Hessian; uncertainties come from the asymptotic covariance, and
the overlap matrix says how well neighbouring states share samples.

Before MBAR, a run's start is discarded where it has not equilibrated
(:func:`detect_equilibration`, Chodera 2016) and the rest is thinned to
uncorrelated samples (:func:`subsample`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    top = a.max(axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.log(np.exp(a - top).sum(axis=axis)) + np.squeeze(top, axis=axis)


@dataclass
class MBARResult:
    """Free energies (kT) of each state, relative to the first; their covariance;
    the overlap matrix; and the weights (samples by states)."""

    f: np.ndarray
    covariance: np.ndarray
    overlap: np.ndarray
    weights: np.ndarray
    iterations: int

    def delta(self, i: int = 0, j: int = -1) -> tuple[float, float]:
        """f_j - f_i (kT) and its standard error."""
        k = len(self.f)
        i, j = i % k, j % k
        d = self.f[j] - self.f[i]
        c = self.covariance
        var = c[i, i] + c[j, j] - 2 * c[i, j]
        return float(d), float(np.sqrt(max(var, 0.0)))


def mbar(u_kn, N_k, tolerance: float = 1e-10, max_iterations: int = 200) -> MBARResult:
    """Solve MBAR for reduced potentials ``u_kn`` (states by samples) of samples
    ``N_k`` per state, in the order they appear in the columns.

    States that contributed no samples are solved for too (as perturbations).
    """
    u = np.asarray(u_kn, dtype=np.float64)
    N = np.asarray(N_k, dtype=np.float64)
    K, n = u.shape
    if N.shape != (K,) or int(N.sum()) != n:
        raise ValueError(f"N_k must have one count per state summing to {n} samples")
    if not np.all(np.isfinite(u)):
        raise ValueError("u_kn has infinite or undefined energies")
    sampled = N > 0
    logN = np.log(np.where(sampled, N, 1.0))
    f = np.zeros(K)
    us, Ns, logNs = u[sampled], N[sampled], logN[sampled]

    def objective(fs):
        # convex in fs: its minimum is the MBAR solution
        return float(_logsumexp((fs + logNs)[:, None] - us, axis=0).sum() - Ns @ fs)

    def self_consistent(fs):
        log_denom = _logsumexp((fs + logNs)[:, None] - us, axis=0)
        new = -_logsumexp(-us - log_denom[None, :], axis=1)
        return new - new[0]

    fs = np.zeros(len(Ns))
    for _ in range(20):  # a safe start: self-consistent iterations never diverge
        fs = self_consistent(fs)
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        log_denom = _logsumexp((fs + logNs)[:, None] - us, axis=0)
        wn = np.exp(fs[:, None] - us - log_denom[None, :]) * Ns[:, None]  # N_k W_kn
        grad = wn.sum(axis=1) - Ns  # of the objective
        H = np.diag(wn.sum(axis=1)) - wn @ wn.T
        g, Hr = grad[1:], H[1:, 1:]  # the first f is held at 0
        if len(g) == 0 or np.abs(g).max() < tolerance * max(1.0, Ns.sum()):
            break
        try:
            step = -np.linalg.solve(Hr, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(Hr, g, rcond=None)[0]
        now, scale, trial = objective(fs), 1.0, None
        while scale > 1e-6:  # backtrack until the objective falls
            attempt = fs.copy()
            attempt[1:] += scale * step
            if objective(attempt) <= now:
                trial = attempt
                break
            scale *= 0.5
        if trial is None:
            # no Newton step lowers the objective: a sample with an enormous energy in
            # some state (1e13-1e17 kT, an atom overlapping where its repulsion is back
            # on) leaves the Hessian near singular and the step huge.  Taking its last,
            # failed fraction anyway sent IGF1R's free energies to -1e6 kT; a
            # self-consistent iteration never diverges
            trial = self_consistent(fs)
        change = np.abs(trial - fs).max()
        fs = trial - trial[0]
        if change < tolerance:
            break
    f[sampled] = fs
    # every state, sampled or not, from the converged denominators
    log_denom = _logsumexp((f + logN)[sampled, None] - u[sampled], axis=0)
    f = -_logsumexp(-u - log_denom[None, :], axis=1)
    f -= f[0]
    log_w = f[:, None] - u - log_denom[None, :]
    W = np.exp(log_w).T  # n x K, each column sums to 1
    cov = _covariance(W, N)
    overlap = W.T @ (W * N[None, :])
    return MBARResult(f=f, covariance=cov, overlap=overlap, weights=W, iterations=iterations)


def _covariance(W: np.ndarray, N: np.ndarray) -> np.ndarray:
    """The asymptotic covariance of f (Shirts and Chodera 2008, eq. D6), through a
    thin SVD of W so nothing is samples by samples."""
    K = W.shape[1]
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T
    inner = np.eye(K) - (S[:, None] * Vt) @ np.diag(N) @ (V * S[None, :])
    middle = np.linalg.pinv(inner, rcond=1e-10)
    theta = V @ (S[:, None] * middle * S[None, :]) @ Vt
    return theta


# --- timeseries ------------------------------------------------------------------


def statistical_inefficiency(a, fast: bool = True) -> float:
    """g = 1 + 2 tau: how many correlated samples make one independent one, from
    the normalized autocorrelation summed until it first goes negative (Chodera
    2007, as pymbar's timeseries; ``fast`` grows the lag as it goes)."""
    x = np.asarray(a, dtype=np.float64)
    T = len(x)
    if T < 3:
        return 1.0
    dx = x - x.mean()
    var = (dx * dx).mean()
    if var == 0:
        return 1.0
    g, t, inc = 1.0, 1, 1
    while t < T - 1:
        c = (dx[: T - t] * dx[t:]).mean() / var
        if c <= 0 and t > 1:
            break
        g += 2.0 * c * (1.0 - t / T) * inc
        t += inc
        if fast:
            inc += 1
    return max(g, 1.0)


def detect_equilibration(a, candidates: int = 100) -> tuple[int, float, float]:
    """The start t0 that leaves the most effectively independent samples
    (Chodera 2016): (t0, g, N_eff) over ``candidates`` evenly spaced starts."""
    x = np.asarray(a, dtype=np.float64)
    T = len(x)
    if T < 3:
        return 0, 1.0, float(T)
    best = (0, 1.0, -1.0)
    for t0 in np.unique(np.linspace(0, T - 2, min(candidates, T - 1)).astype(int)):
        g = statistical_inefficiency(x[t0:])
        neff = (T - t0) / g
        if neff > best[2]:
            best = (int(t0), g, neff)
    return best


def subsample(n: int, g: float) -> np.ndarray:
    """Indices of ``n`` correlated samples, every g-th (rounded, unique)."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.unique(np.round(np.arange(0, n, max(g, 1.0))).astype(np.int64))
    return idx[idx < n]
