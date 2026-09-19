"""Kinetics of binding: how long a ligand stays, how often it arrives, and what that is worth.

    found = boonza.sites(s, runs)
    rate = boonza.kinetics(s, found, 0, interval_ns=0.1)
    rate.residence_ns, rate.dG, rate.dG_interval, rate.events

A site's frames make a two-state time series per ligand copy -- in it, or
not -- and the dwells between transitions give the rates.  Three things
decide whether the numbers mean anything, and all three are handled here
rather than assumed away:

*Hysteresis.*  With one boundary the ligand flickers across it and invents
events that never happened, which inflates the off rate without bound.  A
copy is taken to arrive within ``r_in`` of the site and to leave only past a
looser ``r_out``, so recrossing the same boundary costs nothing.

*Censoring.*  A run ends because the wall clock ran out, or because
``boonza md --early-stop`` saw the ligand leave -- not because the dwell
ended.  Those intervals count their time but not as completed events, which
is the maximum-likelihood treatment for an exponential and the difference
between a rate and a guess.

*Evidence.*  A rate is worth what its event count is worth.  Every number
comes with the count it rests on, an interval from resampling whole runs,
and a check that the occupancy the rates imply matches the occupancy the
frames show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GAS = 0.0019872041  # kcal/mol/K
LITRES = 1660.5390  # A^3 of one molecule per litre-mole: 1e27 / 6.02214076e23


@dataclass
class Dwell:
    """One stretch a copy spent in a state, and whether we saw it end."""

    run: int
    copy: int
    first: int  # first frame of the stretch
    frames: int
    bound: bool
    censored: bool  # the trajectory ended, the dwell did not

    def __len__(self) -> int:
        return self.frames


@dataclass
class Rates:
    """The kinetics of one site, with what they rest on."""

    site: int
    k_off: float  # 1/ns
    k_on: float  # 1/M/ns
    KD: float  # M
    dG: float  # kcal/mol
    residence_ns: float
    events: int  # completed departures: the error bar is made of these
    arrivals: int  # completed entries
    bound_ns: float
    unbound_ns: float
    concentration: float  # mean free-ligand concentration seen, M
    occupancy: float  # from frame counts
    occupancy_from_rates: float  # k_on [L] / (k_on [L] + k_off)
    dG_interval: tuple[float, float]  # 95% from resampling runs
    censored: int
    dwells: list[Dwell] = field(default_factory=list, repr=False)

    @property
    def consistent(self) -> bool:
        """Do the rates and the frame counts tell the same story?

        They are two routes to the same number, so disagreement means the
        states are wrong -- usually a boundary the ligand recrosses.
        """
        a, b = self.occupancy, self.occupancy_from_rates
        if not np.isfinite(b):
            return True  # nothing measured to contradict the frames
        return abs(a - b) <= 0.1 + 0.25 * max(a, b)

    def summary(self) -> str:
        lo, hi = self.dG_interval
        if np.isfinite(self.dG):
            head = (f"dG {self.dG:+.2f} kcal/mol [{lo:+.2f}, {hi:+.2f}] "
                    f"from {self.events} departures and {self.arrivals} arrivals")  # fmt: skip
        else:
            missing = [what for what, n in (("departures", self.events),
                                            ("arrivals", self.arrivals)) if not n]  # fmt: skip
            head = "no dG: no " + " and no ".join(missing) + " completed in these runs"
        out = [f"site {self.site}: {head}"]
        stay = f"{self.residence_ns:.2f} ns" if np.isfinite(self.residence_ns) else "not measured"
        held = f"{1e3 * self.KD:.3g} mM" if np.isfinite(self.KD) else "not measured"
        out.append(
            f"  residence {stay}, KD {held}, [L] {1e3 * self.concentration:.3g} mM, "
            f"bound for {self.bound_ns:.0f} of {self.bound_ns + self.unbound_ns:.0f} ns"
        )
        by_rates = (f", {self.occupancy_from_rates:.3f} by rates"
                    if np.isfinite(self.occupancy_from_rates) else "")  # fmt: skip
        out.append(
            f"  occupancy {self.occupancy:.3f} by frames{by_rates}"
            + ("" if self.consistent else "  <- these disagree: check the boundaries")
        )
        if np.isfinite(self.dG) and self.events < 10:
            out.append(f"  {self.events} departures is too few to rank: treat dG as a bound")
        return "\n".join(out)


def dwells(found, site: int, hysteresis: float = 2.0, quantile: float = 0.9) -> list[Dwell]:
    """The stretches each copy spent in ``site`` and out of it.

    A copy enters when its centroid comes within ``r_in`` -- the distance
    holding ``quantile`` of the site's own frames -- and leaves only past
    ``hysteresis`` times that.  The first and last stretch of every copy are
    censored: they were cut by the trajectory, not by the ligand.
    """
    s = found.sites[site]
    away = np.sqrt(((found.centroids - s.center) ** 2).sum(1))
    r_in = float(np.quantile(away[s.points], quantile))
    r_out = r_in * float(hysteresis)
    rows = found.where
    out = []
    for run, copy in map(tuple, np.unique(rows[:, :2], axis=0)):
        mine = np.flatnonzero((rows[:, 0] == run) & (rows[:, 1] == copy))
        mine = mine[np.argsort(rows[mine, 2])]
        d, frames = away[mine], rows[mine, 2]
        state = bool(d[0] <= r_in)
        start = 0
        for i in range(1, len(d)):
            leaving = state and d[i] > r_out
            entering = not state and d[i] <= r_in
            if leaving or entering:
                out.append(Dwell(int(run), int(copy), int(frames[start]), i - start, state,
                                 censored=start == 0))  # fmt: skip
                state, start = not state, i
        out.append(Dwell(int(run), int(copy), int(frames[start]), len(d) - start, state,
                         censored=True))  # fmt: skip
    return out


def _totals(items, interval_ns: float, free: np.ndarray, volume: float):
    """(bound ns, unbound ns weighted by concentration, departures, arrivals, mean [L])."""
    bound = sum(d.frames for d in items if d.bound) * interval_ns
    unbound_frames = [d for d in items if not d.bound]
    exposure, unbound = 0.0, 0.0
    for d in unbound_frames:
        rows = slice(d.first, d.first + d.frames)
        conc = free[d.run, rows] * LITRES / volume
        exposure += float(conc.sum()) * interval_ns
        unbound += d.frames * interval_ns
    left = sum(1 for d in items if d.bound and not d.censored)
    came = sum(1 for d in items if not d.bound and not d.censored)
    return bound, unbound, exposure, left, came


def kinetics(system, found, site: int, interval_ns: float, temperature: float = 310.0,
             hysteresis: float = 2.0, quantile: float = 0.9, bootstrap: int = 400,
             seed: int = 0, volume_A3: float | None = None) -> Rates:  # fmt: skip
    """Rates, residence time and dG of one site, with an interval from resampling runs.

    ``interval_ns`` is the time between frames.  The free-ligand
    concentration is counted per frame, from the copies not in the site and
    the box volume, so it falls as copies bind.  The volume is the mean of
    the boxes the frames actually had, which under a barostat is not the box
    the structure file carries; ``volume_A3`` overrides it.  The interval comes from
    resampling whole runs with replacement, which carries run-to-run
    disagreement that a Poisson count cannot see; with one run it is the
    dwells that are resampled instead.

    Rates rest on completed events.  A dwell the trajectory cut short counts
    its time and not its ending, so a run stopped early -- by the wall clock
    or by ``--early-stop`` -- biases nothing.
    """
    if volume_A3 is None:  # what the frames had, not what the structure file says
        volume_A3 = getattr(found, "volume", 0.0) or abs(
            float(np.linalg.det(np.asarray(system.cell, float)))
        )
    if volume_A3 <= 0:
        raise ValueError("the system has no periodic cell: give volume_A3")
    items = dwells(found, site, hysteresis, quantile)
    rows = found.where
    runs = int(rows[:, 0].max()) + 1
    nframes = int(rows[:, 2].max()) + 1
    inside = np.zeros((runs, nframes))  # copies of this site per run and frame
    total = np.zeros((runs, nframes))
    for d in items:
        rows_of = slice(d.first, d.first + d.frames)
        total[d.run, rows_of] += 1
        if d.bound:
            inside[d.run, rows_of] += 1
    free = np.maximum(total - inside, 0.0)

    def measure(chosen):
        picked = [d for d in items if d.run in chosen]
        return _totals(picked, interval_ns, free, volume_A3)

    bound, unbound, exposure, left, came = measure(set(range(runs)))
    if bound <= 0 or exposure <= 0:
        raise ValueError(f"site {site} is never entered or never left: no rates from it")
    k_off = left / bound
    k_on = came / exposure
    conc = exposure / unbound if unbound > 0 else 0.0
    KD = (k_off / k_on) if k_on > 0 else np.inf
    dG = GAS * temperature * np.log(KD) if np.isfinite(KD) and KD > 0 else np.nan

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(bootstrap, 0)):
        if runs > 1:
            chosen = rng.integers(0, runs, runs)
            picked = [d for r in chosen for d in items if d.run == r]
        else:
            picked = list(rng.choice(np.array(items, dtype=object), len(items)))
        b, _u, e, lf, cm = _totals(picked, interval_ns, free, volume_A3)
        if b > 0 and e > 0 and lf > 0 and cm > 0:
            draws.append(GAS * temperature * np.log((lf / b) / (cm / e)))
    span = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) if draws \
        else (np.nan, np.nan)  # fmt: skip

    occupancy = float(inside.sum() / max(total.sum(), 1))
    # both rates have to be measured for this to say anything the frames do not
    by_rates = ((k_on * conc) / (k_on * conc + k_off) if k_off > 0 and came and left
                else float("nan"))  # fmt: skip
    return Rates(site=site, k_off=k_off, k_on=k_on, KD=KD, dG=dG,
                 residence_ns=(1.0 / k_off) if k_off > 0 else np.inf,
                 events=left, arrivals=came, bound_ns=bound, unbound_ns=unbound,
                 concentration=conc, occupancy=occupancy, occupancy_from_rates=by_rates,
                 dG_interval=span, censored=sum(1 for d in items if d.censored),
                 dwells=items)  # fmt: skip
