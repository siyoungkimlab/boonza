"""Hamiltonian replica exchange over states that differ in global parameters.

Each state is a set of a context's global parameters (a hybrid's lambdas, by
:func:`boonza.fep.lambdas.values`).  One replica per state: each iteration
moves every replica by Langevin dynamics in its state, computes its reduced
potential in every state, then swaps states between replicas (Metropolis,
"swap-all" as openmmtools does: many random pair attempts, so a replica can
travel far in one iteration).  One OpenMM context serves every replica, which
keeps GPU memory at one copy of the system.

What a run keeps (``<stem>.npz``, rewritten at each checkpoint):

    u_kln      iterations x replicas x states: reduced potentials (energy / kT)
    states     iterations x replicas: the state each replica was in
    and the replicas' positions, velocities and boxes, to continue from.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KB = 0.0083144626  # kJ/mol/K


@dataclass
class Replicas:
    """The replicas' coordinates and where the run has got to."""

    positions: list  # per replica, nm (n, 3)
    velocities: list
    boxes: list  # per replica, (3, 3) nm
    states: np.ndarray  # state index of each replica
    u_kln: list = field(default_factory=list)  # one (replicas, states) array per iteration
    history: list = field(default_factory=list)  # states per iteration
    iteration: int = 0
    equilibration: int = 0  # iterations run before the first one kept

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, positions=np.asarray(self.positions), velocities=np.asarray(self.velocities),
                 boxes=np.asarray(self.boxes), states=self.states,
                 u_kln=np.asarray(self.u_kln).reshape(-1, len(self.states), len(self.states)),
                 history=np.asarray(self.history, dtype=np.int32).reshape(-1, len(self.states)),
                 iteration=self.iteration, equilibration=self.equilibration)  # fmt: skip
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> Replicas:
        d = np.load(path)
        return cls(positions=list(d["positions"]), velocities=list(d["velocities"]),
                   boxes=list(d["boxes"]), states=d["states"], u_kln=list(d["u_kln"]),
                   history=list(d["history"]), iteration=int(d["iteration"]),
                   equilibration=int(d["equilibration"]))  # fmt: skip


class ReplicaExchange:
    """Replica exchange of ``system`` over ``states`` (dicts of global parameters).

    ``temperature`` in K, ``timestep`` in ps, ``steps`` MD steps per iteration
    (openfe: 2.5 ps); ``platform`` an OpenMM platform name or None for the fastest.
    """

    def __init__(self, system, states: list[dict], temperature: float = 298.15,
                 timestep: float = 0.004, steps: int = 625, collision: float = 1.0,
                 platform: str | None = None, properties: dict | None = None,
                 seed: int = 0):  # fmt: skip
        import openmm

        self.mm = openmm
        self.states = [dict(s) for s in states]
        if len(self.states) < 2:
            raise ValueError("replica exchange needs at least two states")
        self.kT = KB * temperature
        self.steps = int(steps)
        self.rng = np.random.default_rng(seed)
        integrator = openmm.LangevinMiddleIntegrator(temperature, collision, timestep)
        if seed:
            integrator.setRandomNumberSeed(int(seed))
        args = [system, integrator]
        if platform:
            args.append(openmm.Platform.getPlatformByName(platform))
            if properties:
                args.append(properties)
        self.context = openmm.Context(*args)
        self.integrator = integrator
        self.temperature = temperature
        names = {k for s in self.states for k in s}
        missing = names - set(self.context.getParameters())
        if missing:
            raise ValueError(f"the system has no global parameters {sorted(missing)}")

    # --- one replica at a time -------------------------------------------------------

    def _set_state(self, k: int) -> None:
        for name, value in self.states[k].items():
            self.context.setParameter(name, value)

    def _load(self, r: Replicas, i: int) -> None:
        u = self.mm.unit
        self.context.setPeriodicBoxVectors(*(r.boxes[i] * u.nanometer))
        self.context.setPositions(r.positions[i] * u.nanometer)
        self.context.setVelocities(r.velocities[i] * u.nanometer / u.picosecond)

    def _store(self, r: Replicas, i: int) -> None:
        u = self.mm.unit
        s = self.context.getState(getPositions=True, getVelocities=True)
        r.positions[i] = s.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        r.velocities[i] = s.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
        r.boxes[i] = np.asarray(s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))

    def _energies(self) -> np.ndarray:
        """The loaded configuration's reduced potential in every state."""
        u = self.mm.unit
        out = np.empty(len(self.states))
        for k in range(len(self.states)):
            self._set_state(k)
            e = self.context.getState(getEnergy=True).getPotentialEnergy()
            out[k] = e.value_in_unit(u.kilojoule_per_mole) / self.kT
        return out

    # --- the run ----------------------------------------------------------------------

    def start(self, positions, box=None, minimize: int = 0) -> Replicas:
        """Replicas all at ``positions`` (nm), each minimized in its state for up to
        ``minimize`` iterations (0: until converged), velocities at temperature."""
        u = self.mm.unit
        K = len(self.states)
        xyz = np.asarray(positions, float)
        r = Replicas(positions=[xyz.copy() for _ in range(K)], velocities=[None] * K,
                     boxes=[None] * K, states=np.arange(K))  # fmt: skip
        vectors = self.context.getSystem().getDefaultPeriodicBoxVectors() if box is None else (
            [v * u.nanometer for v in np.asarray(box, float)])  # fmt: skip
        for i in range(K):
            self.context.setPeriodicBoxVectors(*vectors)
            self.context.setPositions(xyz * u.nanometer)
            self._set_state(i)
            self.mm.LocalEnergyMinimizer.minimize(self.context, 10.0, minimize)
            self.context.setVelocitiesToTemperature(self.temperature,
                                                    int(self.rng.integers(1, 2**31)))  # fmt: skip
            self._store(r, i)
        return r

    def iterate(self, r: Replicas, n: int, keep: bool = True, checkpoint=None,
                every: int = 0, log=None) -> Replicas:  # fmt: skip
        """``n`` iterations; ``keep``: record their energies (production), else
        they count as equilibration.  ``checkpoint``: a path written every
        ``every`` iterations and at the end."""
        K = len(self.states)
        started = time.time()
        for it in range(n):
            u = np.empty((K, K))
            for i in range(K):
                self._load(r, i)
                self._set_state(int(r.states[i]))
                self.integrator.step(self.steps)
                self._store(r, i)
                u[i] = self._energies()
            self._swap(r, u)
            r.iteration += 1
            if keep:
                r.u_kln.append(u)
                r.history.append(r.states.copy())
            else:
                r.equilibration += 1
            if checkpoint and every and (it + 1) % every == 0:
                r.save(checkpoint)
            if log and (it + 1) % max(1, n // 10) == 0:
                rate = (it + 1) / (time.time() - started)
                log(f"  iteration {r.iteration} ({rate:.2f}/s)")
        if checkpoint:
            r.save(checkpoint)
        return r

    def _swap(self, r: Replicas, u: np.ndarray) -> None:
        """Swap-all: K^3 random pair attempts on the state assignment."""
        K = len(self.states)
        s = r.states
        for _ in range(K**3):
            i, j = self.rng.integers(0, K, 2)
            if i == j:
                continue
            a, b = s[i], s[j]
            log_p = -(u[i, b] + u[j, a]) + (u[i, a] + u[j, b])
            if log_p >= 0 or self.rng.random() < math.exp(log_p):
                s[i], s[j] = b, a
