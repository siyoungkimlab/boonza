"""boonza fep run: one leg's replica exchange, one repeat, restartable.

Reads what boonza fep prep wrote (system.xml.gz, state.npz, leg.json), runs
openfe's protocol -- minimization in each state, ``equilibration_ns`` of
exchanges not kept, then ``production_ns`` kept, exchanges every 2.5 ps -- and
writes ``run_<repeat>.npz`` as it goes (boonza.fep.sampler.Replicas); started
again it carries on from there.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from . import lambdas
from .sampler import ReplicaExchange, Replicas

ITERATION_PS = 2.5  # between exchanges (openfe)
CHECKPOINT_ITERATIONS = 40  # 100 ps
#: below this many particles (a vacuum leg) the CPU platform on one thread is
#: the fastest: a 16-atom leg ran 5400 steps/s on one thread, 680 on twelve
ONE_THREAD_BELOW = 2000


def run_leg(directory, repeat: int = 1, platform: str | None = None, minimize: int = 5000,
            log=print) -> Path:  # fmt: skip
    """Run (or carry on) repeat ``repeat`` of the leg in ``directory``; the run's file.
    ``minimize``: iterations of minimization in each state first (openfe: 5000)."""
    import openmm

    d = Path(directory)
    meta = json.loads((d / "leg.json").read_text())
    with gzip.open(d / "system.xml.gz", "rt") as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())
    state = np.load(d / "state.npz")
    dt_ps = meta["timestep_fs"] / 1000.0
    every = float(meta.get("iteration_ps", ITERATION_PS))
    steps = int(round(every / dt_ps))
    n_eq = int(round(meta["equilibration_ns"] * 1000.0 / every))
    n_prod = int(round(meta["production_ns"] * 1000.0 / every))
    states = [lambdas.values(x) for x in meta["lambdas"]]
    properties = None
    if platform in (None, "CPU") and system.getNumParticles() < ONE_THREAD_BELOW:
        platform, properties = "CPU", {"Threads": "1"}  # a GPU is no faster here
    rex = ReplicaExchange(system, states, temperature=meta["temperature"], timestep=dt_ps,
                          steps=steps, platform=platform, properties=properties,
                          seed=1000 * repeat + 1)  # fmt: skip
    path = d / f"run_{repeat}.npz"
    a, b = meta["edge"]
    if path.exists():
        r = Replicas.load(path)
        log(f"{a} -> {b} {meta['leg']}, repeat {repeat}: carrying on from iteration "
            f"{r.iteration}")  # fmt: skip
    else:
        log(f"{a} -> {b} {meta['leg']}, repeat {repeat}: {len(states)} states, minimizing")
        box = state["box"] if np.any(state["box"]) else None  # none in vacuum
        r = rex.start(state["positions"], box=box, minimize=minimize)
    left_eq = max(n_eq - r.equilibration, 0)
    if left_eq:
        log(f"  equilibration: {left_eq} iterations ({left_eq * every / 1000:g} ns)")
        rex.iterate(r, left_eq, keep=False, checkpoint=path, every=CHECKPOINT_ITERATIONS, log=log)
    left = max(n_prod - len(r.u_kln), 0)
    if left:
        log(f"  production: {left} iterations ({left * every / 1000:g} ns)")
        rex.iterate(r, left, keep=True, checkpoint=path, every=CHECKPOINT_ITERATIONS, log=log)
    r.save(path)
    log(f"  done: {len(r.u_kln)} production iterations in {path.name}")
    return path
