"""Well-tempered metadynamics along the drmsd, and the reversible work c(t).

The bias a well-tempered run deposits converges to the free energy only up to a
time-dependent offset, and Tiwary and Parrinello (J. Phys. Chem. B 2015, 119, 736)
showed what that offset is and how to compute it from the bias alone:

    c(t) = 1/beta * log( int ds exp(beta*gamma/(gamma - 1) * V(s,t))
                       / int ds exp(beta/(gamma - 1) * V(s,t)) )

It is the reversible work done by the bias, and CTMD (J. Chem. Inf. Model. 2026,
66, 11929) uses it as a score: a ligand that resists being pushed off its pose takes
more work to move, so c(t) climbs higher.  PLUMED computes it as ``metad.rct``;
OpenMM has no equivalent, so :func:`reversible_work` does it here from the bias grid
``openmm.app.Metadynamics`` already keeps.

The defaults are the published ones (``amber_MD_template/plumed_template.dat`` of the
authors' repository), except that the coordinate is the drmsd (see
:mod:`boonza.ctmd.cv`), so its width and range may need looking at again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: kJ/mol/K
GAS_CONSTANT = 0.008314462618


@dataclass
class Settings:
    """How a CTMD run is biased and how long it goes on for.

    The published values, with the coordinate's width ``sigma``, its range
    ``grid`` and the escape ``cutoff`` in nm, ``height`` in kJ/mol and the times in
    ps.  ``commit`` is how long the ligand must stay past ``cutoff`` to count as
    gone, which stops the replica.
    """

    sigma: float = 0.015
    height: float = 1.75
    bias_factor: float = 10.0
    pace: float = 1.0
    grid: tuple[float, float] = (0.0, 1.5)
    bins: int = 200
    temperature: float = 303.0
    timestep: float = 0.002
    replicas: int = 3
    nanoseconds: float = 5.0
    filter_nanoseconds: float = 5.0
    stride: float = 0.2
    cutoff: float = 0.6
    commit: float = 200.0
    seed: int = 0
    friction: float = 1.0
    pressure: float | None = 1.0
    cv: dict = field(default_factory=dict)

    @property
    def kT(self) -> float:
        return GAS_CONSTANT * self.temperature


def reversible_work(bias, bias_factor: float, temperature: float) -> float:
    """c(t) in kJ/mol from a well-tempered bias grid (kJ/mol).

    The grid's spacing cancels in the ratio, so only the values matter; the sums are
    taken through their logs, since the exponents reach thousands at a bias of a few
    kJ/mol.  A flat bias gives back its own value, and no bias gives 0.
    """
    from scipy.special import logsumexp

    v = np.asarray(bias, float).ravel()
    if not v.size:
        return 0.0
    gamma = float(bias_factor)
    if gamma <= 1.0:
        raise ValueError(f"the bias factor is above 1 in well-tempered metadynamics, not {gamma}")
    kT = GAS_CONSTANT * float(temperature)
    hot = gamma / (gamma - 1.0) / kT
    cold = 1.0 / (gamma - 1.0) / kT
    return float(kT * (logsumexp(hot * v) - logsumexp(cold * v)))


def bias_grid(meta, bias_factor: float) -> np.ndarray:
    """The bias V(s,t) in kJ/mol, from what ``Metadynamics`` reports as free energy.

    ``getFreeEnergy`` returns ``-gamma/(gamma - 1) * V``, so this undoes that rather
    than reaching into the object.
    """
    import openmm.unit as u

    free = np.asarray(meta.getFreeEnergy().value_in_unit(u.kilojoule_per_mole), float)
    gamma = float(bias_factor)
    return -free * (gamma - 1.0) / gamma


def _openmm_system(system, forcefields=None, log=print):
    """``(topology, openmm system, index)``, parameterized if it is not already.

    ``index[a]`` is where atom ``a`` of ``system`` sits in the OpenMM system: the
    force field may reorder the atoms, and both the coordinate's pairs and the
    starting positions have to follow it.
    """
    from ..omm import to_openmm
    from ..viewer.edit import parameterize

    work, index = system, np.arange(system.natoms)
    if not system.tables:
        log("no force field on this structure: parameterizing it")
        work = parameterize(system, forcefields=forcefields, log=log)
        order = np.asarray(work.atoms["view_index"], np.int64)
        if len(order) != system.natoms or (order < 0).any():
            raise ValueError("the force field changed the atoms; prepare the structure first")
        index = np.empty(system.natoms, np.int64)
        index[order] = np.arange(system.natoms)  # the other way about
    topology, omm, _ = to_openmm(work, nonbonded_method="PME", cutoff=10.0)
    return topology, omm, index


def _simulation(topology, omm, settings: Settings, seed: int, platform=None, log=print):
    import openmm as mm
    import openmm.unit as u

    from ..md.platforms import create_simulation

    integrator = mm.LangevinMiddleIntegrator(settings.temperature * u.kelvin,
                                             settings.friction / u.picosecond,
                                             settings.timestep * u.picoseconds)  # fmt: skip
    if seed:
        integrator.setRandomNumberSeed(int(seed))
    sim, notes = create_simulation(topology, omm, integrator, platform, "mixed")
    for note in notes:
        log(note)
    return sim


def _write_colvar(path, time, cv, bias, work) -> None:
    """A COLVAR file in PLUMED's format, so the authors' analysis scripts read it too."""
    with open(path, "w") as f:
        f.write("#! FIELDS time d metad.bias metad.rct\n")
        for row in zip(time, cv, bias, work, strict=True):
            f.write(" {:f} {:f} {:f} {:f}\n".format(*row))


def run(system, workdir, reference=None, *, settings: Settings | None = None, forcefields=None,
        platform: str | None = None, log=print):  # fmt: skip
    """CTMD on one pose: the unbiased filter, then the biased replicas.

    ``system`` is the prepared, equilibrated complex (what ``boonza md`` leaves as
    ``equilibrated.dms``) and ``reference`` the structure the pose is measured
    against, which is the docked pose itself: the published protocol measures drift
    from the pose docking gave, not from where the unbiased run ended, so that the
    drift counts against the ligand.  Without one, ``system``'s own coordinates are
    the reference.

    The filter run is unbiased: a ligand that leaves the site there is scored 0 and
    no replica is run for it.  Each replica then deposits bias along the drmsd until
    its time is up or the ligand is gone (see :class:`Settings`), and writes a
    ``colvar_K.dat`` beside a ``bias_K.npy``.  Returns what
    :func:`boonza.ctmd.score.score_run` makes of them.
    """
    import copy

    import openmm as mm
    import openmm.unit as u
    from openmm.app import BiasVariable, Metadynamics

    from .cv import drmsd_force, drmsd_of
    from .score import score_run

    cfg = settings or Settings()
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    topology, template, index = _openmm_system(system, forcefields, log)
    start = np.asarray(system.positions, float)[np.argsort(index)] / 10.0 * u.nanometer
    _, npairs = drmsd_force(system, reference, index=index, **cfg.cv)
    log(f"drmsd on {npairs} pocket-ligand pairs; it starts at "
        f"{drmsd_of(system, reference=reference, **cfg.cv):.3f} nm")  # fmt: skip
    stride_steps = max(1, round(cfg.stride / cfg.timestep))
    pace_steps = max(1, round(cfg.pace / cfg.timestep))

    def stage():
        """A system and a coordinate of their own.

        Both are made again for every stage: OpenMM takes ownership of a force once it
        is in a system, so one coordinate cannot go into two, and a system carrying
        one replica's bias is not the system the next replica should start from.
        """
        omm = copy.deepcopy(template)
        if cfg.pressure is not None:
            omm.addForce(mm.MonteCarloBarostat(cfg.pressure * u.bar, cfg.temperature * u.kelvin))
        force, _ = drmsd_force(system, reference, index=index, **cfg.cv)
        return omm, force

    if cfg.filter_nanoseconds > 0:
        omm, force = stage()
        # weighted by zero, so it reports the coordinate without pulling on anything:
        # the filter run has to be unbiased
        probe = mm.CustomCVForce("0*cv")
        probe.addCollectiveVariable("cv", force)
        omm.addForce(probe)
        sim = _simulation(topology, omm, cfg, cfg.seed or 1, platform, log)
        sim.context.setPositions(start)
        sim.context.setVelocitiesToTemperature(cfg.temperature * u.kelvin, cfg.seed or 1)
        time, cv, escaped = [], [], None
        for step in range(round(cfg.filter_nanoseconds * 1000 / cfg.stride)):
            sim.step(stride_steps)
            time.append((step + 1) * cfg.stride)
            cv.append(float(probe.getCollectiveVariableValues(sim.context)[0]))
            escaped = _escape(np.array(time), np.array(cv), cfg)
            if escaped is not None:
                break
        _write_colvar(out / "filter.dat", time, cv, [0.0] * len(cv), [0.0] * len(cv))
        if escaped is not None:
            log(f"the ligand left the site after {escaped:g} ps of unbiased MD: scored 0")
            return score_run(out, cfg, log=log)
        start = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        log(f"{time[-1]:g} ps unbiased: drmsd {cv[-1]:.3f} nm, and it stayed")

    for k in range(1, cfg.replicas + 1):
        colvar = out / f"colvar_{k}.dat"
        if colvar.exists():
            log(f"replica {k}: {colvar.name} is there already")
            continue
        seed = (cfg.seed or 1) + k
        omm, force = stage()
        variable = BiasVariable(force, cfg.grid[0], cfg.grid[1], cfg.sigma, gridWidth=cfg.bins)
        meta = Metadynamics(omm, [variable], cfg.temperature * u.kelvin, cfg.bias_factor,
                            cfg.height * u.kilojoule_per_mole, pace_steps)  # fmt: skip
        sim = _simulation(topology, omm, cfg, seed, platform, log)
        sim.context.setPositions(start)
        sim.context.setVelocitiesToTemperature(cfg.temperature * u.kelvin, seed)
        time, cv, bias, work = [], [], [], []
        for step in range(round(cfg.nanoseconds * 1000 / cfg.stride)):
            meta.step(sim, stride_steps)
            grid = bias_grid(meta, cfg.bias_factor)
            time.append((step + 1) * cfg.stride)
            cv.append(float(meta.getCollectiveVariables(sim)[0]))
            bias.append(float(grid.max()))
            work.append(reversible_work(grid, cfg.bias_factor, cfg.temperature))
            if _escape(np.array(time), np.array(cv), cfg) is not None:
                break
        _write_colvar(colvar, time, cv, bias, work)
        np.save(out / f"bias_{k}.npy", bias_grid(meta, cfg.bias_factor))
        log(f"replica {k}: {time[-1]:g} ps, drmsd {cv[-1]:.3f} nm, c(t) {work[-1]:.1f} kJ/mol")
    return score_run(out, cfg, log=log)


def _escape(time, cv, cfg: Settings):
    """When the ligand went, or None, by the rule :mod:`boonza.ctmd.score` scores by."""
    from .score import escape_index

    where = escape_index(time, cv, cfg.cutoff, cfg.commit)
    return None if where is None else float(time[where])
