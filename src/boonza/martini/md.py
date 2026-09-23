"""Starting Martini dynamics in OpenMM."""

from __future__ import annotations

WARM_UP = (0.002, 0.005, 0.010)  # ps


def equilibrate(simulation, temperature: float = 310.0, timestep: float = 0.020,
                steps: int = 2000, seed: int = 1) -> None:  # fmt: skip
    """Minimize, give velocities at ``temperature`` (K), and step up to ``timestep`` (ps).

    A 20 fs step straight from a minimum can blow up even where 20 fs is
    stable afterwards: OpenMM's minimizer stops higher than GROMACS's, and
    the first large steps meet the strain it leaves.  So after a tight
    minimization (1 kJ/mol/nm) this runs ``steps`` steps at each of 2, 5 and
    10 fs before leaving the integrator at ``timestep``, as Martini
    tutorials do.  The integrator must have a step size to set (Langevin
    middle, Verlet, ...).
    """
    import openmm.unit as u

    ctx = simulation.context
    ctx.computeVirtualSites()
    simulation.minimizeEnergy(tolerance=1.0, maxIterations=20000)
    ctx.setVelocitiesToTemperature(temperature * u.kelvin, seed)
    integrator = simulation.integrator
    for dt in (d for d in WARM_UP if d < timestep):
        integrator.setStepSize(dt * u.picosecond)
        simulation.step(steps)
    integrator.setStepSize(timestep * u.picosecond)
