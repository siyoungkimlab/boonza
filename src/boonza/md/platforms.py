"""Choosing the OpenMM platform and precision, and the banner describing a run."""

from __future__ import annotations

import numpy as np

_PRECISION = "Precision"


def _candidates(name):
    import openmm as mm

    if name:
        return [mm.Platform.getPlatformByName(name)]
    platforms = [mm.Platform.getPlatform(k) for k in range(mm.Platform.getNumPlatforms())]
    return sorted(platforms, key=lambda p: p.getSpeed(), reverse=True)


def create_simulation(
    topology,
    system,
    integrator,
    platform: str | None,
    precision: str,
    precision_given: bool = False,
):
    """(Simulation, notes) on the fastest platform that works at ``precision``.

    A platform that cannot honour the default precision (Apple's OpenCL
    rejects mixed) falls back to its own, with a note; a precision given
    explicitly that cannot be met is an error.
    """
    notes, failures = [], []
    for p in _candidates(platform):
        name = p.getName()
        if _PRECISION not in p.getPropertyNames():
            if precision_given:
                notes.append(
                    f"the {name} platform has no {_PRECISION} setting; "
                    f"precision {precision} does not apply to it"
                )
            sim = _try(topology, system, integrator, p, {}, failures)
            if sim is not None:
                return sim, notes
            continue
        sim = _try(topology, system, integrator, p, {_PRECISION: precision}, failures)
        if sim is not None:
            return sim, notes
        if precision_given:
            raise RuntimeError(
                f"the {name} platform cannot run at precision {precision}: " + failures[-1]
            )
        sim = _try(topology, system, integrator, p, {}, failures)
        if sim is not None:
            notes.append(
                f"the {name} platform rejected {_PRECISION}={precision}; using its "
                "own default (give precision explicitly to make this an error)"
            )
            return sim, notes
    raise RuntimeError("no platform could run the simulation:\n  " + "\n  ".join(failures))


def _try(topology, system, integrator, platform, properties, failures):
    from openmm import app

    try:
        return app.Simulation(topology, system, integrator, platform, properties)
    except Exception as e:  # noqa: BLE001 - platforms raise assorted types
        how = ", ".join(f"{k}={v}" for k, v in properties.items()) or "defaults"
        failures.append(f"{platform.getName()} ({how}): {e}")
        return None


def describe_platform(context) -> str:
    """The platform a context runs on, with every property it exposes."""
    p = context.getPlatform()
    lines = [f"Platform: {p.getName()} (speed {p.getSpeed():g})"]
    for name in p.getPropertyNames():
        try:
            value = p.getPropertyValue(context, name)
        except Exception:  # noqa: BLE001 - a property may be unreadable
            value = "<unavailable>"
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)


def describe_system(s, omm_system) -> str:
    """The size and shape of the system about to be integrated."""
    import openmm as mm
    from openmm import unit

    nvs = sum(1 for k in range(omm_system.getNumParticles()) if omm_system.isVirtualSite(k))
    water_res = set(s.atoms["residue"][s.select("water").ids].tolist())
    frag = s.fragids
    single = np.bincount(frag)[frag] == 1 if len(frag) else frag
    ions = int((single & (s.atoms["anum"] > 1)).sum()) if len(frag) else 0
    lines = [
        f"System: {omm_system.getNumParticles()} particles ({nvs} virtual sites)",
        f"  chains: {s.nchains}, residues: {s.nresidues} ({len(water_res)} water, {ions} ion), "
        f"bonds: {s.nbonds}",
        f"  constraints: {omm_system.getNumConstraints()}, forces: "
        + ", ".join(f.getName() or type(f).__name__ for f in omm_system.getForces()),
    ]
    a, b, c = omm_system.getDefaultPeriodicBoxVectors()
    lengths = [v[k].value_in_unit(unit.nanometer) for k, v in enumerate((a, b, c))]
    lines.append(
        f"  box: {lengths[0]:.2f} x {lengths[1]:.2f} x {lengths[2]:.2f} nm "
        f"({lengths[0] * lengths[1] * lengths[2]:.1f} nm^3)"
    )
    for f in omm_system.getForces():
        if isinstance(f, mm.NonbondedForce):
            methods = {
                mm.NonbondedForce.NoCutoff: "NoCutoff",
                mm.NonbondedForce.PME: "PME",
                mm.NonbondedForce.Ewald: "Ewald",
                mm.NonbondedForce.LJPME: "LJPME",
                mm.NonbondedForce.CutoffPeriodic: "CutoffPeriodic",
                mm.NonbondedForce.CutoffNonPeriodic: "CutoffNonPeriodic",
            }
            cutoff = f.getCutoffDistance().value_in_unit(unit.nanometer)
            lines.append(
                f"  nonbonded: {methods.get(f.getNonbondedMethod(), '?')}, cutoff {cutoff:g} nm"
            )
            break
    return "\n".join(lines)
