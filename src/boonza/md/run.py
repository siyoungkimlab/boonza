"""Running a workflow: a new run (build, minimize, NVT, NPT, production) or
a restart of production toward its absolute target."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import monitor as mon
from .config import commit_settings, restore_settings, settings_of, write_settings
from .performance import PerformanceReporter, PerformanceTracker, TimedReporter
from .platforms import create_simulation, describe_platform, describe_system
from .prepare import build_system, save_structure, write_components


@dataclass(frozen=True)
class RunPaths:
    """The files of one work directory (ommflow's names, plus ``solvated.dms``,
    the system with its force field)."""

    workdir: Path

    def __getattr__(self, name):
        files = {
            "solvated_dms": "solvated.dms",
            "solvated_pdb": "solvated.pdb",
            "solvated_mae": "solvated.mae",
            "components_json": "components.json",
            "system_xml": "system.xml",
            "integrator_xml": "integrator.xml",
            "checkpoint": "checkpoint.chk",
            "final_configuration": "final.toml",
            "equilibrated_pdb": "equilibrated.pdb",
            "equilibrated_mae": "equilibrated.mae",
            "final_pdb": "final.pdb",
            "final_mae": "final.mae",
            "equilibration_dcd": "equilibration.dcd",
            "equilibration_csv": "equilibration.csv",
            "trajectory_dcd": "trajectory.dcd",
            "state_csv": "state.csv",
            "pocket_json": "pocket.json",
            "dihedral_restraints_csv": "dihedral_restraints.csv",
            "dihedral_restraints_png": "dihedral_restraints.png",
            "performance_csv": "performance.csv",
            "monitor_csv": "monitor.csv",
            "status_json": "status.json",
        }
        if name not in files:
            raise AttributeError(name)
        return self.workdir / files[name]


def is_restart(workdir: Path) -> bool:
    """A work directory with anything in it is a restart; an empty one (as
    batch schedulers make) is a new run."""
    return workdir.exists() and (not workdir.is_dir() or any(workdir.iterdir()))


def steps_for(ns: float, dt_ns: float) -> int:
    raw = ns / dt_ns
    steps = round(raw)
    if steps < 1:
        raise ValueError("every duration must be at least one time step")
    if abs(raw - steps) > 1e-6:
        raise ValueError(f"{ns:g} ns is not a whole number of {dt_ns * 1e6:g} fs steps")
    return steps


def intervals(args, dt_ns: float) -> dict[str, int]:
    """Durations in steps; checkpoints must line up with frames and checks."""
    n = {
        k: steps_for(getattr(args, k), dt_ns)
        for k in (
            "equilibration_ns",
            "equilibration_report_interval_ns",
            "production_ns",
            "production_report_interval_ns",
            "checkpoint_interval_ns",
            "performance_interval_ns",
        )
    }
    if n["production_report_interval_ns"] % n["checkpoint_interval_ns"]:
        raise ValueError(
            "checkpoint_interval_ns must divide production_report_interval_ns, so "
            "every frame has a checkpoint to resume from"
        )
    if args.early_stop:
        n["monitor_interval_ns"] = steps_for(args.monitor_interval_ns, dt_ns)
        if n["monitor_interval_ns"] % n["checkpoint_interval_ns"]:
            raise ValueError("monitor_interval_ns must be a multiple of checkpoint_interval_ns")
    return n


def _state_reporter(path, steps, append=False):
    from openmm import app

    return app.StateDataReporter(
        str(path),
        steps,
        append=append,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        density=True,
        volume=True,
        separator=",",
    )


def _save_frame(s, state, stem: Path, title: str) -> None:
    from openmm import unit

    pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    box = np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))
    for ext in (".pdb", ".mae"):
        save_structure(s, stem.with_suffix(ext), positions=np.asarray(pos), box=box)


def _current_steps(simulation, dt) -> int:
    """The production step a restored checkpoint is at (OpenMM's clock drifts
    by about 1e-9 per step, so a small tolerance)."""
    raw = simulation.context.getState().getTime() / dt
    steps = round(raw)
    if abs(raw - steps) > 0.01:
        raise ValueError(
            f"the checkpoint time is {raw:g} steps: it was written with another integration_fs"
        )
    return steps


def _new_run(args, paths: RunPaths, src: Path, log):
    """Build a new run's system, OpenMM system and integrator, and write
    its files up to ``final.toml``."""
    import openmm as mm
    from openmm import unit

    from ..io import save
    from ..omm import to_openmm

    shutil.copy2(src, paths.workdir / f"input{''.join(src.suffixes).lower()}")
    # a wrong or missing early-stop target fails before anything is built
    check = (lambda found: mon.select_target(found, args)) if args.early_stop else None
    s, info = build_system(args, paths.workdir, log, check=check)
    write_components(paths.components_json, info, args)
    save(s, paths.solvated_dms)
    for p in (paths.solvated_pdb, paths.solvated_mae):
        save_structure(s, p)
    topology, system, positions = to_openmm(s, nonbonded_method="PME", cutoff=10.0 * args.cutoff_nm)
    if args.dihedral_restraint != "none":
        from .restraints import add_dihedral_restraints, plot_well, write_records

        records, what = add_dihedral_restraints(
            system, s, args.dihedral_restraint, args.dihedral_restraint_kJ,
            getattr(args, "dihedral_restraint_selection", None),
        )  # fmt: skip
        write_records(paths.dihedral_restraints_csv, records)
        plot_well(paths.dihedral_restraints_png, args.dihedral_restraint_kJ)
        log(
            f"Dihedral restraints: {len(records)} backbone torsions over {what}, "
            f"K = {-abs(args.dihedral_restraint_kJ):g} kJ/mol"
        )
    if args.repulsion_selection:
        from .restraints import add_repulsion

        n = add_repulsion(system, s, args.repulsion_selection, args.repulsion_distance_nm,
                          args.repulsion_kJ)  # fmt: skip
        if n < 2:
            raise ValueError(f"repulsion_selection {args.repulsion_selection!r} picks {n} "
                             "molecules; it needs at least two")  # fmt: skip
        log(f"Repulsion between {n} molecules ({args.repulsion_selection}): heavy atoms of "
            f"different ones closer than {args.repulsion_distance_nm:g} nm, "
            f"k = {args.repulsion_kJ:g} kJ/mol/nm^2")  # fmt: skip
    integrator = mm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin, 1 / unit.picosecond, args.integration_fs * unit.femtoseconds
    )
    integrator.setRandomNumberSeed(args.seed)
    write_settings(paths.final_configuration, settings_of(args))
    return s, info, topology, system, positions, integrator


def run_workflow(args, log=print) -> None:
    """A new run in an empty work directory, else a restart of the one there."""
    import openmm as mm
    from openmm import app, unit

    from ..io import load
    from ..omm import _topology

    paths = RunPaths(Path(args.workdir))
    resume = is_restart(paths.workdir)
    if resume and not paths.workdir.is_dir():
        raise NotADirectoryError(f"{paths.workdir} exists and is not a directory")
    info = None
    if not resume:
        if args.input_structure is None:
            raise ValueError("a new run needs INPUT_STRUCTURE")
        src = Path(args.input_structure)
        if not src.is_file():
            raise FileNotFoundError(f"{src} does not exist")
        if not args.early_stop and any(getattr(args, k) is not None for k in mon.MONITOR_SELECTORS):
            raise ValueError("an early-stop target is chosen but early_stop is off")
        created = not paths.workdir.exists()
        paths.workdir.mkdir(parents=True, exist_ok=True)
        try:
            s, info, topology, system, positions, integrator = _new_run(args, paths, src, log)
        except BaseException:
            # the directory was new or empty: leave it so, and the fixed command starts afresh
            if created:
                shutil.rmtree(paths.workdir, ignore_errors=True)
            else:
                for child in paths.workdir.iterdir():
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            raise
    else:
        for p in (
            paths.solvated_dms,
            paths.system_xml,
            paths.integrator_xml,
            paths.checkpoint,
            paths.final_configuration,
        ):
            if not p.is_file():
                raise FileNotFoundError(f"cannot resume: {p} is missing")
        restore_settings(args, paths.final_configuration)
        s = load(paths.solvated_dms)
        topology = _topology(s, app, s.atoms["anum"])
        system = mm.XmlSerializer.deserialize(paths.system_xml.read_text(encoding="utf-8"))
        integrator = mm.XmlSerializer.deserialize(paths.integrator_xml.read_text(encoding="utf-8"))
        if not args.early_stop and any(k in args.specified for k in mon.MONITOR_SELECTORS):
            raise ValueError("an early-stop target is chosen but early_stop is off")

    simulation, notes = create_simulation(
        topology, system, integrator, args.platform, args.precision, "precision" in args.specified
    )
    log(describe_system(s, system))
    log(describe_platform(simulation.context))
    for note in notes:
        log(f"Note: {note}")
    dt = integrator.getStepSize()
    dt_ns = dt.value_in_unit(unit.nanoseconds)
    if (
        resume
        and "integration_fs" in args.specified
        and abs(dt_ns * 1e6 - args.integration_fs) > 1e-8
    ):
        raise ValueError("integration_fs cannot change when resuming: integrator.xml fixes it")
    n = intervals(args, dt_ns)

    monitor = None
    if resume:
        simulation.loadCheckpoint(str(paths.checkpoint))
        log(f"Resuming from {paths.checkpoint}")
        if args.early_stop:
            monitor = mon.load_pocket(paths.pocket_json)
            mon.check_saved(monitor, args, system.getNumParticles())
        commit_settings(args, paths.final_configuration)
    else:
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
        simulation.reporters.append(
            app.DCDReporter(str(paths.equilibration_dcd), n["equilibration_report_interval_ns"])
        )
        simulation.reporters.append(
            _state_reporter(paths.equilibration_csv, n["equilibration_report_interval_ns"])
        )
        log(f"Running {args.equilibration_ns:g} ns NVT equilibration...")
        simulation.step(n["equilibration_ns"])
        log(f"Running {args.equilibration_ns:g} ns NPT equilibration...")
        system.addForce(
            mm.MonteCarloBarostat(args.pressure * unit.bar, args.temperature * unit.kelvin, 25)
        )
        simulation.context.reinitialize(preserveState=True)
        simulation.step(n["equilibration_ns"])
        simulation.reporters.clear()
        state = simulation.context.getState(getPositions=True)
        _save_frame(s, state, paths.equilibrated_pdb, "equilibrated")
        simulation.context.setTime(0 * unit.picoseconds)
        simulation.currentStep = 0
        paths.system_xml.write_text(mm.XmlSerializer.serialize(system), encoding="utf-8")
        paths.integrator_xml.write_text(mm.XmlSerializer.serialize(integrator), encoding="utf-8")
        simulation.saveCheckpoint(str(paths.checkpoint))
        if args.early_stop:
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            box = np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            monitor = mon.initialize(s, np.asarray(pos), box, info, args)
            mon.write_pocket(paths.pocket_json, monitor)
            mon.write_status(
                paths.status_json, mon.status("running", monitor, args.production_ns, 0.0, 0, 0)
            )
    _production(simulation, s, args, paths, n, dt, resume, monitor, log)
    log(f"Finished. Results are in {paths.workdir}")


def _production(simulation, s, args, paths: RunPaths, n, dt, append, monitor, log) -> None:
    from openmm import app, unit

    dt_ns = dt.value_in_unit(unit.nanoseconds)
    step = _current_steps(simulation, dt)
    target = n["production_ns"]
    prior = mon.load_status(paths.status_json)
    if monitor is not None and prior is not None:
        done = prior.get("target_production_ns")
        if prior.get("final_production_step", 0) > step:
            raise ValueError(
                "status.json is newer than the checkpoint; restore the matching "
                "checkpoint before resuming"
            )
        if prior.get("outcome") == "detached" and (
            "production_ns" not in args.specified or done is None or args.production_ns <= done
        ):
            log(
                "Production stays stopped after the confirmed detachment: turn early_stop "
                "off, or give a larger production_ns, to continue."
            )
            return
    tracker = PerformanceTracker()
    for task, reporter in (
        (
            "trajectory",
            app.DCDReporter(
                str(paths.trajectory_dcd), n["production_report_interval_ns"], append=append
            ),
        ),
        ("state", _state_reporter(paths.state_csv, n["production_report_interval_ns"], append)),
        ("checkpoint", app.CheckpointReporter(str(paths.checkpoint), n["checkpoint_interval_ns"])),
    ):
        simulation.reporters.append(TimedReporter(reporter, tracker, task))
    simulation.reporters.append(
        PerformanceReporter(
            paths.performance_csv,
            n["performance_interval_ns"],
            tracker,
            dt_ns,
            append=append,
            initial_step=step,
        )
    )
    tracker.start()
    detached, count = False, 0
    if target > step:
        log(
            f"Running {(target - step) * dt_ns:g} ns to reach {args.production_ns:g} ns of "
            "NPT production..."
        )
        if monitor is None:
            simulation.step(target - step)
            step = target
        else:
            count = mon.restore_count(paths.monitor_csv, step) if append else 0
            detached, count, step = _monitored(
                simulation, monitor, args, paths, n, step, target, count, tracker, dt_ns
            )
    else:
        log(f"The production target of {args.production_ns:g} ns is already reached.")
    log(tracker.summary())
    state = simulation.context.getState(getPositions=True)
    _save_frame(s, state, paths.final_pdb, "final")
    time_ns = state.getTime().value_in_unit(unit.nanoseconds)
    if monitor is not None:
        simulation.saveCheckpoint(str(paths.checkpoint))
        mon.write_status(
            paths.status_json,
            mon.status(
                "detached" if detached else "target_reached",
                monitor,
                args.production_ns,
                time_ns,
                step,
                count,
            ),
        )
        if detached:
            log(
                f"Confirmed detachment of {monitor.ligand_id} at {time_ns:g} ns; production "
                "stopped early."
            )
    elif prior is not None:
        prior.update(
            outcome="target_reached",
            target_production_ns=args.production_ns,
            final_production_time_ns=time_ns,
            final_production_step=step,
            consecutive_detached_count=0,
            early_stop_enabled=False,
        )
        mon.write_status(paths.status_json, prior)


def _monitored(simulation, m, args, paths, n, step, target, count, tracker, dt_ns):
    from openmm import unit

    every = n["monitor_interval_ns"]
    while step < target:
        todo = min((step // every + 1) * every - step, target - step)
        simulation.step(todo)
        step += todo
        if step % every:
            continue
        state = simulation.context.getState(getPositions=True)
        with tracker.measure("checkpoint"):
            simulation.saveCheckpoint(str(paths.checkpoint))
        with tracker.measure("monitor"):
            pos = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            box = np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            dist, contacts = mon.measure(
                m.ligand_heavy_atom_indices, m.pocket_atom_indices, pos, box, args.contact_cutoff_nm
            )
        count, confirmed = mon.detachment_update(
            dist, contacts, args.detach_cutoff_nm, count, args.confirmation_checks
        )
        time_ns = state.getTime().value_in_unit(unit.nanoseconds)
        mon.append_row(
            paths.monitor_csv,
            {
                "production_time_ns": f"{time_ns:.12g}",
                "step": step,
                "min_ligand_pocket_distance_nm": f"{dist:.12g}",
                "contact_count": contacts,
                "initial_contact_count": m.initial_contact_count,
                "contact_fraction": (
                    f"{contacts / m.initial_contact_count:.12g}" if m.initial_contact_count else ""
                ),
                "consecutive_detached_count": count,
                "detached": str(confirmed).lower(),
            },
        )
        if confirmed and step < target:
            return True, count, step
        mon.write_status(
            paths.status_json, mon.status("running", m, args.production_ns, time_ns, step, count)
        )
    return False, count, step


def load_components(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
