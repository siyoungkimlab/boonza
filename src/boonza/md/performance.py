"""Where the wall time of production goes.

OpenMM reporters are duck-typed, so each real reporter is wrapped in a proxy
that times its ``report``; the workflow times its own checkpoint and monitor
calls, and the rest is integration.
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path

FIELDS = (
    "production_time_ns",
    "step",
    "wall_s",
    "interval_s",
    "ns_per_day",
    "md_s",
    "trajectory_s",
    "state_s",
    "checkpoint_s",
    "monitor_s",
    "reported_pct",
)
TASKS = ("trajectory", "state", "checkpoint", "monitor")


class PerformanceTracker:
    """Wall time per task since production started in this process."""

    def __init__(self):
        self.totals = {t: 0.0 for t in TASKS}
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    @property
    def started(self) -> bool:
        return self._start is not None

    def elapsed(self) -> float:
        return 0.0 if self._start is None else time.perf_counter() - self._start

    @contextmanager
    def measure(self, task: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[task] = self.totals.get(task, 0.0) + time.perf_counter() - t0

    def summary(self) -> str:
        wall = self.elapsed()
        reported = sum(self.totals.values())
        lines = [
            f"Production wall time: {wall:.1f} s ({reported / wall * 100:.2f}% in "
            "reporting and checkpointing)"
            if wall > 0
            else "Production wall time: 0.0 s"
        ]
        for task in TASKS:
            sec = self.totals.get(task, 0.0)
            lines.append(f"  {task:<12} {sec:8.2f} s  {sec / wall * 100 if wall else 0:6.2f}%")
        rest = wall - reported
        lines.append(
            f"  {'integration':<12} {rest:8.2f} s  {rest / wall * 100 if wall else 0:6.2f}%"
        )
        return "\n".join(lines)


class TimedReporter:
    """A reporter whose ``report`` time goes to one task."""

    def __init__(self, reporter, tracker: PerformanceTracker, task: str):
        self._reporter, self._tracker, self._task = reporter, tracker, task

    def describeNextReport(self, simulation):  # noqa: N802 - OpenMM's interface
        return self._reporter.describeNextReport(simulation)

    def report(self, simulation, state):
        with self._tracker.measure(self._task):
            self._reporter.report(simulation, state)

    def __getattr__(self, name):
        return getattr(self._reporter, name)


class PerformanceReporter:
    """One row of ``performance.csv`` every ``interval`` steps."""

    def __init__(
        self,
        path,
        interval: int,
        tracker: PerformanceTracker,
        timestep_ns: float,
        append: bool = False,
        initial_step: int = 0,
    ):
        self._path, self._interval = Path(path), interval
        self._tracker, self._dt = tracker, timestep_ns
        self._previous = (initial_step * timestep_ns, 0.0)
        if not append or not self._path.is_file() or self._path.stat().st_size == 0:
            with self._path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    def describeNextReport(self, simulation):  # noqa: N802 - OpenMM's interface
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        wall = self._tracker.elapsed()
        ns = simulation.currentStep * self._dt
        prev_ns, prev_wall = self._previous
        interval = wall - prev_wall
        self._previous = (ns, wall)
        t = self._tracker.totals
        reported = sum(t.values())
        row = {
            "production_time_ns": f"{ns:.12g}",
            "step": simulation.currentStep,
            "wall_s": f"{wall:.3f}",
            "interval_s": f"{interval:.3f}",
            "ns_per_day": f"{(ns - prev_ns) / interval * 86400 if interval > 0 else 0:.3f}",
            "md_s": f"{wall - reported:.3f}",
            **{f"{k}_s": f"{t.get(k, 0.0):.3f}" for k in TASKS},
            "reported_pct": f"{reported / wall * 100:.3f}" if wall > 0 else "0",
        }
        with self._path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
