"""Make boonza's equilibrated Martini 3 water box (src/boonza/data/martini/water.gro).

    python tests/data/martini/make_water_box.py path/to/martini_v3.0.0.itp

1000 W beads on a lattice, minimized, warmed up at 2, 5 and 10 fs (a 20 fs
step straight from a minimum can blow up), then 5 ns of NPT in OpenMM
(300 K, 1 bar, 20 fs) with Martini 3's W parameters; the last frame is kept.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as u

import boonza
from boonza.io.gromacs import load_top
from boonza.martini import OPENMM_OPTIONS

OUT = Path(__file__).resolve().parents[3] / "src" / "boonza" / "data" / "martini" / "water.gro"


def main(martini_itp):
    martini_itp = Path(martini_itp).resolve()
    n, a = 10, 0.5  # nm
    grid = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) * a
    rng = np.random.default_rng(0)
    grid += rng.normal(0, 0.03, grid.shape)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "w.itp").write_text("[ moleculetype ]\nW 1\n[ atoms ]\n1 W 1 W W 1 0\n")
        top = f'#include "{martini_itp}"\n#include "w.itp"\n'
        top += f"[ system ]\nW\n[ molecules ]\nW {len(grid)}\n"
        (tmp / "topol.top").write_text(top)
        rows = [f"{k + 1:5d}W    {'W':>5s}{k + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                for k, (x, y, z) in enumerate(grid)]  # fmt: skip
        box = n * a
        (tmp / "w.gro").write_text(f"W\n{len(grid)}\n" + "\n".join(rows) + f"\n{box} {box} {box}\n")
        s = load_top(tmp / "topol.top", tmp / "w.gro")
    top, system, pos = boonza.to_openmm(s, **OPENMM_OPTIONS)
    system.addForce(mm.MonteCarloBarostat(1 * u.bar, 300 * u.kelvin, 25))
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picosecond)
    integ.setRandomNumberSeed(1)
    sim = mm.app.Simulation(top, system, integ, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(pos)
    sim.minimizeEnergy(tolerance=1.0, maxIterations=20000)
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, 1)
    for dt in (0.002, 0.005, 0.010, 0.020):
        integ.setStepSize(dt * u.picosecond)
        sim.step(2000)
    dens = []
    for _ in range(250):  # 5 ns
        sim.step(1000)
        st = sim.context.getState()
        v = st.getPeriodicBoxVolume().value_in_unit(u.nanometer**3)
        dens.append(len(grid) * 72.0 / 6.02214076e23 / (v * 1e-21))
    st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
    e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / len(grid)
    if not -40 < e < -10:
        raise RuntimeError(f"the run went wrong: {e:.3g} kJ/mol per bead")
    x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    b = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
    rows = []
    for k, p in enumerate(x):
        n = (k + 1) % 100000
        rows.append(f"{n:5d}W    {'W':>5s}{n:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}")
    edges = f"{b[0, 0]:.5f} {b[1, 1]:.5f} {b[2, 2]:.5f}"
    header = f"Martini 3 water, equilibrated in OpenMM (300 K, 1 bar)\n{len(x)}\n"
    OUT.write_text(header + "\n".join(rows) + f"\n{edges}\n")
    print(f"density {np.mean(dens[50:]):.4f} g/mL (last 4 ns), box {b[0, 0]:.3f} nm -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
