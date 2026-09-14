"""Trajectory formats: write and read DCD, Amber NetCDF, XTC and TRR.

Every format goes through the same two calls, ``boonza.open_writer`` and
``boonza.open_trajectory``.  Desmond DTR directories and STK lists are read
the same way (there is no DTR writer).

    python examples/19_trajectory_formats.py
"""

import importlib.util

import numpy as np
from _common import OUT, water_box

import boonza

s = water_box(4)
rng = np.random.default_rng(0)
frames = s.positions + rng.normal(0, 0.2, (5, s.natoms, 3))

formats = ["dcd", "nc"]
if importlib.util.find_spec("MDAnalysis"):
    formats += ["xtc", "trr"]  # through MDAnalysis's compiled XDR library
else:
    print("(install MDAnalysis for XTC and TRR)")
for fmt in formats:
    path = OUT / f"waters.{fmt}"
    # DCD stores a start and a uniform time step, not a time per frame
    options = {"dt": 2.0, "istart": 0} if fmt == "dcd" else {}
    with boonza.open_writer(path, s.natoms, **options) as w:
        for k, f in enumerate(frames):
            w.write(f, box=s.cell, time=2.0 * k)
    traj = boonza.open_trajectory(path, s)
    back = traj.read()
    change = np.abs(back.positions - frames).max()
    box = np.diag(back.boxes[0]).round(2).tolist()
    times = np.round(back.times, 3).tolist()
    print(f"{fmt:>4}: {len(traj)} frames at {times} ps, box {box} Å, "
          f"largest change {change:.4f} Å")  # fmt: skip
print("XTC stores 0.001 nm; the other formats keep float32 coordinates.")

# frames can be read lazily, by slice, and only for the atoms you need
traj = boonza.open_trajectory(OUT / "waters.dcd", s)
print("every other frame:", np.round(traj[::2].read().times, 3).tolist(),
      "| oxygens only:", traj.read(atoms="name O").positions.shape)  # fmt: skip
