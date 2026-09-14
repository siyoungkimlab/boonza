"""PDB, mmCIF and MAE read/write timings: boonza vs msys, gemmi and MDAnalysis.

    python benchmarks/bench_io.py --copies 40

A parameterized DMS (msys's 3.dms by default) is tiled into a grid of
copies, and boonza writes it once as PDB, mmCIF and MAE.  Every program then
loads those same files and saves its own copy; the best of ``--repeat`` runs
is reported.

What each program does on load differs, so rows are labeled:

- boonza and msys guess bonds from distances when reading PDB; boonza also
  applies CONECT/SSBOND/LINK records.  gemmi and MDAnalysis do not guess
  bonds, so boonza is also timed with ``guess_bonds=False``.
- MAE carries the full force field (``ffio_ff``) for boonza and msys.

msys runs under its own Python through tests/msys_oracle.py, gemmi through
tests/gemmi_oracle.py (BOONZA_GEMMI_PYTHON), and MDAnalysis in this process
when it is installed.  Missing programs are shown as "-".
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import boonza

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from conftest import msys_available, run_msys  # noqa: E402

GEMMI_PYTHON = os.path.expanduser(
    os.environ.get("BOONZA_GEMMI_PYTHON", "~/miniforge3/envs/ommflow/bin/python")
)


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def tiled(src: boonza.System, copies: int) -> boonza.System:
    """``copies`` of ``src`` on a grid, each shifted by the cell (or extent) plus 5 Å."""
    step = np.diag(src.cell) if src.cell.any() else np.ptp(src.positions, axis=0)
    step = step + 5.0
    side = int(np.ceil(copies ** (1 / 3)))
    big = boonza.System("tiled")
    for k in range(copies):
        shift = np.array([k % side, (k // side) % side, k // (side * side)]) * step
        piece = src.copy()
        piece.positions = src.positions + shift
        big.append(piece)
    return big


def gemmi_times(pdb, cif, workdir, repeat):
    code = f"""
import json, time, gemmi
def best(fn, n):
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); out.append(time.perf_counter() - t0)
    return min(out)
res = {{}}
for fmt, path in (("pdb", {str(pdb)!r}), ("cif", {str(cif)!r})):
    st = gemmi.read_structure(path)
    res["load " + fmt] = best(lambda: gemmi.read_structure(path), {repeat})
    out = {str(workdir)!r} + "/gemmi_out." + fmt
    save = (lambda: st.write_pdb(out)) if fmt == "pdb" else \\
        (lambda: st.make_mmcif_document().write_file(out))
    res["save " + fmt] = best(save, {repeat})
print(json.dumps(res))
"""
    try:
        r = subprocess.run([GEMMI_PYTHON, "-c", code], capture_output=True, text=True)
    except OSError:
        return {}
    if r.returncode:
        print(f"gemmi failed: {r.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
        return {}
    return json.loads(r.stdout)


def mdanalysis_times(pdb, workdir, repeat):
    try:
        import MDAnalysis as mda
    except ImportError:
        return {}
    import warnings

    warnings.filterwarnings("ignore")
    u = mda.Universe(str(pdb))
    return {
        "load pdb": best(lambda: mda.Universe(str(pdb)), repeat),
        "save pdb": best(lambda: u.atoms.write(str(workdir / "mda_out.pdb")), repeat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path("~/msys/tests/files/3.dms").expanduser()))
    ap.add_argument("--copies", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    big = tiled(boonza.load(args.src), args.copies)
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="boonza-io-"))
    workdir.mkdir(parents=True, exist_ok=True)
    files = {fmt: workdir / f"big.{fmt}" for fmt in ("pdb", "cif", "mae")}
    for path in files.values():
        boonza.save(big, path)
    print(f"system: {big.natoms:,} atoms, {big.nbonds:,} bonds, {len(big.tables)} tables "
          f"-> {workdir}")  # fmt: skip
    boonza.load(files["pdb"])  # compile the numba kernels once

    ours = {
        "load pdb": best(lambda: boonza.load(files["pdb"]), args.repeat),
        "load pdb (no bond guessing)": best(
            lambda: boonza.load(files["pdb"], guess_bonds=False), args.repeat),
        "save pdb": best(lambda: boonza.save(big, workdir / "out.pdb"), args.repeat),
        "load cif": best(lambda: boonza.load(files["cif"]), args.repeat),
        "load cif (no bond guessing)": best(
            lambda: boonza.load(files["cif"], guess_bonds=False), args.repeat),
        "save cif": best(lambda: boonza.save(big, workdir / "out.cif"), args.repeat),
        "load mae": best(lambda: boonza.load(files["mae"]), args.repeat),
        "save mae": best(lambda: boonza.save(big, workdir / "out.mae"), args.repeat),
    }  # fmt: skip
    msys = run_msys("iotime", files["pdb"], files["mae"], workdir, args.repeat) \
        if msys_available() else {}  # fmt: skip
    gemmi = gemmi_times(files["pdb"], files["cif"], workdir, args.repeat)
    mda = mdanalysis_times(files["pdb"], workdir, args.repeat)

    def cell(d, key):
        v = d.get(key)
        return f"{v:9.2f}s" if isinstance(v, int | float) else "        -"

    print(f"{'step':<30} {'boonza':>10} {'msys':>10} {'gemmi':>10} {'MDAnalysis':>11}")
    for key, t in ours.items():
        base = key.split(" (")[0]
        others = [cell(d, base) if "(" not in key else "        -" for d in (msys,)]
        others += [cell(d, base) for d in (gemmi, mda)]
        print(f"{key:<30} {t:9.2f}s {others[0]} {others[1]} {others[2]:>11}")


if __name__ == "__main__":
    main()
