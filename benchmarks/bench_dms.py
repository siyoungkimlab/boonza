"""DMS load/save timings: boonza vs msys on a tiled, fully parameterized system.

    python benchmarks/bench_dms.py --copies 40

msys runs through tests/msys_oracle.py under the msys Python (see tests/conftest.py).
msys computes fragments and residue types during load; boonza computes
fragments lazily, so that step is timed separately.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import boonza

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from conftest import msys_available, run_msys  # noqa: E402


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path("~/msys/tests/files/3.dms").expanduser()))
    ap.add_argument("--copies", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    src = boonza.load(args.src)
    big = boonza.System("tiled")
    t0 = time.perf_counter()
    for _ in range(args.copies):
        big.append(src)
    t_append = time.perf_counter() - t0

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="boonza-bench-"))
    path = workdir / "big.dms"
    boonza.save(big, path)
    nterms = sum(len(t) for t in big.tables.values())
    print(f"system: {big.natoms:,} atoms, {big.nbonds:,} bonds, {nterms:,} terms -> {path}")
    print(f"boonza append x{args.copies}: {t_append:.2f} s")

    s = boonza.load(path)
    _ = s.fragids  # compile the numba kernel once

    def fragments():
        s._cache.clear()
        _ = s.fragids

    rows = [
        ("load", best(lambda: boonza.load(path), args.repeat)),
        ("save", best(lambda: boonza.save(big, workdir / "boonza_out.dms"), args.repeat)),
        ("fragments", best(fragments, args.repeat)),
    ]
    ref = None
    if msys_available():
        ref = run_msys("time", path, workdir / "msys_out.dms", args.repeat)
    print(f"{'step':<10} {'boonza':>9} {'msys':>9}")
    for step, t in rows:
        m = f"{ref[step]:8.2f}s" if ref and step in ref else "      -  "
        print(f"{step:<10} {t:8.2f}s {m}")


if __name__ == "__main__":
    main()
