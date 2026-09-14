"""Molecule (fragment) finding and identical-molecule grouping: boonza vs msys.

    python benchmarks/bench_fragments.py --copies 40

Tiles msys's solvated 3.dms (26k atoms) into a large system, saves it, and
times boonza's fragids / distinct_fragments against msys updateFragids /
FindDistinctFragments on the same file.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import msys_file, run_msys  # noqa: E402

import boonza  # noqa: E402


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    base = boonza.load(msys_file("3.dms"), structure_only=True)
    big = boonza.System("tiled")
    for _ in range(args.copies):
        big.append(base)
    print(f"{big.natoms} atoms, {big.nbonds} bonds")

    def fragids():
        big._cache.clear()
        return big.fragids

    def distinct():
        return boonza.distinct_fragments(big)

    t_frag, ids = best(fragids, args.repeat)
    t_dist, groups = best(distinct, args.repeat)
    print(f"boonza fragids:            {t_frag:.3f} s  ({int(ids.max()) + 1} molecules)")
    print(f"boonza distinct_fragments: {t_dist:.3f} s  ({len(groups)} groups)")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.dms"
        boonza.save(big, path)
        try:
            ref = run_msys("fragtime", path)
        except Exception as e:  # msys missing
            print(f"msys: not available ({e})")
            return
    print(f"msys updateFragids:        {ref['fragids']:.3f} s  ({ref['nfragments']} molecules)")
    print(f"msys FindDistinctFragments:{ref['distinct']:.3f} s  ({ref['ngroups']} groups)")


if __name__ == "__main__":
    main()
