"""fpocket and mdpocket, run with the flags tuned for each bead model.

fpocket was written for all-atom proteins: its alpha-sphere sizes, clustering
distances and pocket score assume atoms 3 to 4 A apart.  The presets in
``data/pockets/presets.json`` were tuned the way fpocket's own were, on its
263-complex training set, for Martini 2, Martini 3 and SIRAH beads, and checked
on its 48 apo/holo pairs.  Each preset passes refitted score coefficients
through ``--score_coefficients``, which only the fpocket build at
https://github.com/siyoungkimlab/fpocket has; :func:`find_fpocket` refuses
any other.

The tuning itself lives with that build (``cg/benchmark/``), which writes the
presets file here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

PRESETS = Path(__file__).resolve().parent.parent / "data" / "pockets" / "presets.json"
#: where the build that can read --score_coefficients comes from
FORK = "https://github.com/siyoungkimlab/fpocket"

__all__ = [
    "Pocket",
    "PocketError",
    "find_fpocket",
    "merge_flags",
    "preset",
    "presets",
    "read_dx",
    "read_pockets",
    "run_fpocket",
    "run_mdpocket",
    "write_isosurface",
]


class PocketError(RuntimeError):
    """fpocket is missing, is not the build the presets need, or failed."""


@cache
def _has_coefficients(exe: str) -> bool:
    try:
        out = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--score_coefficients" in out.stdout + out.stderr


def find_fpocket(path=None, program: str = "fpocket") -> Path:
    """The ``program`` (fpocket or mdpocket) to run: in ``path``, else in
    ``$FPOCKET_HOME/bin``, else the one on ``PATH``.

    ``path`` is the program itself or the directory of an fpocket build (its
    ``bin/`` is looked in).  It must be the build that reads
    ``--score_coefficients``, which the tuned presets pass.
    """
    candidates = []
    for where in (path, os.environ.get("FPOCKET_HOME")):
        if where:
            p = Path(where).expanduser()
            candidates += [p] if p.is_file() else [p / "bin" / program, p / program]
    found = shutil.which(program)
    if found:
        candidates.append(Path(found))
    for exe in candidates:
        if exe.is_file() and os.access(exe, os.X_OK):
            # mdpocket's help does not list the flag, so the fpocket beside it is asked
            if _has_coefficients(str(exe.with_name("fpocket"))):
                return exe
            raise PocketError(
                f"{exe} cannot read --score_coefficients, which the tuned presets pass: build "
                f"fpocket from {FORK} (make ARCH=MACOSXARM64 on Apple silicon, make on Linux) "
                "and set FPOCKET_HOME to that directory"
            )
    raise PocketError(
        f"{program} was not found: build it from {FORK} (make ARCH=MACOSXARM64 on Apple "
        "silicon, make on Linux) and set FPOCKET_HOME to that directory, or put its bin/ on PATH"
    )


@cache
def presets() -> dict:
    """Every model's preset: its fpocket flags, score coefficients, how the N-class
    Martini beads are counted, and the level mdpocket's density map is drawn at."""
    return json.loads(PRESETS.read_text())


def preset(model: str) -> dict:
    """The preset of one model."""
    every = presets()
    if model not in every:
        raise ValueError(f"no preset for {model!r}; presets: {', '.join(every)}")
    return every[model]


def merge_flags(base: list[str], extra: list[str]) -> list[str]:
    """``base`` with every flag that ``extra`` repeats taken from ``extra`` instead."""
    extra = [x for x in extra if x != "--"]
    given = {x for x in extra if x.startswith("-")}
    out, k = [], 0
    while k < len(base):
        if base[k] in given:
            k += 2
            continue
        out.append(base[k])
        k += 1
    return out + extra


@dataclass
class Pocket:
    """One pocket of one fpocket run, in fpocket's own order."""

    centres: np.ndarray  #: its alpha spheres' centres (n, 3), A
    radii: np.ndarray  #: their radii, A
    score: float  #: fpocket's score (the preset's refitted one)
    info: str = ""  #: its block of fpocket's _info.txt
    descriptors: dict = field(default_factory=dict)  #: the numbers in that block

    @property
    def centre(self) -> np.ndarray:
        """The mean of its alpha-sphere centres: fpocket's pocket centre."""
        return self.centres.mean(0)

    @property
    def volume(self) -> float:
        """fpocket's Monte Carlo volume, A^3 (nan when the block has none)."""
        return float(self.descriptors.get("Volume", np.nan))


def _info_blocks(info: Path) -> dict[int, str]:
    blocks, current, lines = {}, None, []
    for line in open(info):
        if line.startswith("Pocket "):
            if current is not None:
                blocks[current] = "".join(lines)
            current, lines = int(line.split()[1]), [line]
        elif current is not None:
            lines.append(line)
    if current is not None:
        blocks[current] = "".join(lines)
    return blocks


def _descriptors(block: str) -> dict:
    out = {}
    for line in block.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            try:
                out[key.strip()] = float(value)
            except ValueError:
                continue
    return out


def read_pockets(out_dir, stem: str) -> list[Pocket]:
    """The pockets of an fpocket output directory (``<stem>_out``), in its ranking."""
    out_dir = Path(out_dir)
    pqr = out_dir / f"{stem}_pockets.pqr"
    if not pqr.exists():
        return []
    spheres: dict[int, list] = {}
    for line in open(pqr):
        if line.startswith("ATOM"):
            spheres.setdefault(int(line[22:26]), []).append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]),
                 float(line[66:].split()[0])))  # fmt: skip
    blocks = _info_blocks(out_dir / f"{stem}_info.txt")
    pockets = []
    for k, rows in sorted(spheres.items()):
        a = np.array(rows)
        d = _descriptors(blocks.get(k, ""))
        pockets.append(Pocket(a[:, :3], a[:, 3], float(d.get("Score", np.nan)),
                              blocks.get(k, ""), d))  # fmt: skip
    return pockets


def run_fpocket(pdb, flags=(), fpocket=None, quiet: bool = False) -> list[Pocket]:
    """fpocket on ``pdb`` with ``flags``: its pockets, best first.

    The output is fpocket's own, ``<stem>_out`` beside ``pdb``, replaced if it
    is there already.
    """
    pdb = Path(pdb).resolve()
    exe = find_fpocket(fpocket)
    out_dir = pdb.with_name(pdb.stem + "_out")
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [str(exe), "-f", str(pdb), *flags]
    pipe = subprocess.DEVNULL if quiet else None
    done = subprocess.run(cmd, stdout=pipe, stderr=pipe, cwd=pdb.parent)  # it writes time.txt
    if done.returncode != 0:
        raise PocketError(f"fpocket failed ({done.returncode}): {' '.join(cmd)}")
    return read_pockets(out_dir, pdb.stem)


def run_mdpocket(pdb, dcd, prefix, flags=(), fpocket=None) -> list[str]:
    """mdpocket over a trajectory (``pdb`` and its frames, ``dcd``); returns the command run.

    Writes ``<prefix>_freq.dx`` (the share of frames each point is in a pocket)
    and ``<prefix>_dens.dx`` (alpha-sphere density) among mdpocket's other files.
    """
    exe = find_fpocket(fpocket, "mdpocket")
    pdb, dcd, prefix = (Path(x).resolve() for x in (pdb, dcd, prefix))
    cmd = [str(exe), "-f", str(pdb), "-t", str(dcd), "-O", "dcd", "-o", str(prefix), *flags]
    # a line per frame on stdout otherwise, and its time.txt where it runs
    done = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=prefix.parent)
    if done.returncode != 0:
        raise PocketError(f"mdpocket failed ({done.returncode}): {' '.join(cmd)}")
    return cmd


def read_dx(dx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(values, origin, the three delta vectors)`` of a .dx map."""
    values, counts, origin, delta = [], None, None, []
    for line in open(dx):
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        if line.startswith("object 1"):
            counts = tuple(int(x) for x in w[-3:])
        elif w[0] == "origin":
            origin = np.array(w[1:4], float)
        elif w[0] == "delta":
            delta.append(np.array(w[1:4], float))
        elif w[0][0].isdigit() or w[0][0] in "-.":
            values += [float(x) for x in w]
    return np.array(values[: int(np.prod(counts))]).reshape(counts), origin, np.array(delta)


def write_isosurface(dx, level: float) -> Path:
    """The points of an mdpocket density map at or above ``level``, as mdpocket writes
    them (``<prefix>_dens_iso_<level>.pdb``, the density in the B-factor column).

    mdpocket draws its map at 8, a level set for all-atom proteins; each bead
    model's preset carries the level that matches it (``mdpocket_density_iso``).
    """
    dx = Path(dx)
    grid, origin, delta = read_dx(dx)
    out = dx.with_name(dx.name.replace("_dens.dx", f"_dens_iso_{level:g}.pdb"))
    lines = []
    for k, (i, j, m) in enumerate(np.argwhere(grid >= level)):
        x, y, z = origin + i * delta[0] + j * delta[1] + m * delta[2]
        lines.append(f"ATOM  {(k + 1) % 100000:5d}  C   PTH     1    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"{0.0:6.2f}{grid[i, j, m]:6.2f}")  # fmt: skip
    out.write_text("\n".join(lines) + "\nEND\n")
    return out
