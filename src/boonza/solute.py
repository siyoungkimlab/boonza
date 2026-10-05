"""The solute of a run, without the water and the salt it swam in.

A coarse-grained swim is mostly solvent: of eight thousand particles, seven
thousand are water beads and ions that no part of the analysis reads.  The sites
are the ligand's, the pocket is measured against the protein, and bulk comes
from the box the frames carry -- so a copy holding the protein and the ligands
answers every question the whole box does, over a quarter of the atoms and a
quarter of the disk.

    boonza solute --workdir run/md -o light      # then: boonza sites --workdir light

What counts as solvent depends on the model, and one name is a trap: a
one-residue probe of a tryptophan is called W, and so is Martini's water.  The
probes of a swim are a chain of their own, so a residue there is never solvent
however it is named.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

#: The files a run directory carries that the analysis reads beside the
#: trajectory: which probes swam, and how far apart the frames are.
BESIDE = ("probes.json", "md.toml", "final.toml")
#: The one beside them that is a structure and not a setting, so it is cut down
#: rather than copied: the file a viewer opens.
#:
#: What it leaves out is the elastic network a viewer would draw as a hairball,
#: which is worth having where the network is made of bonds -- Martini 3 writes
#: its rubber bands as bonds of type 1, and one run's copy holds 1119 of them
#: stretching up to 10.7 A through the protein.  Martini 2.2 writes its own as
#: type 6, a harmonic potential that is no bond at all, so nothing is drawn and
#: its view file differs from the system in name only.  SIRAH holds its fold
#: with torsions and has nothing to leave out either.
VIEW = "view.dms"


def solvent_selection(system) -> str:
    """What to drop: the water and the salt, and nothing the probes are made of.

    All-atom water is recognised by :func:`boonza.analyze.classify` (an oxygen
    and two hydrogens, or a name it knows); a coarse-grained model's is named,
    since a bead has neither.  A chain of probes is spared whatever its residues
    are called.
    """
    from .md.cgswim import PROBE_CHAIN
    from .probemap import SOLVENT_NAMES

    named = " ".join(SOLVENT_NAMES)
    drop = f"(water or ions or resname {named})"
    theirs = sorted(c for c in {str(x).strip() for x in system.chains["name"]}
                    if re.fullmatch(PROBE_CHAIN + r"\d*", c))  # fmt: skip
    return f"{drop} and not chain {' '.join(theirs)}" if theirs else drop


def solute_ids(system, keep: str | None = None) -> np.ndarray:
    """The atoms to keep: ``keep`` if given, else everything but the solvent."""
    if keep:
        return system.select(keep).ids
    return system.select(f"not ({solvent_selection(system)})").ids


def write_solute(system, trajectory, out, keep: str | None = None, beside=()) -> dict:
    """Write ``system`` and ``trajectory`` holding only the solute, into ``out``.

    ``out`` is a directory, written as a run directory is -- ``solvated.dms`` and
    ``trajectory.dcd`` -- so ``boonza sites --workdir`` reads it as it reads the
    run itself.  ``beside`` are files copied in alongside (``probes.json`` and
    the run's settings, which the analysis also reads), except ``view.dms``,
    which is a structure: it holds the same atoms in the same order, so it is cut
    down the same way rather than copied, and the copy a viewer opens has no more
    solvent in it than the trajectory does.

    The box of every frame is carried over: bulk is counted against it, so a copy
    without it would answer a different question than the run did.
    """
    from .io import load as _load_view
    from .io import save
    from .trajectory import open_trajectory, open_writer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = solute_ids(system, keep)
    if not len(ids):
        raise ValueError("nothing but solvent: the selection leaves no atoms")
    save(system.clone(ids), out / "solvated.dms")
    frames = 0
    with open_writer(out / "trajectory.dcd", natoms=len(ids)) as writer:
        for frame in open_trajectory(trajectory, system):
            writer.write(frame.positions[ids], box=frame.box, time=frame.time, step=frame.step)
            frames += 1
    for name in beside:
        src = Path(name)
        if not src.is_file():
            continue
        if src.name == VIEW:
            # the same atoms in the same order as the system, so the same cut:
            # copied whole it would hold the solvent this is written to be rid of
            save(_load_view(src).clone(ids), out / VIEW)
        else:
            shutil.copy(src, out / src.name)
    return {"atoms": int(len(ids)), "of": int(system.natoms), "frames": frames,
            "out": str(out)}  # fmt: skip
