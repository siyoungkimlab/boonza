"""A copy of a run without the water and the salt it swam in."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.md.cgswim import PROBE_CHAIN
from boonza.solute import solute_ids, solvent_selection, write_solute

DATA = Path(__file__).parent / "data"


def _run(tmp_path, resnames, chains, names=None, frames=5, bonds=None):
    """A little run: a system and a trajectory of it, boxes and all.

    The protein's beads are bonded to each other, as a real one's are: a lone
    bead of a heavier element is an ion, and that is how the solvent is known.
    """
    n = len(resnames)
    xyz = np.array([[float(3 * k), 0.0, 0.0] for k in range(n)])
    if bonds is None:
        chain = [k for k, c in enumerate(chains) if c not in ("", None)]
        bonds = [(a, b) for a, b in zip(chain, chain[1:], strict=False)
                 if chains[a] == chains[b]]  # fmt: skip
    s = boonza.System.from_arrays(xyz, names=names or ["BB"] * n, anum=[32] * n,
                                  resnames=resnames, resids=list(range(1, n + 1)),
                                  chains=chains, bonds=bonds or None,
                                  cell=np.diag([40.0, 41.0, 42.0]))  # fmt: skip
    boonza.save(s, tmp_path / "solvated.dms")
    from boonza.trajectory import open_writer

    with open_writer(tmp_path / "trajectory.dcd", natoms=n) as w:
        for f in range(frames):
            w.write(xyz + f, box=np.diag([40.0 + f, 41.0, 42.0]))
    return s


def test_the_water_and_the_salt_go_and_the_probes_stay(tmp_path):
    """Martini's water is called W, and so is a one-residue tryptophan probe: the
    chain the probes are in is what tells them apart, as it does for the ligand
    selection.  Drop the names alone and the probes go with the solvent."""
    s = _run(tmp_path, resnames=["ALA", "ALA", "W", "W", "W", "NA", "W"],
             chains=["A", "A", "", "", "", "", PROBE_CHAIN], bonds=[(0, 1)])  # fmt: skip
    kept = solute_ids(s)
    assert sorted(kept.tolist()) == [0, 1, 6]  # the protein and the probe, not the water
    assert PROBE_CHAIN in solvent_selection(s)  # the chain is spared by name

    out = write_solute(s, tmp_path / "trajectory.dcd", tmp_path / "light")
    assert out == {"atoms": 3, "of": 7, "frames": 5, "out": str(tmp_path / "light")}
    light = boonza.load(tmp_path / "light" / "solvated.dms")
    assert light.natoms == 3
    assert sorted(str(x) for x in light.residues["name"]) == ["ALA", "ALA", "W"]


def test_the_box_of_every_frame_is_carried_over(tmp_path):
    """Bulk is counted against the box, so a copy without it answers another
    question: the enrichment a site is found by divides by that volume."""
    s = _run(tmp_path, resnames=["ALA", "ALA", "W", "W"], chains=["A", "A", "", ""], frames=4)
    write_solute(s, tmp_path / "trajectory.dcd", tmp_path / "light")
    light = boonza.load(tmp_path / "light" / "solvated.dms")
    boxes = [f.box.copy() for f in boonza.open_trajectory(tmp_path / "light" / "trajectory.dcd",
                                                          light)]  # fmt: skip
    assert len(boxes) == 4
    assert [round(float(b[0, 0])) for b in boxes] == [40, 41, 42, 43]  # as written, frame by frame
    assert all(round(float(b[1, 1])) == 41 for b in boxes)


@pytest.mark.parametrize(("water", "ion"), [
    ("HOH", "NA"),    # all-atom, by name
    ("W", "ION"),     # Martini 2 and 3
    ("WT4", "NaW"),   # SIRAH
])  # fmt: skip
def test_every_model_s_solvent_is_known(tmp_path, water, ion):
    """One command for all of them: what a model calls its water is named here,
    since a bead has neither an oxygen nor two hydrogens to be recognised by."""
    s = _run(tmp_path, resnames=["ALA", "ALA", water, water, ion],
             chains=["A", "A", "", "", ""])  # fmt: skip
    assert sorted(solute_ids(s).tolist()) == [0, 1]


def test_what_to_keep_can_be_said_instead(tmp_path):
    s = _run(tmp_path, resnames=["ALA", "ALA", "W"], chains=["A", "A", ""])
    assert sorted(solute_ids(s, keep="resname W").tolist()) == [2]
    with pytest.raises(ValueError, match="no atoms"):
        write_solute(s, tmp_path / "trajectory.dcd", tmp_path / "none", keep="resname ZZZ")
