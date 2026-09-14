"""Shared helpers for the examples: data paths, optional dependencies, small builders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

import boonza

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # PDB files from the RCSB used by the examples
OUT = HERE / "output"  # files the examples write
OUT.mkdir(exist_ok=True)


def require(*modules: str) -> None:
    """Exit politely (status 0) when an optional package is missing."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        print(f"skipped: this example needs {', '.join(missing)} (pip install {' '.join(missing)})")
        sys.exit(0)


def water_box(n: int = 5, spacing: float = 3.1) -> boonza.System:
    """n**3 water molecules on a cubic grid, with bonds and a matching cell."""
    s = boonza.System("water box")
    chain = s.add_chain(name="W")
    nw = n**3
    residues = [s.add_residue(chain, name="HOH", resid=k + 1).id for k in range(nw)]
    grid = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    template = np.array([[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.240, 0.927, 0.0]])
    xyz = (grid[:, None, :] * spacing + template[None]).reshape(-1, 3)
    s.add_atoms(3 * nw, residue=np.repeat(residues, 3), name=np.tile(["O", "H1", "H2"], nw),
                anum=np.tile([8, 1, 1], nw), mass=np.tile([15.999, 1.008, 1.008], nw),
                charge=np.tile([-0.834, 0.417, 0.417], nw), pos=xyz)  # fmt: skip
    first = 3 * np.arange(nw)
    s.add_bonds(np.concatenate([np.column_stack([first, first + 1]),
                                np.column_stack([first, first + 2])]))  # fmt: skip
    s.cell = np.eye(3) * n * spacing
    return s


def remove_incomplete_residues(model) -> None:
    """Drop residues missing backbone atoms from an OpenMM Modeller.

    Crystal structures often end with a residue that has only a few atoms
    (1TEN starts with an arginine that has just C and O); force-field
    templates cannot match those.
    """
    backbone = {"N", "CA", "C"}
    bad = [r for r in model.topology.residues() if not backbone <= {a.name for a in r.atoms()}]
    if bad:
        print("removing incomplete residues:", ", ".join(f"{r.name}{r.id}" for r in bad))
        model.delete(bad)


def amber_system(pdb_name: str) -> boonza.System:
    """A protein from ``data/`` parameterized with OpenMM's Amber14 force field.

    Waters are removed and hydrogens added; the result is cached as a DMS
    file in ``output/`` so later examples load it instantly.
    """
    require("openmm")
    cached = OUT / (Path(pdb_name).stem + "_amber14.dms")
    if cached.exists():
        return boonza.load(cached)
    from openmm import app

    pdb = app.PDBFile(str(DATA / pdb_name))
    model = app.Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    remove_incomplete_residues(model)
    ff = app.ForceField("amber14-all.xml")
    model.addHydrogens(ff)
    omm = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                          removeCMMotion=False)  # fmt: skip
    s = boonza.from_openmm(model.topology, omm, model.positions)
    boonza.save(s, cached)
    return s
