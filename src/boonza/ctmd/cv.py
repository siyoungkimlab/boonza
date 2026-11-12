"""The pocket-ligand drmsd, as an OpenMM collective variable.

CTMD biases the ligand away from its pose along one coordinate.  The published
protocol uses the RMSD of the ligand after superposing the protein, which PLUMED
expresses with per-atom weights in a reference PDB.  OpenMM's ``RMSDForce``
superposes and measures on the same atoms, so it cannot say "align on the protein,
measure the ligand".  drmsd says the same thing without any superposition: it
compares pocket-ligand distances with the ones in the reference, so a rotation or
translation of the whole system leaves it unchanged and only the ligand's motion
relative to the pocket is seen.  That is :func:`boonza.drmsd`, and the pocket here
is the one it uses.

The force's energy *is* the coordinate, in nm, which is how OpenMM takes a
collective variable.
"""

from __future__ import annotations

import numpy as np

#: how the pocket is defined, as in :func:`boonza.drmsd`: the alpha carbons within
#: CUTOFF A of the ligand in the reference.  Backbone rather than every atom, since a
#: side chain swinging about would move the coordinate without the ligand going
#: anywhere
POCKET = "protein and name CA"
#: A of the ligand a pocket atom must be within
CUTOFF = 5.0


class CVError(ValueError):
    """A collective variable that cannot be built from this structure."""


def drmsd_pairs(system, reference=None, ligand: str | None = None, pocket: str = POCKET,
                cutoff: float = CUTOFF, periodic: bool = True):  # fmt: skip
    """The pairs the coordinate is built from: ``(pocket atoms, ligand atoms, distances)``.

    ``reference`` is the structure the pose is measured against (default: ``system``
    itself, so the coordinate starts at 0).  The distances are the reference's, in nm,
    as an (npocket, nligand) array.  Atoms are paired by index, so ``reference`` must
    be the same system -- a frame of it, or the structure it was built from.
    """
    from .. import pbc
    from ..symmetry import DEFAULT_LIGAND

    ref = system if reference is None else reference
    if ref.natoms != system.natoms:
        raise CVError(f"the reference has {ref.natoms} atoms and the system {system.natoms}; "
                      "the coordinate pairs them by index")  # fmt: skip
    lig = np.asarray(system.select(DEFAULT_LIGAND if ligand is None else ligand).ids, np.int64)
    if not len(lig):
        raise CVError(f"no ligand atoms: {ligand or DEFAULT_LIGAND!r} selects none")
    prot = np.asarray(system.select(pocket).ids, np.int64)
    if not len(prot):
        raise CVError(f"no pocket atoms: {pocket!r} selects none")
    pos = np.asarray(ref.positions, float)
    box = np.asarray(ref.cell, float) if periodic and np.asarray(ref.cell).any() else None
    d = pbc.distances(pos[prot], pos[lig], box)
    near = (d.min(1) <= cutoff) & ~np.isin(prot, lig)
    if not near.any():
        raise CVError(f"no pocket atom within {cutoff:g} A of the ligand in the reference")
    return prot[near], lig, d[near] / 10.0  # A -> nm


def drmsd_force(system, reference=None, *, ligand: str | None = None, pocket: str = POCKET,
                cutoff: float = CUTOFF, periodic: bool = True, index=None):  # fmt: skip
    """An OpenMM force whose energy is the pocket-ligand drmsd in nm.

    ``sqrt(mean((d - d_ref)^2))`` over the pairs of :func:`drmsd_pairs`: a
    ``CustomBondForce`` sums the squares, and a ``CustomCVForce`` takes the root.  It
    is 0 at the reference and grows as the ligand leaves, and needs no superposition,
    so the protein is free to move while it is biased.

    ``index``: where each atom of ``system`` sits in the OpenMM system the force goes
    into, for a structure the force field reordered (``parameterize`` leaves that map
    in its ``view_index`` column).  Without it the two are taken to agree.

    Returns ``(force, npairs)``.
    """
    import openmm as mm

    prot, lig, dref = drmsd_pairs(system, reference, ligand, pocket, cutoff, periodic)
    npairs = dref.size
    if index is not None:
        index = np.asarray(index, np.int64)
        if len(index) != system.natoms or (index < 0).any():
            raise CVError("the index must place every atom of the system in the OpenMM one")
        prot, lig = index[prot], index[lig]
    square = mm.CustomBondForce(f"(r - r0)^2*{1.0 / npairs!r}")
    square.addPerBondParameter("r0")
    square.setUsesPeriodicBoundaryConditions(bool(periodic and np.asarray(system.cell).any()))
    for a, row in zip(prot.tolist(), dref, strict=True):
        for b, r0 in zip(lig.tolist(), row.tolist(), strict=True):
            square.addBond(a, b, [r0])
    force = mm.CustomCVForce("sqrt(msd)")
    force.addCollectiveVariable("msd", square)
    force.setName("drmsd")
    return force, npairs


def drmsd_of(system, positions=None, **kwargs) -> float:
    """The coordinate's value (nm) at ``positions``, without running anything.

    For a sanity check against :func:`boonza.drmsd`, and to see where a pose starts.
    """
    import openmm as mm
    import openmm.unit as u

    force, _ = drmsd_force(system, **kwargs)
    probe = mm.System()
    for _ in range(system.natoms):
        probe.addParticle(1.0)
    cell = np.asarray(system.cell, float) / 10.0
    if cell.any():
        probe.setDefaultPeriodicBoxVectors(*(c * u.nanometer for c in cell))
    probe.addForce(force)
    context = mm.Context(probe, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))  # fmt: skip
    xyz = np.asarray(system.positions if positions is None else positions, float) / 10.0
    context.setPositions(xyz * u.nanometer)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return float(energy.value_in_unit(u.kilojoule_per_mole))
