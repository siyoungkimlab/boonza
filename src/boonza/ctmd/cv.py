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

#: the atoms the ligand is measured against: alpha carbons, as :func:`boonza.drmsd`
#: has them.  Backbone rather than every atom, since a side chain swinging about would
#: move the coordinate without the ligand going anywhere -- what the published
#: protocol gets by superposing the protein and measuring only the ligand
POCKET = "protein and name CA"
#: A of the ligand a pocket atom must be within, the same 8 :func:`boonza.drmsd` uses.
#: 5 would be far too tight for a coordinate to be biased along: an alpha carbon sits
#: behind its side chain, so a real pocket may have
#: only a handful that close (PANX1 with Brilliant Blue FCF, a 51-atom ligand in the
#: pore: 3 within 5 A, 20 within 8, 45 within 10).  8 rather than 10 because widening
#: dilutes: the mean runs over every pair, and a second-shell one barely answers the
#: ligand at all, so 1 A of ligand gives 0.59 A of coordinate across the 3 atoms
#: within 5 A, 0.56 across the 20 within 8, and 0.55 across the 45 within 10 -- at a
#: pair count that more than doubles
CUTOFF = 8.0
#: fewer pocket atoms than this and the coordinate rests on too thin a frame.  Not for
#: want of signal: measured on the PANX1 pocket, 3 atoms answer a 1 A displacement as
#: strongly as 45 do, and jittering the protein by 0.3 A moves them no further in the
#: mean.  What a thin frame costs is reproducibility and comparability -- the spread of
#: that spurious motion was 0.104 A across 3 atoms against 0.045 across 20 and 0.036
#: across 45, and
#: whether a given alpha carbon falls inside the cutoff is a knife-edge, so two ligands
#: in one pocket can end up measured against different atoms, which a ranking across
#: ligands cannot afford
MIN_POCKET = 8


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
    if near.sum() < MIN_POCKET:
        raise CVError(f"only {int(near.sum())} pocket atoms within {cutoff:g} A of the ligand, "
                      f"too few to measure against ({MIN_POCKET} is the least this will "
                      "take): widen the cutoff, or name the pocket yourself")  # fmt: skip
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


def drmsd_response(system, reference=None, displacement: float = 1.0, **kwargs) -> float:
    """How much the coordinate moves (nm) per A the ligand is displaced.

    The ligand is translated bodily by ``displacement`` A along each axis in turn and
    the coordinate averaged, so this is the factor between a distance a chemist would
    quote and the coordinate a hill is deposited along.  It is well under 1: a pair
    far from the direction of travel hardly changes, and the coordinate is a mean over
    all of them (about 0.55 for Brilliant Blue FCF in PANX1's pore).

    The published hill width, grid and escape cutoff are for an aligned RMSD, which
    moves 1 A per A.  Multiplying them by this is how to carry them over; it is a
    property of the pocket and the ligand, so it is worth measuring per system.
    """
    lig = system.select(kwargs.get("ligand") or _default_ligand()).ids
    whole = system.select("same residue as index " + " ".join(str(int(a)) for a in lig)).ids
    pos = np.asarray((system if reference is None else reference).positions, float)
    out = []
    for axis in range(3):
        moved = pos.copy()
        moved[whole, axis] += displacement
        out.append(drmsd_of(system, moved, reference=reference, **kwargs))
    return float(np.mean(out) / displacement)


def _default_ligand() -> str:
    from ..symmetry import DEFAULT_LIGAND

    return DEFAULT_LIGAND


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
