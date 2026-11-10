"""MM/GBSA: what binding the ligand is worth, frame by frame, from the run itself.

A pose that holds for 50 ns has proved it is stable, not that it is right: a
ligand parked in a wrong but greasy groove holds as well as the native pose
does, and nothing measured on the trajectory alone -- drift, contacts, how
many runs agree -- asks whether the pose is *favourable*.  This does.  Each
frame's energy is taken three times on the same coordinates, as the complex,
as the receptor alone and as the ligand alone, with the water replaced by a
Generalized Born model:

    dG = E(complex) - E(receptor) - E(ligand)
       = dE_vdw + dE_elec + dG_GB + dG_SA

Single-trajectory MM/GBSA (Kollman et al. 2000; Genheden & Ryde 2015).  The
bonded terms cancel exactly, since the pieces have the coordinates the complex
has, and what is left is the interaction energy screened by the solvent model.
No entropy term: normal modes cost more than the dynamics did, and for ranking
poses of one ligand in one protein -- the same atoms in every pose, so the same
systematic errors too -- they are left out as a rule.  That ranking is what the
number is good for.  Across different ligands the usual MM/GBSA caveats apply,
and an absolute binding free energy it is not.

The GB model is OpenMM's implementation of Amber's (GBn2 by default; OBC2,
OBC1, HCT and GBn as well), with the radii Amber would assign -- mbondi3 for
GBn2, mbondi2 for the others -- read off the elements and bonds, so a GAFF2
ligand is treated exactly as the protein is.  The nonpolar term is Amber's
MM/PBSA convention: surface tension times the solvent-accessible area buried,
0.0072 kcal/mol/A^2 (``surften``), on a Shrake-Rupley surface.

    boonza gbsa md/runs/*/ -o gbsa --ligandsel "chain L"    # one row per run
    boonza gbsa pose.mae --ligandsel "resname LIG"          # one structure

The frames are production as it was written, the ligand imaged next to the
protein, so what is scored is the pose the dynamics relaxed into and not the
one it was started from: a run that left a bad start and settled is scored
where it settled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .omm import KCAL, NM, to_openmm
from .solute import solute_ids
from .symmetry import DEFAULT_LIGAND

__all__ = ["GBSA", "MODELS", "SURFACE_TENSION", "TERMS", "gb_force", "gbsa", "kappa"]

#: OpenMM's Amber GB models, by Amber's names.  GBn2 (igb=8) is the default,
#: being the one Amber recommends and the one with element-level parameters for
#: a ligand's atoms (its alpha, beta and gamma are per element).
MODELS = ("GBn2", "OBC2", "OBC1", "HCT", "GBn")
#: water's dielectric
EPSILON_WATER = 78.5
#: Amber MM/PBSA's surface tension, kcal/mol/A^2 (``surften``), with no offset:
#: the nonpolar term is this times the solvent-accessible area buried
SURFACE_TENSION = 0.0072
TEMPERATURE = 298.15
#: what a binding energy is split into, in the order it is written out
TERMS = ("vdw", "elec", "gb", "sa")
PIECES = ("complex", "receptor", "ligand")

#: the force groups an OpenMM system's energy is read back by.  ``bonded`` is
#: every term inside one molecule; it cancels between the pieces and is kept as
#: the check that it did.
_GROUPS = {"bonded": 0, "elec": 1, "vdw": 2, "gb": 3}


def kappa(salt: float, dielectric: float = EPSILON_WATER,
          temperature: float = TEMPERATURE) -> float:  # fmt: skip
    """Debye screening (1/nm) for ``salt`` mol/L, as Amber and OpenMM take it.

    ``1/sqrt(eps0 kB / (2 NA e^2 1000)) = 50.33355`` in 1/A per sqrt(M/K),
    times Amber's 0.73 for the ions the solute excludes, times 10 for nm --
    what ``AmberPrmtopFile.createSystem(implicitSolventSaltConc=...)`` does.
    """
    if salt < 0:
        raise ValueError(f"a salt concentration is not negative: {salt}")
    return 50.33355 * math.sqrt(salt / dielectric / temperature) * 7.3


def gb_force(system, topology=None, model: str = "GBn2", salt: float = 0.0,
             dielectric: float = EPSILON_WATER, surface_area: str | None = None):  # fmt: skip
    """OpenMM's ``model`` over ``system``'s atoms, ready to add to its OpenMM System.

    Charges come from the force-field tables.  Radii and screening come from
    the elements and bonds as Amber assigns them (mbondi3 for GBn2, mbondi2
    for the others), through OpenMM's own tables -- what OpenMM does for a
    prmtop given ``implicitSolvent=`` -- so a ligand's atoms are treated as
    the protein's are, by what they are bonded to rather than by residue
    name.  ``topology`` is the OpenMM Topology ``to_openmm`` returned, and is
    built when not given.  ``surface_area``: None, or ``"ACE"`` for OpenMM's
    analytical nonpolar term, what it simulates with; a binding energy takes
    its nonpolar term from the buried surface instead.
    """
    from openmm import app
    from openmm.app.internal import customgbforces as gb

    from .omm import _topology

    classes = {"GBn2": gb.GBSAGBn2Force, "OBC2": gb.GBSAOBC2Force, "OBC1": gb.GBSAOBC1Force,
               "HCT": gb.GBSAHCTForce, "GBn": gb.GBSAGBnForce}  # fmt: skip
    if model not in classes:
        raise ValueError(f"the GB model is one of {', '.join(MODELS)}, not {model!r}")
    if topology is None:
        topology = _topology(system, app, np.asarray(system.atoms["anum"]), periodic=False)
    force = classes[model](solventDielectric=dielectric, soluteDielectric=1.0, SA=surface_area,
                           cutoff=None, kappa=kappa(salt, dielectric))  # fmt: skip
    radii = np.asarray(force.getStandardParameters(topology), dtype=np.float64)
    charge = np.asarray(system.atoms["charge"], dtype=np.float64)
    # a list, not an array: given an array, OpenMM keeps its rows as arrays and
    # later adds the radius index to one with `+`, which is elementwise on an array
    force.addParticles(np.column_stack([charge, radii]).tolist())
    force.finalize()
    force.setName("gb")
    return force


def _split_nonbonded(omm) -> None:
    """The NonbondedForce as two: the charges alone and the Lennard-Jones alone.

    OpenMM evaluates them together, and a binding energy wants them apart.
    The exceptions (1-4 pairs) are split the same way.  Where ``to_openmm``
    already put the Lennard-Jones in a CustomNonbondedForce (a force field
    with NBFIX or geometric combining), the NonbondedForce carries charges
    only and is just renamed.
    """
    import copy

    import openmm as mm

    names = [f.getName() for f in omm.getForces()]
    nb = next(f for f in omm.getForces() if isinstance(f, mm.NonbondedForce))
    nb.setName("elec")
    if "nonbonded_vdw" in names:
        return
    lj = copy.deepcopy(nb)
    lj.setName("vdw")
    for i in range(nb.getNumParticles()):
        q, sigma, epsilon = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q, sigma, 0.0)
        lj.setParticleParameters(i, 0.0, sigma, epsilon)
    for i in range(nb.getNumExceptions()):
        a, b, qq, sigma, epsilon = nb.getExceptionParameters(i)
        nb.setExceptionParameters(i, a, b, qq, sigma, 0.0)
        lj.setExceptionParameters(i, a, b, 0.0, sigma, epsilon)
    omm.addForce(lj)


def _group(force) -> str:
    name = force.getName()
    if name in ("elec", "gb"):
        return name
    if name in ("vdw", "nonbonded_vdw"):
        return "vdw"
    return "bonded"


def _platform(name: str | None):
    import openmm as mm

    if name:
        return mm.Platform.getPlatformByName(name)
    for want in ("CPU", "Reference"):
        try:
            return mm.Platform.getPlatformByName(want)
        except Exception:  # not built into this OpenMM
            continue
    raise RuntimeError("no OpenMM platform for single-point energies")


class _Piece:
    """One of the three systems, held in an OpenMM context for repeated energies.

    The full potential, constraints left as the flexible terms they stand
    for: the same geometry goes into the complex and into its pieces, so any
    term inside one molecule cancels whatever it is, and what is measured is
    everything between the molecules.
    """

    def __init__(self, system, model: str, salt: float, dielectric: float, platform):
        import openmm as mm
        from openmm import unit

        self.unit = unit
        topology, omm, _ = to_openmm(system, nonbonded_method="NoCutoff", constraints=False,
                                     dispersion_correction=False)  # fmt: skip
        _split_nonbonded(omm)
        omm.addForce(gb_force(system, topology, model, salt, dielectric))
        for force in omm.getForces():
            force.setForceGroup(_GROUPS[_group(force)])
        self.context = mm.Context(omm, mm.VerletIntegrator(0.001), platform)

    def energies(self, positions) -> dict[str, float]:
        """kcal/mol in each group, for ``positions`` in A."""
        pos = self.unit.Quantity(np.asarray(positions, dtype=np.float64) * NM, self.unit.nanometer)
        self.context.setPositions(pos)
        self.context.computeVirtualSites()
        out = {}
        for name, group in _GROUPS.items():
            e = self.context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
            out[name] = e.value_in_unit(self.unit.kilojoule_per_mole) / KCAL
        return out


@dataclass
class GBSA:
    """A run's binding energies, frame by frame, in kcal/mol.

    ``terms`` holds the differences, complex minus receptor minus ligand, by
    name: ``vdw``, ``elec``, ``gb`` (polar solvation), ``sa`` (nonpolar, from
    the buried surface), and ``bonded``, which is zero to the precision of the
    arithmetic and is kept as the check that it is.  ``pieces`` holds the
    energies the differences were taken from, and ``areas`` the solvent-
    accessible surface (A^2) of each piece.
    """

    model: str
    frames: np.ndarray
    terms: dict[str, np.ndarray] = field(default_factory=dict)
    pieces: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    areas: dict[str, np.ndarray] = field(default_factory=dict)
    #: how many atoms each piece has
    natoms: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def total(self) -> np.ndarray:
        """dG per frame: the four terms summed."""
        return sum(self.terms[k] for k in TERMS)

    def summary(self) -> dict:
        """The numbers a run is ranked by.

        ``dg`` is the mean over the frames and ``sem`` its standard error
        (frames taken as independent, which a nanosecond apart they nearly
        are); ``best`` is the lowest frame and ``best_frame`` which one;
        ``late`` is the mean of the second half, where a pose that settled
        late shows what it settled into.  Then the mean of each term.
        """
        total = self.total
        n = len(total)
        if n == 0:
            return {"frames": 0}
        std = float(total.std(ddof=1)) if n > 1 else 0.0
        out = {"frames": n, "dg": float(total.mean()), "std": std, "sem": std / math.sqrt(n),
               "best": float(total.min()), "best_frame": int(self.frames[int(total.argmin())]),
               "late": float(total[n // 2:].mean())}  # fmt: skip
        for k in (*TERMS, "bonded"):
            out[k] = float(self.terms[k].mean())
        return out


def _frames(system, positions, stride: int, box=None):
    """``(index, positions of every atom, box)`` for each frame asked for.

    ``box`` is for positions given as an array, which carry none of their own:
    one (3, 3) box for every frame or one per frame; the system's cell when
    not given, and all zeros for no periodic images at all.
    """
    from .trajectory import Frames, Trajectory

    if stride < 1:
        raise ValueError(f"stride is a positive number of frames, not {stride}")
    if positions is None:
        yield (
            0,
            np.asarray(system.positions, dtype=np.float64),
            (system.cell if box is None else np.asarray(box, dtype=np.float64).reshape(3, 3)),
        )
        return
    if isinstance(positions, Trajectory):
        for block in positions[::stride].chunks():
            for k in range(len(block)):
                yield int(block.indices[k]), block.positions[k], block.boxes[k]
        return
    if isinstance(positions, Frames):
        for k in range(0, len(positions), stride):
            yield int(positions.indices[k]), positions.positions[k], positions.boxes[k]
        return
    xyz = np.asarray(positions, dtype=np.float64)
    if xyz.ndim == 2:
        xyz = xyz[None]
    boxes = np.asarray(system.cell if box is None else box, dtype=np.float64)
    if boxes.ndim == 2:
        boxes = np.broadcast_to(boxes, (len(xyz), 3, 3))
    for k in range(0, len(xyz), stride):
        yield k, xyz[k], boxes[k]


def _atoms(system, what) -> np.ndarray:
    """Atom indices from a selection string or anything array-like."""
    if isinstance(what, str):
        return np.asarray(system.select(what).ids, dtype=np.int64)
    return np.asarray(getattr(what, "ids", what), dtype=np.int64)


def gbsa(system, positions=None, *, ligand=DEFAULT_LIGAND, receptor=None,
         keep: str | None = None, model: str = "GBn2", salt: float = 0.0,
         dielectric: float = EPSILON_WATER, surface_tension: float = SURFACE_TENSION,
         stride: int = 1, platform: str | None = None, box=None) -> GBSA:  # fmt: skip
    """MM/GBSA binding energy of ``ligand`` in ``system``, over ``positions``.

    ``system`` is the run's structure, water and all: the solvent is dropped
    (everything but water and ions is kept, or ``keep`` says what is -- a
    structural ion, say), the ligand is ``ligand`` and the receptor is the
    rest, or ``receptor``; both are selections or atom indices of ``system``,
    and the ligand is kept whatever it is made of, a water molecule included.
    ``positions`` is a Trajectory, a Frames block, an (nframes, natoms, 3)
    array of every atom, or None for the system's own coordinates; every
    ``stride``-th frame is scored.  Each frame is made whole and the ligand
    imaged beside the receptor first, since a trajectory is written wrapped;
    an array of positions is imaged in ``box`` (the system's cell by default;
    all zeros for a structure whose cell is a crystal's and not a box).

    ``model`` is one of :data:`MODELS`, ``salt`` in mol/L, ``surface_tension``
    in kcal/mol/A^2 over the buried solvent-accessible area, and ``platform``
    an OpenMM platform (CPU when there is one).  A ligand bonded to the
    receptor is refused: its bonded terms would not cancel, and a covalent
    complex is not what MM/GBSA measures.
    """
    from .glue import Glue
    from .shrake_rupley import sasa

    lig_all = _atoms(system, ligand)
    if not len(lig_all):
        raise ValueError(f"the ligand selection {ligand!r} picks no atom")
    # the pieces are named on the whole system and carried into the solute, so an
    # index selection means what it says whatever solvent sits before it
    ids = np.union1d(solute_ids(system, keep), lig_all)
    where = np.full(system.natoms, -1, dtype=np.int64)
    where[ids] = np.arange(len(ids))
    solute = system.clone(ids)
    if "posre_harm" in solute.tables:  # a restraint is the run's, not the molecule's
        solute.del_table("posre_harm")
    lig = where[lig_all]
    if receptor is None:
        rec = np.setdiff1d(np.arange(solute.natoms), lig)
    else:
        rec_all = _atoms(system, receptor)
        if np.intersect1d(rec_all, lig_all).size:
            raise ValueError("the receptor and the ligand share atoms")
        rec = where[rec_all]
        rec = rec[rec >= 0]  # a receptor atom that is solvent is not kept
    if not len(rec):
        raise ValueError("nothing is left for the receptor")
    in_lig = np.zeros(solute.natoms, bool)
    in_lig[lig] = True
    in_rec = np.zeros(solute.natoms, bool)
    in_rec[rec] = True
    i, j = np.asarray(solute.bonds["i"]), np.asarray(solute.bonds["j"])
    across = (in_lig[i] & in_rec[j]) | (in_rec[i] & in_lig[j])
    if across.any():
        raise ValueError(f"{int(across.sum())} bond(s) join the ligand to the receptor: a "
                         "covalent complex cannot be split into the two, and its bonded "
                         "terms would not cancel")  # fmt: skip

    where = _platform(platform)
    members = {"complex": np.arange(solute.natoms), "receptor": rec, "ligand": lig}
    systems = {k: solute if k == "complex" else solute.clone(v) for k, v in members.items()}
    scorers = {k: _Piece(v, model, salt, dielectric, where) for k, v in systems.items()}
    fix = Glue(solute, glue="all", wrap=False)  # whole, and the pieces jointly imaged

    indices: list[int] = []
    energies = {k: {g: [] for g in _GROUPS} for k in PIECES}
    areas = {k: [] for k in PIECES}
    for index, xyz, cell in _frames(system, positions, stride, box):
        pos, _ = fix(np.asarray(xyz)[ids], cell)
        for piece, atoms in members.items():
            here = pos[atoms]
            for g, e in scorers[piece].energies(here).items():
                energies[piece][g].append(e)
            areas[piece].append(float(sasa(systems[piece], here[None]).sum()))
        indices.append(index)

    out = GBSA(model, np.asarray(indices, dtype=np.int64))
    out.natoms = {k: int(len(v)) for k, v in members.items()}
    out.pieces = {k: {g: np.asarray(v, dtype=np.float64) for g, v in e.items()}
                  for k, e in energies.items()}  # fmt: skip
    out.areas = {k: np.asarray(v, dtype=np.float64) for k, v in areas.items()}

    def delta(values):
        return values["complex"] - values["receptor"] - values["ligand"]

    for g in _GROUPS:
        out.terms[g] = delta({k: out.pieces[k][g] for k in PIECES})
    out.terms["sa"] = surface_tension * delta(out.areas)
    return out
