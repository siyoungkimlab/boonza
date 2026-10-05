"""Backbone phi/psi restraints holding the protein near its input conformation.

The restraint on each torsion is a truncated Fourier well,

    V(theta) = sum_{i=1..6} K (-1)^i / i! [1 + cos(i (theta - theta0 - pi))],

with K = -|strength|, so that it sits on theta0. Each term is one OpenMM
periodic torsion with periodicity i and phase i (theta0 + pi). The force is
part of the OpenMM system, so ``system.xml`` carries it into restarts.
"""

from __future__ import annotations

import csv
import math

import numpy as np

FOURIER_TERMS = 6


def backbone_torsions(s) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    """(residue, "phi" or "psi", atoms) for each backbone torsion whose peptide
    bond exists, so termini and chain breaks are left free."""
    names, res = s.atoms["name"], s.atoms["residue"]
    bb: dict[int, dict[str, int]] = {}
    for a in np.flatnonzero(np.isin(names, ["N", "CA", "C"])).tolist():
        bb.setdefault(int(res[a]), {})[str(names[a])] = a
    out = []
    for r in sorted(bb):
        if len(bb[r]) != 3:
            continue
        n, ca, c = bb[r]["N"], bb[r]["CA"], bb[r]["C"]
        prev = [o for o in s.bonded_atoms(n).tolist() if res[o] != r and names[o] == "C"]
        nxt = [o for o in s.bonded_atoms(c).tolist() if res[o] != r and names[o] == "N"]
        if prev:
            out.append((r, "phi", (prev[0], n, ca, c)))
        if nxt:
            out.append((r, "psi", (n, ca, c, nxt[0])))
    return out


def bead_torsions(s) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    """(residue, "bb", beads) for each BB-BB-BB-BB torsion of a coarse-grained
    backbone: four residues in a row, bonded, so chain breaks are left free.

    It is the Martini counterpart of phi and psi, and the torsion Martini's
    own helix term acts on.
    """
    names, res = s.atoms["name"], s.atoms["residue"]
    bb = {int(res[a]): int(a) for a in np.flatnonzero(names == "BB").tolist()}
    out = []
    for r in sorted(bb):
        beads = [bb.get(r + k) for k in range(4)]
        if any(b is None for b in beads):
            continue
        if all(beads[k + 1] in s.bonded_atoms(beads[k]).tolist() for k in range(3)):
            out.append((r, "bb", tuple(beads)))
    return out


def sirah_torsions(s) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    """(residue, kind, beads) for each backbone torsion of a SIRAH chain.

    SIRAH's backbone is three beads a residue -- GN on the amide nitrogen, GC on
    the alpha carbon, GO on the carbonyl oxygen -- bonded GN-GC-GO-GN(+1), so
    phi and psi are there as they are all-atom: psi is GN-GC-GO-GN(+1) where
    all-atom it is N-CA-C-N, and phi is GO(-1)-GN-GC-GO where all-atom it is
    C-N-CA-C.  A residue whose neighbour is missing, or a chain break, leaves
    the torsion out, as the bonds say.
    """
    names, res = s.atoms["name"], s.atoms["residue"]
    bead = {}
    for which in ("GN", "GC", "GO"):
        for a in np.flatnonzero(names == which).tolist():
            bead[(int(res[a]), which)] = int(a)
    out = []
    for r in sorted({k for k, _ in bead}):
        here = [bead.get((r, w)) for w in ("GN", "GC", "GO")]
        if any(b is None for b in here):
            continue
        n, ca, o = here
        after, before = bead.get((r + 1, "GN")), bead.get((r - 1, "GO"))
        for kind, atoms in (("psi", (n, ca, o, after)), ("phi", (before, n, ca, o))):
            if atoms[0] is None or atoms[3] is None:
                continue
            if all(atoms[k + 1] in s.bonded_atoms(atoms[k]).tolist() for k in range(3)):
                out.append((r, kind, tuple(atoms)))
    return out


def coarse_grained(s) -> str:
    """Which coarse-grained backbone this system has: ``"martini"`` (one bead a
    residue, BB), ``"sirah"`` (three, GN-GC-GO), or ``""`` for atoms."""
    if len(s.select("name CA").ids):
        return ""
    if len(s.select("name BB").ids) >= 4:
        return "martini"
    return "sirah" if len(s.select("name GC").ids) >= 4 else ""


def fourier_terms(strength_kj: float):
    """(periodicity, force constant in kJ/mol) of the well."""
    k = -abs(strength_kj)
    for i in range(1, FOURIER_TERMS + 1):
        yield i, k * (-1) ** i / math.factorial(i)


def dihedral(pos, atoms) -> float:
    """The torsion angle of four atoms (radians)."""
    p0, p1, p2, p3 = (np.asarray(pos[a], dtype=np.float64) for a in atoms)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.atan2(float(np.dot(np.cross(b1, v), w)), float(np.dot(v, w)))


def add_dihedral_restraints(omm_system, s, mode: str, strength_kj: float, selection=None,
                            secondary=None):  # fmt: skip
    """Restrain the backbone torsions of ``s`` (``mode`` "bb" or "ss") to their
    values in its positions, only torsions whose atoms ``selection`` picks when
    given; returns (records, description).

    All-atom torsions are phi and psi; a coarse-grained one is BB-BB-BB-BB
    over four residues, the torsion Martini's own helix term acts on.  There
    ``mode`` "ss" needs ``secondary``, the DSSP codes of the protein's
    residues, since DSSP cannot read beads; boonza writes them beside the
    topology it builds.
    """
    import openmm as mm

    model = coarse_grained(s)
    beads = bool(model)
    torsions = {"martini": bead_torsions, "sirah": sirah_torsions,
                "": backbone_torsions}[model](s)  # fmt: skip
    if selection:
        picked = np.zeros(s.natoms, bool)
        picked[s.select(selection).ids] = True
        torsions = [t for t in torsions if picked[list(t[2])].all()]
    if mode == "ss":
        if beads:
            if not secondary:
                raise ValueError("dihedral_restraint = 'ss' needs the secondary structure of a "
                                 "coarse-grained system, which boonza writes as secondary.txt "
                                 "beside the topology it builds; use 'bb' instead")  # fmt: skip
            # the codes are the protein's residues, in order, from its first one
            first = min((t[0] for t in torsions), default=0)
            structured = ("H", "G", "I", "E", "B")
            chosen = {first + k for k, code in enumerate(secondary) if code in structured}
            # a Martini torsion spans four residues; a SIRAH one spans two, and
            # both of them have to be structured for it to be held
            span = 4 if model == "martini" else 2
            torsions = [t for t in torsions if all(t[0] + k in chosen for k in range(span))]
        else:
            from ..secondary import dssp

            codes = dssp(s, simplified=True)[0]
            chosen = {r for r, code in enumerate(codes.tolist()) if code in ("H", "E")}
            torsions = [t for t in torsions if t[0] in chosen]
        description = f"{len(chosen)} residues in helices and sheets"
    else:
        description = f"{len({t[0] for t in torsions})} protein residues"
    if not torsions:
        raise ValueError(
            f"dihedral_restraint = '{mode}' found no backbone torsions ({description})"
        )
    force = mm.PeriodicTorsionForce()
    force.setName("DihedralRestraint")
    pos = s.positions
    chains = s.chains["name"]
    records = []
    for r, kind, atoms in torsions:
        ref = dihedral(pos, atoms)
        for n, k in fourier_terms(strength_kj):
            force.addTorsion(*atoms, n, (n * (ref + math.pi)) % (2 * math.pi), k)
        records.append(
            {
                "angle": kind,
                "chain_id": str(chains[s.residues["chain"][r]]),
                "residue_name": str(s.residues["name"][r]),
                "residue_id": int(s.residues["resid"][r]),
                "atom_indices": " ".join(map(str, atoms)),
                "atom_names": " ".join(str(s.atoms["name"][a]) for a in atoms),
                "reference_degrees": round(math.degrees(ref), 3),
            }
        )
    omm_system.addForce(force)
    return records, description


def add_elastic_network(omm_system, s, selection: str, lower_nm: float = 0.05,
                        upper_nm: float = 0.9, k_kj: float = 500.0,
                        res_min_dist: int = 2) -> int:  # fmt: skip
    """Hold a fold with springs between the beads ``selection`` picks.

    Every pair of selected beads between ``lower_nm`` and ``upper_nm`` apart in
    the starting structure gets a harmonic bond at that distance, skipping pairs
    within ``res_min_dist`` residues of each other, which the bonded terms
    already hold.  ``k`` is in kJ/mol/nm^2.

    It is what Martini calls a rubber band, for a model that has none: SIRAH
    holds its fold with torsions and wanders 4 to 6.5 A of backbone RMSD, which
    is why it finds a cryptic site Martini's pinned backbone never opens.  A
    network here takes that away, so it is worth measuring rather than assuming.

    The springs are forces of the OpenMM system and no part of the topology, so
    they generate no exclusions: a pair held by a band keeps every nonbonded
    interaction it had.  This is what GROMACS bond type 6 is for, which Martini
    2.2 uses for the same reason, where Martini 3's type 1 does create them.
    Nothing is written into the structure either, so a viewer draws no hairball.
    Returns the number of springs.
    """
    import openmm as mm

    from ..spatial import pairs_within

    ids = np.asarray(s.select(selection).ids, int)
    if len(ids) < 2:
        return 0
    xyz = np.asarray(s.positions, float)[ids] / 10.0  # A to nm
    resid = np.array([s.atom(int(a)).residue.resid for a in ids])
    chain = np.array([str(s.atom(int(a)).residue.chain.name) for a in ids])
    ii, jj, d2 = pairs_within(xyz, float(upper_nm))
    d = np.sqrt(d2)
    keep = d >= float(lower_nm)
    # a pair a few residues apart is already held by the bonded terms, and a
    # spring there only stiffens what the force field already says
    same = chain[ii] == chain[jj]
    keep &= ~(same & (np.abs(resid[ii] - resid[jj]) <= int(res_min_dist)))
    if not keep.any():
        return 0
    force = mm.HarmonicBondForce()
    force.setUsesPeriodicBoundaryConditions(omm_system.usesPeriodicBoundaryConditions())
    for a, b, r0 in zip(ii[keep].tolist(), jj[keep].tolist(), d[keep].tolist(), strict=True):
        force.addBond(int(ids[a]), int(ids[b]), float(r0), float(k_kj))
    force.setName("ElasticNetwork")
    omm_system.addForce(force)
    return force.getNumBonds()


def add_repulsion(omm_system, s, selection: str, distance_nm: float, k_kj: float) -> int:
    """Keep the molecules ``selection`` picks from sticking together.

    A flat-bottom wall, E = k (d0 - r)^2 for r < d0, acts between the heavy
    atoms of different selected molecules; not within a molecule, and not
    with the rest of the system. ``k`` in kJ/mol/nm^2, ``d0`` in nm. It is a
    force of the OpenMM system, so ``system.xml`` carries it into restarts.
    Returns the number of molecules.
    """
    import openmm as mm

    anum = s.atoms["anum"]
    frag = np.asarray(s.fragids)
    molecules: dict[int, list[int]] = {}
    for a in s.select(selection).ids.tolist():
        if anum[a] > 1:
            molecules.setdefault(int(frag[a]), []).append(int(a))
    if len(molecules) < 2:
        return len(molecules)
    # atoms of one molecule share a number, and the wall vanishes between them
    # (not by exclusions: OpenMM needs every nonbonded force to exclude alike)
    force = mm.CustomNonbondedForce(
        "repulsion_k*step(repulsion_d0-r)*(repulsion_d0-r)^2*(1-delta(mol1-mol2))"
    )
    force.addGlobalParameter("repulsion_k", float(k_kj))
    force.addGlobalParameter("repulsion_d0", float(distance_nm))
    force.addPerParticleParameter("mol")
    number = np.zeros(omm_system.getNumParticles())
    for k, atoms in enumerate(molecules.values(), start=1):
        number[atoms] = k
    for x in number.tolist():
        force.addParticle([x])
    periodic = omm_system.usesPeriodicBoundaryConditions()
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic if periodic
                             else mm.CustomNonbondedForce.CutoffNonPeriodic)  # fmt: skip
    force.setCutoffDistance(float(distance_nm))
    heavy = [a for atoms in molecules.values() for a in atoms]
    force.addInteractionGroup(heavy, heavy)
    for other in omm_system.getForces():  # the same exclusions as the NonbondedForce
        if isinstance(other, mm.NonbondedForce):
            for e in range(other.getNumExceptions()):
                i, j = other.getExceptionParameters(e)[:2]
                force.addExclusion(i, j)
            break
    force.setName("LigandRepulsion")
    omm_system.addForce(force)
    return len(molecules)


def write_records(path, records) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def plot_well(path, strength_kj: float) -> bool:
    """Plot the well against the equivalent harmonic; False without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    x = np.linspace(-math.pi, math.pi, 721)
    well = sum(k * (1 + np.cos(n * (x - math.pi))) for n, k in fourier_terms(strength_kj))
    well -= well.min()
    harmonic = 0.5 * _curvature(strength_kj) * x**2
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.degrees(x), well, label="restraint")
    ax.plot(np.degrees(x), harmonic, "--", label="harmonic, same curvature")
    ax.set_xlabel("deviation from reference (degrees)")
    ax.set_ylabel("energy (kJ/mol)")
    ax.set_ylim(0, max(float(well.max()) * 1.2, 1e-9))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _curvature(strength_kj: float) -> float:
    """d2V/dtheta2 at the reference: sum of k n^2 cos(n pi) terms."""
    return sum(-k * n * n * math.cos(n * math.pi) for n, k in fourier_terms(strength_kj))
