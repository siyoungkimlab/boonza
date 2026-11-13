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
    if len(s.select("protein and name CA").ids):
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


def residue_keys(s, chains: bool = True) -> list[tuple]:
    """What names each residue of ``s`` across structures: (chain name, residue
    number, insertion code, residue name), or the last three without the chain."""
    names = s.chains["name"]
    res = s.residues
    ins = res["insertion"] if "insertion" in res else np.full(len(res), "")
    out = []
    for c, n, i, nm in zip(res["chain"], res["resid"], ins, res["name"], strict=True):
        key = (int(n), str(i).strip(), str(nm))
        out.append((str(names[c]),) + key if chains else key)
    return out


def paired_residue_keys(s, reference) -> tuple[list[tuple], list[tuple]]:
    """:func:`residue_keys` of ``s`` and of ``reference``, with the chain when
    the two share a chain name and without it otherwise, as a crystal's chain A
    and a model's chain of another name are the same chain."""
    mine, theirs = residue_keys(s), residue_keys(reference)
    if not set(mine) & set(theirs):
        mine, theirs = residue_keys(s, chains=False), residue_keys(reference, chains=False)
    return mine, theirs


def reference_distances(s, reference, pairs) -> list[float]:
    """The distance (nm) in ``reference`` of each atom pair ``(a, b)`` of ``s``,
    atoms paired by residue (:func:`paired_residue_keys`) and atom name; an
    atom the reference lacks is an error, since a spring needs its length."""
    mine, theirs = paired_residue_keys(s, reference)
    where = {}
    names = reference.atoms["name"]
    for a, r in enumerate(reference.atoms["residue"].tolist()):
        where.setdefault((theirs[r], str(names[a])), a)
    out = []
    for a, b in pairs:
        atoms = []
        for i in (a, b):
            key = (mine[int(s.atoms["residue"][i])], str(s.atoms["name"][i]))
            if key not in where:
                raise ValueError(f"distance restraint: the reference has no atom "
                                 f"{_atom_label(s, i)!r} to measure a spring on")  # fmt: skip
            atoms.append(where[key])
        p, q = reference.positions[atoms[0]], reference.positions[atoms[1]]
        out.append(float(np.linalg.norm(q - p)) / 10.0)  # A to nm
    return out


def reference_targets(s, reference, torsions, mode: str) -> dict:
    """{(residue of ``s``, "phi" or "psi"): angle} the ``torsions`` of ``s`` are
    held at, read off ``reference``, another structure of the same protein.

    Residues pair by chain name, number, insertion code and residue name -- or,
    when the two share no chain name, by number, code and name alone, as a
    crystal's chain A and a model's chain of another name are the same chain.
    Under "ss" the residues DSSP puts in the reference's helices and sheets are
    held, so a model is held to the structure it came from and not to its own
    backbone.  A torsion whose residue the reference lacks is left free.
    """
    from ..secondary import dssp

    mine, theirs = paired_residue_keys(s, reference)
    ref = {}
    for r, kind, atoms in backbone_torsions(reference):
        ref[(theirs[r], kind)] = dihedral(reference.positions, atoms)
    structured = None
    if mode == "ss":
        codes = dssp(reference, simplified=True)[0]
        structured = {theirs[r] for r, code in enumerate(codes.tolist()) if code in ("H", "E")}
    out = {}
    for r, kind, _ in torsions:
        key = mine[r]
        if (key, kind) in ref and (structured is None or key in structured):
            out[(r, kind)] = ref[(key, kind)]
    return out


def add_dihedral_restraints(omm_system, s, mode: str, strength_kj: float, selection=None,
                            secondary=None, reference=None):  # fmt: skip
    """Restrain the backbone torsions of ``s`` (``mode`` "bb" or "ss") to their
    values in its positions, only torsions whose atoms ``selection`` picks when
    given; returns (records, description).

    All-atom torsions are phi and psi; a coarse-grained one is BB-BB-BB-BB
    over four residues, the torsion Martini's own helix term acts on.  There
    ``mode`` "ss" needs ``secondary``, the DSSP codes of the protein's
    residues, since DSSP cannot read beads; boonza writes them beside the
    topology it builds.

    ``reference``, another structure of the same protein (a crystal structure,
    say), gives every torsion its target instead of the positions of ``s``,
    and under "ss" the residues DSSP picks (:func:`reference_targets`): a
    backmapped or predicted model is held to the structure it came from, not
    to its own backbone.
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
    targets = None
    if reference is not None:
        if beads:
            raise ValueError("dihedral_restraint_reference holds an all-atom backbone at "
                             "another structure's phi and psi; a coarse-grained run has "
                             "neither")  # fmt: skip
        targets = reference_targets(s, reference, torsions, mode)
        torsions = [t for t in torsions if (t[0], t[1]) in targets]
        held = len({t[0] for t in torsions})
        description = (f"{held} residues in the reference's helices and sheets" if mode == "ss"
                       else f"{held} protein residues the reference has")  # fmt: skip
    elif mode == "ss":
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
        here = dihedral(pos, atoms)
        ref = targets[(r, kind)] if targets is not None else here
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
                "input_degrees": round(math.degrees(here), 3),
            }
        )
    omm_system.addForce(force)
    return records, description


def add_elastic_network(omm_system, s, selection: str, lower_nm: float = 0.05,
                        upper_nm: float = 0.9, k_kj: float = 700.0,
                        res_min_dist: int = 2) -> int:  # fmt: skip
    """Hold a fold with springs between the beads ``selection`` picks.

    Every pair of selected beads between ``lower_nm`` and ``upper_nm`` apart in
    the starting structure gets a harmonic bond at that distance, skipping pairs
    within ``res_min_dist`` residues of each other, which the bonded terms
    already hold.  ``k`` is in kJ/mol/nm^2, 700 as martinize2's -ef is, so that
    an elastic network means the same stiffness whichever model is asked for it.

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


def add_distance_restraints(omm_system, s, pairs, k_kj: float, r0_nm=None,
                            reference=None) -> list[dict]:  # fmt: skip
    """Hold atoms of one selection to atoms of another with harmonic springs.

    For each ``(first, second)`` in ``pairs``, every atom ``first`` picks is
    tied to every atom ``second`` picks -- not to the atoms of its own
    selection -- with E = k/2 (r - r0)^2, OpenMM's harmonic bond.  ``r0`` is
    ``r0_nm`` when given, else each pair's distance in ``reference`` (another
    structure of the same molecules, a crystal structure say, atoms paired by
    residue and name: :func:`reference_distances`) when that is given, else
    each pair's distance in ``s`` (the built system, before minimization).
    ``k`` is in kJ/mol/nm^2.

    The springs are a force of the OpenMM system, as the elastic network's are,
    and no part of the topology: a held pair keeps its nonbonded interactions,
    nothing is written into the structure, and ``system.xml`` carries them into
    restarts.  Returns one record per spring.
    """
    import openmm as mm

    xyz = np.asarray(s.positions, float) / 10.0  # A to nm
    periodic = omm_system.usesPeriodicBoundaryConditions()
    box = None
    if periodic:  # the cell the run uses, in nm
        from openmm import unit

        box = np.array([v.value_in_unit(unit.nanometer)
                        for v in omm_system.getDefaultPeriodicBoxVectors()], float)  # fmt: skip
    force = mm.HarmonicBondForce()
    force.setUsesPeriodicBoundaryConditions(periodic)
    records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    springs: list[tuple[int, int]] = []
    for first, second in pairs:
        a_ids = s.select(first).ids.tolist()
        b_ids = s.select(second).ids.tolist()
        for sel, ids in ((first, a_ids), (second, b_ids)):
            if not ids:
                raise ValueError(f"distance restraint: {sel!r} picks no atoms")
        for a in a_ids:
            for b in b_ids:
                key = (min(a, b), max(a, b))
                if a == b or key in seen:
                    continue
                seen.add(key)
                springs.append((int(a), int(b)))
    lengths = reference_distances(s, reference, springs) if reference is not None else None
    for n, (a, b) in enumerate(springs):
        d = xyz[b] - xyz[a]
        if box is not None:
            d = _minimum_image(d, box)
        r = float(np.linalg.norm(d))
        r0 = float(r0_nm) if r0_nm is not None else lengths[n] if lengths else r
        force.addBond(a, b, r0, float(k_kj))
        records.append({"atom1": a, "label1": _atom_label(s, a),
                        "atom2": b, "label2": _atom_label(s, b),
                        "start_nm": round(r, 4), "r0_nm": round(r0, 4),
                        "k_kJ_per_nm2": float(k_kj)})  # fmt: skip
    if not records:
        raise ValueError("distance restraint: the selections pick no pair of different atoms")
    force.setName("DistanceRestraints")
    omm_system.addForce(force)
    return records


def add_position_restraints(omm_system, s, selections, k_kj: float,
                             scaling: str = "com") -> list[dict]:  # fmt: skip
    """Hold the atoms each of ``selections`` picks where they are in ``s``.

    Every picked atom gets E = k/2 |r - r0|^2, ``r0`` its own position in ``s``
    (the built system, before minimization) and ``k`` in kJ/mol/nm^2.  In a
    periodic system the displacement is the minimum image's, so an atom the
    integrator carries across the cell is not pulled back through it.  An atom
    named by two selections is held once; a massless one (a virtual site) is
    left out, its place being its parents' to decide.

    With ``scaling`` of ``"none"`` the anchors are fixed coordinates (GROMACS's
    ``refcoord_scaling = no``): a barostat's volume move shifts the held atoms
    off them, so the springs resist it.  With ``"com"``, the default, the
    anchors follow the cell as ``refcoord_scaling = com`` has them: a barostat
    moves each molecule as a body, its centre scaled with the box, so each
    anchor is carried the same way -- the centre of its atom's molecule scales
    with the box and its offset from that centre does not.  A
    volume move then costs the springs nothing, and they do not weigh on the
    pressure coupling.  The centre is the centroid of the molecule as OpenMM
    sees it (:func:`_molecule_centres`), which is what its barostats scale.
    The force reads the box as it is at every evaluation
    (:func:`_box_lengths`): per axis in a rectangular cell, so a membrane's
    x-y and z go their own ways, and by one factor in a triclinic one, which
    only an isotropic barostat changes.

    The springs are a force of the OpenMM system, so ``system.xml`` carries
    them into restarts, and they last the whole run.  Returns one record per
    held atom.
    """
    import openmm as mm

    if scaling not in ("com", "none"):
        raise ValueError(f"position restraint: scaling is 'com' or 'none', not {scaling!r}")
    xyz = np.asarray(s.positions, float) / 10.0  # A to nm
    periodic = omm_system.usesPeriodicBoundaryConditions()
    follow = periodic and scaling == "com"
    picked: list[int] = []
    seen: set[int] = set()
    for sel in selections:
        ids = s.select(sel).ids.tolist()
        if not ids:
            raise ValueError(f"position restraint: {sel!r} picks no atoms")
        for a in ids:
            a = int(a)
            if a in seen or omm_system.getParticleMass(a)._value == 0:
                continue
            seen.add(a)
            picked.append(a)
    if not picked:
        raise ValueError("position restraint: the selections pick only massless particles")
    if follow:
        from openmm import unit

        box = np.array([v.value_in_unit(unit.nanometer)
                        for v in omm_system.getDefaultPeriodicBoxVectors()], float)  # fmt: skip
        lengths, scale = _box_lengths(box)
        force = mm.CustomExternalForce(
            "0.5*k*periodicdistance(x,y,z,ax+ux*bx,ay+uy*by,az+uz*bz)^2;" + lengths
        )
        names = ("k", "ax", "ay", "az", "ux", "uy", "uz")
        centres = _molecule_centres(omm_system, xyz, picked)
    else:
        force = mm.CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2" if periodic
                                       else "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")  # fmt: skip
        names = ("k", "x0", "y0", "z0")
    for p in names:
        force.addPerParticleParameter(p)
    records: list[dict] = []
    for a in picked:
        if follow:  # the anchor is (x0 - centre) + centre * (box now / box at the start)
            c = centres[a]
            force.addParticle(a, [float(k_kj), *(xyz[a] - c).tolist(), *(c / scale).tolist()])
        else:
            force.addParticle(a, [float(k_kj), *xyz[a].tolist()])
        records.append({"atom": a, "label": _atom_label(s, a),
                        "x0_nm": round(float(xyz[a][0]), 4),
                        "y0_nm": round(float(xyz[a][1]), 4),
                        "z0_nm": round(float(xyz[a][2]), 4),
                        "k_kJ_per_nm2": float(k_kj)})  # fmt: skip
    force.setName("PositionRestraints")
    omm_system.addForce(force)
    return records


def _box_lengths(box) -> tuple[str, np.ndarray]:
    """The box's lengths now, as an OpenMM expression defining ``bx``, ``by``
    and ``bz`` (nm), and what each is at the start.

    A custom force has no variable for the box, but ``periodicdistance`` knows
    it: a point ``h`` along x from the origin, with L/2 < h < L, is L - h from
    it by the minimum image, so L = h + periodicdistance(0,0,0,h,0,0).  With
    ``h`` at 1/sqrt(2) of the starting length that holds while the box stays
    between 0.71 and 1.41 times it, far beyond what a barostat does to a
    liquid.  A triclinic cell is read along its first vector, which lies on x,
    and the one factor serves all three axes: the same length is returned
    three times.
    """
    box = np.asarray(box, float).reshape(3, 3)
    start = np.diag(box).copy()

    def length(axis: int) -> str:
        h = float(start[axis]) / math.sqrt(2.0)
        point = ",".join(repr(h) if k == axis else "0" for k in range(3))
        return f"{h!r}+periodicdistance(0,0,0,{point})"

    if np.abs(box - np.diag(start)).max() > 1e-6:  # triclinic: one factor, from a
        # OpenMM reads a definition's own variables from the ones after it
        return f"by=bx;bz=bx;bx={length(0)}", np.full(3, start[0])
    return f"bx={length(0)};by={length(1)};bz={length(2)}", start


def _molecule_centres(omm_system, xyz, atoms) -> dict[int, np.ndarray]:
    """The centroid (nm) of the molecule each of ``atoms`` is in: the atoms
    OpenMM takes for one molecule -- joined by bonds, constraints or any other
    bonded term, an elastic network's springs and distance restraints among
    them -- which is the body its barostats move in one piece.  Call it once
    the system has every such force it will get."""
    import openmm as mm

    context = mm.Context(omm_system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))  # fmt: skip
    wanted = set(atoms)
    out = {}
    for molecule in context.getMolecules():
        mine = wanted.intersection(molecule)
        if mine:
            centre = xyz[list(molecule)].mean(axis=0)
            out.update(dict.fromkeys(mine, centre))
    del context
    return out


def _minimum_image(d, box):
    """``d`` wrapped into the cell ``box`` (3x3 rows, nm), triclinic included."""
    box = np.asarray(box, float).reshape(3, 3)
    for k in (2, 1, 0):  # OpenMM's reduced form: c, then b, then a
        d = d - box[k] * np.round(d[k] / box[k][k])
    return d


def _atom_label(s, a) -> str:
    atom = s.atom(int(a))
    res = atom.residue
    return f"{res.chain.name}:{res.name}{res.resid}:{atom.name}"


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
