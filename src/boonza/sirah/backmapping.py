"""SIRAH beads back to atoms.

A SIRAH bead sits on one atom of the structure (:mod:`boonza.sirah.build`), so a
coarse-grained frame says where those atoms are and nothing about the rest.  With the
all-atom structure the run was built from, each of its residues is laid on its beads
as a rigid body and the whole is then minimized with the mapped atoms held to their
beads: names, protonation states, ligands and caps are the reference's own.  Without
it, amino acids are built from the bead names (ideal geometry, the force field's
default protonation states) and treated the same way.
"""

from __future__ import annotations

import numpy as np

from ..align import kabsch
from . import IONS, WATER
from .build import read_map, read_renames

#: the all-atom residue a coarse-grained one is built as when there is no reference:
#: SIRAH's map folds several names into one bead set, and these are the ones the
#: Amber templates know
BUILT_AS = {"sHe": "HIE", "sHd": "HID", "sK": "LYS", "sM": "MET", "sDh": "ASH", "sEh": "GLH",
            "sZ": "CYM", "sX": "CYX", "sC": "CYS", "sD": "ASP", "sE": "GLU"}  # fmt: skip
ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D", "CYS": "C",
              "CYX": "C", "CYM": "C", "GLN": "Q", "GLU": "E", "GLH": "E", "GLY": "G",
              "HIE": "H", "HID": "H", "HIP": "H", "HIS": "H", "ILE": "I", "LEU": "L",
              "LYS": "K", "LYN": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
              "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}  # fmt: skip


#: the nucleic-acid force field the minimization uses when none is named
NUCLEIC_FORCEFIELD = "na.amber.bsc1"


class BackmapError(ValueError):
    """A coarse-grained system that cannot be turned back into atoms as asked."""


def _keys(s) -> list[tuple[str, int, str]]:
    """(chain, resid, insertion) of every residue of ``s``."""
    chain = np.array([str(x).strip() for x in s.chains["name"]])[np.asarray(s.residues["chain"])]
    resid = np.asarray(s.residues["resid"])
    ins = [str(x).strip() for x in s.residues["insertion"]]
    return [(str(c), int(r), i) for c, r, i in zip(chain, resid, ins, strict=True)]


def _residue_atoms(s) -> list[np.ndarray]:
    res = np.asarray(s.atoms["residue"])
    order = np.argsort(res, kind="stable")
    cuts = np.searchsorted(res[order], np.arange(s.nresidues + 1))
    return [order[cuts[r] : cuts[r + 1]] for r in range(s.nresidues)]


def bead_atoms(cg, aa) -> tuple[np.ndarray, np.ndarray]:
    """Which atom of ``aa`` each bead of ``cg`` sits on: ``(beads, atoms)``, indices
    paired.  Residues are matched by chain, number and insertion code; a residue
    SIRAH's map knows by its map (the first of a bead's atom names the residue
    has), any other (a cofactor, a bead a heavy atom) by the atoms' own names.
    Solvent beads and beads of residues ``aa`` does not have are left out."""
    maps = read_map()
    # the name a bead of the map goes by in the topology, where the two differ
    # (SIRAH's dna.arn: the map's C1X, on C1', is the topology's O3')
    renamed = read_renames()
    cg_names = [str(x).strip() for x in cg.atoms["name"]]
    aa_names = [str(x).strip() for x in aa.atoms["name"]]
    aa_resname = [str(x).strip() for x in aa.residues["name"]]
    cg_of = {}
    for r, (key, atoms) in enumerate(zip(_keys(cg), _residue_atoms(cg), strict=True)):
        if str(cg.residues["name"][r]).strip() != WATER:
            cg_of.setdefault(key, {}).update({cg_names[b]: int(b) for b in atoms})
    beads, atoms = [], []
    for r, (key, mine) in enumerate(zip(_keys(aa), _residue_atoms(aa), strict=True)):
        there = cg_of.get(key)
        if not there:
            continue
        have = {aa_names[a]: int(a) for a in mine}
        entry = maps.get(aa_resname[r])
        if entry is not None:
            for bead, sources in entry.beads:
                atom = next((have[n] for n in sources if n in have), None)
                bead = bead if bead in there else renamed.get(bead)
                if atom is not None and bead in there:
                    beads.append(there[bead])
                    atoms.append(atom)
        else:  # a cofactor: a bead for each heavy atom, under the atom's name
            for name, bead in there.items():
                if name in have:
                    beads.append(bead)
                    atoms.append(have[name])
    return np.array(beads, np.int64), np.array(atoms, np.int64)


def _whole(cg, xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """``xyz`` with the solute's molecules unbroken across ``box`` and beside each
    other (a chain cut by the box edge, or two chains a box apart, would be laid
    down that way); the solvent is left where it is."""
    from ..glue import Glue

    names = np.array([str(x).strip() for x in cg.residues["name"]])[np.asarray(cg.atoms["residue"])]
    solute = np.flatnonzero((names != WATER) & ~np.isin(names, list(IONS)))
    if not len(solute):
        return xyz
    out, _ = Glue(cg, glue=[solute.tolist()], whole=solute.tolist(), wrap=False)(xyz, box)
    return np.asarray(out, float)


def _place(aa, pairs_of: dict[int, tuple[np.ndarray, np.ndarray]], log) -> tuple[np.ndarray, list]:
    """Each residue of ``aa`` laid on its beads as a rigid body (three pairs or more:
    a fit; fewer: a shift).  A residue with no bead goes with the placed residue it
    is bonded to (a cap), and is otherwise reported: returns the positions and the
    residues left where they were."""
    pos = np.array(aa.positions, float)
    out = pos.copy()
    mine = _residue_atoms(aa)
    moves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for r, (mobile, target) in pairs_of.items():
        if len(mobile) >= 3:
            R, t = kabsch(pos[mobile], target)
        else:
            R, t = np.eye(3), target.mean(axis=0) - pos[mobile].mean(axis=0)
        moves[r] = (R, t)
        out[mine[r]] = pos[mine[r]] @ R.T + t
    res = np.asarray(aa.atoms["residue"])
    left = []
    for r in range(aa.nresidues):
        if r in moves or not len(mine[r]):
            continue
        near = {int(res[b]) for a in mine[r] for b in np.asarray(aa.bonded_atoms(int(a))).ravel()}
        host = next((h for h in sorted(near) if h in moves), None)
        if host is None:
            left.append(r)
            continue
        R, t = moves[host]
        out[mine[r]] = pos[mine[r]] @ R.T + t
    return out, left


def _held_torsions(aa) -> list[tuple]:
    """What the soft stage must not turn over: each peptide bond (held trans, or cis
    where the structure has it so) and each alpha carbon's hand."""
    names = [str(x).strip() for x in aa.atoms["name"]]
    res = np.asarray(aa.atoms["residue"])
    at = {(int(res[a]), names[a]): a for a in aa.select("protein").ids}
    out = []
    for (r, n), c in at.items():
        if n != "C":
            continue
        ca = at.get((r, "CA"))
        nitrogen = at.get((r, "N"))
        cb = at.get((r, "CB"))
        if ca is not None and nitrogen is not None and cb is not None:
            out.append((nitrogen, ca, c, cb))
        for b in np.asarray(aa.bonded_atoms(int(c))).ravel():
            if names[b] == "N" and res[b] != r and (int(res[b]), "CA") in at and ca is not None:
                out.append((ca, c, int(b), at[(int(res[b]), "CA")]))
    return out


def _relax(aa, atoms: np.ndarray, targets: np.ndarray, restraint: float, forcefields, log) -> dict:
    """``aa`` minimized in place with ``atoms`` on springs (``restraint``
    kcal/mol/A^2) to ``targets``: soft nonbonded terms first (:mod:`boonza.rem`),
    since residues laid down one by one overlap, then the force field."""
    import openmm
    import openmm.unit as u

    from .. import rem
    from ..omm import to_openmm
    from ..viewer.edit import parameterize

    if not (np.asarray(aa.atoms["anum"]) == 1).any():
        raise BackmapError("the all-atom structure has no hydrogens: minimizing needs them "
                           "(prepare it, or pass minimize=False)")  # fmt: skip
    if forcefields is None and len(aa.select("nucleic").ids):
        from ..md.config import DEFAULT_FORCEFIELDS

        # boonza md's defaults have no nucleic acids, and a strand sent to GAFF2 as one
        # molecule fails there: bsc1 for the minimization, which is all it is used for
        forcefields = (*DEFAULT_FORCEFIELDS, (NUCLEIC_FORCEFIELD,))
    p = parameterize(aa, forcefields=forcefields, log=log or (lambda *a: None))
    order = np.asarray(p.atoms["view_index"], dtype=np.int64)
    if len(order) != aa.natoms or (order < 0).any():
        raise BackmapError("the force field changed the atoms of the all-atom structure")
    inv = np.empty(aa.natoms, np.int64)
    inv[order] = np.arange(aa.natoms)
    _, system, _ = to_openmm(p, nonbonded_method="CutoffNonPeriodic", cutoff=12.0,
                             constraints=False)  # fmt: skip
    hold = openmm.CustomExternalForce("0.5*k_bead*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    hold.addGlobalParameter("k_bead", restraint * 4.184 * 100)  # kcal/mol/A^2 -> kJ/mol/nm^2
    for name in ("x0", "y0", "z0"):
        hold.addPerParticleParameter(name)
    for a, xyz in zip(atoms, targets, strict=True):
        hold.addParticle(int(inv[a]), (np.asarray(xyz, float) / 10.0).tolist())
    hold.setName("beads")
    system.addForce(hold)
    start = np.array(aa.positions, float)[order] / 10.0
    torsions = [tuple(int(inv[a]) for a in t) for t in _held_torsions(aa)]
    soft, report = rem.relax(system, start, torsions=torsions)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
    context.setPositions(soft * u.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context, 10.0, 0)
    state = context.getState(getPositions=True, getEnergy=True)
    new = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer)) * 10.0
    pos = np.empty_like(new)
    pos[order] = new
    aa.positions = pos
    report["energy"] = round(state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole), 1)
    return report


def backmap(cg, reference=None, positions=None, *, box=None, minimize: bool = True,
            restraint: float = 50.0, forcefields=None, log=None):  # fmt: skip
    """The all-atom structure of a SIRAH frame.

    ``cg`` is the coarse-grained system (a run's ``sirah/cg.dms``); ``positions``
    its beads' positions in A for the frame wanted (default: its own).
    ``reference`` is the all-atom structure the run was built from: a copy of it is
    returned with every residue laid on its beads -- matched by chain, residue
    number and insertion code -- and, with ``minimize``, relaxed with the mapped
    atoms held to their beads by ``restraint`` kcal/mol/A^2 springs (it needs its
    hydrogens for that).  Residues of the reference with no beads go with the
    residue they are bonded to (ACE and NME caps, which SIRAH drops), and are left
    out otherwise (water, a ligand that was not coarse-grained).  Solvent beads are
    not turned back.

    Without ``reference``, amino acids are built from the bead names
    (:func:`from_beads`) and then treated the same way: default protonation states
    and no ligands, and only for proteins.

    ``box``: the frame's cell.  A trajectory is written wrapped, so with a box the
    solute's molecules are made whole and brought beside each other first (default:
    ``cg``'s own cell when the positions are its own; none for ``positions`` given).

    ``forcefields``: as :func:`boonza.viewer.edit.parameterize` takes them
    (default: boonza md's, with :data:`NUCLEIC_FORCEFIELD` added for a nucleic acid;
    GAFF2 for a ligand, which needs AmberTools).
    """
    say = log or (lambda *a: None)
    xyz = np.array(cg.positions if positions is None else positions, float).reshape(-1, 3)
    if len(xyz) != cg.natoms:
        raise BackmapError(f"{len(xyz)} positions for {cg.natoms} beads")
    if box is None and positions is None:
        box = cg.cell
    if box is not None and np.asarray(box).any():
        xyz = _whole(cg, xyz, np.asarray(box, float))
    if reference is None:
        aa = from_beads(cg, xyz, log=log)
    else:
        aa = reference.copy()
    beads, atoms = bead_atoms(cg, aa)
    if not len(beads):
        raise BackmapError("no residue of the all-atom structure matches a coarse-grained one by "
                           "chain, residue number and insertion code")  # fmt: skip
    res = np.asarray(aa.atoms["residue"])
    pairs_of = {}
    for r in np.unique(res[atoms]).tolist():
        mine = res[atoms] == r
        pairs_of[int(r)] = (atoms[mine], xyz[beads[mine]])
    placed, left = _place(aa, pairs_of, say)
    aa.positions = placed
    if left:
        names = [str(x).strip() for x in aa.residues["name"]]
        kinds: dict[str, int] = {}
        for r in left:
            kinds[names[r]] = kinds.get(names[r], 0) + 1
        say("left out, with no beads to place them on: "
            + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))  # fmt: skip
        index = aa.delete_residues(left)
        atoms = np.asarray(index)[atoms]
    used = set(beads.tolist())
    names = [str(x).strip() for x in cg.residues["name"]]
    cg_res = np.asarray(cg.atoms["residue"])
    stray: dict[str, int] = {}
    for b in range(cg.natoms):
        if b not in used and names[cg_res[b]] != WATER and names[cg_res[b]] not in IONS:
            stray[names[cg_res[b]]] = stray.get(names[cg_res[b]], 0) + 1
    if stray:
        say("beads with no atom in the all-atom structure (not turned back): "
            + ", ".join(f"{n} of {k}" for k, n in sorted(stray.items())))  # fmt: skip
    if minimize:
        report = _relax(aa, atoms, xyz[beads], restraint, forcefields, log)
        say(f"minimized: soft E {report['rem_energy_before']} -> {report['rem_energy_after']}, "
            f"then the force field's {report['energy']} kJ/mol")  # fmt: skip
    off = np.linalg.norm(np.asarray(aa.positions)[atoms] - xyz[beads], axis=1)
    say(f"{len(beads)} beads on their atoms: RMSD {np.sqrt((off**2).mean()):.2f} A, "
        f"largest {off.max():.2f} A")  # fmt: skip
    return aa


def from_beads(cg, positions=None, log=None):
    """Amino acids built from bead names alone, each an ideal residue laid on its
    beads: hydrogens and charged termini from the Amber templates, protonation
    states SIRAH's own (:data:`BUILT_AS`: its neutral epsilon histidine is HIE), a
    disulfide wherever two sulfur beads are bonded.  Nothing is minimized here.
    Only proteins: a nucleotide or a cofactor needs the structure it came from."""
    from .. import peptide
    from ..system import System
    from ..viewer.edit import prepare

    say = log or (lambda *a: None)
    xyz = np.array(cg.positions if positions is None else positions, float).reshape(-1, 3)
    maps = read_map()
    by_cg: dict[str, list[str]] = {}
    for name, entry in maps.items():
        by_cg.setdefault(entry.residue, []).append(name)
    cg_names = [str(x).strip() for x in cg.atoms["name"]]
    cg_resname = [str(x).strip() for x in cg.residues["name"]]
    keys = _keys(cg)
    ideal: dict[str, tuple] = {}

    def template(resname: str):
        """Heavy atoms of an ideal residue: names, elements, positions, bonds."""
        if resname not in ideal:
            t = peptide("G" + ONE_LETTER[resname] + "G", "extended")
            r = np.asarray(t.atoms["residue"])
            heavy = np.flatnonzero((r == 1) & (np.asarray(t.atoms["anum"]) > 1))
            where = {int(a): k for k, a in enumerate(heavy)}
            bonds = [(where[int(i)], where[int(j)])
                     for i, j in zip(t.bonds["i"], t.bonds["j"], strict=True)
                     if int(i) in where and int(j) in where]  # fmt: skip
            ideal[resname] = ([str(t.atoms["name"][a]).strip() for a in heavy],
                              np.asarray(t.atoms["anum"])[heavy],
                              np.asarray(t.positions, float)[heavy], bonds)  # fmt: skip
        return ideal[resname]

    names, anum, pos, resnames, resids, chains, insertions, bonds = [], [], [], [], [], [], [], []
    skipped: dict[str, int] = {}
    backbone: dict[int, dict[str, int]] = {}  # cg residue -> its N and C in the new system
    sulfur: dict[int, int] = {}  # sulfur bead -> SG
    for r, beads in enumerate(_residue_atoms(cg)):
        if cg_resname[r] == WATER or cg_resname[r] in IONS:
            continue
        resname = BUILT_AS.get(cg_resname[r]) or next(
            (n for n in by_cg.get(cg_resname[r], ()) if n in ONE_LETTER), None
        )
        if resname is None:
            skipped[cg_resname[r]] = skipped.get(cg_resname[r], 0) + 1
            continue
        t_names, t_anum, t_pos, t_bonds = template(resname)
        there = {cg_names[b]: int(b) for b in beads}
        mobile, target, on = [], [], {}
        for bead, sources in maps[resname].beads:
            k = next((t_names.index(n) for n in sources if n in t_names), None)
            if k is not None and bead in there:
                mobile.append(t_pos[k])
                target.append(xyz[there[bead]])
                on[there[bead]] = k
        if len(mobile) < 3:
            raise BackmapError(f"{cg_resname[r]}{keys[r][1]} has {len(mobile)} beads to build "
                               "a residue on; three are needed")  # fmt: skip
        R, t = kabsch(np.array(mobile), np.array(target))
        first = len(names)
        names += t_names
        anum += t_anum.tolist()
        pos += (t_pos @ R.T + t).tolist()
        resnames += [resname] * len(t_names)
        resids += [keys[r][1]] * len(t_names)
        chains += [keys[r][0]] * len(t_names)
        insertions += [keys[r][2]] * len(t_names)
        bonds += [(first + i, first + j) for i, j in t_bonds]
        backbone[r] = {n: first + t_names.index(n) for n in ("N", "C")}
        for bead, k in on.items():
            if cg_names[bead] == "BSG":
                sulfur[bead] = first + k
    if not names:
        raise BackmapError("no amino acid among the beads: nucleic acids and cofactors need the "
                           "all-atom structure the run was built from (reference)")  # fmt: skip
    if skipped:
        say("not built (no reference to take them from): "
            + ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())))  # fmt: skip
    cg_res = np.asarray(cg.atoms["residue"])
    for i, j in zip(cg.bonds["i"], cg.bonds["j"], strict=True):
        a, b = sorted((int(i), int(j)))
        ra, rb = int(cg_res[a]), int(cg_res[b])
        if ra == rb or ra not in backbone or rb not in backbone:
            continue
        pair = {cg_names[a], cg_names[b]}
        if pair == {"GO", "GN"}:  # the peptide bond, as SIRAH's topology has it: GO to +GN
            c_res, n_res = (ra, rb) if cg_names[a] == "GO" else (rb, ra)
            bonds.append((backbone[c_res]["C"], backbone[n_res]["N"]))
        elif pair == {"BSG"} and a in sulfur and b in sulfur:
            bonds.append((sulfur[a], sulfur[b]))
    s = System.from_arrays(np.array(pos), names=names, anum=np.array(anum), resnames=resnames,
                           resids=np.array(resids), chains=chains, insertions=insertions,
                           bonds=np.array(bonds, np.int64))  # fmt: skip
    report = prepare(s)
    say(f"built {s.nresidues} residues from their beads, {report['hydrogens']} hydrogens added")
    return s
