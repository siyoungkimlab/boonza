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


#: a probe residue's backbone, what ``probe_side_chains`` leaves out.  CA is here with
#: the rest: the side chain keeps it as the hydrogen that closes CB's valence, not as a
#: carbon
PROBE_BACKBONE = {"N", "H", "H1", "H2", "H3", "CA", "HA", "HA2", "HA3", "C", "O", "OXT",
                  "OC1", "OC2"}  # fmt: skip
#: the caps a probe is built with (:func:`from_beads`), dropped whole with its backbone
PROBE_CAPS = {"ACE", "NME", "NMA"}
#: the C-H a cap hydrogen is given, A
CAP_H = 1.09
#: the bond a hydrogen is given to the element it hangs off, A
BOND_H = {6: 1.09, 7: 1.01, 8: 0.97, 16: 1.34}

#: what holds a peptide bond trans and an alpha carbon's hand while minimizing,
#: kJ/mol/rad^2: a turned-over bond costs thousands of kJ/mol, against the hundred or
#: so a bead's spring (50 kcal/mol/A^2) gains by pulling its atom an angstrom
HOLD_KJ = 1000.0
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


def _molecules_of(aa, residues) -> list[list[int]]:
    """``residues`` grouped into the molecules they are bonded into, each sorted."""
    from ..graph import connected_components

    mine = {int(r) for r in residues}
    order = sorted(mine)
    at = {r: k for k, r in enumerate(order)}
    res = np.asarray(aa.atoms["residue"])
    bi, bj = [], []
    for b in aa.bonds:
        u, v = int(res[b.first.id]), int(res[b.second.id])
        if u != v and u in mine and v in mine:
            bi.append(at[u])
            bj.append(at[v])
    label, _ = connected_components(len(order), np.asarray(bi, int), np.asarray(bj, int))
    out: dict[int, list[int]] = {}
    for r, g in zip(order, np.asarray(label).tolist(), strict=True):
        out.setdefault(int(g), []).append(r)
    return [sorted(v) for _, v in sorted(out.items())]


def _on_a_ring(aa, atoms) -> set[int]:
    """Those of ``atoms`` that lie on a ring made of bonds within ``atoms``.

    A bond is on a ring when its ends are still connected without it, so this is the
    bond-removal test, run over the handful of bonds one residue has.
    """
    mine = {int(a) for a in atoms}
    edges = [(b.first.id, b.second.id) for b in aa.bonds
             if b.first.id in mine and b.second.id in mine]  # fmt: skip
    if not edges:
        return set()
    near: dict[int, list[int]] = {a: [] for a in mine}
    for u, v in edges:
        near[u].append(v)
        near[v].append(u)
    out: set[int] = set()
    for u, v in edges:
        if u in out and v in out:
            continue
        seen, stack = {u}, [u]  # u to v without this bond
        found = False
        while stack:
            x = stack.pop()
            for y in near[x]:
                if (x, y) in ((u, v), (v, u)) and y not in seen:
                    continue  # the bond itself, once
                if y == v:
                    found = True
                    stack = []
                    break
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if found:
            out |= {u, v}
    return out


def _names_in(aa, residue: int) -> set:
    """The atom names already used in ``residue``: a new hydrogen must not repeat one."""
    res = np.asarray(aa.atoms["residue"])
    return {str(aa.atoms["name"][int(a)]).strip() for a in np.flatnonzero(res == residue)}


def _add_hydrogen(aa, pos, host: int, towards, taken: set):
    """A hydrogen on ``host``, in ``host``'s own residue, :data:`BOND_H` away in the
    direction of the point ``towards``.  Returns ``pos`` with its position appended.

    A new atom rather than a retyped one, so it is bonded to its host and to nothing
    else: two of them on atoms that used to be bonded (proline's N and CA) would
    otherwise be left bonded to each other.
    """
    anum = int(np.asarray(aa.atoms["anum"])[host])
    u = np.asarray(towards, float) - pos[host]
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else _any_normal(np.array([0.0, 0.0, 1.0]))
    name = _free_name("H" + str(aa.atoms["name"][host]).strip(), taken)
    taken.add(name)
    h = aa.add_atom(residue=int(np.asarray(aa.atoms["residue"])[host]), name=name, anum=1,
                    formal_charge=0)  # fmt: skip
    aa.add_bond(host, h.id)
    return np.vstack([pos, pos[host] + u * BOND_H.get(anum, CAP_H)])


def _any_normal(u) -> np.ndarray:
    v = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = v - np.dot(v, u) * u
    return v / np.linalg.norm(v)


def _free_name(wanted: str, taken: set) -> str:
    if wanted not in taken:
        return wanted
    return next(f"{wanted}{k}" for k in range(1, 100) if f"{wanted}{k}" not in taken)


def _side_chains_only(aa, residues, log) -> None:
    """``aa``'s single-amino-acid probes among ``residues`` cut down to their side
    chains, in place.

    A dipeptide probe is left exactly as it is: its two side chains belong to one
    molecule, and cutting it would make two that no longer record having been tethered.

    For a lone amino acid the caps go and the backbone goes atom by atom, except that a
    backbone atom bonded to a side-chain atom is replaced by a hydrogen on that atom,
    where it stood, so the fragment closes its own valence instead of leaving a
    radical: CA becomes the hydrogen on CB.  A backbone atom on a ring with the side
    chain stays as itself, so proline keeps N and CA and comes out pyrrolidine rather
    than propane.

    Only the cut is neutral.  The side chain keeps the charge it came with, as a
    probe's own ends are neutral and its side chains are what carry charge
    (:mod:`boonza.sirah.probes`): arginine and lysine come out +1, aspartate and
    glutamate -1.

    What is left is no amino acid, so the protein force field has no template for it
    and :func:`boonza.parameterize` sends it to GAFF2.
    """
    say = log or (lambda *a: None)
    pos = np.array(aa.positions, float)
    names = [str(x).strip() for x in aa.atoms["name"]]
    resname = [str(x).strip() for x in aa.residues["name"]]
    members = _residue_atoms(aa)
    drop: list[int] = []
    kinds: dict[str, int] = {}
    whole: dict[str, int] = {}
    for group in _molecules_of(aa, residues):
        acids = [r for r in group if resname[r] not in PROBE_CAPS]
        if len(acids) != 1:  # a dipeptide probe, or whatever else: left as it is
            kind = "-".join(resname[r] for r in acids) or "no amino acid"
            whole[kind] = whole.get(kind, 0) + 1
            continue
        r = acids[0]
        mine = [int(a) for a in members[r]]
        side = {a for a in mine if names[a] not in PROBE_BACKBONE}
        if not side:  # glycine, alanine: nothing but backbone
            drop.extend(int(a) for a in (x for g in group for x in members[g]))
            bare = f"{resname[r]} (no side chain)"
            kinds[bare] = kinds.get(bare, 0) + 1
            continue
        # proline: its ring runs through N and CA, so they are side chain too
        ring = _on_a_ring(aa, mine)
        keep = set(side)
        keep |= {a for a in ring if np.asarray(aa.atoms["anum"])[a] > 1 and ring & side}
        keep |= {int(b) for a in list(keep) for b in np.asarray(aa.bonded_atoms(a)).ravel()
                 if np.asarray(aa.atoms["anum"])[int(b)] == 1 and int(b) in mine}  # fmt: skip
        taken = _names_in(aa, r)
        caps = []
        for a in (x for g in group for x in members[g]):
            a = int(a)
            if a in keep:
                continue
            hosts = [int(b) for b in np.asarray(aa.bonded_atoms(a)).ravel() if int(b) in keep]
            if hosts:  # it closes a valence: a hydrogen takes its place, where it stood
                caps.append((hosts[0], np.array(pos[a])))
            drop.append(a)
        for h, where in caps:
            pos = _add_hydrogen(aa, pos, h, where, taken)
        kinds[resname[r]] = kinds.get(resname[r], 0) + 1
    aa.positions = pos
    if kinds:
        say("probe side chains kept, their backbones and caps dropped: "
            + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))  # fmt: skip
    if whole:
        say("probes left whole, being no lone amino acid: "
            + ", ".join(f"{n} {k}" for k, n in sorted(whole.items())))  # fmt: skip
    aa.delete_atoms(sorted(set(drop)))


def _dihedral(pos, a, b, c, d) -> float:
    """The dihedral in degrees, in the convention the held torsions are given in
    (:func:`boonza.rem.hold_torsions`, which is OpenMM's)."""
    from ..rem import _dihedral as radians

    return float(np.degrees(radians(np.asarray(pos, float), a, b, c, d)))


def _held_torsions(aa, was=None) -> list[tuple]:
    """What the minimization must not turn over, as ``(a, b, c, d, degrees)``: each
    alpha carbon's hand (N-CA-C-CB, at the angle it has: residues are laid down
    whole, so it is the reference's) and each peptide bond, trans -- or cis where
    ``was``, the positions the structure came with, has it so (a cis proline)."""
    names = [str(x).strip() for x in aa.atoms["name"]]
    res = np.asarray(aa.atoms["residue"])
    pos = np.asarray(aa.positions, float)
    at = {(int(res[a]), names[a]): int(a) for a in aa.select("protein").ids}
    out = []
    for (r, n), c in at.items():
        if n != "C":
            continue
        ca = at.get((r, "CA"))
        nitrogen = at.get((r, "N"))
        cb = at.get((r, "CB"))
        if ca is not None and nitrogen is not None and cb is not None:
            out.append((nitrogen, ca, c, cb, _dihedral(pos, nitrogen, ca, c, cb)))
        for b in np.asarray(aa.bonded_atoms(int(c))).ravel():
            nxt = at.get((int(res[b]), "CA"))
            if names[b] == "N" and res[b] != r and nxt is not None and ca is not None:
                cis = was is not None and abs(_dihedral(was, ca, c, int(b), nxt)) < 90
                out.append((ca, c, int(b), nxt, 0.0 if cis else 180.0))
    return out


def _relax(aa, atoms: np.ndarray, targets: np.ndarray, restraint: float, forcefields, log,
           was=None) -> dict:  # fmt: skip
    """``aa`` minimized in place with ``atoms`` on springs (``restraint``
    kcal/mol/A^2) to ``targets``: soft nonbonded terms first (:mod:`boonza.rem`),
    since residues laid down one by one overlap, then the force field.  Peptide
    bonds and the alpha carbons' hands (:func:`_held_torsions`) are held through
    both: a coarse-grained frame has neither, and beads that sit as a cis bond or
    a D residue would otherwise pull the atoms into one."""
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
    torsions = [(*(int(inv[a]) for a in t[:4]), t[4]) for t in _held_torsions(aa, was)]
    system.addForce(rem.hold_torsions(torsions, start, HOLD_KJ))  # in both stages
    soft, report = rem.relax(system, start)
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
            restraint: float = 50.0, probe_side_chains: bool = False, forcefields=None,
            log=None):  # fmt: skip
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

    ``probe_side_chains``: keep a swim probe's side chains alone.  The whole probe is
    laid on its beads first, so the side chain is placed as the dipeptide's, and only
    then are the caps and the backbone dropped, CA staying as the hydrogen that closes
    CB's valence (:func:`_side_chains_only`).  What is left is a small molecule, not an
    amino acid, so the minimization parameterizes it with GAFF2 (AmberTools): the
    springs that follow hold it by its side-chain beads, the backbone beads having no
    atoms left.  It applies to probes alone -- what ``reference`` does not have -- and
    a probe with no side chain to keep (glycine, alanine in SIRAH) is dropped.

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
        aa, was = from_beads(cg, xyz, log=log), None
    else:
        aa = reference.copy()
        was = np.array(aa.positions, float)  # where a peptide bond is cis, it stays so
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
        index = np.asarray(aa.delete_residues(left))
        atoms = index[atoms]
        was = was[index >= 0]
    used = set(beads.tolist())
    names = [str(x).strip() for x in cg.residues["name"]]
    cg_res = np.asarray(cg.atoms["residue"])
    whole_residues = {int(r) for r in cg_res[beads]}
    extra = sorted({int(cg_res[b]) for b in range(cg.natoms) if b not in used
                    and int(cg_res[b]) not in whole_residues
                    and names[cg_res[b]] != WATER and names[cg_res[b]] not in IONS})  # fmt: skip
    stripped = False
    if extra and reference is not None:  # what the reference has not: a swim's probes
        try:
            built = from_beads(cg, xyz, residues=extra)
        except BackmapError:
            built = None
        if built is not None:
            kinds: dict[str, int] = {}
            for n in built.residues["name"]:
                kinds[str(n).strip()] = kinds.get(str(n).strip(), 0) + 1
            say("not in the all-atom structure, built from their beads: "
                + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))  # fmt: skip
            first, mine = aa.nresidues, aa.natoms
            aa.append(built)
            probes = list(range(first, aa.nresidues))
            if probe_side_chains:
                # this adds atoms as well as dropping them, so `was` is built after
                _side_chains_only(aa, probes, log)
                stripped = True
            # no reference for these: their peptide bonds are held trans
            was = np.vstack([was[:mine], np.full((aa.natoms - mine, 3), np.nan)])
            beads, atoms = bead_atoms(cg, aa)
            used = set(beads.tolist())
    if probe_side_chains and not stripped:
        say("probe_side_chains: no probes here to cut down -- it is for the beads the "
            "all-atom structure does not have, and every bead matched one")  # fmt: skip
    stray: dict[str, int] = {}
    for b in range(cg.natoms):
        if b not in used and names[cg_res[b]] != WATER and names[cg_res[b]] not in IONS:
            stray[names[cg_res[b]]] = stray.get(names[cg_res[b]], 0) + 1
    if stray:
        say("beads with no atom in the all-atom structure (not turned back): "
            + ", ".join(f"{n} of {k}" for k, n in sorted(stray.items())))  # fmt: skip
    if minimize:
        report = _relax(aa, atoms, xyz[beads], restraint, forcefields, log, was)
        say(f"minimized: soft E {report['rem_energy_before']} -> {report['rem_energy_after']}, "
            f"then the force field's {report['energy']} kJ/mol")  # fmt: skip
    off = np.linalg.norm(np.asarray(aa.positions)[atoms] - xyz[beads], axis=1)
    say(f"{len(beads)} beads on their atoms: RMSD {np.sqrt((off**2).mean()):.2f} A, "
        f"largest {off.max():.2f} A, {int((off > 1.0).sum())} over 1 A")  # fmt: skip
    return aa


_TABLE: dict | None = None


def read_table() -> dict:
    """SIRAH Tools' backmapping table, ``{coarse-grained residue: entry}``: the
    all-atom residue it stands for (``name``), the atoms that sit on beads and the
    ones placed from three others by a distance, an angle and a dihedral
    (``build``), what changes at a chain's ends (``first``, ``last``) and the atoms
    written (``print``).  It is SIRAH Tools' own (``sirah_vmdtk.tcl``, GPL v2),
    read from its Tcl by ``scripts/gen_sirah_backmap.py``."""
    global _TABLE
    if _TABLE is None:
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "data" / "sirah" / "backmap.json"
        _TABLE = json.loads(path.read_text(encoding="utf-8"))["residues"]
    return _TABLE


def _add_atom(at3, at2, at1, r: float, angle: float, dihedral: float) -> np.ndarray:
    """The point ``r`` from ``at3`` at ``angle`` (at4-at3-at2) and ``dihedral``
    (at4-at3-at2-at1), degrees, in the convention of SIRAH Tools' ``addatom``."""
    at3, at2, at1 = (np.asarray(x, float) for x in (at3, at2, at1))
    v12, v32 = at1 - at2, at3 - at2
    x = np.cross(v32, v12)
    z = np.cross(x, v32)
    r23 = np.linalg.norm(v32)
    ex, ey, ez = x / np.linalg.norm(x), v32 / r23, z / np.linalg.norm(z)
    d, a = np.radians(dihedral), np.radians(180.0 - angle)
    return (at2 + ex * (r * np.sin(a) * np.sin(d)) + ey * (r * np.cos(a) + r23)
            + ez * (r * np.sin(a) * np.cos(d)))  # fmt: skip


def _chain_neighbors(cg) -> tuple[dict[int, int], dict[int, int]]:
    """The residue before and the one after each residue along its chain, read from
    the bonds between residues (a disulfide, sulfur bead to sulfur bead, is not one)."""
    names = [str(x).strip() for x in cg.atoms["name"]]
    res = np.asarray(cg.atoms["residue"])
    chain = np.asarray(cg.residues["chain"])
    before: dict[int, int] = {}
    after: dict[int, int] = {}
    for i, j in zip(cg.bonds["i"], cg.bonds["j"], strict=True):
        a, b = sorted((int(res[i]), int(res[j])))
        if a == b or chain[a] != chain[b] or names[int(i)] == names[int(j)] == "BSG":
            continue
        if b == a + 1:
            after[a], before[b] = b, a
    return before, after


def _from_table(table, resname: str, here: dict, prev: dict | None, nxt: dict | None):
    """One residue's atoms by SIRAH's instructions: ``(all-atom residue name,
    [(atom, position)])``, or None when the table has none for it.  ``here``,
    ``prev`` and ``nxt`` are ``{bead name: position}`` of the residue and its chain
    neighbors (None at a chain's end)."""
    entry = table.get(resname, {})
    if prev is None and "apply" in entry.get("first", {}):
        resname = entry["first"]["apply"]
        entry = table.get(resname, {})
    if nxt is None and "apply" in entry.get("last", {}):
        resname = entry["last"]["apply"]
        entry = table.get(resname, {})
    if "build" not in entry:
        return None
    build = dict(entry["build"])
    if prev is None:
        build.update(entry.get("first", {}).get("build", {}))
    if nxt is None:
        build.update(entry.get("last", {}).get("build", {}))
    new: dict[str, np.ndarray] = {}
    for atom, how in build.items():
        if atom in new:
            continue
        if how[0] == ":":
            for bead in how[1:]:
                where = prev if bead.endswith("/-") else nxt if bead.endswith("/+") else here
                bead = bead.split("/")[0]
                if where is not None and bead in where:
                    new[atom] = np.asarray(where[bead], float)
                    break
            else:
                raise BackmapError(f"{resname}: no bead to put {atom} on ({' '.join(how[1:])})")
        else:
            a, b, c = how[1]
            if not {a, b, c} <= set(new):
                raise BackmapError(f"{resname}: {atom} is built from {a}, {b}, {c}, not all set")
            new[atom] = _add_atom(new[a], new[b], new[c], *how[2])
    return entry["name"], [(atom, new[atom]) for atom in entry["print"]]


def _template_hydrogens(s, residues, forcefield: str) -> int:
    """The residues' own bonds and hydrogens as ``forcefield``'s templates have them
    (nucleotides: :func:`boonza.viewer.edit.prepare` does this for amino acids
    only), a phosphate's OP2 charged.  Returns how many hydrogens were added."""
    from .. import viparr
    from ..elements import MSYS_MASSES
    from ..viewer.edit import H_BOND, hydrogen_directions

    ff = viparr.load_forcefield(forcefield)
    known = {t.name for t in ff.templates}
    added = 0
    charges = {}
    for r in residues:
        name = str(s.residues["name"][r]).strip()
        if name not in known:
            raise BackmapError(f"{forcefield} has no template for {name}")
        tp = ff.template(name)
        here = {str(s.atoms["name"][a]).strip(): int(a) for a in s.residue_atoms(r)}
        for i, j in tp.bonds:
            ni, nj = tp.names[i], tp.names[j]
            if ni in here and nj in here and s.find_bond(here[ni], here[nj]) is None:
                s.add_bond(here[ni], here[nj], 1)
        if "OP2" in here:
            charges[here["OP2"]] = -1
        for k, heavy in enumerate(tp.names):
            if tp.anum[k] <= 1 or heavy not in here:
                continue
            a = here[heavy]
            hs = []
            for x in tp._nbrs[k]:
                if tp.anum[x] != 1:
                    continue
                # a hydrogen a bead sat on is there already (adenine's H2, under SIRAH's C2A)
                if tp.names[x] in here:
                    if s.find_bond(a, here[tp.names[x]]) is None:
                        s.add_bond(a, here[tp.names[x]], 1)
                else:
                    hs.append(tp.names[x])
            if not hs:
                continue
            shape = "trigonal" if tp.anum[k] in (6, 7) and len(tp._nbrs[k]) == 3 else "tetrahedral"
            pos = np.asarray(s.positions, float)
            dirs = hydrogen_directions(s, a, len(hs), shape=shape)
            new = s.add_atoms(len(hs), residue=r, name=np.array(hs), anum=np.ones(len(hs), int),
                              mass=np.full(len(hs), MSYS_MASSES[1]))  # fmt: skip
            xyz = np.array(s.positions, float)
            for h, d in zip(new.ids.tolist(), dirs, strict=False):
                xyz[h] = pos[a] + H_BOND.get(int(tp.anum[k]), 1.0) * d
            s.positions = xyz
            for h in new.ids.tolist():
                s.add_bond(a, h, 1)
            added += len(hs)
    q = np.asarray(s.atoms["formal_charge"]).copy()
    for a, c in charges.items():
        q[a] = c
    s.atoms["formal_charge"] = q
    return added


def _probe_names(cg, before: dict, after: dict) -> dict[int, str]:
    """The coarse-grained residue name of each residue of a swim probe.  A probe's
    beads carry its sequence as their residue name (``Y``, or ``EK`` for both
    residues of a dipeptide: :mod:`boonza.sirah.probes`); the residue's place along
    the molecule says which letter is its own."""
    maps = read_map()
    three = {}
    for name, letter in ONE_LETTER.items():
        three.setdefault(letter, name)
    three.update({"H": "HIE", "C": "CYS", "D": "ASP", "E": "GLU", "K": "LYS"})
    out = {}
    for r in range(cg.nresidues):
        name = str(cg.residues["name"][r]).strip()
        if not name.isalpha() or not name.isupper() or any(c not in three for c in name):
            continue
        first, place = r, 0
        while first in before:
            first, place = before[first], place + 1
        length, last = place + 1, r
        while last in after:
            last, length = after[last], length + 1
        if length == len(name) and three[name[place]] in maps:
            out[r] = maps[three[name[place]]].residue
    return out


def from_beads(cg, positions=None, log=None, rules: str = "sirah", residues=None):
    """The all-atom structure built from bead names alone, hydrogens included.
    Nothing is minimized here.

    ``rules="sirah"`` follows SIRAH Tools' backmapping table (:func:`read_table`):
    the atoms that sit on beads, and the others placed from them by a distance, an
    angle and a dihedral.  Amino acids and DNA; an amino acid the table lacks is
    built the other way.  ``rules="ideal"`` lays an ideal residue
    (:func:`boonza.peptide`'s) on its beads as a rigid body: amino acids only.

    Either way the hydrogens and charged termini are the Amber templates', the
    protonation states SIRAH's own (:data:`BUILT_AS`: its neutral epsilon histidine
    is HIE), and two bonded sulfur beads are a disulfide.  What the table has no
    way to finish here (lipids, glycans, modified residues) and cofactors are
    reported and left out: they need the structure the run was built from.

    A swim probe, whose beads are named by its sequence, is built as the amino
    acids it is, with an ACE and an NME cap: its ends are neutral beads, and a lone
    amino acid has no Amber template without them.  ``residues``: only these
    residues of ``cg``."""
    from .. import peptide
    from ..system import System
    from ..viewer.edit import prepare

    if rules not in ("sirah", "ideal"):
        raise BackmapError(f"rules is 'sirah' or 'ideal', not {rules!r}")
    say = log or (lambda *a: None)
    xyz = np.array(cg.positions if positions is None else positions, float).reshape(-1, 3)
    maps = read_map()
    table = read_table() if rules == "sirah" else {}
    by_cg: dict[str, list[str]] = {}
    for name, entry in maps.items():
        by_cg.setdefault(entry.residue, []).append(name)
    cg_names = [str(x).strip() for x in cg.atoms["name"]]
    cg_resname = [str(x).strip() for x in cg.residues["name"]]
    keys = _keys(cg)
    members = _residue_atoms(cg)
    before, after = _chain_neighbors(cg)
    probes = set()
    for r, name in _probe_names(cg, before, after).items():
        if cg_resname[r] not in table and cg_resname[r] not in by_cg:
            cg_resname[r] = name
            probes.add(r)
    wanted = None if residues is None else {int(r) for r in residues}
    ideal: dict[str, tuple] = {}
    nucleic = _nucleic_templates()

    def template(resname: str):
        """Heavy atoms of an ideal residue: names, elements, positions."""
        if resname not in ideal:
            t = peptide("G" + ONE_LETTER[resname] + "G", "extended")
            r = np.asarray(t.atoms["residue"])
            heavy = np.flatnonzero((r == 1) & (np.asarray(t.atoms["anum"]) > 1))
            ideal[resname] = ([str(t.atoms["name"][a]).strip() for a in heavy],
                              np.asarray(t.positions, float)[heavy])  # fmt: skip
        return ideal[resname]

    def beads_of(r):
        return None if r is None else {cg_names[b]: xyz[b] for b in members[r]}

    names, pos, resnames, resids, chains, insertions, segids = [], [], [], [], [], [], []
    skipped: dict[str, int] = {}
    made: dict[int, tuple[int, str]] = {}  # cg residue -> its first atom here, its kind
    for r in range(cg.nresidues):
        if cg_resname[r] == WATER or cg_resname[r] in IONS:
            continue
        if wanted is not None and r not in wanted:
            continue
        built = None
        try:
            built = _from_table(table, cg_resname[r], beads_of(r), beads_of(before.get(r)),
                                beads_of(after.get(r)))  # fmt: skip
        except BackmapError:
            if rules == "sirah" and cg_resname[r] not in BUILT_AS and not by_cg.get(cg_resname[r]):
                raise
        kind = None
        if built is not None:
            kind = (
                "protein" if built[0] in ONE_LETTER else "nucleic" if built[0] in nucleic else None
            )
        if kind is None:  # an ideal amino acid on its beads
            resname = BUILT_AS.get(cg_resname[r]) or next(
                (n for n in by_cg.get(cg_resname[r], ()) if n in ONE_LETTER), None)  # fmt: skip
            if resname is None:
                skipped[cg_resname[r]] = skipped.get(cg_resname[r], 0) + 1
                continue
            t_names, t_pos = template(resname)
            there = beads_of(r)
            mobile, target = [], []
            for bead, sources in maps[resname].beads:
                k = next((t_names.index(n) for n in sources if n in t_names), None)
                if k is not None and bead in there:
                    mobile.append(t_pos[k])
                    target.append(there[bead])
            if len(mobile) < 3:
                raise BackmapError(f"{cg_resname[r]}{keys[r][1]} has {len(mobile)} beads to "
                                   "build a residue on; three are needed")  # fmt: skip
            R, t = kabsch(np.array(mobile), np.array(target))
            built, kind = (resname, list(zip(t_names, t_pos @ R.T + t, strict=True))), "protein"
        resname, atoms = built
        resname = BUILT_AS.get(cg_resname[r], resname) if kind == "protein" else resname
        made[r] = (len(names), kind)
        for atom, where in atoms:
            names.append(atom)
            pos.append(np.asarray(where, float))
        resnames += [resname] * len(atoms)
        resids += [keys[r][1]] * len(atoms)
        chains += [keys[r][0]] * len(atoms)
        insertions += [keys[r][2]] * len(atoms)
        first = r  # a probe is a chain of its own, to be capped: its first residue names it
        while first in before:
            first = before[first]
        segids += [f"p{first}" if r in probes else ""] * len(atoms)
    if not names:
        raise BackmapError("nothing among the beads that can be built from its name alone: "
                           "what else there is needs the all-atom structure the run was built "
                           "from (reference)")  # fmt: skip
    if skipped:
        say("not built (no reference to take them from): "
            + ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())))  # fmt: skip
    s = System.from_arrays(np.array(pos), names=names, resnames=resnames, segids=segids,
                           resids=np.array(resids), chains=chains,
                           insertions=insertions)  # fmt: skip
    s.atoms["mass"] = np.array([_mass(z) for z in s.atoms["anum"]])
    atom_res = np.asarray(s.atoms["residue"])
    new_res = {r: int(atom_res[first]) for r, (first, _) in made.items()}

    def atom(r, name):
        found = [int(a) for a in s.residue_atoms(new_res[r]) if str(s.atoms["name"][a]) == name]
        return found[0] if found else None

    cg_res = np.asarray(cg.atoms["residue"])
    for i, j in zip(cg.bonds["i"], cg.bonds["j"], strict=True):
        ra, rb = int(cg_res[i]), int(cg_res[j])
        if ra in made and rb in made and cg_names[int(i)] == cg_names[int(j)] == "BSG" and ra != rb:
            sa, sb = atom(ra, "SG"), atom(rb, "SG")
            if sa is not None and sb is not None and s.find_bond(sa, sb) is None:
                s.add_bond(sa, sb, 1)
    for r, nxt in after.items():  # along each chain: the peptide bond, the phosphodiester
        if r not in made or nxt not in made or made[r][1] != made[nxt][1]:
            continue
        pair = ("C", "N") if made[r][1] == "protein" else ("O3'", "P")
        a, b = atom(r, pair[0]), atom(nxt, pair[1])
        if a is not None and b is not None:
            s.add_bond(a, b, 1)
    hydrogens = 0
    strands = [new_res[r] for r, (_, kind) in made.items() if kind == "nucleic"]
    if strands:  # before prepare, which leaves alone what has its hydrogens
        hydrogens += _template_hydrogens(s, strands, NUCLEIC_FORCEFIELD)
        s.reorder_atoms(np.argsort(np.asarray(s.atoms["residue"]), kind="stable"))
    if probes:
        from ..viewer.edit import add_hydrogens, cap

        capped = sorted({int(s.residues["chain"][new_res[r]]) for r in probes if r in made})
        for c in capped:
            mine = [r for r in range(s.nresidues) if int(s.residues["chain"][r]) == c]

            def named(r, name):
                return next(int(a) for a in s.residue_atoms(r) if str(s.atoms["name"][a]) == name)

            # what a cap replaces: a hydrogen on the first N, an OXT on the last C
            add_hydrogens(s, named(mine[0], "N"), 1)
            xyz = np.asarray(s.positions, float)
            c_, o, ca = (named(mine[-1], n) for n in ("C", "O", "CA"))
            out = -((xyz[o] - xyz[c_]) / np.linalg.norm(xyz[o] - xyz[c_])
                    + (xyz[ca] - xyz[c_]) / np.linalg.norm(xyz[ca] - xyz[c_]))  # fmt: skip
            oxt = s.add_atom(mine[-1], name="OXT", anum=8, mass=_mass(8),
                             pos=xyz[c_] + 1.25 * out / np.linalg.norm(out))  # fmt: skip
            s.add_bond(c_, int(getattr(oxt, "id", oxt)), 1)
            cap(s, c, "N")
            cap(s, c, "C")
        s.chains["segid"] = np.array([""] * s.nchains)
    if any(kind == "protein" for _, kind in made.values()):
        hydrogens += prepare(s)["hydrogens"]
    # each chain's residues in the order of their numbers: a cap is added at the end
    res = np.asarray(s.atoms["residue"])
    chain_of = np.asarray(s.residues["chain"])[res]
    number = np.asarray(s.residues["resid"])[res]
    s.reorder_atoms(np.lexsort((np.arange(s.natoms), res, number, chain_of)))
    say(f"built {s.nresidues} residues from their beads, {hydrogens} hydrogens added")
    return s


def _mass(anum: int) -> float:
    from ..elements import MSYS_MASSES

    return float(MSYS_MASSES[int(anum)])


def _nucleic_templates() -> set[str]:
    from .. import viparr

    return {t.name for t in viparr.load_forcefield(NUCLEIC_FORCEFIELD).templates}
