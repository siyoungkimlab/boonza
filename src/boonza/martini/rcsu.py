"""The contact map a Go model is built from, computed from the structure.

    contacts = boonza.martini.rcsu.contact_map(system)
    boonza.martinize(system, go=True)        # the same map, built in

A residue pair is a contact when the two **overlap** (OV) -- their atoms come
within 1.24 times the sum of their van der Waals radii -- or when the surface
rules of **rCSU** say they touch and the touching is stabilising on balance.
Both are atom-level questions answered first and summed over each residue pair
after, which is how the RCSU program does it (doi:10.5281/zenodo.3817447) and
how vermouth's ``-go`` reimplements it.

CSU asks what each atom can still see.  Every atom is given a sphere of its own
radius plus a water's (2.80 A), sampled at 610 points -- the 14th Fibonacci
number, as the original -- and each point is awarded to the nearest atom whose
own sphere covers it.  A point an atom wins is surface the two share, so the
number of points one atom wins from another is how much of it that atom
touches.  What the touching is worth comes from the two atoms' chemical
characters: a hydrogen bond, a hydrophobic or aromatic contact or an ionic
bridge stabilises (CSU types 1-4), and a pair whose characters disagree
destabilises (type 5).  A residue pair whose stabilising points outnumber its
destabilising ones is an rCSU contact.

Which is to say the map says more than a distance would: two residues can be
close and not touch, because something else is in the way, and can touch
without being close, across a gap a water could not enter.

The radii, the atom characters and the table that reads a pair of them as a
bond type are the RCSU program's own numbers, carried in
``boonza/data/martini/rcsu/protein_map.json``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numba import njit, prange

__all__ = ["ALIASES", "Contact", "ContactMap", "contact_map"]

#: An atom's sphere is its own radius plus a water's: what a water could still
#: reach is not surface the two atoms share.
WATER_RADIUS = 2.80
#: Overlap counts an attraction, so the radii are read 24% larger than they are.
ALPHA = 1.24
#: Points per sphere, the 14th Fibonacci number (with the 13th for the spiral).
FIB_A, FIB_B = 377, 610
#: Bond types 1-4 stabilise a contact, 5 destabilises it, 6 is negligible.
STABILISING = 4
DESTABILISING = 5

#: Residue names the table does not hold, and the residue each one is.  A
#: protonation or a disulfide does not change which atoms a residue has or how
#: big they are, so the name the structure happens to use should not change its
#: contacts.  vermouth's table lists CYM, CYX, HIE and HIP but leaves them
#: empty, so looking an atom of one up raises KeyError there; HID, the CHARMM
#: spellings and the neutral acids it does not list at all, which reads every
#: radius as zero and quietly gives the residue no contacts.
ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "CYP": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG", "TYM": "TYR",
    "MSE": "MET",  # selenomethionine, whose SE sits where a sulfur would
}  # fmt: skip

#: Atom names the tables do not use, and the name each one goes by there.  The
#: tables name an isoleucine's delta carbon CD, as CHARMM does; a PDB calls it
#: CD1.  vermouth never meets the difference because it reads the map after
#: repairing the graph, which renames every atom to its force field's own.
ATOM_ALIASES = {("ILE", "CD1"): "CD"}


@cache
def _tables() -> tuple[dict, np.ndarray]:
    """``({resname: {atomname: (radius, type)}}, bond type table)``."""
    path = Path(__file__).resolve().parent.parent / "data" / "martini" / "rcsu" / "protein_map.json"
    raw = json.loads(path.read_text())
    residues = {
        res: {name: (float(v[0]), int(v[1])) for name, v in atoms.items()}
        for res, atoms in raw["residues"].items()
        if atoms  # the empty ones are names without a table; ALIASES answers them
    }
    return residues, np.asarray(raw["bond_type"], np.int64)


def _in_tables(resname: str) -> bool:
    """Whether the tables hold the residue this name is."""
    residues, _ = _tables()
    return ALIASES.get(resname, resname) in residues


def radius_and_type(resname: str, atomname: str) -> tuple[float, int]:
    """What the tables say about one atom, by the name of its residue.

    An atom of a residue the tables do not hold, or one the residue does not
    list, has no radius and no character: it takes part in no contact, though
    it still takes up room, since it can win a point of another atom's sphere
    and so keep that atom from touching a third.
    """
    residues, _ = _tables()
    residue = ALIASES.get(resname, resname)
    atoms = residues.get(residue)
    if atoms is None:
        return 0.0, 0
    return atoms.get(ATOM_ALIASES.get((residue, atomname), atomname), atoms["default"])


def _sphere(points: int = FIB_B, spiral: int = FIB_A) -> np.ndarray:
    """``points`` directions spread over the unit sphere, as the RCSU program.

    A Fibonacci spiral: the polar angle steps so that every point takes an
    equal band of the sphere, and the azimuth turns by the golden angle, which
    the ratio of two Fibonacci numbers is a whole-numbered stand-in for.
    """
    k = np.arange(points)
    turn = (np.arange(1, points + 1) * spiral) % points
    turn[turn == 0] = points
    theta = np.arccos(1.0 - 2.0 * k / points)
    phi = 2.0 * np.pi * turn / points
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )


@njit(cache=True)
def _adjacency(n, i, j):
    """Both directions of every pair, as offsets into a neighbour array."""
    offsets = np.zeros(n + 1, np.int64)
    for k in range(len(i)):
        offsets[i[k] + 1] += 1
        offsets[j[k] + 1] += 1
    for a in range(n):
        offsets[a + 1] += offsets[a]
    nbr = np.empty(offsets[n], np.int64)
    at = offsets[:n].copy()
    for k in range(len(i)):
        nbr[at[i[k]]] = j[k]
        at[i[k]] += 1
        nbr[at[j[k]]] = i[k]
        at[j[k]] += 1
    for a in range(n):  # ascending, so that the nearer of two equals wins by index
        nbr[offsets[a] : offsets[a + 1]] = np.sort(nbr[offsets[a] : offsets[a + 1]])
    return offsets, nbr


@njit(cache=True, parallel=True)
def _share_surface(xyz, vrad, offsets, nbr, unit, water, won):
    """How many of each atom's sphere points its neighbours win.

    ``won[k]`` counts the points neighbour ``nbr[k]`` wins from the atom whose
    slice holds ``k``.  A point goes to the nearest atom -- nearest by the
    distance between the two atoms, not by the distance to the point -- whose
    own sphere covers it, and ties go to the lower index, which is why the
    neighbours are in order.
    """
    npoints = unit.shape[0]
    natoms = len(offsets) - 1
    for i in prange(natoms):
        start, end = offsets[i], offsets[i + 1]
        if end == start:
            continue
        best = np.full(npoints, np.inf)
        winner = np.full(npoints, -1, np.int64)
        ri = vrad[i] + water
        for slot in range(start, end):
            j = nbr[slot]
            dx = xyz[j, 0] - xyz[i, 0]
            dy = xyz[j, 1] - xyz[i, 1]
            dz = xyz[j, 2] - xyz[i, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d >= vrad[i] + vrad[j] + water:  # too far for a shared surface
                continue
            rj = vrad[j] + water
            rj2 = rj * rj
            for p in range(npoints):
                if d >= best[p]:  # another atom already holds this point
                    continue
                qx = ri * unit[p, 0] - dx
                qy = ri * unit[p, 1] - dy
                qz = ri * unit[p, 2] - dz
                if qx * qx + qy * qy + qz * qz <= rj2:
                    best[p] = d
                    winner[p] = slot
        for p in range(npoints):
            if winner[p] >= 0:
                won[winner[p]] += 1


@dataclass(frozen=True)
class Contact:
    """One residue pair the map examined, and what it found."""

    resid_a: int
    chain_a: str
    resname_a: str
    insertion_a: str
    resid_b: int
    chain_b: str
    resname_b: str
    insertion_b: str
    dca: float  # between the alpha carbons, A, nan if either is missing
    overlap: int  # 1 when any pair of their atoms overlaps
    count: int  # surface points their atoms share
    stabilising: int
    destabilising: int

    @property
    def rcsu(self) -> bool:
        """Whether the shared surface is stabilising on balance."""
        return self.stabilising - self.destabilising > 0

    @property
    def kept(self) -> bool:
        """Whether this is a contact a Go model puts a well on."""
        return self.overlap == 1 or self.rcsu


@dataclass
class ContactMap:
    """Every residue pair a structure's atoms brought within reach."""

    contacts: list  # list[Contact], both ways round, in residue order

    def kept(self) -> list[tuple]:
        """``(resid, chain, insertion, resid, chain, insertion)`` per contact.

        The insertion code is the one thing a map read back from a file cannot
        carry, the format having no column for it, so a map kept in hand says
        more than one written down.  :func:`boonza.martini.go.go_pairs` takes
        either.
        """
        return [(c.resid_a, c.chain_a, c.insertion_a,
                 c.resid_b, c.chain_b, c.insertion_b) for c in self.contacts if c.kept]  # fmt: skip

    def text(self, version: str = "") -> str:
        """The map as the 18-column file the RCSU program writes.

        Which is the format :func:`~boonza.martini.go.read_contact_map` reads,
        so a map boonza computes can be kept, read back, and compared with one
        from the program itself or from the Martini contact-map server.
        """
        from .. import __version__

        head = [
            f"Go contact map calculated with boonza {version or __version__}\n",
            "\n",
            "Residue-Residue Contacts\n\n",
            "ID       - atom identification\n",
            "I1,I2    - serial residue id\n",
            "AA       - 3-letter code of aminoacid\n",
            "C        - chain\n",
            "I(PDB)   - residue number in PDB file\n",
            "DCA      - distance between CA\n",
            "CMs      - OV , CSU , oCSU , rCSU\n",
            "rCSU     - net contact from rCSU\n",
            "Count    - number of contacts between residues\n",
            "MODEL    - model number\n\n",
            "      ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs"
            "    rCSU   Count  Model\n",
            "=" * 93 + "\n",
        ]
        coded = {(c.chain_a, c.resid_a, c.insertion_a) for c in self.contacts
                 if c.insertion_a} | {(c.chain_b, c.resid_b, c.insertion_b)
                                      for c in self.contacts if c.insertion_b}  # fmt: skip
        if coded:
            # the 18-column format writes I(PDB) as a plain number, so a code
            # has nowhere to go and the rows it names stop picking out one
            # residue.  The map is still right; what is written down is not.
            warnings.warn(
                f"{len(coded)} residues carry an insertion code, which this format has no "
                f"column for: the map itself holds them apart and so does any model built "
                f"from it, but a reader of this file could not, so it is a record and not a "
                f"map to give back to --go-map",
                stacklevel=2,
            )
        serial = {}
        for c in self.contacts:
            serial.setdefault((c.chain_a, c.resid_a, c.insertion_a), len(serial) + 1)
            serial.setdefault((c.chain_b, c.resid_b, c.insertion_b), len(serial) + 1)
        rows = []
        for n, c in enumerate(self.contacts, start=1):
            rcsu = 1 if c.rcsu else 0
            rows.append(
                f"R {n:6d} {serial[c.chain_a, c.resid_a, c.insertion_a]:5d}  {c.resname_a:3s} "
                f"{c.chain_a:1s} {c.resid_a:4d}    "
                f"{serial[c.chain_b, c.resid_b, c.insertion_b]:5d}  {c.resname_b:3s} "
                f"{c.chain_b:1s} {c.resid_b:4d}    "
                f"{c.dca:9.4f}     {c.overlap:1d} {1 if c.count else 0} "
                f"{1 if c.stabilising else 0} {rcsu}"
                f"{rcsu: 6d}  {c.count: 6d}     0\n"
            )
        return "".join(head) + "".join(rows)


def contact_map(system, atoms: str = "protein") -> ContactMap:
    """The RCSU/OV contact map of an all-atom structure.

    ``atoms`` is what to read it over, as a selection; every chain is read
    together, so a contact between two of them counts as the structure shows
    it.  The residue numbers and chains are the structure's own, which is what
    a Go model keys its wells by.

    An atom with no coordinates takes no part, and neither does one whose
    residue the tables do not hold -- a cofactor, a ligand, a water -- though
    such an atom still takes up room: it can win a point of another atom's
    sphere and so keep it from touching a third.
    """
    from ..atomsel import select

    chosen = select(system, atoms) if isinstance(atoms, str) else np.asarray(atoms)
    if len(chosen) == 0:
        raise ValueError(f"nothing to read a contact map over: {atoms!r} selects no atoms")
    xyz = np.asarray(system.positions, float)[chosen]
    keep = np.isfinite(xyz).all(axis=1)
    # one conformation, the first of any alternates, as the RCSU program and
    # vermouth both take: two copies of a side chain would each shade the other
    if "altloc" in system.atoms._t._data:
        alt = np.array([str(a).strip() for a in system.atoms["altloc"][chosen]])
        keep &= (alt == "") | (alt == sorted(set(alt[alt != ""]) or {""})[0])
    chosen, xyz = chosen[keep], xyz[keep]

    # boonza groups atoms into residues already, so a residue is a residue
    residue = np.asarray(system.atoms["residue"])[chosen]
    used, residue_of = np.unique(residue, return_inverse=True)
    nres = len(used)
    resids = np.asarray(system.residues["resid"])[used]
    codes = ([str(x or "").strip() for x in np.asarray(system.residues["insertion"])[used]]
             if "insertion" in system.residues._t._data else [""] * nres)  # fmt: skip
    names_of_chain = np.asarray(system.chains["name"])
    chains = [str(names_of_chain[c]).strip()
              for c in np.asarray(system.residues["chain"])[used]]  # fmt: skip
    residue_name = [str(n).strip() for n in np.asarray(system.residues["name"])[used]]

    names = [str(n).strip() for n in system.atoms["name"][chosen]]
    anum = np.asarray(system.atoms["anum"])[chosen]
    vrad = np.zeros(len(chosen))
    atype = np.zeros(len(chosen), np.int64)
    unnamed = set()
    for k, name in enumerate(names):
        resname = residue_name[residue_of[k]]
        vrad[k], atype[k] = radius_and_type(resname, name)
        if vrad[k] == 0.0 and anum[k] != 1 and _in_tables(resname):
            unnamed.add((resname, name))
    if unnamed:
        # the tables hold this residue but not this atom of it, so the atom
        # takes no part in any contact.  Silence here would be a map that is
        # quietly short of what the structure shows
        atoms_said = ", ".join(f"{r} {a}" for r, a in sorted(unnamed))
        warnings.warn(f"no radius for {atoms_said}: these atoms take no part in "
                      f"the contact map", stacklevel=2)  # fmt: skip
    ca = np.full((nres, 3), np.nan)
    for k, name in enumerate(names):
        if name == "CA":
            ca[residue_of[k]] = xyz[k]

    residues, bond_type = _tables()
    reach = 2 * max(r for atoms_ in residues.values() for r, _ in atoms_.values()) + WATER_RADIUS
    overlap, count, stab, destab = _residue_matrices(
        xyz, vrad, atype, residue_of, nres, bond_type, reach
    )

    out = []
    for i, j in sorted(set(zip(*np.nonzero(overlap + count), strict=True))):
        if i == j:
            continue
        out.append(Contact(
            resid_a=int(resids[i]), chain_a=chains[i], resname_a=residue_name[i],
            insertion_a=codes[i],
            resid_b=int(resids[j]), chain_b=chains[j], resname_b=residue_name[j],
            insertion_b=codes[j],
            dca=float(np.linalg.norm(ca[i] - ca[j])), overlap=int(overlap[i, j]),
            count=int(count[i, j]), stabilising=int(stab[i, j]),
            destabilising=int(destab[i, j]),
        ))  # fmt: skip
    return ContactMap(out)


def _residue_matrices(xyz, vrad, atype, residue_of, nres, bond_type, reach):
    """Overlap, shared surface, and what it is worth, per residue pair."""
    from ..spatial import pairs_within

    i, j, _ = pairs_within(xyz, reach)
    d = np.linalg.norm(xyz[i] - xyz[j], axis=1)

    # OV: the atoms come within 1.24 times the sum of their radii
    touching = d < ALPHA * (vrad[i] + vrad[j])
    overlap = np.zeros((nres, nres), np.int64)
    ri, rj = residue_of[i[touching]], residue_of[j[touching]]
    overlap[ri, rj] = 1
    overlap[rj, ri] = 1

    # CSU: how much of each atom's sphere its neighbours take
    offsets, nbr = _adjacency(len(xyz), i, j)
    won = np.zeros(len(nbr), np.int64)
    _share_surface(xyz, vrad, offsets, nbr, _sphere(), WATER_RADIUS, won)
    owner = np.repeat(np.arange(len(xyz)), np.diff(offsets))
    shared = won > 0
    # a contact needs a character at both ends, though an atom without one
    # still took part above, where it could stand in another's way
    shared &= (atype[owner] > 0) & (atype[nbr] > 0)
    a, b, points = owner[shared], nbr[shared], won[shared]

    kind = bond_type[atype[a], atype[b]]
    count = np.zeros((nres, nres), np.int64)
    stab = np.zeros((nres, nres), np.int64)
    destab = np.zeros((nres, nres), np.int64)
    ra, rb = residue_of[a], residue_of[b]
    np.add.at(count, (ra, rb), points)
    keep = (kind > 0) & (kind <= STABILISING)
    np.add.at(stab, (ra[keep], rb[keep]), points[keep])
    keep = kind == DESTABILISING
    np.add.at(destab, (ra[keep], rb[keep]), points[keep])
    return overlap, count, stab, destab
