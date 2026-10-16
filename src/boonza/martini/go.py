"""GoMartini: a fold held by native contacts rather than by rubber bands.

An elastic network holds every pair it spans at a fixed length, so a protein
cannot unfold and cannot open.  A Go model instead puts a Lennard-Jones well on
each pair of residues that touch in the structure -- so the fold is a minimum
rather than a constraint, and a contact can break and re-form.

The contacts themselves come from a map: a residue pair is in it when the two
overlap (OV) or when the RCSU surface rules say they do.  ``read_contact_map``
reads the 18-column format the RCSU program and the Martini contact-map server
write (doi:10.5281/zenodo.3817447), and :func:`go_pairs` turns a map into the
wells, which are what a topology writes as ``[ nonbond_params ]`` between one
virtual site per backbone bead.

Which pairs survive, and with what, follows martinize2:

* a pair must be in the map,
* the two residues must be more than ``res_dist`` residues apart along the
  chain -- what the bonded terms already hold is not the Go model's business,
* their backbone beads must be between ``low`` and ``up`` apart, both bounds
  strict,
* and the map must name the pair **both ways**.  A contact is a statement about
  a pair, so one written only one way is one the map is unsure of, and the RCSU
  rules are not symmetric: whether A reaches B is asked of A's surface and of
  B's separately, and the two can disagree.  On a fibronectin domain seven of
  354 pairs are one-way, and dropping them is the difference between 185 wells
  and 189.
* The well then sits where the pair is: sigma = d / 2^(1/6), epsilon = ``eps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["GoPair", "GO_EPS", "GO_LOW", "GO_RES_DIST", "GO_UP", "go_pairs", "read_contact_map"]

#: The well's depth (kJ/mol), martinize2's -go-eps.
GO_EPS = 9.414
#: Backbone beads nearer than this are already held by the bonded terms (nm).
GO_LOW = 0.3
#: And beyond this they are not touching (nm).  martinize2's -go-up.
GO_UP = 1.1
#: Residues this close along the chain are the bonded terms' business.
GO_RES_DIST = 3
#: A Lennard-Jones well sits at 2^(1/6) sigma, which is where the pair is.
WELL = 2.0 ** (1.0 / 6.0)

#: The columns of the map that mean anything here.  The format has eighteen and
#: the rest are the program's own workings; a line is a contact when it starts
#: with R and has all eighteen, which is how the C code says "this is a row".
_CHAIN_A, _RESID_A, _CHAIN_B, _RESID_B, _OV, _RCSU = 4, 5, 8, 9, 11, 14
_COLUMNS = 18


@dataclass(frozen=True)
class GoPair:
    """One well: the two beads it joins, where it sits, and how deep."""

    i: int
    j: int
    distance: float  # nm, between the backbone beads
    sigma: float
    epsilon: float


def read_contact_map(path) -> list[tuple[int, str, int, str]]:
    """``(resid, chain, resid, chain)`` for every contact the map holds.

    A row counts when it is an overlap (OV), or when it is not but the RCSU
    rules call it a contact anyway -- which is the filter the original script
    applies, kept here so a map means the same thing to boonza as to it.

    The residue numbers are the structure's own, not a count from one: a map is
    written against the structure it was computed from, and matching it up by
    position would quietly pair the wrong residues on anything numbered its own
    way -- a chymotrypsin, an antibody, or a chain that starts at 802.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        token = line.split()
        if not token or token[0] != "R" or len(token) != _COLUMNS:
            continue
        if token[_OV] == "1" or (token[_OV] == "0" and token[_RCSU] == "1"):
            out.append((int(token[_RESID_A]), token[_CHAIN_A],
                        int(token[_RESID_B]), token[_CHAIN_B]))  # fmt: skip
    if not out:
        raise ValueError(f"{path} holds no contacts: is it an 18-column RCSU map?")
    return out


def go_pairs(molecule, contacts, *, eps: float = GO_EPS, low: float = GO_LOW,
             up: float = GO_UP, res_dist: int = GO_RES_DIST,
             backbone: str = "BB") -> list[GoPair]:  # fmt: skip
    """The wells a contact map asks for, on one coarse-grained molecule.

    ``molecule`` is a :class:`~boonza.martini.build.CGMolecule`, whose
    positions are in nanometres, as ``low`` and ``up`` are.  A contact naming a
    residue the molecule does not hold is skipped: a map is written for a whole
    structure and a molecule is one chain of it.
    """
    from .build import _residues_within

    bead_of: dict[tuple[str, int], int] = {}
    for k, node in enumerate(molecule.nodes):
        if str(node.get("atomname", "")) != backbone:
            continue
        key = (str(node.get("chain", "") or ""), int(node["input_resid"]))
        bead_of.setdefault(key, k)
    if not bead_of:
        raise ValueError(f"no {backbone!r} beads: a Go model needs one backbone bead a residue")

    near = _residues_within(molecule, res_dist)  # what the bonded terms hold
    residue_of = {k: int(molecule.nodes[k]["resid"]) for k in bead_of.values()}
    xyz = np.asarray(molecule.positions, float)

    seen: set[tuple[int, int]] = set()
    out: dict[tuple[int, int], GoPair] = {}
    for resid_a, chain_a, resid_b, chain_b in contacts:
        i = bead_of.get((chain_a, resid_a))
        j = bead_of.get((chain_b, resid_b))
        if i is None or j is None or i == j:
            continue
        if residue_of[j] in near.get(residue_of[i], ()):
            continue
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        if not (low < d < up):
            continue
        if (j, i) not in seen:  # the other way round has not come past yet
            seen.add((i, j))
            continue
        out.setdefault((min(i, j), max(i, j)), GoPair(min(i, j), max(i, j), d, d / WELL, eps))
    return sorted(out.values(), key=lambda p: (p.i, p.j))


#: What the virtual sites are called in the topology, and the files that carry
#: them.  A Go model's wells are nonbonded terms between sites that stand for
#: residues, so they live in files of their own that the topology includes,
#: which is how martinize2 writes them.  ``#define GO_VIRT`` goes at the top
#: because martinize2 writes that too; nothing boonza carries reads it, and a
#: parameter file of your own may.
ATOMTYPES_ITP = "go_atomtypes.itp"
NBPARAMS_ITP = "go_nbparams.itp"
#: Where a build writes a map it computed itself, beside the topology: the same
#: 18-column format, so it can be read back, kept, or compared with another.
CONTACTS_OUT = "go_contacts.out"
GO_DEFINE = "#define GO_VIRT"
#: The name a virtual site takes, which a viewer and GROMACS both read as an
#: alpha carbon standing for its residue.
SITE_NAME = "CA"


@dataclass
class GoModel:
    """The wells of one build, and the files that carry them."""

    pairs: list  # list[GoPair], over every molecule, by built bead index
    sites: int  # how many virtual sites were added
    atomtypes: str
    nbparams: str
    #: The map the wells came from, when boonza computed it rather than read
    #: it, so that a build writes down what it held the fold by
    contact_map: str = ""


def add_go_sites(molecule, pairs, moltype: str, backbone: str = "BB") -> dict[int, str]:
    """Put one virtual site on every backbone bead, and hold the wells' pairs apart.

    A well is a nonbonded term, so it cannot act between beads that already
    see each other through the force field: the site carries the Go
    interaction and the bead carries Martini's, and the two never mix.  Every
    residue gets a site whether or not a well reaches it, as martinize2 does --
    a type with no pair in :data:`NBPARAMS_ITP` simply never interacts, and
    numbering them by residue keeps a map readable against the structure.

    Returns ``{bead index: its site's type name}``.
    """
    from .links import Interaction

    spine = [k for k, n in enumerate(molecule.nodes)
             if str(n.get("atomname", "")) == backbone]  # fmt: skip
    type_of: dict[int, str] = {}
    for serial, bead in enumerate(spine, start=1):
        node = molecule.nodes[bead]
        atype = f"{moltype}_{serial}"
        site = molecule.add_node(
            {"atype": atype, "resname": node.get("resname", ""), "atomname": SITE_NAME,
             "charge": 0.0, "mass": 0.0, "resid": node["resid"],
             "chain": node.get("chain", ""), "input_resid": node.get("input_resid"),
             "insertion": node.get("insertion", ""), "go_site": True},
            molecule.positions[bead],
        )  # fmt: skip
        molecule.add("virtual_sitesn", Interaction((site, bead), [1], {"comment": "go site"}))
        type_of[bead] = atype
    # the beads a well joins keep their own nonbonded terms out of its way
    for pair in pairs:
        molecule.add("exclusions", Interaction((pair.i, pair.j), [], {"group": "Go model"}))
    return type_of


def go_files(models) -> tuple[str, str]:
    """``(atomtypes, nonbond_params)`` for every molecule's sites and wells.

    ``models`` is ``[(type_of, pairs), ...]``, one per molecule.
    """
    types, params = ["[ atomtypes ]"], ["[ nonbond_params ]"]
    for type_of, pairs in models:
        for atype in dict.fromkeys(type_of.values()):
            types.append(f"{atype} 0.0 0 A 0.00000000 0.00000000 ")
        for pair in pairs:
            params.append(f"{type_of[pair.i]} {type_of[pair.j]} 1 {pair.sigma:.8f} "
                          f"{pair.epsilon:.8f} ;go bond {pair.distance}")  # fmt: skip
    return "\n".join(types) + "\n", "\n".join(params) + "\n"
