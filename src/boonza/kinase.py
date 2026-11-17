"""The conformational state of a protein kinase domain, as Modi and Dunbrack name it.

A structure gets two labels (Modi & Dunbrack, PNAS 2019, 116:6818; the per-structure
rule is the one Kincore applies, Nucleic Acids Res 2022, 50:D654):

**Where the DFG-Phe ring is** (the spatial group), from two distances to its tip (Cζ):
D1 to the Cα of the fourth residue after the conserved αC Glu, and D2 to the Cα of the
conserved β3 Lys.

    DFGin     D1 <= 11 Å and D2 >= 11 Å    under the C-helix
    DFGout    D1 >  11 Å and D2 <= 14 Å    in the ATP site
    DFGinter  D1 <= 11 Å and D2 <  11 Å    in between, pointing up at the β sheet

A Phe anywhere else has no spatial group.

**How the backbone puts it there** (the dihedral cluster), from φ and ψ of the residue
before the DFG (X), of the Asp and of the Phe, and the Phe χ1 rotamer (plus 0..120°,
trans 120..240°, minus 240..360°).  The clusters are named by the Ramachandran region
of X, Asp and Phe (A alpha, B beta, L left) and the rotamer; their centroids are in
:data:`CENTROIDS` (Table S2 of the 2019 paper).  A structure is compared with the
centroids of its own spatial group and rotamer, by the mean over the six backbone
angles of ``2 (1 - cos Δθ)``, and takes the nearest one closer than 0.45 (about 40° per
angle); otherwise it has no cluster.  ``BLAminus`` is the active conformation; the
other five DFGin clusters are inactive.

A dihedral that would reach across residues missing from the structure is not measured,
and neither is D1 when the residue four after the Glu is one of them.

Also reported, each as the paper defines it: the C-helix (in when the Cβ atoms of the
αC Glu and the β3 Lys are within 10 Å), the regulatory spine contacts (side chains
within 4.5 Å) and whether the activation loop is extended.

The conserved residues are found from sequence motifs (:func:`find_motifs`) and, where
those leave a choice, from the structure; this is not a sequence alignment, and for a
kinase it misses the residue numbers can be given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CENTROIDS", "KinaseState", "find_motifs", "kinase_state", "kinase_states"]

#: cluster centroids, degrees: φ, ψ of X-DFG, DFG-Asp, DFG-Phe (Modi & Dunbrack 2019, Table S2)
CENTROIDS = {
    "DFGin": {
        "BLAminus": (-129, 179, 61, 81, -97, 20),
        "BLAplus": (-119, 168, 59, 34, -89, -8),
        "ABAminus": (-112, -8, -141, 148, -128, 23),
        "BLBminus": (-135, 175, 60, 65, -79, 145),
        "BLBplus": (-125, 172, 60, 33, -85, 145),
        "BLBtrans": (-106, 157, 69, 21, -62, 134),
    },
    "DFGout": {"BBAminus": (-138, -176, -144, 104, -82, -9)},
    "DFGinter": {"BABtrans": (-80, 128, -117, 24, -85, 133)},
}

D_CUT = 11.0  # Å, on D1 and on D2
D2_OUT = 14.0  # Å, the farthest a DFGout Phe sits from the β3 Lys
CLUSTER_CUT = 0.45  # mean 2(1 - cos Δθ), about 40° per angle
CHELIX_CUT = 10.0  # Å, Glu Cβ to Lys Cβ
CONTACT = 4.5  # Å, closest side-chain atoms of two spine residues
#: the side-chain atom that stands for the Phe Cζ, by residue (as in Kincore's program)
TIP = {"F": "CZ", "Y": "OH", "L": "CG", "M": "CE", "W": "CZ3"}
LINK = 2.5  # Å, C to N of two residues that follow one another in the chain
LYS_REACH = 17.0  # Å: the β3 Lys Cα is 9 to 14 Å from the DFG-Asp Cα; farther is another Lys
HBOND = 3.5  # Å, N to O of the hydrogen bond an extended activation loop makes

_BACKBONE = ("N", "CA", "C", "O", "OXT", "OT1", "OT2")
_CATALYTIC_LOOP = re.compile(r"(?=[HY].D.{4}N)")  # HRDLAARN, HRDLKPEN, YRDLKLEN
_CATALYTIC_LOOSE = re.compile(r"(?=[HY].[DN])")  # HRDIKSDS in PAK4-6, HRN in ERBB3
_DFG = re.compile(r"(?=D[FLMWY]G)")  # DFG; DLG in WEE1, DMG in CDK8, DYG, DWG
_BETA3 = re.compile(r"(?=[AC].K)")  # VAIK, VAVK, CAVK, MARK
_BETA3_MUTANT = re.compile(r"(?=[AC].R)")  # the Lys of β3 mutated to Arg


@dataclass
class KinaseState:
    """What :func:`kinase_state` found for one kinase domain.

    Residue numbers are the file's (``lys``, ``glu``, ``hrd``, the His or Tyr of the
    HRD motif, None when the chain has no such motif, and ``dfg``, its Asp).  Distances
    are in Å, angles in degrees; a value whose atoms are missing is NaN, and ``notes``
    says what was missing or unusual.
    """

    chain: str
    lys: int
    glu: int
    hrd: int | None
    dfg: int
    hrd_motif: str
    xdfg_motif: str
    d1: float = np.nan
    d2: float = np.nan
    spatial: str | None = None  # DFGin, DFGout, DFGinter
    dihedrals: tuple = (np.nan,) * 6  # φ, ψ of X-DFG, DFG-Asp, DFG-Phe
    chi1: float = np.nan
    rotamer: str | None = None  # minus, plus, trans
    cluster: str | None = None  # BLAminus, ..., or None when no centroid is close
    cluster_distance: float = np.nan  # to ``nearest``
    nearest: str | None = None  # the nearest centroid of this spatial group and rotamer
    cb_distance: float = np.nan  # αC Glu Cβ to β3 Lys Cβ
    chelix: str | None = None  # in, out
    salt_bridge: float = np.nan  # closest Lys Nζ (Arg Nη) to Glu Oε
    spine_his_phe: float = np.nan  # HRD-His and DFG-Phe side chains
    spine_phe_c: float = np.nan  # DFG-Phe and the fourth residue after the αC Glu
    loop_hbond: float = np.nan  # N of the sixth loop residue (DFGxxX) to O of X-HRD
    notes: list = field(default_factory=list)

    @property
    def label(self) -> str:
        """``DFGin-BLAminus``, ``DFGout-None``: the two labels as Kincore writes them."""
        return f"{self.spatial}-{self.cluster}"

    @property
    def active(self) -> bool:
        """BLAminus with the C-helix in: what an active kinase needs of these residues.
        It says nothing of the activation loop's phosphorylation or of the substrate."""
        return self.cluster == "BLAminus" and self.chelix == "in"

    @property
    def spine(self) -> bool:
        """Whether both regulatory-spine contacts measured here are made (the third,
        to the β4 residue, is not looked for)."""
        return bool(self.spine_his_phe <= CONTACT and self.spine_phe_c <= CONTACT)

    @property
    def loop(self) -> str | None:
        """``extended`` when the activation loop makes the hydrogen bond of a loop that
        lets substrate in, ``folded`` when it does not, None when it cannot be seen."""
        if np.isnan(self.loop_hbond):
            return None
        return "extended" if self.loop_hbond <= HBOND else "folded"


def find_motifs(sequence: str, numbers=None) -> list[dict]:
    """Where the conserved kinase residues are in a one-letter ``sequence``.

    One dict per kinase domain found (a catalytic loop and DFG without a Lys and Glu
    before them only when there is nothing better), of positions in the sequence: ``hrd`` (the His
    or Tyr of the catalytic loop, H/Y-x-D-x4-N), ``dfg`` (the Asp of the first
    D-F/L/M/W/Y-G that follows the catalytic Asp by 14 to 40 residues), and ``pairs``, the
    candidate (β3 Lys, αC Glu) pairs: a Lys in A/C-x-K 95 to 260 residues before the
    DFG and a Glu 8 to 30 residues after it.  More than one pair is common; the
    structure decides (:func:`kinase_state`).  A β3 Lys mutated to Arg is accepted when
    there is no Lys, and a catalytic loop without its Asn or with Asn for the Asp
    (PAK4, ERBB3) when there is no other.

    ``numbers``: the residue number of each letter.  Spacings are counted in these, so
    that residues missing from a structure do not shorten them.
    """
    n = np.arange(len(sequence)) if numbers is None else np.asarray(numbers, np.int64)
    dfgs = [g.start() for g in _DFG.finditer(sequence)]
    found, seen = [], set()
    for loop in (_CATALYTIC_LOOP, _CATALYTIC_LOOSE):
        for m in loop.finditer(sequence):
            asp = m.start() + 2
            dfg = next((g for g in dfgs if 14 <= n[g] - n[asp] <= 40), None)
            if dfg is None or dfg in seen:
                continue
            pairs = []
            for pattern in (_BETA3, _BETA3_MUTANT):
                for k in pattern.finditer(sequence):
                    lys = k.start() + 2
                    if 95 <= n[dfg] - n[lys] <= 260:
                        pairs += [(lys, e) for e in range(lys + 1, dfg)
                                  if sequence[e] == "E" and 8 <= n[e] - n[lys] <= 30]  # fmt: skip
                if pairs:
                    break
            seen.add(dfg)
            found.append({"hrd": m.start(), "dfg": dfg, "pairs": pairs})
        if any(f["pairs"] for f in found):
            break
    return [f for f in found if f["pairs"]] or found


def dihedral_distance(a, b) -> float:
    """Mean of ``2 (1 - cos Δθ)`` over two sets of angles in degrees: the squared chord
    between points on the unit circle, so 180° and -180° are the same angle."""
    delta = np.radians(np.asarray(a, float) - np.asarray(b, float))
    return float(np.mean(2.0 * (1.0 - np.cos(delta))))


def spatial_group(d1: float, d2: float) -> str | None:
    """DFGin, DFGout or DFGinter from D1 and D2 (Å); None outside all three."""
    if np.isnan(d1) or np.isnan(d2):
        return None
    if d1 <= D_CUT:
        return "DFGin" if d2 >= D_CUT else "DFGinter"
    return "DFGout" if d2 <= D2_OUT else None


def rotamer(chi1: float) -> str | None:
    """minus, plus or trans for a χ1 in degrees."""
    if np.isnan(chi1):
        return None
    x = chi1 % 360.0
    return "plus" if 0.0 < x <= 120.0 else "trans" if x <= 240.0 and x > 120.0 else "minus"


def assign_cluster(spatial, dihedrals, chi1, cutoff: float = CLUSTER_CUT):
    """(cluster or None, distance, nearest centroid or None) for six backbone dihedrals
    and a χ1, among the centroids of ``spatial`` that have this rotamer."""
    rot = rotamer(chi1)
    if spatial is None or rot is None or np.isnan(dihedrals).any():
        return None, np.nan, None
    near = {name: dihedral_distance(dihedrals, c)
            for name, c in CENTROIDS[spatial].items() if name.endswith(rot)}  # fmt: skip
    if not near:
        return None, np.nan, None
    best = min(near, key=near.get)
    return (best if near[best] < cutoff else None), near[best], best


def _chains(system, chain):
    if chain is None:
        return list(range(system.nchains))
    if isinstance(chain, str):
        found = [c for c in range(system.nchains) if system.chains["name"][c] == chain]
        if not found:
            raise ValueError(f"no chain {chain!r}")
        return found
    return [int(chain)]


class _Atoms:
    """Atoms of residues by name, for one frame."""

    def __init__(self, system, xyz):
        self.s, self.xyz = system, xyz
        self.names = system.atoms["name"]
        self.heavy = system.atoms["anum"] > 1

    def get(self, residue, name):
        if residue is None:
            return None
        ids = self.s.residue_atoms(residue)
        hit = ids[self.names[ids] == name]
        return self.xyz[hit[0]] if len(hit) else None

    def side_chain(self, residue):
        if residue is None:
            return self.xyz[:0]
        ids = self.s.residue_atoms(residue)
        keep = self.heavy[ids] & ~np.isin(self.names[ids], _BACKBONE)
        return self.xyz[ids[keep]]


def _distance(a, b) -> float:
    return np.nan if a is None or b is None else float(np.linalg.norm(a - b))


def _closest(a, b) -> float:
    if a is None or b is None or not len(a) or not len(b):
        return np.nan
    return float(np.linalg.norm(a[:, None] - b[None], axis=2).min())


def _dihedral(*p) -> float:
    from .pbc import dihedrals

    if any(x is None for x in p):
        return np.nan
    return float(np.degrees(dihedrals(*(x[None] for x in p)))[0])


def _tip(atoms, residue, letter):
    """The atom of the DFG residue that D1 and D2 are measured to: Cζ of Phe, and for
    the other residues some kinases have there the atom Kincore's program takes
    (:data:`TIP`).  None when it is not in the structure."""
    return atoms.get(residue, TIP[letter]) if letter in TIP else None


def _pair(atoms, at, seq, motif):
    """The β3 Lys and αC Glu among the candidates, from the C-terminal lobe, which does
    not move with the C-helix.  The Lys is the one whose Cα is nearest the DFG-Asp Cα
    (β3 lies over the DFG); None when even that one is too far to be β3, as when the
    Lys is mutated or not in the structure.  The Glu is, of that Lys's candidates, the
    one whose Cα is nearest the HRD-His Cα: the conserved Glu is the residue of the
    helix that sits over the catalytic loop, C-helix in or out.  A salt bridge is not
    asked for: an inactive kinase has none."""
    far = lambda d: np.inf if np.isnan(d) else d  # noqa: E731
    asp, his = atoms.get(at(motif["dfg"]), "CA"), atoms.get(at(motif["hrd"]), "CA")
    reach = lambda k: far(_distance(atoms.get(at(k), "CA"), asp))  # noqa: E731
    lys = min({p[0] for p in motif["pairs"]}, key=lambda k: (reach(k), k))
    if reach(lys) > LYS_REACH:
        return None
    glu = min((e for k, e in motif["pairs"] if k == lys),
              key=lambda e: (far(_distance(atoms.get(at(e), "CA"), his)), e))  # fmt: skip
    return lys, glu


def _state(system, atoms, res, seq, chain_name, motif, lys, glu):
    """The state of one domain; ``res`` are the chain's residue indices, in order."""
    resid = system.residues["resid"]
    hrd, dfg = motif["hrd"], motif["dfg"]
    at = lambda k: int(res[k]) if k is not None and 0 <= k < len(res) else None  # noqa: E731
    notes = []

    def step(k, n):
        """The residue ``n`` along the chain from ``k``, None when the chain is broken
        on the way: a residue missing from the structure is not counted over."""
        if k is None:
            return None
        for i in range(min(k, k + n), max(k, k + n)):
            gap = _distance(atoms.get(at(i), "C"), atoms.get(at(i + 1), "N"))
            if not gap <= LINK:
                return None
        return k + n

    if lys is None:
        pair = _pair(atoms, at, seq, motif)
        if pair is None:
            return None
        lys, glu = pair
    if seq[lys] != "K":
        notes.append(f"β3 Lys is {seq[lys]} here")
    if seq[glu] != "E":
        notes.append(f"αC Glu is {seq[glu]} here")
    x, asp, phe, c4 = step(dfg, -1), dfg, step(dfg, 1), step(glu, 4)
    st = KinaseState(
        chain=chain_name, lys=int(resid[at(lys)]), glu=int(resid[at(glu)]),
        hrd=None if hrd is None else int(resid[at(hrd)]), dfg=int(resid[at(asp)]),
        hrd_motif="" if hrd is None else seq[hrd : hrd + 3],
        xdfg_motif=seq[max(dfg - 1, 0) : dfg + 3], notes=notes,
    )  # fmt: skip
    letter = seq[dfg + 1]
    if letter != "F":
        notes.append(f"DFG is D{letter}G: its {TIP[letter]} stands for the Phe Cζ")
    tip = _tip(atoms, at(phe), letter)
    if tip is None:
        notes.append("DFG-Phe ring missing: no spatial group")
    if c4 is None:
        notes.append("the residue four after the αC Glu is not in the structure: no D1")
    st.d1 = _distance(atoms.get(at(c4), "CA"), tip)
    st.d2 = _distance(atoms.get(at(lys), "CA"), tip)
    st.spatial = spatial_group(st.d1, st.d2)

    def phi_psi(k):
        r, before, after = at(k), at(step(k, -1)), at(step(k, 1))
        n, ca, c = (atoms.get(r, name) for name in ("N", "CA", "C"))
        return (_dihedral(atoms.get(before, "C"), n, ca, c),
                _dihedral(n, ca, c, atoms.get(after, "N")))  # fmt: skip

    st.dihedrals = phi_psi(x) + phi_psi(asp) + phi_psi(phe)
    st.chi1 = _dihedral(*(atoms.get(at(phe), name) for name in ("N", "CA", "CB", "CG")))
    st.rotamer = rotamer(st.chi1)
    st.cluster, st.cluster_distance, st.nearest = assign_cluster(
        st.spatial, np.array(st.dihedrals), st.chi1
    )
    if np.isnan(st.dihedrals).any() or st.rotamer is None:
        notes.append("X-DFG atoms missing: no dihedral cluster")

    st.cb_distance = _distance(atoms.get(at(glu), "CB"), atoms.get(at(lys), "CB"))
    if not np.isnan(st.cb_distance):
        st.chelix = "in" if st.cb_distance <= CHELIX_CUT else "out"
    ids = system.residue_atoms(at(lys))
    amine = atoms.xyz[ids[np.isin(atoms.names[ids], ("NZ", "NH1", "NH2", "NE"))]]
    ids = system.residue_atoms(at(glu))
    st.salt_bridge = _closest(amine, atoms.xyz[ids[np.isin(atoms.names[ids], ("OE1", "OE2"))]])
    ring = atoms.side_chain(at(phe))
    st.spine_phe_c = _closest(ring, atoms.side_chain(at(c4)))
    if hrd is None:
        notes.append("no catalytic loop (H/Y-x-D-x4-N) before this DFG")
    else:
        st.spine_his_phe = _closest(atoms.side_chain(at(hrd)), ring)
        st.loop_hbond = _distance(atoms.get(at(step(dfg, 5)), "N"),
                                  atoms.get(at(step(hrd, -1)), "O"))  # fmt: skip
    return st


def kinase_states(system, chain=None, lys=None, glu=None, dfg=None, positions=None) -> list:
    """The :class:`KinaseState` of every kinase domain in ``system`` (or in ``chain``,
    an index or a name): an empty list when there is none.

    ``lys``, ``glu`` and ``dfg`` are the residue numbers of the β3 Lys, the αC Glu and
    the DFG Asp, for a kinase whose motifs are not the usual ones or that has another
    residue there (a Lys mutated away, the His of ERBB3 for the Glu); give ``chain``
    with them when the numbers are not unique.  ``positions``: one frame, (natoms, 3), in
    place of the system's own.
    """
    from .matchmaker import chain_sequence

    xyz = np.asarray(system.positions if positions is None else positions, float)
    if xyz.shape != (system.natoms, 3):
        raise ValueError(f"positions must be one frame, ({system.natoms}, 3)")
    atoms = _Atoms(system, xyz)
    resid = system.residues["resid"]
    out = []
    for c in _chains(system, chain):
        seq, res, _ = chain_sequence(system, c)
        if len(seq) < 100:
            continue
        name = str(system.chains["name"][c])
        where = {int(resid[r]): k for k, r in reversed(list(enumerate(res.tolist())))}
        given = {}
        for key, number in (("lys", lys), ("glu", glu), ("dfg", dfg)):
            if number is None:
                continue
            if number not in where:
                break
            if key == "dfg" and seq[where[number]] != "D":
                raise ValueError(f"dfg={number} is {seq[where[number]]} in chain {name!r}, "
                                 "not the Asp of a DFG")  # fmt: skip
            given[key] = where[number]
        else:
            motifs = find_motifs(seq, resid[res])
            if "dfg" in given:
                k = given["dfg"]
                motifs = [m for m in motifs if m["dfg"] == k] or [
                    {"hrd": None, "dfg": k, "pairs": []}
                ]
            if ("lys" in given) != ("glu" in given):
                raise ValueError("give lys and glu together")
            for m in motifs:
                if not m["pairs"] and "lys" not in given:
                    continue
                st = _state(system, atoms, res, seq, name, m, given.get("lys"), given.get("glu"))
                if st is not None:
                    out.append(st)
    return out


def kinase_state(system, chain=None, lys=None, glu=None, dfg=None, positions=None) -> KinaseState:
    """The :class:`KinaseState` of the one kinase domain in ``system``; arguments as
    :func:`kinase_states`.  Raises when there is none, or more than one (name a chain)."""
    found = kinase_states(system, chain, lys, glu, dfg, positions)
    if not found:
        raise ValueError("no kinase domain found: no catalytic loop (H/Y-x-D-x4-N) with a "
                         "DFG after it and a β3 Lys before; pass lys=, glu= and dfg=")  # fmt: skip
    if len(found) > 1:
        chains = ", ".join(f"{s.chain or '_'} (DFG {s.dfg})" for s in found)
        raise ValueError(f"{len(found)} kinase domains: {chains}; pass chain= or use kinase_states")
    return found[0]
