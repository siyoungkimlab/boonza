"""Physicochemical and ADME properties of a ligand design, from RDKit alone.

What a series' ``html`` button computes for each member: the numbers a medicinal
chemist reads before deciding whether a design is worth making -- size and polarity,
lipophilicity at pH (logD, which is logP corrected for whatever is ionised there),
solubility, passive absorption and brain penetration, the druglikeness filters, and
the substructures worth a second look.

Everything here is RDKit and arithmetic: no extra dependency, nothing downloaded,
nothing sent anywhere.  Two of the numbers come from random forests, which are data
files in ``boonza/data/admet`` walked with plain comparisons -- no sklearn, nothing to
unpack -- and which degrade to the published equations if the files are missing.  The
methods are published ones, cited where they are used:

  logD          Henderson-Hasselbalch on RDKit's Crippen logP, with the pKa of each
                ionisable group located by the table below (``ACIDS``, ``BASES``) and
                priced by the fitted forest, and an ionised group left partitioning as
                an ion pair rather than not at all (the "diff 3-4" approximation;
                Scherrer 1984, Avdeef ch. 4).  The design is neutralised first
                (:func:`neutralised`), because that is the species a pKa belongs to
  pKa           a random forest over the local environment of the ionisable atom,
                fitted from Baltruschat & Czodrowski's collection (CC BY 4.0), with
                extra sites from Dimorphite-DL (Ropp et al., J Cheminform 2019, 11, 14)
  log S         a random forest fitted from AqSolDB (Sorkun, Khetan & Er, Sci Data
                2019, 6, 143; CC0), printed beside the two published equations
  ESOL          Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000
  Ali logS      Ali, Camilleri, Brown, Hutt & Kirton, J. Chem. Inf. Model. 2012, 52, 420
  Log Kp        Potts & Guy, Pharm. Res. 1992, 9, 663
  BOILED-Egg    Daina & Zoete, ChemMedChem 2016, 11, 1117
  QED           Bickerton, Paolini, Besnard, Muresan & Hopkins, Nat. Chem. 2012, 4, 90
  SAscore       Ertl & Schuffenhauer, J. Cheminform. 2009, 1, 8
  Lipinski      Adv. Drug Deliv. Rev. 1997, 23, 3;  Ghose, J. Comb. Chem. 1999, 1, 55;
  and the other Veber, J. Med. Chem. 2002, 45, 2615;  Egan, J. Med. Chem. 2000, 43, 3867;
  filters       Muegge, J. Med. Chem. 2001, 44, 1841

Beside the properties there are three substructure tables and one filter, none of them
a model: :data:`SOFT_SPOTS` (where CYP is likely to attack), :data:`PHASE2_RULES` (which
conjugation routes are open, and which of those conjugates is itself reactive), and
:data:`TRIAGE_FIELDS`, which reads the numbers above into one verdict per design so a
series can be read as a shortlist -- always with the rule that produced it.

These are estimates from the structure alone.  logD carries about 0.5-1.0 log unit of
error even when the ionisation is right, and the filters are rules of thumb that
plenty of marketed drugs break; :func:`caveats` says where a number should not be
trusted at all.  None of this is a measurement.  The BOILED-Egg's own verdicts are weak
where they have been benchmarked -- about 0.75 AUROC for gut absorption and 0.73 for
brain penetration, and the Abbott bioavailability score about 0.57, which is hardly
better than calling every design bioavailable -- and the report says so.

Deliberately not computed: iLOGP (needs CHARMM), XLOGP3 (needs its reference
database), MLOGP, the CYP and P-gp models (a 7 GB neural network, and a different
question from anything here) and sites of metabolism.  A row that cannot be computed
is absent rather than guessed at.
"""

from __future__ import annotations

import math
from pathlib import Path

# --- ionisation ---------------------------------------------------------------------
#
# One representative pKa per functional class.  These are **medians of measured
# drug-like compounds**, not the textbook pKa of the unsubstituted parent: a table of
# parent values prices the chemistry a medicinal chemist actually draws wrongly, since
# nearly every measured example of a class carries substituents that shift it.  Where
# the two disagree the parent value is in a comment.
#
# It is still one value per class, which is the crude part of logD: every aliphatic
# amine is treated as 8.8 whether its real value is 7 or 11.  A design whose ionisation
# matters deserves a measured pKa, which overrides this table completely -- give it as
# a ``pka_acid`` / ``pka_base`` field on the design (see :func:`ionisable`).
#
# Each entry is (label, SMARTS, which atom of the match ionises, pKa); the first
# pattern to claim a heavy atom keeps it, so the specific ones come before the general.

_NOT_AMIDE = ("!$(N~[!#6;!#1]);!$(N[CX3]=[OX1,SX1,NX2]);!$(N[SX4](=O)=O);"
              "!$(N[CX2]#N);!$(Na);!$([N+]);!$(N=*)")  # fmt: skip
_AMINE = f"[NX3;H0,H1,H2;{_NOT_AMIDE}]"
# electron-withdrawing groups, for the phenol and pyridine subclasses
_EWG = "[F,Cl,Br,I,$([N+](=O)[O-]),$(C#N),$([CX3]=[OX1]),$([SX4](=O)=O),$(C(F)(F)F)]"
_FREE_NH = "[NX3;H1,H2;!$(N[CX3]=O);!$(NS(=O)=O)]"

ACIDS: list[tuple[str, str, int, float]] = [
    ("sulfonic acid", "[SX4](=[OX1])(=[OX1])[OX2H1]", 3, -1.0),
    ("phosphonate", "[PX4](=[OX1])[OX2H1]", 2, 2.1),
    ("acylsulfonamide", "[SX4](=[OX1])(=[OX1])[NX3H1][CX3]=[OX1]", 3, 4.5),
    # both tautomers: the 1H pattern alone missed every 2H-tetrazole
    (
        "tetrazole",
        "[$([nX3H1]1[n][n][n][c]1),$([nX3H1]1[c][n][n][n]1),"
        "$([nX3H1]1[n][n][c][n]1),$([nX3H1]1[n][c][n][n]1)]",
        0,
        4.9,
    ),  # fmt: skip
    # a vinylogous acid is a much weaker acid than the enol it looks like: 6.9 measured
    ("vinylogous acid", "[OX2H1][c][c][c]=[OX1]", 0, 6.9),
    ("vinylogous acid", "[OX2H1][c][c]=[OX1]", 0, 6.9),
    ("enol (1,3-dicarbonyl)", "[OX2H1][CX3]=[CX3][CX3]=[OX1]", 0, 5.0),
    ("aryl carboxylic acid", "[c][CX3](=[OX1])[OX2H1]", 3, 3.9),
    ("carboxylic acid", "[CX3](=[OX1])[OX2H1]", 2, 4.4),
    # already drawn deprotonated -- a zwitterion whose carboxylate cannot be
    # neutralised without unbalancing a permanent cation.  Same site, same pKa
    ("carboxylate (as drawn)", "[CX3](=[OX1])[OX1H0-]", 2, 4.4),
    ("hydroxamic acid", "[OX2H1][NX3][CX3]=[OX1]", 0, 9.0),
    (
        "rhodanine / thiazolidinedione NH",
        "[NX3H1;R]([CX3;R]=[OX1,SX1])[CX3;R]=[OX1,SX1]",
        0,
        8.3,
    ),  # fmt: skip
    ("imide", "[CX3](=[OX1])[NX3H1][CX3]=[OX1]", 2, 9.1),
    (
        "N-acyl on a heteroarene",
        "[$([NX3H1]([CX3]=[OX1])c1[n,s,o][c,n][c,n][c,n]1),"
        "$([NX3H1]([CX3]=[OX1])c1ccncc1),$([NX3H1]([CX3]=[OX1])c1ncccn1)]",
        0,
        8.1,
    ),  # fmt: skip
    # a lactam NH inside a conjugated ring: 2-pyridone, uracil, quinazolin-4(3H)-one.
    # The parents are 9.5-11.6; the drug-like median is 7.5 because most measured
    # examples carry ring EWGs
    (
        "conjugated lactam NH",
        "[$([nX3H1;r5,r6]:[c]=[OX1]),$([nX3H1;r5,r6][c]=[OX1])]",
        0,
        7.5,
    ),  # fmt: skip
    (
        "sulfonamide (N-heteroaryl)",
        "[$([SX4](=[OX1])(=[OX1])[NX3H1]c1[n,s,o][c,n][c,n][c,n]1),"
        "$([SX4](=O)(=O)[NX3H1]c1ccncc1),$([SX4](=O)(=O)[NX3H1]c1ncccn1)]",
        0,
        7.0,
    ),  # fmt: skip
    ("sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3;H1,H2]", 3, 8.5),  # parent 10.1
    # an activated ring drops a phenol about 1.5 units off the parent's 10.0
    (
        "phenol (EWG-substituted)",
        f"[$([OX2H1][c]1[c][c][c]({_EWG})[c][c]1),$([OX2H1][c]1[c]({_EWG})[c][c][c][c]1),"
        "$([OX2H1][c]1[n,o,s][c,n][c,n][c,n]1)]",
        0,
        7.5,
    ),  # fmt: skip
    ("phenol", "[c][OX2H1]", 1, 8.8),  # parent 10.0
    # plain azole NH.  The parents are far weaker acids (imidazole 14.4, pyrazole 14.2,
    # benzimidazole 12.8, 1,2,4-triazole 10.3), but drug-like examples are activated:
    # 9.0 splits the difference and leaves the site under 2% ionised at pH 7.4
    ("azole NH", "[nX3H1;r5]", 0, 9.0),
    ("aryl thiol", "[c][SX2H1]", 1, 6.6),
    ("thiol", "[#6][SX2H1]", 1, 10.5),
]
BASES: list[tuple[str, str, int, float]] = [
    # guanidine 13.6 and amidine 11.6 are the aliphatic parents (arginine 12.5,
    # benzamidine 11.6).  Conjugation to an aromatic ring takes 3-4 units off, and
    # nearly every drug-like example is of that kind.  Every branch claims the central
    # carbon, the same atom the parent pattern claims, so only one of them can match
    ("guanidine (conjugated)", "[NX3;$(Na)][CX3](=[NX2])[NX3]", 1, 10.0),
    ("guanidine (conjugated)", "[NX3][CX3](=[NX2])[NX3;$(Na)]", 1, 10.0),
    ("guanidine (conjugated)", "[NX3][CX3](=[NX2;$(Na)])[NX3]", 1, 10.0),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]", 1, 13.6),
    (
        "amidine (cyclic / conjugated)",
        "[$([CX3;!$(C[NX3][CX3]=[NX2])](=[NX2])[NX3]);$([CX3;R]),$(Ca)]",
        0,
        9.0,
    ),  # fmt: skip
    ("amidine", "[CX3;!$(C[NX3][CX3]=[NX2])](=[NX2])[NX3]", 0, 11.6),
    # a nitrogen next to an electron-withdrawing group is a weaker base
    (
        "amine (next to EWG)",
        f"[NX3;H0,H1,H2;{_NOT_AMIDE};$(NCC=[OX1]),$(NC[OX2]),$(NCC#N)]",
        0,
        7.7,
    ),  # fmt: skip
    (
        "piperazine N (far N sulfonylated)",
        f"[NX3;H0,H1;{_NOT_AMIDE};$(N1CCN(CC1)[SX4](=O)=O)]",
        0,
        6.5,
    ),  # fmt: skip
    (
        "piperazine N (far N acylated)",
        f"[NX3;H0,H1;{_NOT_AMIDE};$(N1CCN(CC1)[CX3]=[OX1])]",
        0,
        7.7,
    ),  # fmt: skip
    ("piperazine N (far N aryl)", f"[NX3;H0,H1;{_NOT_AMIDE};$(N1CCN(CC1)a)]", 0, 8.2),
    ("morpholine N (substituted)", "[NX3;H0;$(N1CCOCC1)]", 0, 6.2),
    ("morpholine NH", "[NX3;H1;$(N1CCOCC1)]", 0, 8.4),
    ("amine (beta to an ether)", f"[NX3;H0,H1,H2;{_NOT_AMIDE};$(NCC[OX2H0])]", 0, 9.0),
    # 8.8, not 9.7: the median of measured drug-like amines.  9.7 is a simple
    # alkylamine, and it made every cation's logD about 0.9 too low
    ("amine", _AMINE, 0, 8.8),
    (
        "aniline",
        "[NX3;H0,H1,H2;$(Na);!$(N[CX3]=[OX1]);!$(N[SX4](=O)=O);!$([N+])]",
        0,
        3.5,  # parent 4.6
    ),  # fmt: skip
    # ring N subclasses, each from a measured split: an amino group para or ortho to a
    # pyridine N feeds it electron density and makes it a stronger base, a second ring
    # N takes density off it
    ("pyridine (4-amino)", f"[$([nX2;r6]:[c]:[c]:[c]-{_FREE_NH})]", 0, 6.2),
    ("pyridine (2-amino)", f"[$([nX2;r6]:[c]-{_FREE_NH})]", 0, 5.3),
    ("diazine N", "[$([nX2;r6]:[c]:[nX2;r6]),$([nX2;r6]:[nX2;r6])]", 0, 4.7),
    ("pyridine", "[nX2;r6]", 0, 4.9),
    # an imidazole-type ring N is much more basic than one sitting next to another ring
    # N (pyrazole, triazole, tetrazole), where the neighbour pulls density off it
    ("azole N (next to a ring N)", "[nX2;r5;$(n:[n])]", 0, 2.9),
    (
        "azole N (imidazole-type)",
        "[$([nX2;r5]1:[c]:[n,nH]:[c]:[c]1),$([nX2;r5]1:[c]:[c]:[n,nH]:[c]1)]",
        0,
        5.5,
    ),  # fmt: skip
    ("azole N", "[nX2;r5]", 0, 3.3),
]

_PATTERNS: dict[str, object] = {}


def _pattern(smarts: str):
    from rdkit import Chem

    if smarts not in _PATTERNS:
        p = Chem.MolFromSmarts(smarts)
        if p is None:  # a typo in the tables above, not in anything a user supplies
            raise ValueError(f"bad SMARTS in the pKa table: {smarts}")
        _PATTERNS[smarts] = p
    return _PATTERNS[smarts]


# --- the fitted models, as data walked with plain comparisons -------------------------
#
# Two random forests and a substructure table live in ``boonza/data/admet`` as gzipped
# JSON.  They are not boonza's own work: each is a published dataset, or a model fitted
# from one, redistributed under a licence that permits it (CC BY 4.0, CC0 1.0 and Apache
# 2.0 -- see that directory's README for the attribution each one requires).  They add
# no dependency: a tree is five flat arrays, and walking it is the six lines below.
#
# Delete any of the files and everything here degrades to what it did before: the
# SMARTS table's pKa, no extra sites, and Delaney's ESOL.

DATA = Path(__file__).resolve().parent.parent / "data" / "admet"

_MODELS: dict[str, dict | None] = {}


def _model(name: str) -> dict | None:
    """``{"features": [...], "info": {...}, "trees": [...]}``, or None when the file is
    not there.  Read once and kept: the pKa forest is 100 000 nodes."""
    if name not in _MODELS:
        import gzip
        import json

        try:
            _MODELS[name] = json.loads(gzip.decompress((DATA / f"{name}.json.gz").read_bytes()))
        except (OSError, ValueError):
            _MODELS[name] = None
    return _MODELS[name]


def _walk(tree, x: list[float]) -> float:
    """One decision tree, as (feature, threshold, left child, right child, value);
    ``-2`` in the feature array marks a leaf."""
    feature, threshold, left, right, value = tree
    node = 0
    while feature[node] != -2:
        node = left[node] if x[feature[node]] <= threshold[node] else right[node]
    return value[node]


def _forest(model: dict, x: list[float]) -> float:
    """The mean over every tree, which is what a random forest predicts."""
    trees = model["trees"]
    return sum(_walk(tree, x) for tree in trees) / len(trees)


def _soft_base(mol, atom: int, label: str) -> bool:
    """An aromatic ring N or an aniline N: a weak base, mostly neutral at pH 7.4, so it
    does not weaken a basic centre beside it."""
    return mol.GetAtomWithIdx(atom).GetIsAromatic() or label.startswith("aniline")


# The local environment of one ionisable atom, in the order the pKa forest was fitted
# in (its ``features`` list, which :func:`_pka_features` is checked against in the
# tests).  An electron-withdrawing group near an acid strengthens it and near a base
# weakens it; a donating one does the opposite, and how far away it is matters, which is
# why the counts come at two, three and five bonds.
_PKA_EWG = "[$([N+](=O)[O-]),$(C#N),$(S(=O)=O),$(C(F)(F)F),F,Cl,Br,I,$([CX3]=[OX1])]"
_PKA_EDG = "[$([OX2H1]),$([OX2][CX4]),$([NX3;H2,H1;!$(N[CX3]=O)]),$([CX4H3])]"


def _pka_features(mol, atom: int, kind: str, table_pka: float | None) -> list[float]:
    """What the pKa forest reads about one atom: the table's guess, the atom itself,
    the charges and substituents around it, and a little of the whole molecule."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdPartialCharges

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:  # a charge it cannot assign is not a reason to lose the site
        pass

    def charge(a) -> float:
        try:
            value = float(a.GetProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    distance = Chem.GetDistanceMatrix(mol)
    this = mol.GetAtomWithIdx(atom)

    def within(bonds: int) -> list:
        return [mol.GetAtomWithIdx(j) for j in range(mol.GetNumAtoms())
                if j != atom and distance[atom][j] <= bonds]  # fmt: skip

    near1, near2, near3 = within(1), within(2), within(3)
    ewg = [m[0] for m in mol.GetSubstructMatches(_pattern(_PKA_EWG))]
    edg = [m[0] for m in mol.GetSubstructMatches(_pattern(_PKA_EDG))]

    def count(hits: list[int], bonds: int) -> float:
        return float(sum(1 for j in hits if distance[atom][j] <= bonds))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    rings = mol.GetRingInfo()
    return [
        1.0 if kind == "acid" else 0.0,
        float(table_pka) if table_pka is not None else -1.0,
        1.0 if table_pka is not None else 0.0,
        float(this.GetAtomicNum()), float(this.GetDegree()),
        float(this.GetTotalNumHs()), float(this.GetFormalCharge()),
        float(this.GetIsAromatic()), float(this.IsInRing()),
        float(rings.MinAtomRingSize(atom) if this.IsInRing() else 0),
        charge(this), mean([charge(a) for a in near1]),
        min([charge(a) for a in near2], default=0.0),
        max([charge(a) for a in near2], default=0.0),
        mean([charge(a) for a in near3]),
        count(ewg, 2), count(ewg, 3), count(ewg, 5),
        count(edg, 2), count(edg, 3),
        float(sum(1 for a in near2 if a.GetIsAromatic())),
        float(sum(1 for a in near3 if a.GetIsAromatic())),
        float(sum(1 for a in near2 if a.GetSymbol() == "N")),
        float(sum(1 for a in near2 if a.GetSymbol() == "O")),
        float(sum(1 for a in near3 if a.GetSymbol() in ("N", "O", "S"))),
        float(len(near1)), float(len(near2)), float(len(near3)),
        Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol) / 100.0,
        Descriptors.MolWt(mol) / 100.0, float(mol.GetNumHeavyAtoms()) / 10.0,
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
    ]  # fmt: skip


_DIMORPHITE: list[tuple[str, object, dict[int, int], list]] | None = None


def _dimorphite_patterns():
    """Dimorphite-DL's ionisable substructures, compiled.  Each carries the atom map
    numbers its pKa values belong to, which is how the pattern says *which* atom
    ionises."""
    global _DIMORPHITE
    if _DIMORPHITE is None:
        import json

        from rdkit import Chem

        _DIMORPHITE = []
        try:
            raw = json.loads((DATA / "dimorphite_sites.json").read_text())["sites"]
        except (OSError, ValueError, KeyError):
            return _DIMORPHITE
        for site in raw:
            patt = Chem.MolFromSmarts(site["smarts"])
            if patt is None:
                continue
            maps = {a.GetAtomMapNum(): a.GetIdx() for a in patt.GetAtoms()
                    if a.GetAtomMapNum()}  # fmt: skip
            _DIMORPHITE.append((site["name"], patt, maps, site["values"]))
    return _DIMORPHITE


def _readable(name: str) -> str:
    """``Aromatic_nitrogen_unprotonated`` as "aromatic nitrogen unprotonated".  The
    names come from a file of SMARTS and read like code; the report is read by a
    chemist, beside the table's own labels."""
    import re

    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.replace("_", " "))
    return re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", text).lower()


def extra_sites(mol, sites, claimed=()):
    """Ionisable atoms Dimorphite-DL finds that the table above did not claim.

    The pKa carried out of here is Dimorphite's mean over its own training set, and it
    is only a placeholder: :func:`refine_sites` replaces it with the forest's value for
    that exact atom, which is the whole point of looking -- the means are a class
    average much like the table's, while the forest sees the substituents.

    ``claimed``: atoms to leave alone beyond the ones ``sites`` already occupies.  That
    is how a measured pKa stays the only value of its kind: the table's own sites were
    dropped in favour of the measurement, and the atoms they stood on must not be found
    again here -- as something else, at that.  Dimorphite calls any atom with a
    hydrogen an acid, so the NH of a secondary amine comes back as a weak acid unless
    the amine is accounted for."""
    found = list(sites)
    claimed = {s[2] for s in sites if s[2] is not None} | set(claimed)
    for name, patt, maps, values in _dimorphite_patterns():
        for match in mol.GetSubstructMatches(patt):
            for map_num, mean, _std in values:
                index = maps.get(map_num + 1, maps.get(map_num))
                if index is None or index >= len(match):
                    continue
                atom = match[index]
                if atom in claimed:
                    continue
                this = mol.GetAtomWithIdx(atom)
                # an atom carrying a hydrogen can lose it; a neutral nitrogen without
                # one can gain it.  Anything else is not an ionisable site here
                if this.GetTotalNumHs() >= 1:
                    kind = "acid"
                elif this.GetSymbol() == "N" and this.GetFormalCharge() == 0:
                    kind = "base"
                else:
                    continue
                claimed.add(atom)
                found.append((kind, f"{_readable(name)} [Dimorphite-DL]", atom, mean))
    return found


# Where the forest is allowed to speak: a table value inside this window.  Outside it
# the collection the forest was fitted on has very few measurements, and a forest cannot
# extrapolate past its training data -- it averages leaves, so it pulls anything extreme
# toward the middle.  Left to itself it prices butylguanidine, measured 13.6, at 8.9 and
# metformin, measured 12.4, at 5.8, which turns a permanent cation into a half-ionised
# one.  A class that is strong by definition is exactly where the table is reliable.
REFINE_RANGE: tuple[float, float] = (1.0, 12.0)


def refine_sites(mol, sites):
    """Each site's pKa replaced by the forest's value for that exact atom.

    A site with no atom index carries a measured pKa and is left alone: a measurement
    always beats a model.  So does the whole molecule if the forest cannot be loaded,
    or if anything about one atom goes wrong -- the table's value stands.  So does a
    site the table already calls very strong (:data:`REFINE_RANGE`)."""
    model = _model("pka_forest")
    if model is None or not sites:
        return sites
    low, high = REFINE_RANGE
    out = []
    for kind, label, atom, pka in sites:
        if atom is None or not low <= pka <= high:
            out.append((kind, label, atom, pka))
            continue
        try:
            value = _forest(model, _pka_features(mol, atom, kind, pka))
        except Exception:
            out.append((kind, label, atom, pka))
            continue
        out.append((kind, f"{label} [refined]", atom, round(value, 2)))
    return out


def neutralised(mol):
    """The neutral parent of a molecule drawn in one of its ionised forms.

    Every pattern in :data:`ACIDS` and :data:`BASES` matches the *protonated* acid and
    the *unprotonated* base -- the neutral species, which is what a pKa is a property
    of, and several of the base patterns exclude ``[N+]`` outright.  A design that comes
    out of a prepared structure is not drawn that way: an amine arrives as ``[NH3+]``
    and a carboxylic acid as ``[O-]``, because that is what it is in the simulation it
    was built for.  Matched as drawn, such a design shows *no* ionisable site at all,
    and then Henderson-Hasselbalch has nothing to correct, so its logD comes back
    exactly equal to its logP -- a number that looks computed and is not.

    RDKit's ``Uncharger`` adds and removes the protons that balance, and leaves alone a
    charge that nothing can balance: a quaternary ammonium stays ``+1`` (so the
    permanent-charge caveat still fires) and a betaine stays a zwitterion (whose
    carboxylate the ``"carboxylate (as drawn)"`` class is there to catch).  The
    molecule is copied, never neutralised in place: the caller is still drawing it.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    try:
        # the Uncharger narrates itself at rdApp.info ("Running Uncharger", "Removed
        # positive charge"), and this runs on every property of every design: blocked
        # here rather than relying on the level being off, since anything in the process
        # may turn it on
        with rdBase.BlockLogs():
            neutral = rdMolStandardize.Uncharger().uncharge(Chem.Mol(mol))
    except Exception:  # a valence the standardiser rejects: read it as drawn
        return mol
    return mol if neutral is None else neutral


def ionisable(mol, measured=None, refine: bool = True) -> list[tuple[str, str, int, float]]:
    """Every ionisable group of ``mol``: (acid or base, label, atom, pKa).  A heavy
    atom belongs to one group only, the first pattern that claims it, so a phenol in
    a carboxylic acid's molecule is not counted twice.

    ``measured``: pKa values measured for *this* molecule, as ``{"acid": [2.2],
    "base": [9.0, 12.5]}``.  A measured value beats any prediction, so where one is
    given it replaces the table's sites of that kind entirely and is reported with no
    atom (``None``): the number is known, which atom carries it need not be.  Give a
    design's values as a ``pka_acid`` / ``pka_base`` field on its SDF record.

    The table locates the groups; :func:`extra_sites` adds the ones it misses and
    :func:`refine_sites` then prices each one by its own substituents rather than by
    its class, so a label reading ``[refined]`` means the number came from the forest
    and the table's value was only its starting point.  ``refine=False`` asks for what
    the table alone says, which is what the report falls back to anyway when the data
    files are not installed."""
    from rdkit import Chem

    found: list[tuple[str, str, int, float]] = []
    taken: set[int] = set()
    for kind, table in (("acid", ACIDS), ("base", BASES)):
        # a measured value replaces this kind's sites, but the atoms they stand on are
        # still claimed: that is what keeps the other kind, and the substructure search
        # below, off the group the measurement is about
        given = bool((measured or {}).get(kind))
        if given:
            found += [(kind, f"measured {kind}", None, float(p)) for p in measured[kind]]
        for label, smarts, which, pka in table:
            for match in mol.GetSubstructMatches(_pattern(smarts)):
                atom = match[which]
                if atom in taken:
                    continue
                taken.add(atom)
                if not given:
                    found.append((kind, label, atom, pka))
    # A second basic centre near a stronger one is much weaker than on its own
    # (piperazine: 9.8 and 5.6, not 9.8 twice), and counting them as independent drags
    # logD far too low.  Only an *aliphatic* base counts as the stronger centre: the
    # drop comes from the positive charge it carries once protonated, and a pyridine,
    # azole or aniline is mostly neutral at pH 7.4 so it has no charge to lend.  Before
    # that proviso the second ring N of every pyrimidine, purine and quinazoline was
    # pushed to about 1.2 where the measured median is 4.4.  An aromatic N next to a
    # protonated amine is still weakened, as in 2-picolylamine.
    bases = sorted((k for k, s in enumerate(found) if s[0] == "base" and s[2] is not None),
                   key=lambda k: -found[k][3])  # fmt: skip
    strong: list[int] = []
    distance = Chem.GetDistanceMatrix(mol) if bases else None
    for k in bases:
        kind, label, atom, pka = found[k]
        if any(distance[atom][other] <= 3 for other in strong):
            found[k] = (kind, f"{label} (second basic centre)", atom, pka - 4.0)
        elif not _soft_base(mol, atom, label):
            strong.append(atom)
    if not refine:
        return found
    # and now the two fitted pieces, in this order: find the sites the table missed,
    # then give every site that has an atom the forest's pKa rather than its class's
    return refine_sites(mol, extra_sites(mol, found, claimed=taken))


# How much less an ion pair partitions into octanol than the neutral molecule does.
#
# Textbook Henderson-Hasselbalch treats an ionised molecule as not partitioning at all,
# which has no floor: a strongly ionised design's logD keeps falling without limit, and
# every cation comes out roughly 0.9 log units too low.  An ion does partition, as an
# ion pair, and the plateau between logP and logD is about 3 log units for a base and
# about 4 for an acid -- the published "diff 3-4" approximation (Scherrer, 1984; Avdeef,
# *Absorption and Drug Development*, ch. 4).
#
# Set either to ``None`` for textbook Henderson-Hasselbalch on that charge type.
ION_PAIR_DROP_ACID: float | None = 4.0
ION_PAIR_DROP_BASE: float | None = 3.0


def log_d(logp: float, sites, ph: float = 7.4, charge: int = 0) -> float:
    """Crippen logP corrected for whatever is ionised at ``ph``.

    Henderson-Hasselbalch gives how much of the molecule is neutral::

        acid = SUM_acids 10^(pH - pKa)     base = SUM_bases 10^(pKa - pH)
        neutral fraction = 1 / (1 + acid + base)

    and what partitions is the neutral fraction plus whatever the ionised fractions
    manage as ion pairs (:data:`ION_PAIR_DROP_ACID`, :data:`ION_PAIR_DROP_BASE`)::

        logD = logP + log10(neutral + acid/total * 10^-drop + base/total * 10^-drop)

    ``charge`` is a *permanent* formal charge -- a quaternary ammonium, a sulfonate --
    which no pKa governs and which therefore appears in none of the sites above.  Such a
    molecule is fully ionised at every pH, so it can never partition better than the ion
    pair does: the plateau ``logP - drop`` becomes a ceiling on the result, and whatever
    else is ionisable can only push it lower.  Without that, a permanently charged
    design had nothing for Henderson-Hasselbalch to correct and its logD came back equal
    to its logP, some 3 log units too lipophilic.
    """
    acid = base = 0.0
    for kind, _label, _atom, pka in sites:
        # a long way either side of the pKa the exponent overflows; it is saturated
        # instead, since by then the group is fully ionised and the sum is what it is
        power = min(ph - pka if kind == "acid" else pka - ph, 30.0)
        if kind == "acid":
            acid += 10.0**power
        else:
            base += 10.0**power
    total = 1.0 + acid + base
    if ION_PAIR_DROP_ACID is None and ION_PAIR_DROP_BASE is None:
        return logp - math.log10(total)
    share = 1.0 / total
    if ION_PAIR_DROP_ACID is not None:
        share += (acid / total) * 10.0**-ION_PAIR_DROP_ACID
    if ION_PAIR_DROP_BASE is not None:
        share += (base / total) * 10.0**-ION_PAIR_DROP_BASE
    value = logp + math.log10(share)
    # a permanent charge: the ion-pair plateau is a ceiling, not a correction to add
    drop = ION_PAIR_DROP_BASE if charge > 0 else ION_PAIR_DROP_ACID
    if charge and drop is not None:
        value = min(value, logp - drop)
    return value


def ionised_form(sites, ph: float = 7.4, charge: int = 0) -> str:
    """What the molecule mostly is at ``ph``: neutral, anion, cation or zwitterion.
    A permanent formal charge (a quaternary nitrogen) settles it on its own."""
    anion = any(k == "acid" and pka < ph for k, _l, _a, pka in sites)
    cation = any(k == "base" and pka > ph for k, _l, _a, pka in sites)
    if charge > 0:
        cation = True
    if charge < 0:
        anion = True
    if anion and cation:
        return "zwitterion"
    return "anion" if anion else "cation" if cation else "neutral"


def caveats(sites, form: str, charge: int) -> list[str]:
    """Where this logD should not be believed, in plain words.  Worth printing: the
    cases below are exactly the ones where the number looks precise and is not."""
    out = []
    if charge:
        out.append(f"permanently charged ({charge:+d}): it is ionised at every pH, so "
                   "logD rests entirely on the ion-pair term and is a lower bound")  # fmt: skip
    if form == "zwitterion":
        out.append("zwitterion: both centres are corrected for, but its ion pair is "
                   "intramolecular and partitions better than the model allows")  # fmt: skip
    if any((k == "base" and pka >= 11) or (k == "acid" and pka <= 1) for k, _l, _a, pka in sites):
        out.append("a very strong acid or base: fully ionised at this pH, so logD is "
                   "the ion-pair plateau rather than a Henderson-Hasselbalch number")  # fmt: skip
    if any(atom is None for _k, _l, atom, _p in sites):
        out.append("a measured pKa was given and used in place of the table")
    return out


# --- solubility, skin, the egg --------------------------------------------------------


def esol(logp: float, mw: float, rotatable: int, aromatic_fraction: float) -> float:
    """Delaney's ESOL (2004): log S from logP, weight, flexibility and aromaticity."""
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotatable - 0.74 * aromatic_fraction


def ali_logs(logp: float, tpsa: float) -> float:
    """Ali et al. (2012), their model 03: log S from logP and TPSA alone."""
    return -1.0377 * logp - 0.0210 * tpsa + 0.4488


# The fitted alternative to both.  ESOL and Ali are linear in logP and solubility is
# not: on a scaffold split of AqSolDB, ESOL reaches RMSE 1.68 and a ridge regression on
# the descriptors below only 1.57, while a small random forest reaches 1.31.  The two
# published equations stay exactly as they are and are still printed -- a fitted number
# and two published ones disagreeing is itself worth seeing.
SOLUBILITY_FEATURES = ("wlogp", "wlogp_sq", "mw100", "tpsa100", "rotatable",
                       "aromatic_fraction", "heavy10", "hbd", "hba", "fsp3",
                       "aromatic_rings", "rings", "is_acid", "is_base", "molmr10",
                       "halogens", "wlogp_x_tpsa")  # fmt: skip


def solubility_features(mol, sites=None, ph: float = 7.4) -> list[float]:
    """What the solubility forest reads, in :data:`SOLUBILITY_FEATURES` order: size,
    polarity, flexibility, aromaticity and whether the molecule is charged at ``ph``."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    if sites is None:
        sites = ionisable(mol)
    form = ionised_form(sites, ph, Chem.GetFormalCharge(mol))
    return [
        logp, logp * logp, Descriptors.MolWt(mol) / 100.0, tpsa / 100.0,
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        aromatic / heavy if heavy else 0.0, heavy / 10.0,
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(form in ("anion", "zwitterion")),
        float(form in ("cation", "zwitterion")),
        Descriptors.MolMR(mol) / 10.0,
        float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))),
        logp * tpsa / 100.0,
    ]  # fmt: skip


def fitted_logs(mol, sites=None, ph: float = 7.4) -> float | None:
    """Fitted log S in mol/L, or None when the forest is not there."""
    model = _model("logs_forest")
    if model is None:
        return None
    try:
        return _forest(model, solubility_features(mol, sites, ph))
    except Exception:
        return None


def solubility_class(log_s: float) -> str:
    """The scale SwissADME prints beside a log S, in mol/L."""
    for limit, name in ((0, "highly soluble"), (-2, "very soluble"), (-4, "soluble"),
                        (-6, "moderately soluble"), (-10, "poorly soluble")):  # fmt: skip
        if log_s >= limit:
            return name
    return "insoluble"


def log_kp(logp: float, mw: float) -> float:
    """Potts & Guy skin permeation, in cm/s (the equation gives cm/h).  More negative
    is less skin permeant."""
    return (-2.7 + 0.71 * logp - 0.0061 * mw) - math.log10(3600.0)


# The BOILED-Egg: two ellipses in TPSA against WLOGP.  Inside the white one, passive
# gastrointestinal absorption is likely; inside the yolk, so is passive brain
# penetration.  Daina & Zoete give each region by the range it spans -- white
# TPSA 0.0 to 142.1 with WLOGP -2.3 to 6.8, yolk TPSA -2.9 to 79.1 with WLOGP 0.4 to
# 6.0 -- which is the bounding box of an ellipse centred at the middle of each range
# with half the range as its semi-axis.  The published ellipses are tilted by about a
# degree, too little to change a verdict, so they are taken upright here.
# (center TPSA, semi-axis TPSA, center WLOGP, semi-axis WLOGP)
EGG_WHITE = (71.05, 71.05, 2.25, 4.55)
EGG_YOLK = (38.10, 41.00, 3.20, 2.80)


def _in_ellipse(tpsa: float, logp: float, ellipse) -> bool:
    cx, ax, cy, ay = ellipse
    return ((tpsa - cx) / ax) ** 2 + ((logp - cy) / ay) ** 2 <= 1.0


def boiled_egg(tpsa: float, logp: float) -> tuple[str, str]:
    """(gastrointestinal absorption, brain penetration), each "high"/"low" or
    "yes"/"no", from where the molecule sits in the egg."""
    return ("high" if _in_ellipse(tpsa, logp, EGG_WHITE) else "low",
            "yes" if _in_ellipse(tpsa, logp, EGG_YOLK) else "no")  # fmt: skip


# --- druglikeness filters -------------------------------------------------------------
#
# Each returns the list of what it fails, so an empty list is a pass and the caller can
# say which rule was broken rather than only that one was.  These use RDKit's Crippen
# logP; SwissADME drives several of them with XLOGP3 instead, so a borderline molecule
# can fall on the other side of a cutoff there.


def lipinski(mw: float, logp: float, hbd: int, hba: int) -> list[str]:
    """Lipinski's rule of five.  One violation is allowed by the rule itself."""
    fails = []
    if mw > 500:
        fails.append("MW > 500")
    if logp > 5:
        fails.append("logP > 5")
    if hbd > 5:
        fails.append("HBD > 5")
    if hba > 10:
        fails.append("HBA > 10")
    return fails


def ghose(mw: float, logp: float, mr: float, heavy: int) -> list[str]:
    fails = []
    if not 160 <= mw <= 480:
        fails.append("MW outside 160-480")
    if not -0.4 <= logp <= 5.6:
        fails.append("logP outside -0.4-5.6")
    if not 40 <= mr <= 130:
        fails.append("molar refractivity outside 40-130")
    if not 20 <= heavy <= 70:
        fails.append("heavy atoms outside 20-70")
    return fails


def veber(rotatable: int, tpsa: float) -> list[str]:
    fails = []
    if rotatable > 10:
        fails.append("rotatable bonds > 10")
    if tpsa > 140:
        fails.append("TPSA > 140")
    return fails


def egan(logp: float, tpsa: float) -> list[str]:
    fails = []
    if logp > 5.88:
        fails.append("logP > 5.88")
    if tpsa > 131.6:
        fails.append("TPSA > 131.6")
    return fails


def muegge(mw: float, logp: float, tpsa: float, rings: int, carbons: int,
           heteroatoms: int, rotatable: int, hba: int, hbd: int) -> list[str]:  # fmt: skip
    fails = []
    if not 200 <= mw <= 600:
        fails.append("MW outside 200-600")
    if not -2 <= logp <= 5:
        fails.append("logP outside -2-5")
    if tpsa > 150:
        fails.append("TPSA > 150")
    if rings > 7:
        fails.append("rings > 7")
    if carbons <= 4:
        fails.append("carbons <= 4")
    if heteroatoms <= 1:
        fails.append("heteroatoms <= 1")
    if rotatable > 15:
        fails.append("rotatable bonds > 15")
    if hba > 10:
        fails.append("HBA > 10")
    if hbd > 5:
        fails.append("HBD > 5")
    return fails


def leadlike(mw: float, logp: float, rotatable: int) -> list[str]:
    """Room to grow: a lead is smaller and simpler than the drug it becomes."""
    fails = []
    if not 250 <= mw <= 350:
        fails.append("MW outside 250-350")
    if logp > 3.5:
        fails.append("logP > 3.5")
    if rotatable > 7:
        fails.append("rotatable bonds > 7")
    return fails


def bioavailability(form: str, tpsa: float, lipinski_fails: int) -> float:
    """The Abbott bioavailability score, from charge, TPSA and the rule of five: the
    probability of at least 10% oral bioavailability in rat."""
    if form in ("anion", "zwitterion"):
        return 0.56 if tpsa <= 150 else 0.11
    return 0.55 if lipinski_fails <= 1 else 0.17


# --- alerts ---------------------------------------------------------------------------

_CATALOGS: dict[str, object] = {}


def _catalog(which: str):
    """RDKit's own filter catalogs: PAINS (the frequent hitters) and Brenk (groups
    better kept out of a screening library)."""
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    if which not in _CATALOGS:
        params = FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, which))
        _CATALOGS[which] = FilterCatalog(params)
    return _CATALOGS[which]


def alerts(mol, which: str) -> list[str]:
    """What ``mol`` matches in a catalog, by name.  A hit is a substructure worth a
    second look, not a verdict: plenty of real drugs carry one."""
    out = []
    for match in _catalog(which).GetMatches(mol):
        name = match.GetDescription().strip()
        out.append(name or "unnamed alert")
    return out


def sascore(mol) -> float | None:
    """Ertl & Schuffenhauer synthetic accessibility, 1 (easy) to 10 (hard).  It ships
    with RDKit as a contributed script rather than in the library, so it is imported
    by path and left out if that ever moves."""
    import os
    import sys

    from rdkit.Chem import RDConfig

    path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    try:
        if path not in sys.path:
            sys.path.append(path)
        import sascorer  # type: ignore[import-not-found]

        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


# --- everything, for one molecule -----------------------------------------------------


def _round(x: float, places: int = 2) -> float:
    """Rounded, and never -0.0, which reads as a different number than it is."""
    v = round(float(x), places)
    return 0.0 if v == 0 else v


def properties(mol, ph: float = 7.4, measured=None) -> dict:
    """Every property of one RDKit molecule, flat, ready for a table or a report.
    ``ph``: where logD and the ionised form are worked out.  ``measured``: pKa values
    measured for this molecule, which override the table -- see :func:`ionisable`.

    The keys that are numbers are plain floats and ints so they sort and plot; the
    filters are "pass" or what they fail; ``caveats`` is a list, empty when the
    numbers can be read as they stand."""
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.RemoveHs(mol)  # the descriptors count hydrogens themselves
    mol = neutralised(mol)  # and the pKa table wants the neutral parent
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    charge = Chem.GetFormalCharge(mol)

    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hbd, hba = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
    mr = Crippen.MolMR(mol)

    sites = ionisable(mol, measured)
    form = ionised_form(sites, ph, charge)
    logd = log_d(logp, sites, ph, charge)
    gi, bbb = boiled_egg(tpsa, logp)
    s_esol = esol(logp, mw, rotatable, aromatic / heavy if heavy else 0.0)
    s_ali = ali_logs(logp, tpsa)
    s_fit = fitted_logs(mol, sites, ph)

    def verdict(fails: list[str]) -> str:
        return "pass" if not fails else "fails " + ", ".join(fails)

    lip = lipinski(mw, logp, hbd, hba)
    out = {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "MW": _round(mw),
        "heavy atoms": heavy,
        "aromatic atoms": aromatic,
        "fraction Csp3": _round(Descriptors.FractionCSP3(mol)),
        "rotatable bonds": rotatable,
        "HBD": hbd,
        "HBA": hba,
        "molar refractivity": _round(mr),
        "TPSA": _round(tpsa, 1),
        "formal charge": charge,
        "cLogP": _round(logp),
        "logD": _round(logd),
        "pH": ph,
        "ionised form": form,
        "ionisable groups": "; ".join(f"{label} {pka:g}" for _k, label, _a, pka in sites) or "none",
        "log S (ESOL)": _round(s_esol),
        # the class follows the best number there is, which is the forest where it
        # could be loaded and Delaney's equation where it could not
        "solubility": solubility_class(s_esol if s_fit is None else s_fit),
        "log S (Ali)": _round(s_ali),
        "log Kp (skin)": _round(log_kp(logp, mw)),
        "GI absorption": gi,
        "BBB permeant": bbb,
        "QED": _round(QED.qed(mol)),
        "Lipinski": verdict(lip),
        "Lipinski violations": len(lip),
        "Ghose": verdict(ghose(mw, logp, mr, heavy)),
        "Veber": verdict(veber(rotatable, tpsa)),
        "Egan": verdict(egan(logp, tpsa)),
        "Muegge": verdict(
            muegge(
                mw,
                logp,
                tpsa,
                rdMolDescriptors.CalcNumRings(mol),
                symbols.count("C"),
                sum(1 for s in symbols if s not in ("C", "H")),
                rotatable,
                hba,
                hbd,
            )
        ),  # fmt: skip
        "lead-like": verdict(leadlike(mw, logp, rotatable)),
        "bioavailability score": bioavailability(form, tpsa, len(lip)),
        "caveats": caveats(sites, form, charge),
    }
    if s_fit is not None:
        out["log S (fitted)"] = _round(s_fit)
    pains, brenk = alerts(mol, "PAINS"), alerts(mol, "BRENK")
    out["PAINS"] = len(pains)
    out["PAINS alerts"] = "; ".join(pains) or "none"
    out["Brenk"] = len(brenk)
    out["Brenk alerts"] = "; ".join(brenk) or "none"
    score = sascore(mol)
    if score is not None:
        out["synthetic accessibility"] = _round(score)
    out["triage"], reasons = triage(out)
    out["triage reasons"] = "; ".join(reasons) or "nothing breached"
    return out


# which of the above are worth a column in the series' table: the numbers a chemist
# ranks designs by.  The rest are in the report, where there is room for them
TABLE_COLUMNS = ("MW", "heavy atoms", "cLogP", "logD", "TPSA", "HBD", "HBA",
                 "rotatable bonds", "QED",
                 "synthetic accessibility", "GI absorption", "BBB permeant",
                 "Lipinski violations", "PAINS", "triage")  # fmt: skip


# --- triage -----------------------------------------------------------------------------
#
# One verdict per design, so a series of forty rows can be read as a shortlist.  Nothing
# here is new chemistry: every rule reads a property computed above.  What it adds is a
# single answer and, more importantly, the reason for it -- a filtered list that hides
# the rule that produced it is worse than no filter at all.
#
# A design fails on any "fail" rule and is flagged for review on any "review" rule.  The
# thresholds are the conventional ones; they are defaults, not findings, and the page
# lets them be moved.
#
# (key, label, which property it reads, comparison, default threshold, level)
TRIAGE_FIELDS: list[tuple[str, str, str, str, float, str]] = [
    ("lipinski", "Lipinski violations", "Lipinski violations", ">=", 2, "fail"),
    ("pains", "PAINS alerts", "PAINS", ">=", 1, "fail"),
    ("tpsa", "TPSA", "TPSA", ">", 140.0, "fail"),
    ("brenk", "Brenk alerts", "Brenk", ">=", 1, "review"),
    ("qed", "QED below", "QED", "<", 0.30, "review"),
    ("logd_low", "logD below", "logD", "<", -1.0, "review"),
    ("logd_high", "logD above", "logD", ">", 4.0, "review"),
    ("gi", "GI absorption not high", "GI absorption", "is not", "high", "review"),
    ("sascore", "synthetic accessibility above", "synthetic accessibility", ">", 6.0,
     "review"),
]  # fmt: skip

TRIAGE_LEVELS = ("pass", "review", "fail")


def triage(props: dict, limits: dict | None = None) -> tuple[str, list[str]]:
    """``props`` (one design's :func:`properties`) read as a verdict: ("pass", "review"
    or "fail", the rules it breached).  ``limits``: thresholds to move, keyed as
    :data:`TRIAGE_FIELDS` keys.

    A property that could not be computed is skipped rather than counted against the
    design: an absent SA score is not a bad SA score."""
    worst, why = 0, []
    for key, label, field, operator, default, level in TRIAGE_FIELDS:
        limit = (limits or {}).get(key, default)
        value = props.get(field)
        if value is None:
            continue
        if operator == "is not":
            # compared without case, so a verdict written "High" rather than "high"
            # cannot quietly make the rule fire on every design
            hit = str(value).lower() != str(limit).lower()
        elif not isinstance(value, (int, float)):
            continue
        else:
            hit = (value >= limit if operator == ">=" else
                   value > limit if operator == ">" else value < limit)  # fmt: skip
        if hit:
            why.append(label if operator == "is not" else f"{label} {value:g}")
            worst = max(worst, 2 if level == "fail" else 1)
    return TRIAGE_LEVELS[worst], why


# --- metabolic soft spots --------------------------------------------------------------
#
# Where a molecule is likely to be attacked, from the chemistry of the group alone.
#
# This is NOT SMARTCyp and must not be read as it: SMARTCyp ranks every atom by a
# DFT activation energy from a published rule table, and says which site goes first.
# What follows only recognises the textbook liabilities -- the groups a medicinal
# chemist already looks for when a compound turns over too fast -- and says where
# they are.  It gives no order and no rate: two molecules both carrying "N-dealkylation"
# are not thereby equally labile.  Blocking a flagged site (deuterium, fluorine, a
# ring) is the usual response, and a real answer needs microsomes.
#
# (label, SMARTS, which atoms of the match are the site, what happens there)
SOFT_SPOTS: list[tuple[str, str, tuple[int, ...], str]] = [
    (
        "N-dealkylation",
        f"[NX3;H0,H1;{_NOT_AMIDE}][CH3,CH2;!$(C=O)]",
        (1,),
        "CYP takes the alkyl off the nitrogen; the commonest route of all",
    ),  # fmt: skip
    (
        "O-dealkylation",
        "[c,C][OX2;H0][CH3,CH2;!$(C=O)]",
        (2,),
        "CYP takes the alkyl off the oxygen, leaving the phenol or alcohol",
    ),  # fmt: skip
    (
        "benzylic oxidation",
        "[CH1,CH2,CH3;!$(C=O);!$(C[OX2,NX3]);$(C[a])]",
        (0,),
        "a carbon next to a ring: hydroxylated easily, the C-H being weak",
    ),  # fmt: skip
    (
        "allylic oxidation",
        "[CH1,CH2,CH3;!$(C=O);$(CC=C)]",
        (0,),
        "a carbon next to a double bond, hydroxylated for the same reason",
    ),  # fmt: skip
    (
        "aromatic hydroxylation",
        "[cH;$(c1ccccc1);$(c:c[OX2,NX3,CH3])]",
        (0,),
        "an electron-rich ring position, hydroxylated through an epoxide",
    ),  # fmt: skip
    (
        "terminal alkyl (omega-1)",
        "[CH3][CH2][CH2]",
        (1,),
        "the carbon next to the end of a chain, the usual place a chain is hit",
    ),  # fmt: skip
    (
        "S-oxidation",
        "[SX2;$(S(-[#6])-[#6]);!$(S(=O))]",
        (0,),
        "a thioether goes to the sulfoxide and on to the sulfone, readily",
    ),  # fmt: skip
    (
        "N-oxidation",
        f"[NX3;H0;{_NOT_AMIDE};!$(Na)]",
        (0,),
        "a tertiary amine gives the N-oxide",
    ),  # fmt: skip
    (
        "alkene epoxidation",
        "[CX3H1,CX3H2]=[CX3;!$(C=O)]",
        (0, 1),
        "an epoxide, which may go on to bind covalently: watch it",
    ),  # fmt: skip
    (
        "aldehyde oxidation",
        "[CX3H1]=[OX1]",
        (0,),
        "oxidised to the acid quickly, by aldehyde oxidase as well as CYP",
    ),  # fmt: skip
    (
        "ester hydrolysis",
        "[CX3](=[OX1])[OX2][#6]",
        (0,),
        "esterases cut it, often before CYP gets near: a prodrug handle",
    ),  # fmt: skip
    (
        "amide hydrolysis",
        "[CX3](=[OX1])[NX3;H1,H2]",
        (0,),
        "amidases cut it, far more slowly than an ester",
    ),  # fmt: skip
]


def soft_spots(mol) -> list[dict]:
    """Where ``mol`` is likely to be oxidised, by group.  Each is {label, atoms, note};
    ``atoms`` are the heavy atoms of the site, for highlighting.

    A flag is a liability to look at, not a rate and not an order -- see the note on
    :data:`SOFT_SPOTS`.  Nothing here predicts which site goes first, which is what
    SMARTCyp does and this does not.  :func:`phase2_sites` asks the complementary
    question: which groups are handles for conjugation."""
    out = []
    for label, smarts, which, note in SOFT_SPOTS:
        atoms: set[int] = set()
        for match in mol.GetSubstructMatches(_pattern(smarts)):
            atoms.update(match[k] for k in which if k < len(match))
        if atoms:
            out.append({"label": label, "atoms": sorted(atoms), "note": note})
    return out


# --- phase II: where a conjugating enzyme can attach -----------------------------------
#
# :data:`SOFT_SPOTS` above asks where CYP oxidises a molecule.  This asks the
# complementary question: which groups are already handles for conjugation, so the
# molecule can be cleared without being oxidised first.  A compound with a free phenol
# is often cleared by UGT whatever its CYP liabilities, and a report that lists only
# oxidation misses the route that actually dominates.
#
# This is a substructure table, not a model.  It says "this group is a substrate type
# for this enzyme family", which is a statement about chemistry, and it is written from
# the standard descriptions of each family's substrates.  It does NOT predict rate,
# extent, or which route wins in vivo -- those depend on tissue expression, competing
# routes and dose.  Read a hit as "this route is open", not "this route dominates".
#
# Two things are worth more than the site list itself: the groups whose conjugate is
# *reactive* (an acyl glucuronide can rearrange and acylate protein), and the enzymes
# that are polymorphic in people, where a slow metaboliser reaches a quite different
# exposure from the same dose.
#
# (enzyme family, group, SMARTS, which atom of the match reacts, note)
PHASE2_RULES: list[tuple[str, str, str, int, str]] = [
    # UGT: glucuronidation, usually the highest-capacity route
    ("UGT", "phenol O-glucuronide", "[c][OX2H1]", 1, ""),
    ("UGT", "aliphatic OH O-glucuronide", "[CX4][OX2H1]", 1, ""),
    ("UGT", "acyl glucuronide (COOH)", "[CX3](=[OX1])[OX2H1]", 0,
     "an acyl glucuronide can rearrange and acylate protein: reactive metabolite risk"),
    ("UGT", "N-glucuronide (arylamine)",
     "[NX3;H1,H2;$(Na);!$(N[CX3]=[OX1]);!$(N[SX4](=O)=O)]", 0, ""),
    ("UGT", "N-glucuronide (aliphatic amine)",
     "[NX3;H1,H2;!$(Na);!$(N[CX3]=[OX1]);!$(N[SX4](=O)=O);!$(N~[!#6;!#1])]", 0, ""),
    ("UGT", "N-glucuronide (acidic NH)",
     "[$([nX3H1;r5]),$([NX3H1]([SX4](=O)=O)a)]", 0, ""),
    ("UGT", "S-glucuronide (thiol)", "[#6][SX2H1]", 1, ""),
    # SULT: sulfation.  Low capacity, and it saturates before UGT does
    ("SULT", "phenol O-sulfate", "[c][OX2H1]", 1,
     "a low-capacity route: it saturates with dose, shifting the load to UGT"),
    ("SULT", "aliphatic OH O-sulfate", "[CX4;H1,H2][OX2H1]", 1, ""),
    ("SULT", "N-sulfate (arylamine)",
     "[NX3;H1,H2;$(Na);!$(N[CX3]=[OX1]);!$(N[SX4](=O)=O)]", 0, ""),
    ("SULT", "O-sulfate of an N-OH", "[NX3][OX2H1]", 1,
     "an N-OH sulfate can ionise to a nitrenium ion: reactive metabolite risk"),
    # NAT: N-acetylation, polymorphic (slow and fast acetylators)
    ("NAT", "arylamine N-acetylation",
     "[NX3;H2;$(Na);!$(N[CX3]=[OX1]);!$(N[SX4](=O)=O)]", 0,
     "NAT2 is polymorphic: a slow acetylator reaches a higher exposure"),
    ("NAT", "hydrazine N-acetylation", "[NX3;H1,H2][NX3;H1,H2]", 0,
     "NAT2 is polymorphic: a slow acetylator reaches a higher exposure"),
    ("NAT", "sulfonamide N4-acetylation",
     "[NX3;H2;$(c1ccc([SX4](=O)(=O)[NX3])cc1)]", 0, ""),
    # COMT: O-methylation of catechols
    ("COMT", "catechol O-methylation", "[OX2H1][c][c][OX2H1]", 0,
     "a catechol is also oxidised to a quinone: reactive metabolite risk"),
    # GST: glutathione conjugation.  These are electrophiles already
    ("GST", "epoxide", "[CX4]1[OX2][CX4]1", 0,
     "a direct electrophile: reactive metabolite risk"),
    ("GST", "Michael acceptor", "[CX3]=[CX3][CX3]=[OX1]", 0,
     "a direct electrophile: reactive metabolite risk"),
    ("GST", "quinone", "[OX1]=[CX3]1[CX3]=[CX3][CX3](=[OX1])[CX3]=[CX3]1", 0,
     "a direct electrophile: reactive metabolite risk"),
    ("GST", "SNAr-activated aryl halide",
     "[c]([F,Cl,Br,I])[c,n][c,n][$([NX3+](=O)[OX1-]),$(C#N),$(S(=O)=O)]", 0, ""),
    ("GST", "alkyl halide", "[CX4;H1,H2][Cl,Br,I]", 0, ""),
    ("GST", "isothiocyanate", "[NX2]=[CX2]=[SX1]", 0, ""),
    # TPMT and the other thiol methyltransferases
    ("TPMT", "S-methylation (thiol)", "[#6][SX2H1]", 1, "TPMT is polymorphic"),
]  # fmt: skip

#: the order the families are read in: capacity first, then the reactive ones
PHASE2_FAMILIES = ("UGT", "SULT", "NAT", "COMT", "GST", "TPMT")


def phase2_sites(mol) -> list[dict]:
    """``mol``'s conjugation handles, one entry per enzyme family that has any:
    {family, atoms, groups, notes}.

    An atom is reported once per family, so a phenol that both UGT and SULT can take
    appears under each but not twice under either.  See the note on
    :data:`PHASE2_RULES`: a hit means the route is open, not that it dominates."""
    seen: dict[str, dict] = {}
    claimed: set[tuple[str, int]] = set()
    for family, group, smarts, which, note in PHASE2_RULES:
        for match in mol.GetSubstructMatches(_pattern(smarts)):
            if which >= len(match):
                continue
            atom = match[which]
            if (family, atom) in claimed:
                continue
            claimed.add((family, atom))
            out = seen.setdefault(family, {"family": family, "atoms": [], "groups": [],
                                           "notes": []})  # fmt: skip
            out["atoms"].append(atom)
            if group not in out["groups"]:
                out["groups"].append(group)
            if note and note not in out["notes"]:
                out["notes"].append(note)
    for out in seen.values():
        out["atoms"] = sorted(out["atoms"])
    return [seen[f] for f in PHASE2_FAMILIES if f in seen]


def phase2_warnings(sites) -> list[str]:
    """The reactive-metabolite and polymorphism notes of :func:`phase2_sites`, once
    each: the part of the phase II panel that changes what a chemist does next."""
    out: list[str] = []
    for site in sites:
        for note in site["notes"]:
            if note not in out:
                out.append(note)
    return out


# --- the report ------------------------------------------------------------------------


#: the metabolic soft spots' mark, light and dark.  Purple because the page's accent
#: is amber: an amber mark read as more of the same rather than as a mark
SPOT_COLOUR = {False: (0.70, 0.53, 0.91), True: (0.79, 0.65, 0.96)}


def depiction(mol, width: int = 240, height: int = 170, dark: bool = False,
              highlight=None) -> str:  # fmt: skip
    """A 2D drawing of ``mol`` as SVG.  ``dark``: drawn for a dark background, so the
    page can carry one of each and show whichever its theme asks for.  ``highlight``:
    atoms to mark (the metabolic soft spots).  Transparent, so it takes the card's own
    background; "" if RDKit cannot lay the molecule out.

    Each mark is given a ``soft`` class, so the page can hide them all with one rule
    and show the structures plain."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    try:
        mol = AllChem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol)
        draw = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = draw.drawOptions()
        if dark:
            rdMolDraw2D.SetDarkMode(opts)
        opts.setBackgroundColour((0, 0, 0, 0))
        opts.clearBackground = False
        hi = [int(a) for a in (highlight or [])]
        colour = SPOT_COLOUR[bool(dark)]
        draw.DrawMolecule(mol, highlightAtoms=hi,
                          highlightAtomColors={a: colour for a in hi})  # fmt: skip
        draw.FinishDrawing()
        svg = draw.GetDrawingText()
    except Exception:
        return ""
    # the <?xml?> header belongs to a standalone file, not to a span of a page
    svg = svg[svg.index("<svg") :] if "<svg" in svg else ""
    return _mark_spots(svg) if hi else svg


def _mark_spots(svg: str) -> str:
    """The soft spots' marks given a class of their own, so the page can switch them
    off.  RDKit draws them first, each an ellipse, and everything after the first bond
    is the structure itself; so the ellipses before that bond are the marks.

    Matching on where they are rather than on their colour is deliberate: RDKit rounds
    a colour its own way when it writes it out (0.65 came back as A5, not A6), so the
    hex in the file need not be the hex that was asked for."""
    import re

    first = svg.find("<path class='bond")
    head, rest = (svg[:first], svg[first:]) if first > 0 else (svg, "")
    head = re.sub(r"(<ellipse\b[^>]*?)class='([^']*)'", r"\1class='\2 soft'", head)
    return head + rest


def egg_svg(points, width: int = 560, height: int = 330) -> str:
    """The BOILED-Egg as inline SVG: TPSA across, WLOGP up, the two regions as
    ellipses and ``points`` ([(index, name, tpsa, wlogp)]) on them.  Each point carries
    its design's number, so the page can scroll to it when it is clicked."""
    left, right, top, bottom = 52, width - 14, 16, height - 36
    x0, x1, y0, y1 = 0.0, 200.0, -4.0, 8.0  # the axes, wide enough for the whole egg
    sx = lambda v: left + (min(max(v, x0), x1) - x0) / (x1 - x0) * (right - left)  # noqa: E731
    sy = lambda v: bottom - (min(max(v, y0), y1) - y0) / (y1 - y0) * (bottom - top)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="BOILED-Egg plot of TPSA against WLOGP">']  # fmt: skip
    out.append(f'<rect x="{left}" y="{top}" width="{right - left}" '
               f'height="{bottom - top}" fill="#6b7280" fill-opacity="0.18" '
               'stroke="#9aa1ad" stroke-width="0.7"/>')  # fmt: skip
    for (cx, ax, cy, ay), fill, stroke in ((EGG_WHITE, "#ffffff", "#c8ccd4"),
                                           (EGG_YOLK, "#f5c518", "#c79b0d")):  # fmt: skip
        out.append(f'<ellipse cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" '
                   f'rx="{(sx(cx + ax) - sx(cx)):.1f}" ry="{(sy(cy) - sy(cy + ay)):.1f}" '
                   f'fill="{fill}" fill-opacity="0.92" stroke="{stroke}" '
                   'stroke-width="1"/>')  # fmt: skip
    for value in (0, 50, 100, 150, 200):  # the axes, labelled where it helps
        out.append(f'<line x1="{sx(value):.1f}" y1="{bottom}" x2="{sx(value):.1f}" '
                   f'y2="{bottom + 4}" stroke="currentColor" stroke-width="0.7"/>'
                   f'<text x="{sx(value):.1f}" y="{bottom + 16}" text-anchor="middle" '
                   f'font-size="10" fill="currentColor">{value}</text>')  # fmt: skip
    for value in (-4, -2, 0, 2, 4, 6, 8):
        out.append(f'<line x1="{left - 4}" y1="{sy(value):.1f}" x2="{left}" '
                   f'y2="{sy(value):.1f}" stroke="currentColor" stroke-width="0.7"/>'
                   f'<text x="{left - 7}" y="{sy(value) + 3.5:.1f}" text-anchor="end" '
                   f'font-size="10" fill="currentColor">{value}</text>')  # fmt: skip
    out.append(f'<text x="{(left + right) / 2:.0f}" y="{height - 4}" text-anchor="middle" '
               'font-size="11" fill="currentColor">TPSA (&#8491;&#178;)</text>'
               f'<text x="13" y="{(top + bottom) / 2:.0f}" font-size="11" fill="currentColor" '
               f'transform="rotate(-90 13 {(top + bottom) / 2:.0f})" '
               'text-anchor="middle">WLOGP</text>')  # fmt: skip
    for index, name, tpsa, logp in points:
        out.append(f'<circle class="pt" data-design="{index}" cx="{sx(tpsa):.1f}" '
                   f'cy="{sy(logp):.1f}" r="4.5" fill="#d2384a" stroke="#ffffff" '
                   f'stroke-width="1.2"><title>{_escape(name)}: TPSA {tpsa:g}, '
                   f'WLOGP {logp:g} &#8212; click to go to it</title></circle>')  # fmt: skip
    out.append("</svg>")
    return "".join(out)


def _escape(text) -> str:
    import html

    return html.escape(str(text), quote=True)


# what the report shows in each panel, in the order a chemist reads them: the key in
# ``properties`` and how the row is labelled
PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Physicochemical",
        [
            ("formula", "Formula"),
            ("MW", "Molecular weight"),
            ("heavy atoms", "Heavy atoms"),
            ("aromatic atoms", "Aromatic atoms"),
            ("fraction Csp3", "Fraction Csp3"),
            ("rotatable bonds", "Rotatable bonds"),
            ("HBA", "H-bond acceptors"),
            ("HBD", "H-bond donors"),
            ("molar refractivity", "Molar refractivity"),
            ("TPSA", "TPSA (&#8491;&#178;)"),
            ("formal charge", "Formal charge"),
        ],
    ),  # fmt: skip
    (
        "Lipophilicity and ionisation",
        [
            ("cLogP", "Log P (Crippen/WLOGP)"),
            ("logD", "Log D"),
            ("ionised form", "Mostly present as"),
            ("ionisable groups", "Ionisable groups (pKa used)"),
        ],
    ),  # fmt: skip
    (
        "Water solubility",
        [
            ("log S (fitted)", "Log S (fitted, AqSolDB)"),
            ("solubility", "Class"),
            ("log S (ESOL)", "Log S (ESOL)"),
            ("log S (Ali)", "Log S (Ali)"),
        ],
    ),  # fmt: skip
    (
        "Pharmacokinetics",
        [
            ("GI absorption", "Gastrointestinal absorption"),
            ("BBB permeant", "Blood-brain barrier"),
            ("log Kp (skin)", "Log Kp, skin (cm/s)"),
        ],
    ),  # fmt: skip
    (
        "Druglikeness",
        [
            ("Lipinski", "Lipinski"),
            ("Ghose", "Ghose"),
            ("Veber", "Veber"),
            ("Egan", "Egan"),
            ("Muegge", "Muegge"),
            ("lead-like", "Lead-likeness"),
            ("QED", "QED"),
            ("bioavailability score", "Bioavailability score"),
        ],
    ),  # fmt: skip
    (
        "Medicinal chemistry",
        [
            ("PAINS alerts", "PAINS"),
            ("Brenk alerts", "Brenk"),
            ("synthetic accessibility", "Synthetic accessibility"),
        ],
    ),  # fmt: skip
]

_CSS = """
:root { color-scheme: light dark;
  --bg: #ffffff; --card: #f7f8fa; --line: #d9dde3; --text: #1d2026; --muted: #69707c;
  --accent: #c9711a; --warn: #b4520f; --good: #1c7a3e; --flash: rgba(201, 113, 26, 0.22);
  --spot: #7c4dbe; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15171c; --card: #1c1f26; --line: #2b3039; --text: #e6e8ec; --muted: #8a919e;
  --accent: #f0a23b; --warn: #ffab6b; --good: #5fd089; --flash: rgba(240, 162, 59, 0.26);
  --spot: #c9a7f5; } }
:root[data-theme="dark"] {
  --bg: #15171c; --card: #1c1f26; --line: #2b3039; --text: #e6e8ec; --muted: #8a919e;
  --accent: #f0a23b; --warn: #ffab6b; --good: #5fd089; --flash: rgba(240, 162, 59, 0.26);
  --spot: #c9a7f5; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 20px 60px; background: var(--bg); color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 21px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 32px 0 10px; padding-bottom: 5px;
  border-bottom: 1px solid var(--line); }
h3 { font-size: 14px; margin: 0 0 2px; }
.sub { color: var(--muted); font-size: 12.5px; margin: 0 0 6px; }
.top { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
button.ui { background: var(--card); color: var(--text); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 11px; font: inherit; font-size: 12.5px; cursor: pointer; }
button.ui:hover { border-color: var(--accent); color: var(--accent); }
/* the SMILES is the molecule: it is read and copied, so it is not greyed out */
.smiles { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
  color: var(--text); word-break: break-all; margin: 0 0 7px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; white-space: nowrap; }
#summary th { cursor: pointer; user-select: none; }
#summary th:hover { color: var(--accent); }
#summary td.n, #summary th.n { text-align: right; font-variant-numeric: tabular-nums; }
#summary tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--card); }
.arrow { color: var(--accent); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px 16px; margin: 14px 0; }
.design { display: grid; grid-template-columns: 250px 1fr; gap: 18px; align-items: start; }
/* the drawing sits on the card, in the page's own colours */
.struct { background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: 6px; text-align: center; position: relative; }
/* inline SVG has no "Copy image" in the browser's own menu, so each drawing
   carries a button; it shows when the drawing is pointed at */
.figbtns { display: inline-flex; gap: 4px; }
.figbtns button { padding: 2px 8px; font-size: 11px; }
.struct .figbtns, .egg .figbtns { position: absolute; top: 5px; right: 5px; opacity: 0;
  transition: opacity .12s; }
.struct:hover .figbtns, .struct:focus-within .figbtns,
.egg:hover .figbtns, .egg:focus-within .figbtns { opacity: 1; }
@media (hover: none) { .struct .figbtns, .egg .figbtns { opacity: 1; } }
.axes .figbtns { margin-left: auto; }
:root[data-theme="dark"] .light-only { display: none; }
:root[data-theme="light"] .dark-only { display: none; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .light-only { display: none; } }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) .dark-only { display: none; } }
@keyframes flash { from { background: var(--flash); } to { background: var(--card); } }
.design.lit { animation: flash 1.6s ease-out; }
.panels { display: grid; gap: 4px 22px;
  grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); }
.panel table { font-size: 12px; }
.panel th { width: 52%; font-weight: 400; }
.panel td { text-align: right; font-variant-numeric: tabular-nums; }
.pt2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--accent); margin: 10px 0 2px; }
.pt2.spot { color: var(--spot); }
.flag { color: var(--warn); }
.ok { color: var(--good); }
.note { background: var(--bg); border-left: 3px solid var(--accent); padding: 7px 11px;
  margin: 8px 0; font-size: 12.5px; }
.card.egg { position: relative; }
.egg svg { display: block; margin: 0 auto; max-width: 620px; }
circle.pt { cursor: pointer; }
circle.pt:hover { fill: var(--accent); r: 6; }
select.ui { background: var(--card); color: var(--text); border: 1px solid var(--line);
  border-radius: 5px; padding: 3px 6px; font: inherit; font-size: 12.5px; }
.axes { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  font-size: 12.5px; color: var(--muted); margin-bottom: 8px; }
.spots li { margin: 2px 0; }
.spots { margin: 4px 0 0; padding-left: 18px; font-size: 12.5px; }
.spots .what { color: var(--spot); }
/* the soft spots switched off: the marks on every drawing and the lists with them */
:root.no-spots .soft, :root.no-spots .spots-block { display: none; }
.switch { display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
  font-size: 12.5px; color: var(--muted); user-select: none; }
.switch input { accent-color: var(--spot); margin: 0; }
.tools { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
/* triage: the verdict, the rules that make it, and what a filter hides */
.verdict { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  border: 1px solid currentColor; border-radius: 10px; padding: 1px 7px;
  vertical-align: 2px; }
.v-pass { color: var(--good); }
.v-review { color: var(--accent); }
.v-fail { color: var(--warn); }
#summary td[data-verdict] { font-size: 11.5px; text-transform: uppercase;
  letter-spacing: 0.04em; }
.why { margin: 0 0 6px; }
.rules { display: grid; gap: 2px 18px;
  grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); }
.rule { display: flex; align-items: center; gap: 7px; font-size: 12.5px;
  padding: 2px 0; cursor: pointer; }
.rule input[type=checkbox] { accent-color: var(--accent); margin: 0; }
.rlabel { flex: 1; }
.thr { width: 62px; background: var(--bg); color: var(--text);
  border: 1px solid var(--line); border-radius: 5px; padding: 2px 5px; font: inherit;
  font-size: 12px; }
.rlevel { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.05em;
  width: 44px; text-align: right; }
.tally { font-size: 12.5px; color: var(--text); }
.gone { display: none; }
footer { margin-top: 42px; padding-top: 14px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 11.5px; }
footer p { margin: 5px 0; }
@media (max-width: 760px) { .design { grid-template-columns: 1fr; } }
@media print { body { padding: 0; } .card { break-inside: avoid; } button.ui { display: none; } }
"""

# The page's only script.  It sorts the table, moves to a design when its row or its
# point is clicked, draws the scatter plot, and remembers the theme.  It reads nothing
# but the page itself and fetches nothing.
_JS = r"""
(function () {
  var DATA = __DATA__, NUMERIC = __NUMERIC__, RULES = __RULES__;
  var root = document.documentElement;

  // --- theme: the system's unless the reader says otherwise -------------------------
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };
  var saved = store.get('boonza.report.theme');
  if (saved === 'dark' || saved === 'light') root.dataset.theme = saved;
  function isDark() {
    if (root.dataset.theme) return root.dataset.theme === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  var themeButton = document.getElementById('theme');
  function markTheme() {
    themeButton.textContent = isDark() ? 'Light mode' : 'Dark mode';
    themeButton.title = 'show this report ' + (isDark() ? 'light' : 'dark');
  }
  themeButton.onclick = function () {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    store.set('boonza.report.theme', root.dataset.theme);
    markTheme();
    plot();
  };
  markTheme();

  // --- the soft spots, on or off ------------------------------------------------------
  // off hides every mark on every drawing and the lists with them, leaving the
  // structures plain; it is remembered, like the theme
  var spotsBox = document.getElementById('spots');
  if (spotsBox) {
    if (store.get('boonza.report.spots') === 'off') {
      spotsBox.checked = false;
      root.classList.add('no-spots');
    }
    spotsBox.onchange = function () {
      root.classList.toggle('no-spots', !spotsBox.checked);
      store.set('boonza.report.spots', spotsBox.checked ? 'on' : 'off');
    };
  }

  // --- take a figure away: a drawing, the plot, the egg ---------------------------------
  // Each is inline SVG, which the browser will not copy for you.  They go as vector,
  // never as pixels: a figure wants shapes that stay sharp at any size.  Where the
  // clipboard cannot be reached (an older browser, a page served oddly) the file is
  // saved instead, which gets the reader the figure either way
  function figureOf(button) {
    var holder = button.closest('[data-figure]') || button.parentNode;
    var shown = null, svgs = holder.querySelectorAll('svg');
    for (var i = 0; i < svgs.length; i++) {
      if (svgs[i].getBoundingClientRect().width > 0) { shown = svgs[i]; break; }
    }
    return shown ? { svg: shown, name: holder.getAttribute('data-figure') || 'figure' } : null;
  }
  // the SVG as it stands, with the marks left out when they are switched off
  function figureSVG(found) {
    var copy = found.svg.cloneNode(true);
    if (root.classList.contains('no-spots')) {
      var marks = copy.querySelectorAll('.soft');
      for (var k = 0; k < marks.length; k++) marks[k].parentNode.removeChild(marks[k]);
    }
    return copy;
  }
  function saveFigure(button) {
    var found = figureOf(button);
    if (!found) return;
    var text = new XMLSerializer().serializeToString(figureSVG(found));
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'image/svg+xml' }));
    var named = found.name;
    if (named === 'plot' && xs && ys) named = ys.value + '-vs-' + xs.value;
    a.download = named.replace(/[^\w.-]+/g, '_') + '.svg';
    a.click();
    button.textContent = 'saved';
    setTimeout(function () { button.textContent = 'svg'; }, 1400);
  }
  // A figure is copied as vector, not as pixels: SVG goes on the clipboard, and
  // Illustrator, Inkscape and Office all take it as shapes that stay sharp at any
  // size.  Nothing is painted behind it either, so it drops onto a slide or a page of
  // any colour.
  function figureText(found) {
    var shown = found.svg;
    var box = shown.viewBox && shown.viewBox.baseVal;
    var copy = figureSVG(found);
    copy.setAttribute('width', (box && box.width) || shown.clientWidth || 240);
    copy.setAttribute('height', (box && box.height) || shown.clientHeight || 170);
    // a plot draws its axes in the page's own colours (currentColor), which a cloned
    // node knows nothing about: the colour is written onto it so it survives the trip
    var ink = getComputedStyle(shown).color;
    if (ink) copy.setAttribute('color', ink);
    return new XMLSerializer().serializeToString(copy);
  }
  function copyDrawing(button) {
    var found = figureOf(button);
    if (!found) return;
    var text = figureText(found);
    var said = function (what) {
      button.textContent = what;
      setTimeout(function () { button.textContent = 'copy'; }, 1600);
    };
    var asText = function () {
      if (!navigator.clipboard) { saveFigure(button); return; }
      navigator.clipboard.writeText(text).then(function () { said('copied'); },
                                               function () { saveFigure(button); });
    };
    try {
      if (navigator.clipboard && window.ClipboardItem) {
        var items = {
          'image/svg+xml': new Blob([text], { type: 'image/svg+xml' }),
          'text/plain': new Blob([text], { type: 'text/plain' })
        };
        navigator.clipboard.write([new ClipboardItem(items)])
          .then(function () { said('copied'); }, asText);
      } else { asText(); }
    } catch (e) { asText(); }
  }

  // --- go to a design, from its row or from a point -----------------------------------
  function go(index) {
    var card = document.getElementById('design-' + index);
    if (!card) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.remove('lit');
    void card.offsetWidth;  // so the animation restarts when the same one is clicked
    card.classList.add('lit');
  }
  document.addEventListener('click', function (e) {
    if (!e.target.closest) return;
    var button = e.target.closest('[data-copy]');
    if (button) { copyDrawing(button); return; }  // not a jump to the design it is on
    var vector = e.target.closest('[data-svg]');
    if (vector) { saveFigure(vector); return; }
    var node = e.target.closest('[data-design]');
    if (node) go(+node.getAttribute('data-design'));
  });

  // --- the table sorts on any column, or on several ------------------------------------
  // "2PM1" is not a number, though parseFloat would call it 2 and make every name
  // equal; and "2PM10" comes after "2PM2", which a plain text compare gets backwards
  var collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
  function isNumber(text) { return text !== '' && isFinite(Number(text)); }
  function compare(a, b) {
    if (isNumber(a) && isNumber(b)) return Number(a) - Number(b);
    if (isNumber(a) !== isNumber(b)) return isNumber(a) ? -1 : 1;  // numbers first
    return collator.compare(a, b);
  }
  var table = document.getElementById('summary');
  if (table) {
    var headers = [].slice.call(table.tHead.rows[0].cells);
    // the keys in force, most significant first: [{col, down}].  A plain click starts
    // again from one column; a shift-click adds a column to sort *within* the keys
    // already chosen, which is what "first by X, then by Y" means
    var keys = [];
    function mark() {
      headers.forEach(function (th) {
        var arrow = th.querySelector('.arrow');
        if (arrow) arrow.textContent = '';
      });
      keys.forEach(function (key, rank) {
        var arrow = headers[key.col].querySelector('.arrow');
        if (!arrow) return;
        // the number is the order the keys are applied in; one key alone needs no number
        arrow.textContent = ' ' + (keys.length > 1 ? (rank + 1) : '') + (key.down ? '↓' : '↑');
      });
    }
    function apply() {
      if (!keys.length) return;
      var body = table.tBodies[0];
      var rows = [].slice.call(body.rows);
      rows.sort(function (x, y) {
        for (var k = 0; k < keys.length; k++) {
          var c = compare(x.cells[keys[k].col].textContent.trim(),
                          y.cells[keys[k].col].textContent.trim());
          if (c) return keys[k].down ? -c : c;
        }
        // every key equal: the order the designs came in, so a sort is repeatable
        return (+x.getAttribute('data-row')) - (+y.getAttribute('data-row'));
      });
      rows.forEach(function (r) { body.appendChild(r); });
    }
    headers.forEach(function (th, col) {
      th.addEventListener('click', function (e) {
        var at = -1;
        for (var k = 0; k < keys.length; k++) if (keys[k].col === col) at = k;
        if (e.shiftKey) {
          // already a key: turn that one round, leaving its place in the order
          if (at >= 0) keys[at].down = !keys[at].down;
          else keys.push({ col: col, down: false });
        } else {
          // the only key already: turn it round.  Otherwise it becomes the only key
          keys = [{ col: col, down: keys.length === 1 && at === 0 ? !keys[0].down : false }];
        }
        mark();
        apply();
      });
    });
  }

  // --- triage: one verdict per design, and the rules that made it ----------------------
  // The rules come from the same table the report was written with, so the verdict the
  // page recomputes and the one printed into it cannot disagree.  Nothing is recomputed
  // from the structure here: every rule reads a number already on the page
  var tally = document.getElementById('tally');
  var show = document.getElementById('show');
  var ORDER = ['pass', 'review', 'fail'];
  function limits() {
    var out = {};
    RULES.forEach(function (rule) {
      var on = document.getElementById('c-' + rule.key);
      if (on && !on.checked) { out[rule.key] = null; return; }   // the rule is ignored
      var box = document.getElementById('t-' + rule.key);
      if (box && box.value !== '' && isFinite(Number(box.value))) out[rule.key] = Number(box.value);
    });
    return out;
  }
  function judge(design, moved) {
    var worst = 0, why = [];
    RULES.forEach(function (rule) {
      if (moved[rule.key] === null) return;
      var value = design.t[rule.field];
      if (value === undefined || value === null) return;
      var limit = moved.hasOwnProperty(rule.key) ? moved[rule.key] : rule.limit;
      var hit;
      // without case, as triage() compares it: "High" must not fire a rule set on "high"
      if (rule.op === 'is not') {
        hit = String(value).toLowerCase() !== String(rule.limit).toLowerCase();
      }
      else if (typeof value !== 'number') return;
      else hit = rule.op === '>=' ? value >= limit : rule.op === '>' ? value > limit
                                                                     : value < limit;
      if (hit) {
        why.push(rule.op === 'is not' ? rule.label : rule.label + ' ' + value);
        worst = Math.max(worst, rule.level === 'fail' ? 2 : 1);
      }
    });
    return { level: ORDER[worst], why: why };
  }
  function retriage() {
    if (!RULES.length) return;
    var moved = limits(), counts = { pass: 0, review: 0, fail: 0 };
    var keep = show ? show.value : 'all';
    DATA.forEach(function (design) {
      var out = judge(design, moved);
      counts[out.level]++;
      var badges = document.querySelectorAll('[data-verdict="' + design.i + '"]');
      for (var k = 0; k < badges.length; k++) {
        badges[k].textContent = out.level;
        badges[k].className = badges[k].className.replace(/\bv-\w+\b/g, '') + ' v-' + out.level;
      }
      var why = document.querySelector('[data-why="' + design.i + '"]');
      if (why) why.textContent = out.why.length ? out.why.join('; ') : 'nothing breached';
      // hidden, not deleted: untick a rule and it is back, with nothing recomputed
      var hide = keep === 'pass' ? out.level !== 'pass'
               : keep === 'review' ? out.level === 'fail' : false;
      var row = document.querySelector('#summary tr[data-row="' + design.i + '"]');
      if (row) row.classList.toggle('gone', hide);
      var card = document.getElementById('design-' + design.i);
      if (card) card.classList.toggle('gone', hide);
    });
    if (tally) {
      tally.textContent = counts.pass + ' pass, ' + counts.review + ' to review, '
                        + counts.fail + ' fail';
    }
  }
  RULES.forEach(function (rule) {
    var on = document.getElementById('c-' + rule.key);
    var box = document.getElementById('t-' + rule.key);
    if (on) on.onchange = retriage;
    if (box) box.oninput = retriage;
  });
  if (show) show.onchange = retriage;
  var resetButton = document.getElementById('reset');
  if (resetButton) {
    resetButton.onclick = function () {
      RULES.forEach(function (rule) {
        var on = document.getElementById('c-' + rule.key);
        var box = document.getElementById('t-' + rule.key);
        if (on) on.checked = true;
        if (box) box.value = rule.limit;
      });
      if (show) show.value = 'all';
      retriage();
    };
  }
  retriage();

  // --- any property against any other ---------------------------------------------------
  var xs = document.getElementById('plot-x'), ys = document.getElementById('plot-y');
  var holder = document.getElementById('plot');
  function css(name) {
    return getComputedStyle(root).getPropertyValue(name).trim() || '#888';
  }
  function plot() {
    if (!holder || !xs || !ys) return;
    var kx = xs.value, ky = ys.value;
    var pts = [];
    for (var i = 0; i < DATA.length; i++) {
      var x = DATA[i].p[kx], y = DATA[i].p[ky];
      if (typeof x === 'number' && typeof y === 'number') {
        pts.push({ i: DATA[i].i, name: DATA[i].n, x: x, y: y });
      }
    }
    if (!pts.length) { holder.innerHTML = '<p class="sub">nothing to plot here.</p>'; return; }
    var W = 560, H = 330, L = 58, R = 14, T = 14, B = 40;
    var xv = pts.map(function (p) { return p.x; }), yv = pts.map(function (p) { return p.y; });
    function span(v) {
      var lo = Math.min.apply(null, v), hi = Math.max.apply(null, v);
      if (lo === hi) { lo -= 1; hi += 1; }
      var pad = (hi - lo) * 0.08;
      return [lo - pad, hi + pad];
    }
    var ax = span(xv), ay = span(yv);
    var sx = function (v) { return L + (v - ax[0]) / (ax[1] - ax[0]) * (W - L - R); };
    var sy = function (v) { return H - B - (v - ay[0]) / (ay[1] - ay[0]) * (H - T - B); };
    function ticks(lo, hi) {
      var step = Math.pow(10, Math.floor(Math.log(Math.abs(hi - lo) / 4) / Math.LN10));
      var k = (hi - lo) / step;
      if (k > 8) step *= 5; else if (k > 4) step *= 2;
      var out = [], start = Math.ceil(lo / step) * step;
      for (var v = start; v <= hi + step * 1e-6 && out.length < 12; v += step) out.push(v);
      return out;
    }
    function label(v) {
      return Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)
        ? v.toExponential(1) : String(Math.round(v * 1000) / 1000);
    }
    var line = css('--line'), text = css('--text');
    var muted = css('--muted'), accent = css('--accent');
    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="'
             + ky + ' against ' + kx + '">'];
    s.push('<rect x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="'
           + (H - T - B) + '" fill="none" stroke="' + line + '"/>');
    ticks(ax[0], ax[1]).forEach(function (v) {
      s.push('<line x1="' + sx(v).toFixed(1) + '" y1="' + T + '" x2="' + sx(v).toFixed(1)
             + '" y2="' + (H - B) + '" stroke="' + line + '" stroke-width="0.6"/>'
             + '<text x="' + sx(v).toFixed(1) + '" y="' + (H - B + 15)
             + '" text-anchor="middle" font-size="10" fill="' + muted + '">'
             + label(v) + '</text>');
    });
    ticks(ay[0], ay[1]).forEach(function (v) {
      s.push('<line x1="' + L + '" y1="' + sy(v).toFixed(1) + '" x2="' + (W - R) + '" y2="'
             + sy(v).toFixed(1) + '" stroke="' + line + '" stroke-width="0.6"/>'
             + '<text x="' + (L - 6) + '" y="' + (sy(v) + 3.5).toFixed(1)
             + '" text-anchor="end" font-size="10" fill="' + muted + '">' + label(v) + '</text>');
    });
    s.push('<text x="' + ((L + W - R) / 2) + '" y="' + (H - 4) + '" text-anchor="middle"'
           + ' font-size="11" fill="' + text + '">' + kx + '</text>');
    s.push('<text x="14" y="' + ((T + H - B) / 2) + '" font-size="11" fill="' + text
           + '" text-anchor="middle" transform="rotate(-90 14 ' + ((T + H - B) / 2) + ')">'
           + ky + '</text>');
    pts.forEach(function (p) {
      s.push('<circle class="pt" data-design="' + p.i + '" cx="' + sx(p.x).toFixed(1)
             + '" cy="' + sy(p.y).toFixed(1) + '" r="4.5" fill="' + accent
             + '" stroke="' + css('--bg') + '" stroke-width="1.2"><title>' + p.name + ': '
             + kx + ' ' + label(p.x) + ', ' + ky + ' ' + label(p.y)
             + ' — click to go to it</title></circle>');
    });
    s.push('</svg>');
    holder.innerHTML = s.join('');
  }
  if (xs && ys) {
    xs.onchange = ys.onchange = plot;
    plot();
  }
})();
"""


def _figure_buttons(what: str) -> str:
    """Copy and save, for any figure on the page: the drawing of a design, the plot,
    the egg.  Both give vector -- shapes that stay sharp and drop onto any colour."""
    return (
        f'<span class="figbtns"><button class="ui copy" type="button" data-copy '
        f'title="copy {_escape(what)} as vector (SVG): it pastes into Illustrator, '
        'Inkscape or Office as shapes, and carries no background">copy</button>'
        '<button class="ui" type="button" data-svg '
        f'title="save {_escape(what)} as an SVG file">svg</button></span>'
    )


def _triage_controls() -> str:
    """The triage rules as the page's own controls: a tick and a threshold each, and a
    tally.  Every rule is shown whether it bites or not, so a shortlist can never hide
    the rule that produced it, and a threshold a reader disagrees with can be moved."""
    rows = []
    for key, label, _field, operator, default, level in TRIAGE_FIELDS:
        box = ""
        if operator != "is not":
            step = "0.1" if isinstance(default, float) else "1"
            box = (f'<input type="number" id="t-{key}" value="{default:g}" step="{step}" '
                   'class="thr">')  # fmt: skip
        rows.append(f'<label class="rule"><input type="checkbox" id="c-{key}" checked>'
                    f'<span class="rlabel">{_escape(label)}</span>{box}'
                    f'<span class="rlevel v-{level}">{level}</span></label>')  # fmt: skip
    return (
        '<div class="card" id="controls"><p class="pt2">Triage rules</p>'
        '<p class="sub">Untick a rule to ignore it, or change a threshold: the table, '
        "the tally and every panel follow. A design fails on any <em>fail</em> rule "
        "and is flagged on any <em>review</em> rule. These thresholds are the "
        "conventional ones, which is not the same as being right for your series."
        "</p>"
        f'<div class="rules">{"".join(rows)}</div>'
        '<div class="axes" style="margin:10px 0 0">'
        '<span class="tally" id="tally"></span>'
        '<label for="show">show</label>'
        '<select class="ui" id="show">'
        '<option value="all">every design</option>'
        '<option value="review">pass and review</option>'
        '<option value="pass">pass only</option></select>'
        '<button class="ui" id="reset" type="button">reset to defaults</button>'
        "</div></div>"
    )


def _verdict_class(value) -> str:
    """How a filter's verdict is coloured: a pass green, a failure amber."""
    text = str(value)
    if text in ("pass", "high", "yes", "none"):
        return ' class="ok"'
    if text.startswith("fails") or text == "low":
        return ' class="flag"'
    return ""


def _plot_data(designs, columns) -> tuple[str, str, str]:
    """What the page's one script needs, as JSON: every design's numbers (for the
    scatter plot) and the properties triage reads, the properties that can go on an
    axis, and the triage rules themselves -- the same table :func:`triage` uses, so the
    verdict the page recomputes and the one computed here cannot drift apart.

    ``<`` is escaped so nothing in a name can end the script element early."""
    import json

    def dump(value) -> str:
        return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c")

    numeric = [c for c in columns
               if any(isinstance(p.get(c), (int, float)) and not isinstance(p.get(c), bool)
                      for _n, _s, _o, p, _m in designs)]  # fmt: skip
    wanted = [f[2] for f in TRIAGE_FIELDS]
    rows = []
    for k, (name, _smiles, _note, props, _mol) in enumerate(designs):
        mine = {c: props[c] for c in numeric if isinstance(props.get(c), (int, float))}
        reads = {c: props[c] for c in wanted if props.get(c) is not None}
        rows.append({"i": k, "n": name, "p": mine, "t": reads})
    rules = [{"key": k, "label": lab, "field": f, "op": o, "limit": d, "level": lv}
             for k, lab, f, o, d, lv in TRIAGE_FIELDS]  # fmt: skip
    return dump(rows), dump(numeric), dump(rules)


def report_html(designs, group: str, ph: float = 7.4, title: str = "") -> str:
    """A whole series as one self-contained HTML page: the BOILED-Egg, a plot of any
    property against any other, a sortable table of every design, and a panel each.

    ``designs``: [(name, smiles, note, properties, mol)], as :func:`properties` gives
    them; ``mol`` may be None, and then that design has no drawing.  The page carries
    its own style and the structures as inline SVG, so it opens anywhere, offline, and
    can be emailed as one file.  Clicking a design anywhere -- a row of the table, a
    point of either plot -- goes to its panel."""
    import datetime

    everything: list[str] = []
    for _n, _s, _o, p, _m in designs:
        for key in p:
            if key not in everything and key != "caveats":
                everything.append(key)
    columns = ["name", *[c for c in TABLE_COLUMNS if c in everything]]
    numeric = {"MW", "heavy atoms", "cLogP", "logD", "TPSA", "HBD", "HBA", "rotatable bonds",
               "QED", "synthetic accessibility", "Lipinski violations", "PAINS"}  # fmt: skip
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = _escape(title or group)
    data, axes, rules = _plot_data(designs, everything)

    out = [
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{head} &middot; boonza</title>",
        f"<style>{_CSS}</style>\n</head>\n<body>\n<main>",
        '<div class="top"><div>',
        f"<h1>{head}</h1>",
        f'<p class="sub">{len(designs)} design{"" if len(designs) == 1 else "s"} &middot; '
        f"properties computed at pH {ph:g} &middot; {when} &middot; boonza</p>",
        '<p class="sub">Every number here is estimated from the structure alone. '
        "Treat them as a way to rank designs against each other, not as measurements.</p>",
        '</div><div class="tools">',
        '<label class="switch" for="spots"><input type="checkbox" id="spots" checked>'
        "metabolic soft spots</label>",
        '<button class="ui" id="theme" type="button">Dark mode</button></div></div>',
    ]

    # one row per design
    out.append(f"<h2>{_escape(group)} at a glance</h2>")
    out.append(_triage_controls())
    out.append('<p class="sub">Click a column to sort by it; <strong>shift-click</strong> '
               "another to sort within that, and again to turn it round &mdash; the "
               "headings number the keys in the order they are applied. Click a row to go "
               "to that design.</p>")  # fmt: skip
    out.append('<table id="summary"><thead><tr>')
    notes = any(o for _n, _s, o, _p, _m in designs)
    for c in [*columns, *(["note"] if notes else [])]:
        cls = ' class="n"' if c in numeric else ""
        out.append(f'<th{cls} data-label="{_escape(c)}">{_escape(c)}'
                   '<span class="arrow"></span></th>')  # fmt: skip
    out.append("</tr></thead><tbody>")
    for k, (name, _smiles, note, props, _mol) in enumerate(designs):
        out.append(f'<tr data-design="{k}" data-row="{k}">')
        for c in columns:
            value = name if c == "name" else props.get(c, "")
            cls = ' class="n"' if c in numeric else ""
            if c == "triage":
                out.append(f'<td data-verdict="{k}" class="v-{_escape(value)}">'
                           f"{_escape(value)}</td>")  # fmt: skip
                continue
            out.append(f"<td{cls}>{_escape(value)}</td>")
        if notes:
            out.append(f"<td>{_escape(note or '')}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")

    # any property against any other
    if len(axes) > 2:
        pick = [c for c in everything if f'"{c}"' in axes]
        default_x = "cLogP" if "cLogP" in pick else pick[0]
        default_y = "logD" if "logD" in pick else pick[min(1, len(pick) - 1)]
        out.append("<h2>One property against another</h2>")
        out.append('<div class="card" data-figure="plot"><div class="axes">')
        for which, label, chosen in (("y", "up", default_y), ("x", "across", default_x)):
            out.append(f'<label for="plot-{which}">{label}</label>'
                       f'<select class="ui" id="plot-{which}">')  # fmt: skip
            for c in pick:
                mark = " selected" if c == chosen else ""
                out.append(f'<option value="{_escape(c)}"{mark}>{_escape(c)}</option>')
            out.append("</select>")
        out.append('<span class="sub" style="margin:0">click a point to go to that design'
                   "</span>" + _figure_buttons("the plot") + "</div>")  # fmt: skip
        out.append('<div id="plot"></div></div>')

    # the egg, with every design on it
    points = [(k, n, p["TPSA"], p["cLogP"])
              for k, (n, _s, _o, p, _m) in enumerate(designs) if "TPSA" in p]  # fmt: skip
    if points:
        out.append("<h2>Passive absorption and brain penetration</h2>")
        buttons = _figure_buttons("the BOILED-Egg")
        out.append(f'<div class="card egg" data-figure="BOILED-Egg">{buttons}'
                   + egg_svg(points) + "</div>")  # fmt: skip

    # a panel per design
    out.append("<h2>Every design</h2>")
    for k, (name, smiles, note, props, mol) in enumerate(designs):
        spots, conjugates = [], []
        if mol is not None:
            try:
                spots = soft_spots(mol)
            except Exception:
                spots = []
            try:
                conjugates = phase2_sites(mol)
            except Exception:
                conjugates = []
        marked = sorted({a for s in spots for a in s["atoms"]}
                        | {a for s in conjugates for a in s["atoms"]})  # fmt: skip
        out.append(f'<div class="card design" id="design-{k}">')
        if mol is not None:
            # one drawing for each theme, since a structure drawn for white paper
            # disappears on a dark page; the stylesheet shows whichever is wanted
            light = depiction(mol, dark=False, highlight=marked)
            dark = depiction(mol, dark=True, highlight=marked)
            drawing = (f'<span class="light-only">{light}</span>'
                       f'<span class="dark-only">{dark}</span>') if light else ""  # fmt: skip
        else:
            drawing = ""
        if not drawing:
            out.append('<div class="struct"><p class="sub">no drawing</p></div>')
        else:
            out.append(f'<div class="struct" data-figure="{_escape(name)}">'
                       f"{_figure_buttons(name)}{drawing}</div>")  # fmt: skip
        out.append("<div>")
        verdict = props.get("triage")
        badge = (f'<span class="verdict v-{_escape(verdict)}" data-verdict="{k}">'
                 f"{_escape(verdict)}</span>") if verdict else ""  # fmt: skip
        out.append(f"<h3>{_escape(name)} {badge}</h3>")
        if props.get("triage reasons"):
            out.append(f'<p class="sub why" data-why="{k}">'
                       f'{_escape(props["triage reasons"])}</p>')  # fmt: skip
        if smiles:
            out.append(f'<p class="smiles">{_escape(smiles)}</p>')
        if note:
            out.append(f'<div class="note">{_escape(note)}</div>')
        for caveat in props.get("caveats", []):
            out.append(f'<div class="note flag">Log D caveat: {_escape(caveat)}</div>')
        out.append('<div class="panels">')
        for heading, rows in PANELS:
            shown = [(key, lab) for key, lab in rows if key in props]
            if not shown:
                continue
            out.append(f'<div class="panel"><p class="pt2">{heading}</p><table>')
            for key, lab in shown:
                value = props[key]
                if key == "logD":
                    lab = f"Log D (pH {ph:g})"
                out.append(f"<tr><th>{lab}</th><td{_verdict_class(value)}>"
                           f"{_escape(value)}</td></tr>")  # fmt: skip
            out.append("</table></div>")
        out.append("</div>")
        if spots or conjugates:
            out.append('<div class="spots-block">')
        if spots:
            out.append('<p class="pt2 spot">Metabolic soft spots &mdash; oxidation (phase I)'
                       "</p>")  # fmt: skip
            out.append('<ul class="spots">')
            for spot in spots:
                out.append(f'<li><span class="what">{_escape(spot["label"])}</span> '
                           f'&mdash; {_escape(spot["note"])}</li>')  # fmt: skip
            out.append("</ul>")
        if conjugates:
            out.append('<p class="pt2 spot">Conjugation handles &mdash; phase II</p>')
            out.append('<ul class="spots">')
            for site in conjugates:
                groups = ", ".join(site["groups"])
                out.append(f'<li><span class="what">{_escape(site["family"])}</span> '
                           f'&mdash; {_escape(groups)}</li>')  # fmt: skip
            out.append("</ul>")
            for warning in phase2_warnings(conjugates):
                out.append(f'<div class="note flag">{_escape(warning)}</div>')
        if spots or conjugates:
            out.append('<p class="sub">Marked on the drawing. Both lists are substructure '
                       "matches: they say which groups are labile and which conjugation "
                       "routes are open, never a rate and never which route wins. Nothing "
                       "says which site goes first.</p>")  # fmt: skip
            out.append("</div>")
        out.append("</div></div>")

    out.append(
        "<footer><p><strong>How these were computed.</strong> All of it from the structure, "
        "with RDKit and published equations; nothing was sent anywhere.</p>"
        "<p>Log P is RDKit's Crippen estimate (the WLOGP of SwissADME). Log D is that, "
        "corrected for ionisation by Henderson-Hasselbalch, with one representative pKa per "
        "functional group &mdash; so it carries about 0.5&ndash;1.0 log unit of error even "
        "when the ionisation is right, and more when a group's real pKa is far from its "
        "class average. The pKa values are medians of measured drug-like compounds rather "
        "than the textbook value of the unsubstituted parent. An ionised molecule is taken "
        "to partition as an ion pair, three log units below the neutral form for a base "
        "and four for an acid (the published &ldquo;diff 3&ndash;4&rdquo; approximation): "
        "without that floor logD falls without limit and every cation reads about 0.9 too "
        "low. A measured pKa, given as a design's <code>pka_acid</code> or "
        "<code>pka_base</code> field, replaces the table for that design and is the single "
        "biggest thing that improves its logD. A design is neutralised before its pKa is "
        "read, since that is the species a pKa belongs to: a design taken from a prepared "
        "structure arrives drawn as the simulation needed it &mdash; the amine protonated, "
        "the acid deprotonated &mdash; and read that way it would show no ionisable group "
        "at all, leaving its logD equal to its logP. A charge no proton can balance, such "
        "as a quaternary ammonium, survives that step, and holds logD down at the ion-pair "
        "plateau for good.</p>"
        "<p><strong>The pKa of each group is then fitted, not looked up.</strong> The table "
        "locates the ionisable atom; a random forest over its local environment &mdash; the "
        "Gasteiger charges around it, the electron-withdrawing and donating groups at two, "
        "three and five bonds, its ring and aromatic context &mdash; prices it, and such a "
        "group is marked <code>[refined]</code>. Groups the table misses are found with "
        "Dimorphite-DL's substructures and priced the same way. On 1000 measured pKa values "
        "the site is located for 927 and the mean error is 0.79, against 1.21 for the table "
        "alone and 0.93 for Marvin, which is commercial; logD improves from 0.85 to 0.80 on "
        "1000 measured values. It is fitted from Baltruschat &amp; Czodrowski's collection "
        "(CC BY 4.0), and a measured pKa still beats it.</p>"
        "<p><strong>The BOILED-Egg</strong> (Daina &amp; Zoete 2016). A design in the white "
        "is likely to be absorbed in the gut; one in the yolk is likely to cross the "
        "blood-brain barrier as well. Outside both, neither is likely by passive diffusion. "
        "Click a point to go to that design. It knows nothing of transporters or efflux, so "
        "a molecule that is carried across (or pumped back out) sits in the wrong place: "
        "caffeine, which does reach the brain, falls outside the yolk. Published "
        "benchmarks put the egg's GI verdict near 0.75 and its brain verdict near 0.73 "
        "AUROC against measured sets, and the Abbott bioavailability score near 0.57 "
        "&mdash; which is barely better than calling every design bioavailable. Read all "
        "three as a sanity check, never as a decision.</p>"
        "<p><strong>Log S is fitted too.</strong> Solubility is not linear in logP, and "
        "the published equations are: on a scaffold split of AqSolDB a random forest "
        "reaches a mean error of 0.92 against 1.18 for Delaney's ESOL, so the fitted value "
        "is the one the class is read from. ESOL and Ali are printed unchanged beside it "
        "&mdash; three numbers disagreeing is itself worth seeing. Fitted from AqSolDB "
        "(Sorkun, Khetan &amp; Er 2019; CC0).</p>"
        "<p>Log S: Delaney 2004 (ESOL) and Ali et al. 2012. Log Kp: Potts &amp; Guy 1992. "
        "QED: Bickerton et al. 2012. Synthetic accessibility: Ertl &amp; Schuffenhauer 2009. "
        "Filters: Lipinski 1997, Ghose 1999, Egan 2000, Muegge 2001, Veber 2002. Alerts: the "
        "PAINS and Brenk catalogues that ship with RDKit.</p>"
        "<p><strong>Metabolic soft spots are substructure flags, not a prediction.</strong> "
        "They mark groups known to be labile &mdash; what a chemist looks for when a "
        "compound turns over too fast. They give no rate and no order, and they are not "
        "SMARTCyp, which ranks every atom by a computed activation energy. Microsomes "
        "settle it.</p>"
        "<p><strong>The phase II list says which conjugation routes are open</strong>, not "
        "which one dominates. UGT is usually the highest-capacity route and SULT saturates "
        "with dose; NAT2 and TPMT are polymorphic in people, so a slow metaboliser reaches "
        "a quite different exposure from the same dose. The groups whose conjugate is "
        "itself reactive are called out: an acyl glucuronide can rearrange and acylate "
        "protein, an N-OH sulfate can ionise to a nitrenium ion, and a catechol is also "
        "oxidised to a quinone.</p>"
        "<p><strong>Triage is a filter, not a finding.</strong> Every rule reads a number "
        "shown above it, and the thresholds are the conventional ones, which is not the "
        "same as being right for a series. Move them, or untick a rule, and the verdicts "
        "follow.</p>"
        "<p>PAINS and Brenk alerts are substructures worth a second look, not verdicts; "
        "a good many marketed drugs carry one. The filters are rules of thumb, and plenty "
        "of useful molecules break them.</p></footer>"
    )
    script = (_JS.replace("__DATA__", data).replace("__NUMERIC__", axes)
              .replace("__RULES__", rules))  # fmt: skip
    out.append(f"</main>\n<script>{script}</script>\n</body>\n</html>\n")
    return "\n".join(out)
