"""Physicochemical and ADME properties of a ligand design, from RDKit alone.

What a series' ``html`` button computes for each member: the numbers a medicinal
chemist reads before deciding whether a design is worth making -- size and polarity,
lipophilicity at pH (logD, which is logP corrected for whatever is ionised there),
solubility, passive absorption and brain penetration, the druglikeness filters, and
the substructures worth a second look.

Everything here is RDKit and arithmetic: no extra dependency, nothing downloaded,
nothing sent anywhere.  The methods are published ones, cited where they are used:

  logD          Henderson-Hasselbalch on RDKit's Crippen logP, with the pKa of each
                ionisable group taken from the table below (``ACIDS``, ``BASES``)
  ESOL          Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000
  Ali logS      Ali, Camilleri, Brown, Hutt & Kirton, J. Chem. Inf. Model. 2012, 52, 420
  Log Kp        Potts & Guy, Pharm. Res. 1992, 9, 663
  BOILED-Egg    Daina & Zoete, ChemMedChem 2016, 11, 1117
  QED           Bickerton, Paolini, Besnard, Muresan & Hopkins, Nat. Chem. 2012, 4, 90
  SAscore       Ertl & Schuffenhauer, J. Cheminform. 2009, 1, 8
  Lipinski      Adv. Drug Deliv. Rev. 1997, 23, 3;  Ghose, J. Comb. Chem. 1999, 1, 55;
  and the other Veber, J. Med. Chem. 2002, 45, 2615;  Egan, J. Med. Chem. 2000, 43, 3867;
  filters       Muegge, J. Med. Chem. 2001, 44, 1841

These are estimates from the structure alone.  logD carries about 0.5-1.0 log unit of
error even when the ionisation is right, and the filters are rules of thumb that
plenty of marketed drugs break; :func:`caveats` says where a number should not be
trusted at all.  None of this is a measurement.

Deliberately not computed: iLOGP (needs CHARMM), XLOGP3 (needs its reference
database), MLOGP, the CYP and P-gp models (a 7 GB neural network, and a different
question from anything here) and sites of metabolism.  A row that cannot be computed
is absent rather than guessed at.
"""

from __future__ import annotations

import math

# --- ionisation ---------------------------------------------------------------------
#
# One representative pKa per functional class, as textbooks give them.  This is the
# crude part of logD: every aliphatic amine is treated as 9.7 whether its real value
# is 8 or 11, so a design whose ionisation matters deserves a measured pKa.  Each
# entry is (label, SMARTS, which atom of the match ionises, pKa); the first pattern
# to claim a heavy atom keeps it, so the specific ones come before the general.

_NOT_AMIDE = ("!$(N~[!#6;!#1]);!$(N[CX3]=[OX1,SX1,NX2]);!$(N[SX4](=O)=O);"
              "!$(N[CX2]#N);!$(Na);!$([N+]);!$(N=*)")  # fmt: skip
_AMINE = f"[NX3;H0,H1,H2;{_NOT_AMIDE}]"

ACIDS: list[tuple[str, str, int, float]] = [
    ("sulfonic acid", "[SX4](=[OX1])(=[OX1])[OX2H1]", 3, -1.0),
    ("phosphonate", "[PX4](=[OX1])[OX2H1]", 2, 2.1),
    ("acylsulfonamide", "[SX4](=[OX1])(=[OX1])[NX3H1][CX3]=[OX1]", 3, 4.5),
    ("tetrazole", "[nH]1nnnc1", 0, 4.9),
    ("aryl carboxylic acid", "[c][CX3](=[OX1])[OX2H1]", 3, 4.2),
    ("carboxylic acid", "[CX3](=[OX1])[OX2H1]", 2, 4.4),
    ("aryl thiol", "[c][SX2H1]", 1, 6.6),
    ("imide", "[CX3](=[OX1])[NX3H1][CX3]=[OX1]", 2, 9.5),
    ("phenol", "[c][OX2H1]", 1, 10.0),
    ("sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3;H1,H2]", 3, 10.1),
    ("thiol", "[#6][SX2H1]", 1, 10.5),
]
BASES: list[tuple[str, str, int, float]] = [
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]", 1, 13.6),
    ("amidine", "[CX3;!$(C[NX3][CX3]=[NX2])](=[NX2])[NX3]", 0, 11.6),
    # a nitrogen next to an electron-withdrawing group is a weaker base
    (
        "amine (next to EWG)",
        f"[NX3;H0,H1,H2;{_NOT_AMIDE};$(NCC=[OX1]),$(NC[OX2]),$(NCC#N)]",
        0,
        8.0,
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
    ("piperazine N (far N aryl)", f"[NX3;H0,H1;{_NOT_AMIDE};$(N1CCN(CC1)a)]", 0, 8.7),
    ("morpholine N (substituted)", "[NX3;H0;$(N1CCOCC1)]", 0, 6.9),
    ("morpholine NH", "[NX3;H1;$(N1CCOCC1)]", 0, 8.4),
    ("amine (beta to an ether)", f"[NX3;H0,H1,H2;{_NOT_AMIDE};$(NCC[OX2H0])]", 0, 9.0),
    ("amine", _AMINE, 0, 9.7),
    (
        "aniline",
        "[NX3;H0,H1,H2;$(Na);!$(N[CX3]=[OX1]);!$(N[SX4](=O)=O);!$([N+])]",
        0,
        4.6,
    ),  # fmt: skip
    ("pyridine", "[nX2;r6]", 0, 5.2),
    ("azole N", "[nX2;r5]", 0, 5.5),
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


def ionisable(mol) -> list[tuple[str, str, int, float]]:
    """Every ionisable group of ``mol``: (acid or base, label, atom, pKa).  A heavy
    atom belongs to one group only, the first pattern that claims it, so a phenol in
    a carboxylic acid's molecule is not counted twice."""
    from rdkit import Chem

    found: list[tuple[str, str, int, float]] = []
    taken: set[int] = set()
    for kind, table in (("acid", ACIDS), ("base", BASES)):
        for label, smarts, which, pka in table:
            for match in mol.GetSubstructMatches(_pattern(smarts)):
                atom = match[which]
                if atom in taken:
                    continue
                taken.add(atom)
                found.append((kind, label, atom, pka))
    # a second basic centre near a stronger one is much weaker than on its own
    # (piperazine: 9.8 and 5.6, not 9.8 twice), and counting them as independent
    # drags logD far too low
    strong: list[int] = []
    distance = Chem.GetDistanceMatrix(mol) if found else None
    for k, (kind, label, atom, pka) in enumerate(found):
        if kind != "base":
            continue
        if any(distance[atom][other] <= 3 for other in strong):
            found[k] = (kind, f"{label} (second basic centre)", atom, pka - 4.0)
        else:
            strong.append(atom)
    return found


def log_d(logp: float, sites, ph: float = 7.4) -> float:
    """Crippen logP corrected for whatever is ionised at ``ph``:

        logD = logP - log10(1 + SUM_acids 10^(pH - pKa) + SUM_bases 10^(pKa - pH))

    which is Henderson-Hasselbalch for each group, assuming only the neutral species
    partitions.  An ion pair does partition a little, so a charged design's real logD
    is higher than this; :func:`caveats` says so where it matters."""
    term = 0.0
    for kind, _label, _atom, pka in sites:
        # a long way either side of the pKa the exponent overflows; it is saturated
        # instead, since by then the group is fully ionised and the sum is what it is
        power = min(ph - pka if kind == "acid" else pka - ph, 30.0)
        term += 10.0**power
    return logp - math.log10(1.0 + term)


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
        out.append(f"permanently charged ({charge:+d}): it is ionised at every pH, "
                   "so logD is a lower bound")  # fmt: skip
    if form == "zwitterion":
        out.append("zwitterion: both centres are corrected for, but the ion pair "
                   "partitions, so the real logD is higher")  # fmt: skip
    if any((k == "base" and pka >= 11) or (k == "acid" and pka <= 1) for k, _l, _a, pka in sites):
        out.append("a very strong acid or base: fully ionised, so logD is under-estimated")
    return out


# --- solubility, skin, the egg --------------------------------------------------------


def esol(logp: float, mw: float, rotatable: int, aromatic_fraction: float) -> float:
    """Delaney's ESOL (2004): log S from logP, weight, flexibility and aromaticity."""
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotatable - 0.74 * aromatic_fraction


def ali_logs(logp: float, tpsa: float) -> float:
    """Ali et al. (2012), their model 03: log S from logP and TPSA alone."""
    return -1.0377 * logp - 0.0210 * tpsa + 0.4488


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


def properties(mol, ph: float = 7.4) -> dict:
    """Every property of one RDKit molecule, flat, ready for a table or a report.
    ``ph``: where logD and the ionised form are worked out.

    The keys that are numbers are plain floats and ints so they sort and plot; the
    filters are "pass" or what they fail; ``caveats`` is a list, empty when the
    numbers can be read as they stand."""
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.RemoveHs(mol)  # the descriptors count hydrogens themselves
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

    sites = ionisable(mol)
    form = ionised_form(sites, ph, charge)
    logd = log_d(logp, sites, ph)
    gi, bbb = boiled_egg(tpsa, logp)
    s_esol = esol(logp, mw, rotatable, aromatic / heavy if heavy else 0.0)
    s_ali = ali_logs(logp, tpsa)

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
        "solubility": solubility_class(s_esol),
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
    pains, brenk = alerts(mol, "PAINS"), alerts(mol, "BRENK")
    out["PAINS"] = len(pains)
    out["PAINS alerts"] = "; ".join(pains) or "none"
    out["Brenk"] = len(brenk)
    out["Brenk alerts"] = "; ".join(brenk) or "none"
    score = sascore(mol)
    if score is not None:
        out["synthetic accessibility"] = _round(score)
    return out


# which of the above are worth a column in the series' table: the numbers a chemist
# ranks designs by.  The rest are in the report, where there is room for them
TABLE_COLUMNS = ("MW", "heavy atoms", "cLogP", "logD", "TPSA", "HBD", "HBA",
                 "rotatable bonds", "QED",
                 "synthetic accessibility", "GI absorption", "BBB permeant",
                 "Lipinski violations", "PAINS")  # fmt: skip


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
# (label, SMARTS, which atoms of the match are the site, what happens there, phase)
SOFT_SPOTS: list[tuple[str, str, tuple[int, ...], str, int]] = [
    (
        "N-dealkylation",
        f"[NX3;H0,H1;{_NOT_AMIDE}][CH3,CH2;!$(C=O)]",
        (1,),
        "CYP takes the alkyl off the nitrogen; the commonest route of all",
        1,
    ),  # fmt: skip
    (
        "O-dealkylation",
        "[c,C][OX2;H0][CH3,CH2;!$(C=O)]",
        (2,),
        "CYP takes the alkyl off the oxygen, leaving the phenol or alcohol",
        1,
    ),  # fmt: skip
    (
        "benzylic oxidation",
        "[CH1,CH2,CH3;!$(C=O);!$(C[OX2,NX3]);$(C[a])]",
        (0,),
        "a carbon next to a ring: hydroxylated easily, the C-H being weak",
        1,
    ),  # fmt: skip
    (
        "allylic oxidation",
        "[CH1,CH2,CH3;!$(C=O);$(CC=C)]",
        (0,),
        "a carbon next to a double bond, hydroxylated for the same reason",
        1,
    ),  # fmt: skip
    (
        "aromatic hydroxylation",
        "[cH;$(c1ccccc1);$(c:c[OX2,NX3,CH3])]",
        (0,),
        "an electron-rich ring position, hydroxylated through an epoxide",
        1,
    ),  # fmt: skip
    (
        "terminal alkyl (omega-1)",
        "[CH3][CH2][CH2]",
        (1,),
        "the carbon next to the end of a chain, the usual place a chain is hit",
        1,
    ),  # fmt: skip
    (
        "S-oxidation",
        "[SX2;$(S(-[#6])-[#6]);!$(S(=O))]",
        (0,),
        "a thioether goes to the sulfoxide and on to the sulfone, readily",
        1,
    ),  # fmt: skip
    (
        "N-oxidation",
        f"[NX3;H0;{_NOT_AMIDE};!$(Na)]",
        (0,),
        "a tertiary amine gives the N-oxide",
        1,
    ),  # fmt: skip
    (
        "alkene epoxidation",
        "[CX3H1,CX3H2]=[CX3;!$(C=O)]",
        (0, 1),
        "an epoxide, which may go on to bind covalently: watch it",
        1,
    ),  # fmt: skip
    (
        "aldehyde oxidation",
        "[CX3H1]=[OX1]",
        (0,),
        "oxidised to the acid quickly, by aldehyde oxidase as well as CYP",
        1,
    ),  # fmt: skip
    (
        "ester hydrolysis",
        "[CX3](=[OX1])[OX2][#6]",
        (0,),
        "esterases cut it, often before CYP gets near: a prodrug handle",
        1,
    ),  # fmt: skip
    (
        "amide hydrolysis",
        "[CX3](=[OX1])[NX3;H1,H2]",
        (0,),
        "amidases cut it, far more slowly than an ester",
        1,
    ),  # fmt: skip
    (
        "glucuronidation / sulfation",
        "[c][OX2H1]",
        (1,),
        "a phenol is conjugated straight away (phase II), often the main route",
        2,
    ),  # fmt: skip
    (
        "glucuronidation",
        "[CX4][OX2H1]",
        (1,),
        "an alcohol is conjugated, more slowly than a phenol",
        2,
    ),  # fmt: skip
    (
        "acyl glucuronidation",
        "[CX3](=[OX1])[OX2H1]",
        (0,),
        "an acyl glucuronide, which can be reactive: a known liability",
        2,
    ),  # fmt: skip
]


def soft_spots(mol) -> list[dict]:
    """Where ``mol`` is likely to be metabolised, by group.  Each is {label, atoms,
    note, phase}; ``atoms`` are the heavy atoms of the site, for highlighting.

    A flag is a liability to look at, not a rate and not an order -- see the note on
    :data:`SOFT_SPOTS`.  Nothing here predicts which site goes first, which is what
    SMARTCyp does and this does not."""
    out = []
    for label, smarts, which, note, phase in SOFT_SPOTS:
        atoms: set[int] = set()
        for match in mol.GetSubstructMatches(_pattern(smarts)):
            atoms.update(match[k] for k in which if k < len(match))
        if atoms:
            out.append({"label": label, "atoms": sorted(atoms), "note": note, "phase": phase})
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
        [("log S (ESOL)", "Log S (ESOL)"), ("solubility", "Class"), ("log S (Ali)", "Log S (Ali)")],
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
.struct .copy { position: absolute; top: 5px; right: 5px; opacity: 0; transition: opacity .12s;
  padding: 2px 8px; font-size: 11px; }
.struct:hover .copy, .struct .copy:focus { opacity: 1; }
@media (hover: none) { .struct .copy { opacity: 1; } }
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
.egg { display: grid; grid-template-columns: 1fr 230px; gap: 18px; align-items: center; }
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
footer { margin-top: 42px; padding-top: 14px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 11.5px; }
footer p { margin: 5px 0; }
@media (max-width: 760px) { .design, .egg { grid-template-columns: 1fr; } }
@media print { body { padding: 0; } .card { break-inside: avoid; } button.ui { display: none; } }
"""

# The page's only script.  It sorts the table, moves to a design when its row or its
# point is clicked, draws the scatter plot, and remembers the theme.  It reads nothing
# but the page itself and fetches nothing.
_JS = r"""
(function () {
  var DATA = __DATA__, NUMERIC = __NUMERIC__;
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

  // --- copy a drawing as a picture ------------------------------------------------------
  // the drawing is inline SVG, which the browser will not copy for you: it is painted
  // onto a canvas at three times the size and put on the clipboard as a PNG.  Where
  // the clipboard cannot be reached (an older browser, a page served oddly) it is
  // saved instead, which gets the reader the picture either way
  function copyDrawing(button) {
    var holder = button.parentNode;
    var shown = null, svgs = holder.querySelectorAll('svg');
    for (var i = 0; i < svgs.length; i++) {
      if (svgs[i].getBoundingClientRect().width > 0) { shown = svgs[i]; break; }
    }
    if (!shown) return;
    var box = shown.viewBox && shown.viewBox.baseVal;
    var w = (box && box.width) || shown.clientWidth || 240;
    var h = (box && box.height) || shown.clientHeight || 170;
    var copy = shown.cloneNode(true);
    copy.setAttribute('width', w);
    copy.setAttribute('height', h);
    if (root.classList.contains('no-spots')) {
      var marks = copy.querySelectorAll('.soft');
      for (var k = 0; k < marks.length; k++) marks[k].parentNode.removeChild(marks[k]);
    }
    var text = new XMLSerializer().serializeToString(copy);
    var image = new Image();
    image.onload = function () {
      var scale = 3;  // so it is still sharp pasted into a slide
      var canvas = document.createElement('canvas');
      canvas.width = w * scale;
      canvas.height = h * scale;
      var g = canvas.getContext('2d');
      g.fillStyle = getComputedStyle(holder).backgroundColor || '#ffffff';
      g.fillRect(0, 0, canvas.width, canvas.height);
      g.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        if (!blob) return;
        var said = function (what) {
          button.textContent = what;
          setTimeout(function () { button.textContent = 'copy'; }, 1400);
        };
        var save = function () {
          var a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = (button.getAttribute('title') || 'molecule').replace(
            /^copy\s+|\s+as a picture$/g, '') + '.png';
          a.click();
          said('saved');
        };
        try {
          if (navigator.clipboard && window.ClipboardItem) {
            navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
              .then(function () { said('copied'); }, save);
          } else { save(); }
        } catch (e) { save(); }
      }, 'image/png');
    };
    image.onerror = function () { button.textContent = 'cannot copy'; };
    image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(text);
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
    var node = e.target.closest('[data-design]');
    if (node) go(+node.getAttribute('data-design'));
  });

  // --- the table sorts on any column ---------------------------------------------------
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
    headers.forEach(function (th, col) {
      th.addEventListener('click', function () {
        var down = th.dataset.down !== '1';
        headers.forEach(function (other) {
          other.dataset.down = '';
          var mark = other.querySelector('.arrow');
          if (mark) mark.textContent = '';
        });
        th.dataset.down = down ? '1' : '0';
        th.querySelector('.arrow').textContent = down ? ' ↓' : ' ↑';
        var body = table.tBodies[0];
        var rows = [].slice.call(body.rows);
        rows.sort(function (x, y) {
          return compare(x.cells[col].textContent.trim(),
                         y.cells[col].textContent.trim()) * (down ? -1 : 1);
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  }

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


def _verdict_class(value) -> str:
    """How a filter's verdict is coloured: a pass green, a failure amber."""
    text = str(value)
    if text in ("pass", "high", "yes", "none"):
        return ' class="ok"'
    if text.startswith("fails") or text == "low":
        return ' class="flag"'
    return ""


def _plot_data(designs, columns) -> tuple[str, str]:
    """The numbers the scatter plot needs, as JSON for the page's one script, and the
    properties that can go on an axis.  ``<`` is escaped so nothing in a name can end
    the script element early."""
    import json

    numeric = [c for c in columns
               if any(isinstance(p.get(c), (int, float)) and not isinstance(p.get(c), bool)
                      for _n, _s, _o, p, _m in designs)]  # fmt: skip
    rows = []
    for k, (name, _smiles, _note, props, _mol) in enumerate(designs):
        mine = {c: props[c] for c in numeric if isinstance(props.get(c), (int, float))}
        rows.append({"i": k, "n": name, "p": mine})
    dumped = json.dumps(rows, ensure_ascii=True).replace("<", "\\u003c")
    return dumped, json.dumps(numeric, ensure_ascii=True).replace("<", "\\u003c")


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
    data, axes = _plot_data(designs, everything)

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
    out.append('<p class="sub">Click a column to sort by it, or a row to go to that design.</p>')
    out.append('<table id="summary"><thead><tr>')
    notes = any(o for _n, _s, o, _p, _m in designs)
    for c in [*columns, *(["note"] if notes else [])]:
        cls = ' class="n"' if c in numeric else ""
        out.append(f'<th{cls} data-label="{_escape(c)}">{_escape(c)}'
                   '<span class="arrow"></span></th>')  # fmt: skip
    out.append("</tr></thead><tbody>")
    for k, (name, _smiles, note, props, _mol) in enumerate(designs):
        out.append(f'<tr data-design="{k}">')
        for c in columns:
            value = name if c == "name" else props.get(c, "")
            cls = ' class="n"' if c in numeric else ""
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
        out.append('<div class="card"><div class="axes">')
        for which, label, chosen in (("y", "up", default_y), ("x", "across", default_x)):
            out.append(f'<label for="plot-{which}">{label}</label>'
                       f'<select class="ui" id="plot-{which}">')  # fmt: skip
            for c in pick:
                mark = " selected" if c == chosen else ""
                out.append(f'<option value="{_escape(c)}"{mark}>{_escape(c)}</option>')
            out.append("</select>")
        out.append('<span class="sub" style="margin:0">click a point to go to that design'
                   "</span></div>")  # fmt: skip
        out.append('<div id="plot"></div></div>')

    # the egg, with every design on it
    points = [(k, n, p["TPSA"], p["cLogP"])
              for k, (n, _s, _o, p, _m) in enumerate(designs) if "TPSA" in p]  # fmt: skip
    if points:
        out.append("<h2>Passive absorption and brain penetration</h2>")
        out.append('<div class="card egg"><div>' + egg_svg(points) + "</div><div>")
        out.append('<p class="sub">The BOILED-Egg (Daina &amp; Zoete 2016). A design in the '
                   "white is likely to be absorbed in the gut; one in the yolk is likely to "
                   "cross the blood-brain barrier as well. Outside both, neither is likely by "
                   "passive diffusion. Click a point to go to that design.</p>")  # fmt: skip
        out.append('<p class="sub">It knows nothing of transporters or efflux, so a molecule '
                   "that is carried across (or pumped back out) sits in the wrong place: "
                   "caffeine, which does reach the brain, falls outside the yolk.</p>"
                   "</div></div>")  # fmt: skip

    # a panel per design
    out.append("<h2>Every design</h2>")
    for k, (name, smiles, note, props, mol) in enumerate(designs):
        spots = []
        if mol is not None:
            try:
                spots = soft_spots(mol)
            except Exception:
                spots = []
        marked = sorted({a for s in spots for a in s["atoms"]})
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
            copy = (
                '<button class="ui copy" type="button" data-copy '
                f'title="copy {_escape(name)} as a picture">copy</button>'
            )
            out.append(f'<div class="struct">{copy}{drawing}</div>')
        out.append("<div>")
        out.append(f"<h3>{_escape(name)}</h3>")
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
        if spots:
            out.append('<div class="spots-block">')
            out.append('<p class="pt2 spot">Metabolic soft spots</p>')
            out.append('<ul class="spots">')
            for spot in spots:
                phase = "phase II" if spot["phase"] == 2 else "phase I"
                out.append(f'<li><span class="what">{_escape(spot["label"])}</span> '
                           f'({phase}) &mdash; {_escape(spot["note"])}</li>')  # fmt: skip
            out.append("</ul>")
            out.append('<p class="sub">Marked on the drawing. These are the groups known '
                       "to be labile, found by substructure: a liability to look at, not "
                       "a rate. Nothing says which site goes first.</p>")  # fmt: skip
            out.append("</div>")
        out.append("</div></div>")

    out.append(
        "<footer><p><strong>How these were computed.</strong> All of it from the structure, "
        "with RDKit and published equations; nothing was sent anywhere.</p>"
        "<p>Log P is RDKit's Crippen estimate (the WLOGP of SwissADME). Log D is that, "
        "corrected for ionisation by Henderson-Hasselbalch, with one representative pKa per "
        "functional group &mdash; so it carries about 0.5&ndash;1.0 log unit of error even "
        "when the ionisation is right, and more when a group's real pKa is far from its "
        "class average.</p>"
        "<p>Log S: Delaney 2004 (ESOL) and Ali et al. 2012. Log Kp: Potts &amp; Guy 1992. "
        "Absorption and brain penetration: Daina &amp; Zoete 2016. QED: Bickerton et al. 2012. "
        "Synthetic accessibility: Ertl &amp; Schuffenhauer 2009. Filters: Lipinski 1997, "
        "Ghose 1999, Egan 2000, Muegge 2001, Veber 2002. Alerts: the PAINS and Brenk "
        "catalogues that ship with RDKit.</p>"
        "<p><strong>Metabolic soft spots are substructure flags, not a prediction.</strong> "
        "They mark groups known to be labile &mdash; what a chemist looks for when a "
        "compound turns over too fast. They give no rate and no order, and they are not "
        "SMARTCyp, which ranks every atom by a computed activation energy. Microsomes "
        "settle it.</p>"
        "<p>PAINS and Brenk alerts are substructures worth a second look, not verdicts; "
        "a good many marketed drugs carry one. The filters are rules of thumb, and plenty "
        "of useful molecules break them.</p></footer>"
    )
    script = _JS.replace("__DATA__", data).replace("__NUMERIC__", axes)
    out.append(f"</main>\n<script>{script}</script>\n</body>\n</html>\n")
    return "\n".join(out)
