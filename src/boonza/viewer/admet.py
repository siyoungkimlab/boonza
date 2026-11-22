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


# --- the report ------------------------------------------------------------------------


def depiction(mol, width: int = 240, height: int = 170) -> str:
    """A 2D drawing of ``mol`` as SVG, for the report.  Transparent, so it sits on the
    card's own background; returns "" if RDKit cannot lay the molecule out."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    try:
        mol = AllChem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol)
        draw = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = draw.drawOptions()
        opts.setBackgroundColour((0, 0, 0, 0))
        opts.clearBackground = False
        draw.DrawMolecule(mol)
        draw.FinishDrawing()
        svg = draw.GetDrawingText()
    except Exception:
        return ""
    # the <?xml?> header belongs to a standalone file, not to a span of a page
    return svg[svg.index("<svg") :] if "<svg" in svg else ""


def egg_svg(points, width: int = 560, height: int = 330) -> str:
    """The BOILED-Egg as inline SVG: TPSA across, WLOGP up, the two regions as
    ellipses and ``points`` ([(name, tpsa, wlogp)]) on top of them.  A design in the
    yolk is likely to reach the brain, one in the white to be absorbed in the gut,
    and one in the grey outside neither."""
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
    for name, tpsa, logp in points:
        out.append(f'<circle cx="{sx(tpsa):.1f}" cy="{sy(logp):.1f}" r="4.5" fill="#d2384a" '
                   f'stroke="#ffffff" stroke-width="1.2"><title>{_escape(name)}: TPSA '
                   f'{tpsa:g}, WLOGP {logp:g}</title></circle>')  # fmt: skip
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
  --accent: #c9711a; --warn: #b4520f; --good: #1c7a3e; }
@media (prefers-color-scheme: dark) { :root {
  --bg: #15171c; --card: #1c1f26; --line: #2b3039; --text: #e6e8ec; --muted: #8a919e;
  --accent: #f0a23b; --warn: #ffab6b; --good: #5fd089; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 20px 60px; background: var(--bg); color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 21px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 32px 0 10px; padding-bottom: 5px;
  border-bottom: 1px solid var(--line); }
h3 { font-size: 14px; margin: 0 0 2px; }
.sub { color: var(--muted); font-size: 12.5px; margin: 0 0 6px; }
.smiles { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
  color: var(--muted); word-break: break-all; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; white-space: nowrap; }
#summary th { cursor: pointer; user-select: none; }
#summary th:hover { color: var(--accent); }
#summary td.n, #summary th.n { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:hover { background: var(--card); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px 16px; margin: 14px 0; }
.design { display: grid; grid-template-columns: 250px 1fr; gap: 18px; align-items: start; }
.struct { background: #ffffff; border: 1px solid var(--line); border-radius: 6px; padding: 6px;
  text-align: center; }
.panels { display: grid; gap: 4px 22px;
  grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); }
.panel table { font-size: 12px; }
.panel th { width: 52%; font-weight: 400; }
.panel td { text-align: right; font-variant-numeric: tabular-nums; }
.pt { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent);
  margin: 10px 0 2px; }
.flag { color: var(--warn); }
.ok { color: var(--good); }
.note { background: var(--card); border-left: 3px solid var(--accent); padding: 7px 11px;
  margin: 8px 0; font-size: 12.5px; }
.egg { display: grid; grid-template-columns: 1fr 230px; gap: 18px; align-items: center; }
footer { margin-top: 42px; padding-top: 14px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 11.5px; }
footer p { margin: 5px 0; }
@media (max-width: 760px) { .design, .egg { grid-template-columns: 1fr; } }
@media print { body { padding: 0; } .card { break-inside: avoid; } }
"""

# clicking a column sorts by it, again the other way.  The only script in the page: it
# reads the table already there and reorders its rows, and touches nothing else
_SORT_JS = """
document.querySelectorAll('#summary th').forEach((th, col) => {
  th.onclick = () => {
    const body = th.closest('table').tBodies[0];
    const down = th.dataset.down !== '1';
    th.closest('tr').querySelectorAll('th').forEach(o => {
      o.dataset.down = ''; o.textContent = o.dataset.label; });
    th.dataset.down = down ? '1' : '0';
    th.textContent = th.dataset.label + (down ? ' \\u2193' : ' \\u2191');
    const key = tr => { const t = tr.cells[col].textContent.trim();
      const n = parseFloat(t); return isNaN(n) ? t.toLowerCase() : n; };
    [...body.rows].sort((a, b) => { const x = key(a), y = key(b);
      return (x < y ? -1 : x > y ? 1 : 0) * (down ? -1 : 1); }).forEach(r => body.append(r));
  };
});
"""


def _verdict_class(value) -> str:
    """How a filter's verdict is coloured: a pass green, a failure amber."""
    text = str(value)
    if text in ("pass", "high", "yes", "none"):
        return ' class="ok"'
    if text.startswith("fails") or text in ("low",):
        return ' class="flag"'
    return ""


def report_html(designs, group: str, ph: float = 7.4, title: str = "") -> str:
    """A whole series as one self-contained HTML page: the BOILED-Egg, a sortable
    table of every design, and a panel each.

    ``designs``: [(name, smiles, note, properties, mol)], as :func:`properties` gives
    them; ``mol`` may be None, and then that design has no drawing.  The page carries
    its own style and the structures as inline SVG, so it opens anywhere, offline,
    and can be emailed as one file."""
    import datetime

    columns = ["name", *[c for c in TABLE_COLUMNS
                         if any(c in p for _n, _s, _o, p, _m in designs)]]  # fmt: skip
    numeric = {"MW", "cLogP", "logD", "TPSA", "HBD", "HBA", "rotatable bonds", "QED",
               "synthetic accessibility", "Lipinski violations", "PAINS"}  # fmt: skip
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = _escape(title or group)

    out = [
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{head} &middot; boonza</title>",
        f"<style>{_CSS}</style>\n</head>\n<body>\n<main>",
        f"<h1>{head}</h1>",
        f'<p class="sub">{len(designs)} design{"" if len(designs) == 1 else "s"} &middot; '
        f"properties computed at pH {ph:g} &middot; {when} &middot; boonza</p>",
        '<p class="sub">Every number here is estimated from the structure alone. '
        "Treat them as a way to rank designs against each other, not as measurements.</p>",
    ]

    # the egg, with every design on it
    points = [(n, p["TPSA"], p["cLogP"]) for n, _s, _o, p, _m in designs if "TPSA" in p]
    if points:
        out.append("<h2>Passive absorption and brain penetration</h2>")
        out.append('<div class="card egg"><div>' + egg_svg(points) + "</div><div>")
        out.append('<p class="sub">The BOILED-Egg (Daina &amp; Zoete 2016). A design in the '
                   "white is likely to be absorbed in the gut; one in the yolk is likely to "
                   "cross the blood-brain barrier as well. Outside both, neither is likely by "
                   "passive diffusion.</p>")  # fmt: skip
        out.append('<p class="sub">It knows nothing of transporters or efflux, so a molecule '
                   "that is carried across (or pumped back out) sits in the wrong place: "
                   "caffeine, which does reach the brain, falls outside the yolk.</p>"
                   "</div></div>")  # fmt: skip

    # one row per design
    out.append(f"<h2>{_escape(group)} at a glance</h2>")
    out.append('<p class="sub">Click a column to sort by it.</p>')
    out.append('<table id="summary"><thead><tr>')
    for c in columns:
        cls = ' class="n"' if c in numeric else ""
        out.append(f'<th{cls} data-label="{_escape(c)}">{_escape(c)}</th>')
    if any(o for _n, _s, o, _p, _m in designs):
        out.append('<th data-label="note">note</th>')
    out.append("</tr></thead><tbody>")
    notes = any(o for _n, _s, o, _p, _m in designs)
    for name, _smiles, note, props, _mol in designs:
        out.append("<tr>")
        for c in columns:
            value = name if c == "name" else props.get(c, "")
            cls = ' class="n"' if c in numeric else ""
            out.append(f"<td{cls}>{_escape(value)}</td>")
        if notes:
            out.append(f"<td>{_escape(note or '')}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")

    # a panel per design
    out.append("<h2>Every design</h2>")
    for name, smiles, note, props, mol in designs:
        out.append('<div class="card design">')
        svg = depiction(mol) if mol is not None else ""
        drawing = svg or '<p class="sub">no drawing</p>'
        out.append(f'<div class="struct">{drawing}</div>')
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
            shown = [(key, label) for key, label in rows if key in props]
            if not shown:
                continue
            out.append(f'<div class="panel"><p class="pt">{heading}</p><table>')
            for key, label in shown:
                value = props[key]
                if key == "logD":
                    label = f"Log D (pH {ph:g})"
                out.append(f"<tr><th>{label}</th><td{_verdict_class(value)}>"
                           f"{_escape(value)}</td></tr>")  # fmt: skip
            out.append("</table></div>")
        out.append("</div></div></div>")

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
        "<p>PAINS and Brenk alerts are substructures worth a second look, not verdicts; "
        "a good many marketed drugs carry one. The filters are rules of thumb, and plenty "
        "of useful molecules break them.</p></footer>"
    )
    out.append(f"</main>\n<script>{_SORT_JS}</script>\n</body>\n</html>\n")
    return "\n".join(out)
