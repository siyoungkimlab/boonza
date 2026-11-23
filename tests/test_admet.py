"""The properties a series' ``html`` button computes, and the report it writes.

The numbers are checked against drugs whose behaviour is known, not against another
program: the point is that a chemist reading the report is not misled, so what matters
is that an acid comes out an acid, that logD drops where the molecule is ionised, and
that the famous non-CNS drug is not called brain-penetrant.
"""

import re

import pytest

from boonza.viewer import admet

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

# name, SMILES, what is known about it
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"
ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ARGININE = "NC(CCCNC(N)=N)C(=O)O"


def props(smiles, ph=7.4):
    return admet.properties(Chem.MolFromSmiles(smiles), ph)


def test_an_acid_is_an_anion_and_a_base_a_cation():
    """The ionised form at pH 7.4, which is what logD is corrected for.  An amine is
    protonated there, a carboxylic acid is not, and an amino acid is both."""
    assert props(ASPIRIN)["ionised form"] == "anion"
    assert props(IBUPROFEN)["ionised form"] == "anion"
    assert props(PROPRANOLOL)["ionised form"] == "cation"
    assert props(CAFFEINE)["ionised form"] == "neutral"  # its nitrogens are amides
    assert props(ARGININE)["ionised form"] == "zwitterion"


def test_logd_is_logp_only_when_nothing_is_ionised():
    """Henderson-Hasselbalch: a neutral molecule's logD is its logP, and an ionised
    one's is lower, by more the further the pKa is from the pH."""
    caffeine = props(CAFFEINE)
    assert caffeine["logD"] == caffeine["cLogP"]
    for smiles in (ASPIRIN, IBUPROFEN, PROPRANOLOL, ATENOLOL, ARGININE):
        p = props(smiles)
        assert p["logD"] < p["cLogP"], smiles
    # a carboxylic acid at pH 7.4 is about three log units below its logP (7.4 - 4.4)
    assert props(IBUPROFEN)["cLogP"] - props(IBUPROFEN)["logD"] == pytest.approx(3.0, abs=0.3)


def test_logd_follows_the_ph():
    """Below its pKa an acid is neutral, so its logD climbs to its logP; above it the
    acid is ionised and logD falls.  The lipophilicity a formulation sees changes with
    where in the gut it is, which is the point of computing logD at all."""
    low, high = props(IBUPROFEN, ph=2.0), props(IBUPROFEN, ph=9.0)
    assert low["logD"] == pytest.approx(low["cLogP"], abs=0.05)
    assert high["logD"] < low["logD"] - 3
    # a base is the other way round
    assert props(PROPRANOLOL, ph=11.0)["logD"] > props(PROPRANOLOL, ph=2.0)["logD"]


def test_logd_against_measured_values():
    """Where an experimental logD(7.4) is well established, the estimate lands near
    it.  Crippen's logP carries its own error, so this is deliberately loose -- it
    catches a sign error or a factor of ten, not a tenth of a log unit."""
    for smiles, measured in ((ATENOLOL, -1.8), (ASPIRIN, -2.0), (IBUPROFEN, 0.6)):
        assert props(smiles)["logD"] == pytest.approx(measured, abs=1.5), smiles


def test_the_egg_knows_what_does_not_reach_the_brain():
    """The BOILED-Egg, as published: imatinib is absorbed but does not get into the
    brain, and atenolol -- polar, famously not CNS-active -- does not either."""
    assert props(IMATINIB)["GI absorption"] == "high"
    assert props(IMATINIB)["BBB permeant"] == "no"
    assert props(ATENOLOL)["BBB permeant"] == "no"
    assert props(PROPRANOLOL)["BBB permeant"] == "yes"  # it is CNS-active
    # the ellipses are the published ranges: white TPSA 0-142.1 and WLOGP -2.3-6.8,
    # yolk TPSA -2.9-79.1 and WLOGP 0.4-6.0.  The yolk sits inside the white, so the
    # three places a molecule can be are yolk, white-but-not-yolk, and outside both
    assert admet.boiled_egg(38.10, 3.20) == ("high", "yes")  # the yolk's own centre
    assert admet.boiled_egg(120.0, 2.0) == ("high", "no")  # too polar for the brain
    assert admet.boiled_egg(190.0, 7.5) == ("low", "no")  # outside both


def test_a_permanent_charge_and_a_zwitterion_are_flagged():
    """logD is a Henderson-Hasselbalch estimate that assumes only the neutral species
    partitions, which is wrong for these; the report says so rather than printing a
    confident number."""
    assert any("zwitterion" in c for c in props(ARGININE)["caveats"])
    quaternary = props("C[N+](C)(C)CCO")  # choline: charged at every pH
    assert any("permanently charged" in c for c in quaternary["caveats"])
    assert not props(CAFFEINE)["caveats"]


def test_a_second_basic_centre_is_weakened():
    """Two basic nitrogens close together are not two independent bases (piperazine is
    9.8 and 5.6, not 9.8 twice).  Counted as independent, logD comes out far too low."""
    mol = Chem.MolFromSmiles("C1CN(C)CCN1C")  # dimethylpiperazine
    sites = admet.ionisable(mol, refine=False)
    bases = sorted(pka for kind, _l, _a, pka in sites if kind == "base")
    assert len(bases) == 2 and bases[0] < bases[1] - 3
    assert any("second basic centre" in label for _k, label, _a, _p in sites)
    # the forest sees the same two centres and still keeps them apart, by less: it was
    # fitted on measured values and the table's flat 4 units is a worst case
    refined = sorted(pka for kind, _l, _a, pka in admet.ionisable(mol) if kind == "base")
    assert len(refined) == 2 and refined[0] < refined[1] - 1


def test_each_atom_is_claimed_by_one_group_only():
    """A molecule with a phenol and a carboxylic acid has one of each, not a phenol
    counted again as something else: a double count would double its logD correction."""
    sites = admet.ionisable(Chem.MolFromSmiles("OC(=O)c1ccccc1O"))  # salicylic acid
    atoms = [a for _k, _l, a, _p in sites]
    assert len(atoms) == len(set(atoms))
    # which holds across the extra sites too: Dimorphite-DL is only asked for atoms the
    # table left unclaimed, so nothing is corrected for twice
    classes = {label.split(" (")[0].replace(" [refined]", "") for _k, label, _a, _p in sites}
    assert classes == {"aryl carboxylic acid", "phenol"}


def test_the_filters_say_what_was_broken():
    """A filter returns what it fails, so the report can name the rule rather than
    only saying no."""
    assert admet.lipinski(mw=200, logp=1, hbd=1, hba=2) == []
    assert admet.lipinski(mw=700, logp=7, hbd=9, hba=20) == ["MW > 500", "logP > 5",
                                                             "HBD > 5", "HBA > 10"]  # fmt: skip
    assert props(IMATINIB)["Lipinski"] == "pass"  # 493 Da: inside, just
    assert props(IMATINIB)["Lipinski violations"] == 0
    assert "fails" in props("CCCCCCCCCCCCCCCCCCCCCCCC")["Muegge"]


def test_alerts_and_accessibility_come_from_rdkit():
    """PAINS and Brenk from RDKit's own catalogues, and the Ertl synthetic
    accessibility, so nothing has to be shipped or downloaded."""
    p = props(ASPIRIN)
    assert p["PAINS"] == 0 and p["PAINS alerts"] == "none"
    assert p["Brenk"] >= 1 and p["Brenk alerts"] != "none"  # its ester
    assert 1 <= p["synthetic accessibility"] <= 10
    # a catechol is the classic PAINS/frequent-hitter substructure
    assert props("Oc1ccccc1O")["Brenk"] >= 1


def test_a_molecule_rdkit_cannot_read_does_not_stop_the_rest(tmp_path):
    """One odd design must not empty the table or fail the report."""
    from boonza.viewer.outputs import _properties

    class NotAMolecule:
        natoms = 0

    assert _properties(NotAMolecule()) == {}


def test_the_report_is_one_self_contained_page():
    """It has to open on a machine that has never heard of boonza, and carry nothing
    that reaches the network: a chemist mails it to a colleague."""
    designs = []
    for name, smiles in (("LIG1", ASPIRIN), ("LIG2", IMATINIB)):
        mol = Chem.MolFromSmiles(smiles)
        designs.append((name, smiles, "", admet.properties(mol), mol))
    page = admet.report_html(designs, "series 1", 7.4)
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    for name in ("LIG1", "LIG2"):
        assert f">{name}<" in page
    assert "<svg" in page  # the egg and a drawing of each design
    # nothing is fetched: no stylesheet, frame, image or script from anywhere.  The
    # only http: left are XML namespaces, which name a vocabulary and are never
    # fetched -- RDKit's drawings carry the SVG one and its own
    assert "<link" not in page and "<iframe" not in page and "src=" not in page
    assert "@import" not in page and "url(" not in page
    for url in re.findall(r"https?://[^\s\"'<>]+", page):
        assert url.startswith(("http://www.w3.org/", "http://www.rdkit.org/")), url
    # one inline script, which only sorts the table that is already there
    assert page.count("<script>") == 1 and "fetch(" not in page and "XMLHttpRequest" not in page


def test_the_report_names_what_cannot_be_trusted():
    """A caveat belongs beside the number it is about, and the method behind every
    row has to be findable: these are estimates, and the page says so."""
    mol = Chem.MolFromSmiles(ARGININE)
    page = admet.report_html([("LIG1", ARGININE, "a note of mine", admet.properties(mol), mol)],
                             "series 1", 7.4)  # fmt: skip
    assert "zwitterion" in page and "Log D caveat" in page
    assert "a note of mine" in page
    assert "estimated from the structure alone" in page
    for cited in ("Daina", "Delaney", "Ertl", "Lipinski", "Crippen"):
        assert cited in page, cited
    # the egg's own limitation, where a reader will otherwise think it is a bug
    assert "caffeine" in page


def test_the_report_escapes_what_a_design_is_called():
    """A design's name and note are whatever was typed, and they land in HTML."""
    mol = Chem.MolFromSmiles(CAFFEINE)
    page = admet.report_html([("<script>x</script>", CAFFEINE, 'a "quoted" & <b>bold</b> note',
                               admet.properties(mol), mol)], "series 1")  # fmt: skip
    assert "<script>x</script>" not in page.replace("<script>\n", "")
    assert "&lt;script&gt;" in page and "&amp;" in page
    assert page.count("<script>") == 1  # still only the sort one


def test_the_table_columns_are_a_subset_of_the_report():
    """The series' table shows the columns worth ranking by; every one of them has to
    be something the report actually computes, or the table would show blanks."""
    everything = admet.properties(Chem.MolFromSmiles(IMATINIB))
    for column in admet.TABLE_COLUMNS:
        assert column in everything, column


def test_the_table_sorts_names_that_start_with_a_digit():
    """A design called 2PM1 is not the number 2, though parseFloat says it is -- that
    made every name compare equal, so clicking the column did nothing at all.  And
    2PM10 comes after 2PM2, which a plain text compare gets backwards.  The page's
    sort reads a cell as a number only when the whole of it is one, and compares text
    with its digits as digits."""
    page = admet.report_html([], "series 1")
    script = page[page.index("<script>") :]
    assert "parseFloat(" not in script  # the bug: it stops at the first letter
    assert "isFinite(Number(text))" in script  # the whole cell, or it is text
    assert "numeric: true" in script  # 2PM2 before 2PM10


def test_a_design_can_be_reached_from_the_table_and_from_a_point():
    """Clicking a design anywhere -- its row, its point on the egg, its point on the
    plot -- goes to its panel, which is the whole use of an overview."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    designs = [(f"2PM{k}", ASPIRIN, "", admet.properties(mol), mol) for k in (1, 2)]
    page = admet.report_html(designs, "series 1")
    assert '<div class="card design" id="design-0"' in page
    assert '<tr data-design="0" data-row="0">' in page
    assert '<tr data-design="1" data-row="1">' in page
    assert 'class="pt" data-design="0"' in page  # on the egg
    assert "scrollIntoView" in page and "classList.add('lit')" in page


def test_any_property_can_go_on_either_axis():
    """A plot of one property against another, chosen by the reader, below the
    summary: the numbers are embedded for it, and nothing is fetched to draw it."""
    mol = Chem.MolFromSmiles(IMATINIB)
    designs = [(f"2PM{k}", IMATINIB, "", admet.properties(mol), mol) for k in (1, 2, 3)]
    page = admet.report_html(designs, "series 1")
    assert 'id="plot-x"' in page and 'id="plot-y"' in page and 'id="plot"' in page
    for column in ("MW", "cLogP", "logD", "TPSA", "QED"):
        assert f'<option value="{column}"' in page, column
    # the data the plot draws from is in the page, with nothing that could close the
    # script element early
    assert '"n": "2PM1"' in page or '"n":"2PM1"' in page
    assert "</script" not in page[page.index("<script>") + 8 : page.rindex("</script>")]


def test_the_report_has_a_theme_of_its_own():
    """It follows the reader's system by default and can be set either way, which is
    what a report opened beside a paper needs."""
    mol = Chem.MolFromSmiles(CAFFEINE)
    page = admet.report_html([("2PM1", CAFFEINE, "", admet.properties(mol), mol)], "series 1")
    assert 'id="theme"' in page and "prefers-color-scheme: dark" in page
    assert ':root[data-theme="dark"]' in page and ':root:not([data-theme="light"])' in page
    assert "boonza.report.theme" in page  # remembered for next time
    # a structure drawn for white paper vanishes on a dark page, so there is one of each
    assert 'class="light-only"' in page and 'class="dark-only"' in page
    # the SMILES is the molecule: it is read and copied, so it is not greyed out
    css = page[page.index("<style>") : page.index("</style>")]
    assert ".smiles" in css and "color: var(--text)" in css.split(".smiles")[1][:160]
    assert ".struct { background: var(--bg)" in css


def test_soft_spots_find_the_routes_these_drugs_are_known_to_take():
    """The flags are checked against metabolism that is not in doubt."""

    def labels(smiles):
        return {s["label"] for s in admet.soft_spots(Chem.MolFromSmiles(smiles))}

    # aspirin is hydrolysed to salicylic acid before anything else happens to it
    assert "ester hydrolysis" in labels(ASPIRIN)
    # propranolol: 4-hydroxylation and N-dealkylation are its two main routes
    assert {"aromatic hydroxylation", "N-dealkylation"} <= labels(PROPRANOLOL)
    # imatinib's major metabolite is N-desmethyl imatinib
    assert "N-dealkylation" in labels(IMATINIB)
    # a thioether oxidises; caffeine's nitrogens are amides and are not basic sites
    assert "S-oxidation" in labels("CSc1ccccc1")
    assert "N-oxidation" not in labels(CAFFEINE)
    # every flag carries where it is and what happens there, for the drawing and the list
    for spot in admet.soft_spots(Chem.MolFromSmiles(PROPRANOLOL)):
        assert spot["atoms"] and spot["note"]


def test_the_report_says_the_soft_spots_are_not_a_prediction():
    """They are substructure flags.  SMARTCyp ranks atoms by a computed activation
    energy; this does not, and the page must not let a reader think it does."""
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    page = admet.report_html([("2PM1", PROPRANOLOL, "", admet.properties(mol), mol)], "series 1")
    assert "Metabolic soft spots" in page and "N-dealkylation" in page
    assert "not SMARTCyp" in page and "no rate and no order" in page


def test_the_soft_spots_can_be_switched_off():
    """Sometimes the structure wants reading plain.  One switch hides every mark on
    every drawing and the lists with them, and it is remembered."""
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    page = admet.report_html([("2PM1", PROPRANOLOL, "", admet.properties(mol), mol)], "series 1")
    assert 'id="spots"' in page and "metabolic soft spots</label>" in page
    assert ":root.no-spots .soft, :root.no-spots .spots-block { display: none; }" in page
    assert 'class="spots-block"' in page
    assert "boonza.report.spots" in page  # remembered, like the theme


def test_every_mark_is_tagged_in_both_drawings():
    """The mark is tagged by where it is, not by its colour: RDKit rounds a colour its
    own way when it writes it out (0.65 came back as A5, not A6), so matching the hex
    missed every mark on the dark drawing and left it unswitchable."""
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    spots = sorted({a for s in admet.soft_spots(mol) for a in s["atoms"]})
    assert spots
    for dark in (False, True):
        svg = admet.depiction(mol, dark=dark, highlight=spots)
        assert svg.count(" soft") == len(spots), dark
        assert "class='bond" in svg  # the structure itself is untouched
    # a drawing with nothing to mark gains nothing
    assert " soft" not in admet.depiction(mol)
    # and the purple is not the page's amber, which is why an amber mark did not read
    assert admet.SPOT_COLOUR[False] != admet.SPOT_COLOUR[True]
    page = admet.report_html([("2PM1", PROPRANOLOL, "", admet.properties(mol), mol)], "series 1")
    assert "--spot: #" in page


def test_every_figure_is_copied_as_vector_with_no_background():
    """Inline SVG gives the browser no "Copy image" of its own, and a picture of a
    drawing is the wrong thing anyway: a figure wants shapes that stay sharp.  Each
    drawing, the plot and the egg carry copy (to the clipboard) and svg (to a file),
    both vector, and nothing is painted behind them, so they drop onto a slide or a
    page of any colour."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    designs = [(f"2PM{k}", ASPIRIN, "", admet.properties(mol), mol) for k in (1, 2, 3)]
    page = admet.report_html(designs, "series 1")
    # a figure each: the three drawings, the plot and the egg (the script mentions
    # the attribute once more, which is why the count is taken over the body alone)
    body = page[: page.index("<script>")]
    assert body.count("data-copy") == 5 and body.count("data-svg") == 5
    for figure in ('data-figure="2PM1"', 'data-figure="plot"', 'data-figure="BOILED-Egg"'):
        assert figure in page, figure
    # the egg's pair sits where every drawing's does, the top right of its own card,
    # rather than in a row under it
    css = page[page.index("<style>") : page.index("</style>")]
    assert ".struct .figbtns, .egg .figbtns { position: absolute; top: 5px; right: 5px;" in css
    assert ".card.egg { position: relative; }" in css
    assert "figrow" not in page
    script = page[page.index("<script>") :]
    assert "'image/svg+xml'" in script  # vector on the clipboard
    assert "toBlob" not in script and "canvas" not in script  # nothing is rasterised
    assert "fillRect" not in script  # and nothing is painted behind it
    # the drawings themselves carry no background either
    assert "opts.setBackgroundColour" not in page
    assert "<svg" in page
    # a plot's axes are drawn in the page's colours, which a copy would otherwise lose
    assert "copy.setAttribute('color', ink)" in script
    # the marks are left out of a copy when they are switched off
    assert "no-spots" in script and "removeChild" in script
    # and a copy is not also a jump to the design the button sits on
    assert "e.target.closest('[data-copy]')" in script


def test_the_plot_comes_before_the_egg():
    """The egg is a fixed pair of ellipses; the plot is whatever the reader asks for,
    so it is the one they want first."""
    mol = Chem.MolFromSmiles(IMATINIB)
    designs = [(f"2PM{k}", IMATINIB, "", admet.properties(mol), mol) for k in (1, 2, 3)]
    page = admet.report_html(designs, "series 1")
    import re

    assert re.findall(r"<h2>(.*?)</h2>", page) == [
        "series 1 at a glance",
        "One property against another",
        "Passive absorption and brain penetration",
        "Every design",
    ]


# --- ion-pair partitioning, the pKa table, phase II and triage -------------------------


def test_an_ionised_molecule_keeps_a_floor_under_its_logd():
    """Textbook Henderson-Hasselbalch treats an ion as not partitioning at all, which
    has no floor: a strongly ionised design's logD falls without limit.  An ion does
    partition, as an ion pair, three log units below the neutral form for a base and
    four for an acid."""
    # a base far above its pKa: fully protonated, so logD is the plateau and nothing
    # more.  Without the floor this would be logP - 7
    sites = [("base", "amine", 0, 14.4)]
    assert admet.log_d(3.0, sites, 7.4) == pytest.approx(3.0 - admet.ION_PAIR_DROP_BASE, abs=0.01)
    sites = [("acid", "sulfonic acid", 0, 0.4)]
    assert admet.log_d(3.0, sites, 7.4) == pytest.approx(3.0 - admet.ION_PAIR_DROP_ACID, abs=0.01)
    # at its own pKa the molecule is half neutral, so logD is about logP - 0.3 either way
    assert admet.log_d(3.0, [("base", "amine", 0, 7.4)], 7.4) == pytest.approx(2.7, abs=0.01)
    # a neutral molecule is untouched, whichever rule is in force
    assert admet.log_d(3.0, [], 7.4) == pytest.approx(3.0)


def test_the_floor_is_what_takes_the_bias_off_the_cations():
    """Propranolol is a protonated amine at pH 7.4, and without an ion-pair floor its
    logD was read about a log unit too low.  On the 1000 measured logD values the
    package ships, the floor takes a bias of -0.82 on the cations to +0.11."""
    def two_ways(smiles):
        mol = Chem.MolFromSmiles(smiles)
        sites = admet.ionisable(mol)
        logp = props(smiles)["cLogP"]
        try:  # textbook Henderson-Hasselbalch, which is what the floor replaced
            admet.ION_PAIR_DROP_ACID = admet.ION_PAIR_DROP_BASE = None
            without = admet.log_d(logp, sites)
        finally:
            admet.ION_PAIR_DROP_ACID, admet.ION_PAIR_DROP_BASE = 4.0, 3.0
        return admet.log_d(logp, sites), without

    # a guanidine is protonated several log units past pH 7.4, which is where the
    # textbook rule runs away and the floor is the whole difference
    floored, textbook = two_ways("CN(C)C(=N)NC(=N)N")  # metformin
    assert floored > textbook + 2.0
    # a plain amine is only two units past it, so the floor moves it very little: the
    # bias it takes off the cations is not a constant, it is worst where logD is lowest
    floored, textbook = two_ways(PROPRANOLOL)
    assert floored >= textbook
    assert props(PROPRANOLOL)["ionised form"] == "cation"
    assert props(IMATINIB)["ionised form"] == "cation"
    # an acid still drops a long way below its logP: that much was never wrong
    assert props(ASPIRIN)["logD"] < props(ASPIRIN)["cLogP"] - 2.5


def test_a_measured_pka_replaces_the_table():
    """The table has one value per functional class, so it is the crude part of logD.
    A measured value is the single biggest thing that improves one, and it wins
    outright rather than being averaged in."""
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    table = admet.properties(mol)
    assert "amine [refined]" in table["ionisable groups"]
    given = admet.properties(mol, measured={"base": [10.5]})
    assert given["ionisable groups"] == "measured base 10.5"
    assert given["logD"] < table["logD"]  # a stronger base is more ionised
    # and the design says so, since the number no longer came from the table
    assert any("measured pKa" in c for c in given["caveats"])
    # a measured acid does not silence the bases, or the other way round
    both = admet.ionisable(mol, {"acid": [4.0]})
    assert ("acid", "measured acid", None, 4.0) in both
    assert any(label.startswith("amine") for _k, label, _a, _p in both)
    # and nothing else of that kind creeps back in beside it: an extra site found by
    # substructure would quietly undo the measurement it was given instead of
    assert [s for s in both if s[0] == "acid"] == [("acid", "measured acid", None, 4.0)]


def test_an_aromatic_ring_n_does_not_weaken_the_ring_n_beside_it():
    """A second basic centre near a stronger one is much weaker, which is right for
    piperazine.  It was applied to any base, so the second ring N of every pyrimidine,
    purine and quinazoline was pushed near 1 where the measured median is 4.4 -- an
    aromatic N is mostly neutral at pH 7.4 and has no charge to lend."""

    def values(smiles):  # the table's own values: this is the table's rule
        mol = Chem.MolFromSmiles(smiles)
        sites = admet.ionisable(mol, refine=False)
        return [pka for _k, _l, _a, pka in sites if _k == "base"]

    assert all(v > 4 for v in values("c1ccc2ncncc2c1"))  # quinazoline: both ring N stand
    assert all(v > 4 for v in values("c1cnc2[nH]cnc2c1"))  # purine
    # an aliphatic base still weakens its neighbour, which is the case the rule is for
    assert min(values("CN1CCNCC1")) < 5  # piperazine
    assert min(values("NCc1ccccn1")) < 3  # 2-picolylamine: the ring N is pulled right down


def test_the_pka_table_prices_the_classes_a_chemist_draws():
    """The values are medians of measured drug-like compounds, not the textbook pKa of
    the unsubstituted parent, and the classes a parent-value table had no entry for are
    there now.  These are the table's own values, before the forest prices each site by
    its substituents."""

    def groups(smiles):
        sites = admet.ionisable(Chem.MolFromSmiles(smiles), refine=False)
        return "; ".join(f"{label} {pka:g}" for _k, label, _a, pka in sites)

    assert "amine 8.8" in groups(PROPRANOLOL)  # not the alkylamine 9.7
    # classes that were invisible before: a tetrazole drawn either way, a lactam NH, a
    # hydroxamic acid, an azole NH
    for smiles, label in [("c1ccc(-c2nn[nH]n2)cc1", "tetrazole"),
                          ("O=c1cccc[nH]1", "conjugated lactam NH"),
                          ("CC(=O)NO", "hydroxamic acid"),
                          ("c1cc[nH]n1", "azole NH")]:  # fmt: skip
        assert label in groups(smiles), smiles
    # a carboxylate drawn as an anion is the same site, not an absent one
    assert "carboxylate (as drawn)" in groups("CC(=O)[O-]")
    # an activated phenol is a stronger acid than a plain one
    assert "phenol (EWG-substituted)" in groups("Oc1ccc([N+](=O)[O-])cc1")


def test_phase_two_finds_the_conjugation_routes_these_drugs_are_known_to_take():
    """Soft spots ask where CYP oxidises a molecule; this asks which groups are already
    handles for conjugation, so it can be cleared without being oxidised first.  A
    compound with a free phenol is often cleared by UGT whatever its CYP liabilities."""

    def families(smiles):
        return {s["family"] for s in admet.phase2_sites(Chem.MolFromSmiles(smiles))}

    def groups(smiles):
        return {g for s in admet.phase2_sites(Chem.MolFromSmiles(smiles)) for g in s["groups"]}

    # paracetamol's two routes are exactly glucuronidation and sulfation of the phenol
    assert families("CC(=O)Nc1ccc(O)cc1") == {"UGT", "SULT"}
    # morphine gives the 3- (phenol) and the 6- (aliphatic OH) glucuronide
    morphine = "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5"
    assert {"phenol O-glucuronide", "aliphatic OH O-glucuronide"} <= groups(morphine)
    # isoniazid is the classic slow/fast acetylator drug
    assert "NAT" in families("NNC(=O)c1ccncc1")
    # entacapone's catechol is COMT's substrate
    assert "COMT" in families("CCN(CC)C(=O)C(=Cc1cc([N+](=O)[O-])c(O)c(O)c1)C#N")
    # caffeine is cleared by CYP1A2 and has no phase II handle at all
    assert not admet.phase2_sites(Chem.MolFromSmiles(CAFFEINE))


def test_a_reactive_conjugate_and_a_polymorphic_enzyme_are_called_out():
    """The site list is the least of it: what changes what a chemist does next is that
    the conjugate is itself reactive, or that the enzyme is polymorphic in people."""
    diclofenac = admet.phase2_sites(Chem.MolFromSmiles("OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"))
    warnings = admet.phase2_warnings(diclofenac)
    assert any("acylate protein" in w for w in warnings)
    isoniazid = admet.phase2_sites(Chem.MolFromSmiles("NNC(=O)c1ccncc1"))
    assert any("polymorphic" in w for w in admet.phase2_warnings(isoniazid))
    # each warning once, however many sites earn it
    assert len(warnings) == len(set(warnings))


def test_phase_two_says_it_is_a_route_and_not_a_rate():
    """It is a substructure table.  It says a route is open, never that it dominates,
    and the page must not let a reader read it as a prediction."""
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    page = admet.report_html([("2PM1", "", "", admet.properties(mol), mol)], "series 1")
    assert "Conjugation handles" in page and "UGT" in page
    assert "which conjugation routes are open, never a rate" in page
    assert "never which route wins" in page


def test_triage_gives_one_verdict_and_the_reason_for_it():
    """A series of forty rows is read as a shortlist, which is only safe if the rule
    that produced a verdict comes with it."""
    clean = {"Lipinski violations": 0, "PAINS": 0, "TPSA": 60.0, "Brenk": 0, "QED": 0.7,
             "logD": 2.0, "GI absorption": "high", "synthetic accessibility": 3.0}  # fmt: skip
    assert admet.triage(clean) == ("pass", [])
    # a "fail" rule fails it; a "review" rule only flags it
    level, why = admet.triage({**clean, "PAINS": 2})
    assert level == "fail" and why == ["PAINS alerts 2"]
    level, why = admet.triage({**clean, "QED": 0.1})
    assert level == "review" and why == ["QED below 0.1"]
    # the worst rule decides, and every breach is named
    level, why = admet.triage({**clean, "QED": 0.1, "TPSA": 180.0})
    assert level == "fail" and len(why) == 2
    # a threshold can be moved
    assert admet.triage({**clean, "QED": 0.1}, {"qed": 0.05})[0] == "pass"
    # a property that could not be computed is not counted against the design
    short = {k: v for k, v in clean.items() if k != "synthetic accessibility"}
    assert admet.triage(short) == ("pass", [])


def test_the_report_recomputes_triage_from_the_same_rules_it_was_written_with():
    """The page lets a threshold be moved, so it computes verdicts itself.  It is handed
    the very table :func:`triage` uses, since two copies of the rules would drift."""
    mol = Chem.MolFromSmiles(IMATINIB)
    designs = [(f"2PM{k}", IMATINIB, "", admet.properties(mol), mol) for k in (1, 2)]
    page = admet.report_html(designs, "series 1")
    assert "Triage rules" in page and 'id="tally"' in page
    for key, label, _f, _o, _d, _lv in admet.TRIAGE_FIELDS:
        assert f'id="c-{key}"' in page and label in page
    script = page[page.index("<script>") :]
    assert '"key": "qed"' in script and '"level": "review"' in script
    # the verdict is in the table and on the panel, both keyed so the script finds them
    assert 'data-verdict="0"' in page and 'class="verdict v-' in page
    # and it can be moved without the page recomputing anything from the structure
    assert "every rule reads a number already on the page" in script
    assert "Triage is a filter, not a finding" in page


def test_the_table_sorts_on_several_columns_at_once():
    """One column at a time answers "the best logD"; two answer "the best logD among
    the ones that pass", which is how a series is actually read."""
    mol = Chem.MolFromSmiles(IBUPROFEN)
    designs = [(f"2PM{k}", IBUPROFEN, "", admet.properties(mol), mol) for k in (1, 2)]
    page = admet.report_html(designs, "series 1")
    assert "shift-click" in page
    script = page[page.index("<script>") :]
    # the keys are a list, most significant first, and a tie falls to the next one
    assert "keys.push({ col: col, down: false })" in script
    assert "if (c) return keys[k].down ? -c : c;" in script
    # the headings number the keys, so the order they are applied in is never a guess
    assert "keys.length > 1 ? (rank + 1)" in script


# --- the fitted models ----------------------------------------------------------------


def test_the_forests_are_the_ones_that_were_fitted():
    """Both are data files, so the first thing worth checking is that they loaded and
    that their feature lists are the ones the code computes -- a forest walked with its
    columns in the wrong order gives plausible numbers that are entirely wrong."""
    pka, logs = admet._model("pka_forest"), admet._model("logs_forest")
    assert pka is not None and logs is not None, "the data files should ship"
    assert len(pka["trees"]) == 200 and len(logs["trees"]) == 120
    assert tuple(logs["features"]) == admet.SOLUBILITY_FEATURES
    assert pka["features"][:3] == ["is_acid", "table_pka", "has_table_value"]
    # the feature vectors are as long as the forests expect
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    assert len(admet.solubility_features(mol)) == len(logs["features"])
    assert len(admet._pka_features(mol, 3, "base", 8.8)) == len(pka["features"])
    # every tree reads only features that exist
    for tree in pka["trees"]:
        assert max(tree[0]) < len(pka["features"])


def test_a_pka_is_priced_by_its_substituents_not_only_by_its_class():
    """The table gives one value per class, which is the crude part of logD: every
    aliphatic amine was 8.8 whatever sat beside it.  The forest keeps the class value as
    its strongest input and corrects it from the local environment."""
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    table = [p for k, _l, _a, p in admet.ionisable(mol, refine=False) if k == "base"]
    refined = [p for k, _l, _a, p in admet.ionisable(mol) if k == "base"]
    assert table == [8.8]
    assert refined != table  # propranolol's amine is measured at 9.5, not the class 8.8
    assert 8.5 < refined[0] < 10.5
    # an amine beside an electron-withdrawing group must come out weaker than one that
    # is not, which a single class value cannot express at all
    def amine(smiles):
        sites = admet.ionisable(Chem.MolFromSmiles(smiles))
        return max(p for k, _l, _a, p in sites if k == "base")

    assert amine("CCCCNC") > amine("FC(F)(F)CNC")
    # and the label says where the number came from, since a reader cannot tell
    # otherwise which of the two it is looking at
    assert all("[refined]" in label for _k, label, _a, _p in admet.ionisable(mol))


def test_the_forest_is_not_allowed_to_flatten_a_class_that_is_strong_by_definition():
    """A forest averages leaves, so it cannot extrapolate past what it was fitted on,
    and the collection has almost nothing above 12.  Left alone it prices butylguanidine
    (measured 13.6) at 8.9 and metformin (12.4) at 5.8 -- which would make a permanent
    cation read as half neutral.  Outside :data:`REFINE_RANGE` the table stands."""
    for smiles in ("CCCCNC(=N)N", "CN(C)C(=N)NC(=N)N"):
        mol = Chem.MolFromSmiles(smiles)
        bases = [p for k, _l, _a, p in admet.ionisable(mol) if k == "base"]
        assert max(bases) >= 12.0, smiles
        assert admet.ionised_form(admet.ionisable(mol)) == "cation"
    # inside the window the forest does speak: this costs 0.014 of mean error on the
    # 1000 measured values the window was chosen against, which have nothing above 12
    assert admet.REFINE_RANGE == (1.0, 12.0)
    mol = Chem.MolFromSmiles(IBUPROFEN)
    assert "[refined]" in admet.ionisable(mol)[0][1]


def test_an_ionisable_group_the_table_misses_is_still_found():
    """The forest improves the number, not the coverage: finding a site the table has no
    pattern for is what Dimorphite-DL's substructures are for.  On the 1000 measured
    values the site is located for 927 rather than 916."""
    # caffeine's ring nitrogens: the table claims the imidazole-type one and has no
    # pattern for the amide-flanked ones, which are weak bases all the same
    mol = Chem.MolFromSmiles(CAFFEINE)
    plain = admet.ionisable(mol, refine=False)
    found = admet.ionisable(mol)
    assert len(found) > len(plain)
    extra = [s for s in found if "Dimorphite-DL" in s[1]]
    assert extra
    # and their values came from the forest, not from Dimorphite's own class means,
    # which on exactly these sites score 2.17 against the forest's 1.27
    assert all("[refined]" in label for _k, label, _a, _p in extra)
    # an extra site never lands on an atom the table already claimed
    atoms = [a for _k, _l, a, _p in found]
    assert len(atoms) == len(set(atoms))


def test_a_measured_pka_is_not_undercut_by_a_site_found_by_substructure():
    """A measured value replaces the table's sites of its kind.  The atoms those sites
    stood on must stay claimed, or the substructure search finds the same group again --
    and it calls any atom carrying a hydrogen an acid, so a secondary amine came back
    as a weak acid beside the very base that was measured."""
    mol = Chem.MolFromSmiles(PROPRANOLOL)
    sites = admet.ionisable(mol, {"base": [9.5]})
    assert sites == [("base", "measured base", None, 9.5)]
    assert not [s for s in sites if s[0] == "acid"]


def test_solubility_is_fitted_and_the_published_equations_are_still_printed():
    """Solubility is not linear in logP and the published equations are.  The forest
    scores 0.92 mean error on 1000 measured values against ESOL's 1.18, so it is what
    the class is read from -- but all three are printed, because three numbers
    disagreeing is itself worth seeing."""
    p = props(IBUPROFEN)
    assert "log S (fitted)" in p and "log S (ESOL)" in p and "log S (Ali)" in p
    assert p["solubility"] == admet.solubility_class(p["log S (fitted)"])
    # measured: ibuprofen is about -3.6, caffeine about -0.9
    assert p["log S (fitted)"] == pytest.approx(-3.6, abs=1.3)
    assert props(CAFFEINE)["log S (fitted)"] == pytest.approx(-0.9, abs=1.3)
    # and the report says which is which
    mol = Chem.MolFromSmiles(IBUPROFEN)
    page = admet.report_html([("2PM1", IBUPROFEN, "", admet.properties(mol), mol)], "s")
    assert "Log S (fitted, AqSolDB)" in page and "Log S (ESOL)" in page
    assert "Log S is fitted too" in page


def test_the_report_degrades_to_the_published_equations_without_the_data_files():
    """The forests are data files, and a wheel built without them, or a source tree
    with them deleted, must still write a report rather than fail."""
    saved = dict(admet._MODELS), admet._DIMORPHITE
    try:
        admet._MODELS["pka_forest"] = admet._MODELS["logs_forest"] = None
        admet._DIMORPHITE = []
        mol = Chem.MolFromSmiles(PROPRANOLOL)
        p = admet.properties(mol)
        assert "log S (fitted)" not in p  # absent rather than guessed at
        assert p["solubility"] == admet.solubility_class(p["log S (ESOL)"])
        assert "amine 8.8" in p["ionisable groups"]  # the table, unrefined
        assert "[refined]" not in p["ionisable groups"]
        page = admet.report_html([("2PM1", PROPRANOLOL, "", p, mol)], "series 1")
        assert "Log S (ESOL)" in page and "Log S (fitted" not in page
    finally:
        admet._MODELS.clear()
        admet._MODELS.update(saved[0])
        admet._DIMORPHITE = saved[1]


def test_the_forests_carry_the_attribution_their_licences_require():
    """Both are redistributed under licences that permit it only with attribution, and
    Dimorphite-DL's Apache licence also asks that the changes be stated.  The notices
    ship beside the data, not only in a commit message."""
    readme = (admet.DATA / "README.md").read_text()
    for owed in ("Baltruschat", "CC BY 4.0", "AqSolDB", "CC0", "Durrant",
                 "Apache License 2.0", "doi:10.1186/s13321-019-0336-9"):
        assert owed in readme, owed
    assert "Changes from the original" in readme
    licence = (admet.DATA / "LICENSE.dimorphite-dl").read_text()
    assert "Apache License" in licence and "Version 2.0, January 2004" in licence
