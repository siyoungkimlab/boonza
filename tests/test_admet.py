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
    sites = admet.ionisable(mol)
    bases = sorted(pka for kind, _l, _a, pka in sites if kind == "base")
    assert len(bases) == 2 and bases[0] < bases[1] - 3
    assert any("second basic centre" in label for _k, label, _a, _p in sites)


def test_each_atom_is_claimed_by_one_group_only():
    """A molecule with a phenol and a carboxylic acid has one of each, not a phenol
    counted again as something else: a double count would double its logD correction."""
    sites = admet.ionisable(Chem.MolFromSmiles("OC(=O)c1ccccc1O"))  # salicylic acid
    atoms = [a for _k, _l, a, _p in sites]
    assert len(atoms) == len(set(atoms))
    assert {label.split(" (")[0] for _k, label, _a, _p in sites} == {"aryl carboxylic acid",
                                                                     "phenol"}  # fmt: skip


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
    assert '<tr data-design="0">' in page and '<tr data-design="1">' in page
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
        assert spot["atoms"] and spot["note"] and spot["phase"] in (1, 2)


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


def test_a_drawing_can_be_copied_as_a_picture():
    """Inline SVG gives the browser no "Copy image" of its own, so each drawing
    carries a button that puts a PNG on the clipboard, and saves one where the
    clipboard cannot be reached."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    page = admet.report_html([("2PM1", ASPIRIN, "", admet.properties(mol), mol)], "series 1")
    assert "data-copy" in page and ">copy</button>" in page
    script = page[page.index("<script>") :]
    assert "ClipboardItem" in script and "toBlob" in script
    assert "image/png" in script
    # the marks are left out of the copy when they are switched off
    assert "no-spots" in script and "removeChild" in script
    # a copy is not also a jump to the design the button sits on
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
