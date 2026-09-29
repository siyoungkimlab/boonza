"""boonza draw: pictures of molecules from SMILES."""

import struct

import pytest

from boonza.cli import main
from boonza.draw import common_core, draw, molecules, orient, read_smiles

# a congeneric series: one acetanilide core, five substituents
SERIES = [
    "CC(=O)Nc1ccc(O)cc1",
    "CCC(=O)Nc1ccc(O)cc1",
    "CC(=O)Nc1ccc(OC)cc1",
    "CC(=O)Nc1ccc(Cl)cc1",
    "CC(=O)Nc1ccc(F)cc1",
]


def png_size(path) -> tuple[int, int]:
    """A PNG's width and height from its IHDR, so the test needs no imaging library."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", data[16:24])


def test_read_smiles_takes_names_and_skips_comments():
    text = "# a series\nCCO ethanol\n\nc1ccccc1   benzene\nCC\n"
    assert read_smiles(text) == [("CCO", "ethanol"), ("c1ccccc1", "benzene"), ("CC", "")]


def test_one_molecule_fills_the_picture(tmp_path):
    out = tmp_path / "one.png"
    d = draw("CCO", out, size=(300, 250))
    assert d.files == [out] and d.drawn == 1 and not d.failures
    assert png_size(out) == (300, 250)


def test_a_grid_is_as_wide_as_its_columns(tmp_path):
    out = tmp_path / "grid.png"
    d = draw(SERIES, out, columns=3, size=(200, 150))
    w, h = png_size(out)
    assert (w, h) == (600, 300)  # 3 across, 2 rows for 5 molecules
    assert d.drawn == 5


def test_rows_page_the_grid_rather_than_truncating_it(tmp_path):
    """RDKit will happily draw one unreadable strip; --rows splits it instead,
    and every molecule still reaches a page."""
    out = tmp_path / "lib.png"
    d = draw(SERIES, out, columns=2, rows=1, size=(150, 120))
    assert [f.name for f in d.files] == ["lib-1.png", "lib-2.png", "lib-3.png"]
    assert png_size(d.files[0]) == (300, 120)
    assert png_size(d.files[-1]) == (150, 120)  # the last page holds the odd one


def test_every_molecule_is_drawn_even_past_rdkits_old_limit(tmp_path):
    """MolsToGridImage has stopped at 50 molecules in some versions; a library
    that quietly loses its tail is worse than one that fails."""
    out = tmp_path / "many.png"
    many = ["C" * (i % 8 + 1) for i in range(60)]
    draw(many, out, columns=6, size=(100, 80), labels=False)
    assert png_size(out) == (600, 800)  # 6 across, 10 rows


def test_a_bad_smiles_costs_only_itself(tmp_path):
    out = tmp_path / "some.png"
    d = draw([SERIES[0], "not a smiles[", SERIES[1]], out)
    assert d.drawn == 2 and d.failures == ["not a smiles["]
    with pytest.raises(ValueError, match="none of the"):
        draw(["bad[", "worse("], tmp_path / "none.png")


def test_the_mcs_is_the_scaffold_the_series_shares(tmp_path):
    mols, _, _ = molecules(SERIES)
    core = common_core(mols)
    assert core is not None
    assert all(m.HasSubstructMatch(core) for m in mols)
    # the core is the acetanilide, not just any two atoms
    assert core.GetNumAtoms() >= 9

    d = draw(SERIES, tmp_path / "mcs.png", mcs=True)
    assert d.core and "#7" in d.core  # the amide nitrogen is in it


def test_molecules_sharing_nothing_have_no_core_to_mark(tmp_path):
    """Reported rather than drawn as a picture with no highlight, which would
    look like the option had failed."""
    mols, _, _ = molecules(["C1CC1", "c1ccccc1"])
    assert common_core(mols) is None
    d = draw(["C1CC1", "c1ccccc1"], tmp_path / "none.png", mcs=True)
    assert d.core is None and d.drawn == 2


def test_aligning_puts_the_core_the_same_way_up(tmp_path):
    """Every molecule's core atoms land on the same 2D coordinates, which is
    what makes a series comparable by eye."""
    import numpy as np

    mols, _, _ = molecules(SERIES)
    core = common_core(mols)
    assert orient(mols, core) == len(mols)
    first = None
    for mol in mols:
        hit = mol.GetSubstructMatch(core)
        xy = np.array([list(mol.GetConformer().GetAtomPosition(a))[:2] for a in hit])
        if first is None:
            first = xy
        else:
            assert np.allclose(xy, first, atol=1e-6)


def test_highlight_and_mcs_are_not_both_given(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        draw(SERIES, tmp_path / "x.png", mcs=True, highlight="c1ccccc1")
    with pytest.raises(ValueError, match="as SMARTS"):
        draw(SERIES, tmp_path / "x.png", highlight="not)a(smarts")


def test_svg_comes_out_as_text(tmp_path):
    out = tmp_path / "v.svg"
    draw(SERIES[:2], out)
    assert out.read_text().lstrip().startswith("<?xml") and "</svg>" in out.read_text()


def test_only_png_and_svg_are_written(tmp_path):
    with pytest.raises(ValueError, match="draw writes"):
        draw("CCO", tmp_path / "x.jpg")


def test_the_command_line_draws_a_file_of_smiles(tmp_path, capsys):
    lib = tmp_path / "lib.smi"
    lib.write_text("# series\n" + "\n".join(f"{s} name{i}" for i, s in enumerate(SERIES)))
    out = tmp_path / "series.png"
    assert main(["draw", "--input", str(lib), "-o", str(out), "--mcs", "--align",
                 "--columns", "3"]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "5 molecules" in printed and "common core" in printed
    assert "5 of 5 laid out" in printed
    assert png_size(out)[0] == 900  # 3 across at the default width


def test_the_command_line_reports_what_it_could_not_read(tmp_path, capsys):
    out = tmp_path / "x.png"
    assert main(["draw", SERIES[0], "bad[", "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 molecule" in printed and "could not be read" in printed
    assert out.is_file()
