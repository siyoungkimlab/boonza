"""Notebook views: the HTML carries the right atoms, styles and frames."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.view import THREEDMOL_URL

DATA = Path(__file__).parent / "data"


def test_view_html(tmp_path):
    s = boonza.load(DATA / "1HHO.pdb")
    v = s.view()
    html = v._repr_html_()
    assert THREEDMOL_URL in html and "addModel(" in html
    assert html.count("ATOM  ") + html.count("HETATM") == s.natoms
    assert '"cartoon"' in html and '"stick"' in html and '"line"' not in html  # water hidden
    assert html.count("CONECT") > 0  # heme bonds are given, not guessed by 3Dmol
    ids = [int(x) for x in html.split('"serial": [')[1].split("]")[0].split(",")]
    assert ids[0] == 1 and len(ids) == len(s.select("polymer"))
    v.save(tmp_path / "hb.html")
    assert (tmp_path / "hb.html").read_text().startswith("<!doctype html>")

    frames = np.stack([s.positions, s.positions + 1.0, s.positions + 2.0])
    anim = boonza.view(s, "protein and chain A", positions=frames, style="sticks")
    text = anim.to_html()
    n = len(s.select("protein and chain A"))
    assert "addModelsAsFrames" in text and text.count("MODEL ") == 3
    assert text.count("ATOM  ") == 3 * n and "animate" in text
    assert boonza.view(s, water=True).to_html().count('"line"') == 1
    with pytest.raises(ValueError, match="style"):
        boonza.view(s, style="ribbons")
