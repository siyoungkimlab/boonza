import numpy as np
from conftest import msys_file

import boonza
from boonza.cli import main


def test_info(capsys):
    assert main(["info", str(msys_file("ww.dms"))]) == 0
    out = capsys.readouterr().out
    assert "atoms" in out and "stretch_harm" in out and "nonbonded" in out


def test_convert_select_and_diff(tmp_path, capsys):
    src = str(msys_file("ww.dms"))
    out = tmp_path / "protein.pdb"
    assert main(["convert", src, str(out), "-s", "protein"]) == 0
    s = boonza.load(src)
    assert boonza.load(out).natoms == len(s.select("protein"))
    capsys.readouterr()
    assert main(["select", src, "name CA and resid 5 to 7"]) == 0
    ids = [int(x) for x in capsys.readouterr().out.split()]
    assert ids == s.select("name CA and resid 5 to 7").ids.tolist()

    copy = tmp_path / "copy.dms"
    assert main(["convert", src, str(copy)]) == 0
    assert main(["diff", src, str(copy)]) == 0
    moved = s.copy()
    moved.positions = moved.positions + 1.0
    boonza.save(moved, tmp_path / "moved.dms")
    capsys.readouterr()
    assert main(["diff", src, str(tmp_path / "moved.dms")]) == 1
    assert "pos differs" in capsys.readouterr().out
    assert main(["diff", src, str(tmp_path / "moved.dms"), "--no-positions"]) == 0


def test_validate_knots_describe_dssp(capsys):
    assert main(["validate", str(msys_file("ww.dms"))]) == 0
    assert main(["knots", str(msys_file("knot.mae"))]) == 1
    assert "2 knots" in capsys.readouterr().out
    assert main(["describe", str(msys_file("ww.dms")), "resid 5 and name CA"]) == 0
    assert "stretch_harm" in capsys.readouterr().out
    assert main(["dssp", str(msys_file("ww.dms")), "--simplified"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines and set("".join(line.split(": ", 1)[1] for line in lines)) <= set("HEC-")
    assert np.all([": " in line for line in lines])
