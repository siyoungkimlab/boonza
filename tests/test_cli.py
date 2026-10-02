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


def test_info_reads_a_trajectory(tmp_path, capsys):
    """A .dcd or .xtc is not a structure, so `boonza info` reads what a trajectory
    has instead: its frames, their atoms, the time they cover and the box they
    were under.  Only the first and last frames are read, so it costs the same on
    a gigabyte as on a megabyte."""
    import boonza
    from boonza.cli import main
    from boonza.trajectory import is_trajectory

    assert is_trajectory("x.dcd") and is_trajectory("x.xtc") and is_trajectory("x.trr")
    assert not is_trajectory("x.dms") and not is_trajectory("x.pdb")

    s = boonza.peptide("AAA")
    xyz = np.asarray(s.positions)
    for suffix in (".dcd", ".xtc"):
        path = tmp_path / f"run{suffix}"
        box = np.diag([30.0, 30.0, 30.0])
        with boonza.open_writer(path, s.natoms) as w:
            for k in range(4):
                w.write(xyz + k, box=box * (1 + 0.01 * k))
        assert main(["info", str(path)]) == 0
        printed = capsys.readouterr().out
        assert f"{suffix.lstrip('.')}, 4 frames of {s.natoms} atoms" in printed
        assert "first frame: box 30.00 x 30.00 x 30.00 A" in printed
        assert "last frame: box 30.90 x 30.90 x 30.90 A" in printed
        assert "volume" in printed

    # and a structure is still read as a structure
    structure = tmp_path / "one.dms"
    boonza.save(s, structure)
    assert main(["info", str(structure)]) == 0
    assert f"{s.natoms} atoms, {s.nbonds} bonds" in capsys.readouterr().out
