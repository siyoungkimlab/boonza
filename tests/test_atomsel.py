import json

import pytest
from conftest import MSYS_FILES, msys_file, run_msys
from test_system import water_box

import boonza
from boonza.atomsel import SelectionError

EXTRA = [
    "chain A and resid 1 to 5",
    "index 1 to 10 and not index 5",
    "same fragment as index 10",
    "same residue as index 100 200",
    "withinbonds 1 of index 5",
    "mass 12.011",
    "mass 12",
    "x > 0 and y < 0",
    "charge < -0.5",
    "resid -1 0",
    "degree 1",
    "numbonds 0",
    "ct 0",
    'resname "L.*"',
    "name 'CA' 'CB'",
    "residue 3 to 5",
    "-x < 2",
    "x < -2 or x > +2",
    "(x) < 3",
    "sqr(x) + sqr(y) < 50 and protein",
    "protein and within 3 of water",
    "water and exwithin 3 of protein",
    "pbwithin 6 of resid 5",
    "pbnearest 3 to protein",
    "nearest 3 to index 0",
    "not not protein",
    "protein or water and name OH2",
    "within 3 of protein and name CA",
    "sequence TW.",
    "insertion ''",
    "element C N and not backbone",
    "fragid 0 1 2",
]


def _reference_cases():
    path = MSYS_FILES.parent / "atomsel_tests.json"
    if not path.exists():
        pytest.skip("msys atomsel_tests.json not found")
    return json.loads(path.read_text())["tests/files/2f4k.dms"]


def _mismatches(system, cases):
    bad = []
    for sel, want in cases:
        try:
            got = system.select(sel).ids.tolist()
        except (SelectionError, NotImplementedError) as e:
            got = f"ERROR: {e}"
        if isinstance(want, str):  # msys raised
            if not isinstance(got, str):
                bad.append((sel, "msys raised, boonza selected", len(got)))
        elif got != want:
            bad.append((sel, got if isinstance(got, str) else len(got), len(want)))
    return bad


def test_msys_reference_selections():
    s = boonza.load(msys_file("2f4k.dms"))
    assert _mismatches(s, _reference_cases()) == []


def test_matches_live_msys(tmp_path):
    path = msys_file("2f4k.dms")
    sels = [sel for sel, _ in _reference_cases()] + EXTRA
    f = tmp_path / "sels.json"
    f.write_text(json.dumps(sels))
    want = run_msys("select", path, f)
    s = boonza.load(path)
    assert _mismatches(s, list(zip(sels, want, strict=True))) == []


def test_selections_on_built_system():
    s = water_box(3)
    assert s.select("water").ids.tolist() == list(range(9))
    assert s.select("name OW").ids.tolist() == [0, 3, 6]
    assert s.select("within 1.0 of index 0").ids.tolist() == [0, 1, 2]
    assert s.select("same residue as index 4").ids.tolist() == [3, 4, 5]
    assert s.select("withinbonds 1 of index 1").ids.tolist() == [0, 1]
    assert s.select("withinbonds 2 of index 1").ids.tolist() == [0, 1, 2]
    assert s.select("nearest 1 to index 0").ids.tolist() == [1]
    assert s.select("x > 2.9 and x < 3.1").ids.tolist() == [3]
    s.atoms["flag"] = [0, 1, 0, 0, 1, 0, 0, 1, 0]
    assert s.select("flag 1").ids.tolist() == [1, 4, 7]
    assert s.select("flag == 1 and resid 2").ids.tolist() == [4]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "name",
        "resid 1 to",
        "(protein",
        "within 3 protein",
        "bogus",
        "name 1",
        "x < ",
        "resid 1.5",
    ],
)
def test_bad_selections_raise(bad):
    with pytest.raises(SelectionError):
        water_box(1).select(bad)
