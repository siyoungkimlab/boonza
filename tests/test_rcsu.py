"""The RCSU/OV contact map boonza computes, against martinize2's own.

The references in tests/data/martini/<name>.go were written by martinize2 with
``-go``, which computes the map the same way vermouth's reimplementation of the
RCSU program does; tests/data/martini/regenerate.py remakes them.  Each is
compared column for column, over a structure of one chain (1TEN) and two
(2TRX, 1HHO).
"""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.martini import martinize
from boonza.martini.go import read_contact_map
from boonza.martini.rcsu import (
    ALIASES,
    ATOM_ALIASES,
    FIB_A,
    FIB_B,
    WATER_RADIUS,
    _sphere,
    _tables,
    contact_map,
    radius_and_type,
)

DATA = Path(__file__).parent / "data"
REF = DATA / "martini"
#: The structures regenerate.py writes a Go reference for.
CASES = ("1TEN", "2TRX", "1HHO")


def _input(name, tmp_path):
    """The structure as regenerate.py hands it to martinize2."""
    s = boonza.load(DATA / f"{name}.pdb").select("protein").clone()
    boonza.save(s, tmp_path / f"{name}.pdb")
    return boonza.load(tmp_path / f"{name}.pdb")


def _rows(text):
    """``{(chain, resid, chain, resid): columns}`` of an 18-column map."""
    out = {}
    for line in text.splitlines():
        t = line.split()
        if t and t[0] == "R" and len(t) == 18:
            out[(t[4], int(t[5]), t[8], int(t[9]))] = {
                "resnames": (t[3], t[7]),
                "dca": t[10],
                "flags": (t[11], t[12], t[13], t[14]),
                "count": int(t[16]),
            }
    return out


def _reference(name, tmp_path):
    text = gzip.decompress((REF / f"{name}.go" / "go_contacts.out.gz").read_bytes()).decode()
    (tmp_path / f"{name}.out").write_text(text)
    return text, tmp_path / f"{name}.out"


# ---------------------------------------------------------------------------
# against martinize2


@pytest.mark.parametrize("name", CASES)
def test_the_map_is_martinize2s(name, tmp_path):
    """Which residue pairs the map holds, their alpha-carbon distances, and
    every one of the four flags that say what kind of contact each is.

    The fifth column, the count of surface points two residues share, is
    informational -- nothing reads it back -- and agrees everywhere except
    where a structure labels a protonated acid's two carboxyl oxygens the
    other way round from the reference residue, which moves one radius by
    0.04 A.  See ``test_a_protonated_acids_oxygens``.
    """
    text, _ = _reference(name, tmp_path)
    mine = _rows(contact_map(_input(name, tmp_path)).text())
    theirs = _rows(text)
    assert set(mine) == set(theirs)
    for key, ours in mine.items():
        assert ours["resnames"] == theirs[key]["resnames"]
        assert ours["dca"] == theirs[key]["dca"], key
        assert ours["flags"] == theirs[key]["flags"], key


@pytest.mark.parametrize("name", CASES)
def test_the_contacts_are_martinize2s(name, tmp_path):
    """And so the contacts a Go model would put a well on are the same ones."""
    text, path = _reference(name, tmp_path)
    mine = contact_map(_input(name, tmp_path))
    assert set(mine.kept()) == set(read_contact_map(path))
    assert len(mine.kept()) == len(read_contact_map(path))


@pytest.mark.parametrize(("name", "rows", "contacts"),
                         [("1TEN", 777, 701), ("2TRX", 2059, 1898),
                          ("1HHO", 2959, 2753)])  # fmt: skip
def test_how_much_each_structure_holds(name, rows, contacts, tmp_path):
    """The numbers themselves, so a change shows up as a change."""
    mine = contact_map(_input(name, tmp_path))
    assert len(mine.contacts) == rows
    assert len(mine.kept()) == contacts


def test_a_protonated_acids_oxygens(tmp_path):
    """The one place the two tools read a structure differently.

    The tables give an aspartate's OD1 and a glutamate's OE1 a radius 0.04 A
    larger than their OD2/OE2, which is the reference residue's hydroxyl
    oxygen.  Which of the two a structure calls OE1 is the structure's
    business, and martinize2 relabels them to match its reference while
    reading, where boonza takes the names as given.  On 1TEN, whose glutamate
    810 is protonated on the oxygen the reference does not use, that moves
    five of 777 counts by at most five points of some hundreds -- and no flag,
    so not one contact.
    """
    text, _ = _reference("1TEN", tmp_path)
    mine = _rows(contact_map(_input("1TEN", tmp_path)).text())
    theirs = _rows(text)
    differ = {k: (mine[k]["count"], theirs[k]["count"])
              for k in mine if mine[k]["count"] != theirs[k]["count"]}  # fmt: skip
    assert len(differ) == 5
    assert max(abs(a - b) for a, b in differ.values()) <= 5
    assert all(mine[k]["flags"] == theirs[k]["flags"] for k in differ)
    # every one of them is the protonated glutamate 810 or a residue it touches
    assert {k[1] for k in differ} | {k[3] for k in differ} == {808, 810, 812, 822}


# ---------------------------------------------------------------------------
# the tables


def test_the_tables_are_the_rcsu_programs():
    """Carried as data, not retyped: 24 residue names and an 11x11 bond table."""
    residues, bond_type = _tables()
    assert len(residues) == 20  # the empty names are dropped; ALIASES answers them
    assert bond_type.shape == (11, 11)
    assert radius_and_type("ALA", "CA") == (1.88, 7)
    assert radius_and_type("MET", "SD") == (1.77, 8)
    assert radius_and_type("SER", "OG") == (1.46, 1)  # a hydroxyl, and a donor
    assert radius_and_type("GLN", "OE1") == (1.42, 2)  # a carbonyl, and an acceptor


def test_an_atom_the_tables_do_not_name_has_no_radius():
    """A hydrogen, which the tables hold none of, and anything else unnamed."""
    assert radius_and_type("ALA", "HB1") == (0.0, 0)
    assert radius_and_type("HOH", "O") == (0.0, 0)
    assert radius_and_type("HEM", "FE") == (0.0, 0)


def test_a_residue_the_tables_call_something_else():
    """A protonation or a disulfide does not change which atoms a residue has,
    so the name the structure happens to use must not change its contacts.

    vermouth's own table lists CYM, CYX, HIE and HIP but leaves them empty, so
    looking an atom of one up raises KeyError there, and does not list HID or
    the CHARMM spellings at all, which reads every radius as zero.
    """
    for name in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
        assert radius_and_type(name, "ND1") == (1.64, 1)
    for name in ("CYS", "CYX", "CYM"):
        assert radius_and_type(name, "SG") == (1.77, 6)
    assert radius_and_type("ASH", "OD1") == radius_and_type("ASP", "OD1")
    assert radius_and_type("MSE", "CA") == radius_and_type("MET", "CA")
    assert set(ALIASES.values()) <= set(_tables()[0])


def test_an_isoleucines_delta_carbon_is_found_by_either_name():
    """The tables call it CD, as CHARMM does; a PDB calls it CD1.  Without the
    alias it would have no radius and quietly take part in nothing -- and the
    isoleucines are a fold's core.
    """
    assert ATOM_ALIASES[("ILE", "CD1")] == "CD"
    assert radius_and_type("ILE", "CD1") == radius_and_type("ILE", "CD") == (1.88, 4)


def test_a_heavy_atom_with_no_radius_is_said():
    """Silence would be a map quietly short of what the structure shows -- an
    isoleucine's CD1 was exactly that until it was aliased.

    The atoms are given by index here, since renaming a backbone atom is
    enough for ``select("protein")`` not to know the residue any more.
    """
    s = boonza.load(DATA / "1TEN.pdb").select("protein").clone()
    renamed = next(a for a in s.residue(5).atoms if a.anum != 1)  # a heavy one
    resname = str(s.residues["name"][5]).strip()
    renamed.name = "QQ1"
    with pytest.warns(UserWarning, match=f"no radius for {resname} QQ1"):
        contact_map(s, np.arange(s.natoms))
    # and a hydrogen is expected to have none, so it says nothing about those
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        contact_map(boonza.load(DATA / "1TEN.pdb").select("protein").clone())


# ---------------------------------------------------------------------------
# how it is computed


def test_the_sphere_is_the_fibonacci_spiral():
    """610 points, the 14th Fibonacci number, each a unit from the centre and
    spread so every one takes an equal band of the sphere."""
    points = _sphere()
    assert (FIB_A, FIB_B) == (377, 610)
    assert points.shape == (610, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    # equal bands: z steps uniformly, which is what makes the area equal
    z = np.sort(points[:, 2])
    assert np.allclose(np.diff(z), np.diff(z)[0], atol=1e-12)
    assert points.mean(axis=0) == pytest.approx([0, 0, 0], abs=0.01)


def test_an_atom_the_tables_do_not_hold_still_takes_up_room(tmp_path):
    """A cofactor has no radius and no character, so it shares no surface with
    anything -- but it is still in the way: a point of a residue's surface it
    wins is surface that residue no longer shares across the pocket.

    1HHO's haem sits in one.  Reading the map over the protein and the haem
    together takes a share of the surface of 161 residue pairs and ends nine
    contacts between residues altogether.  It also makes two contacts of its
    own, both pure overlaps: overlap is a distance against the sum of two
    radii, and a radius of zero makes that threshold small rather than
    impossible.  Which is to say that whether a cofactor is in the structure
    changes the fold a Go model holds, and is worth deciding on purpose.
    """
    s = boonza.load(DATA / "1HHO.pdb")
    bare, shaded = contact_map(s, "protein"), contact_map(s, "protein or resname HEM")
    haem = [c for c in shaded.contacts if "HEM" in (c.resname_a, c.resname_b)]
    assert radius_and_type("HEM", "FE") == (0.0, 0)
    # no shared surface with the haem, since a contact needs a character at
    # both ends, but an overlap is a distance and two of those it has
    assert [c.count for c in haem] == [0, 0]
    assert [c.overlap for c in haem] == [1, 1]

    def counts(m):
        return {(c.resid_a, c.chain_a, c.resid_b, c.chain_b): c.count for c in m.contacts
                if "HEM" not in (c.resname_a, c.resname_b)}  # fmt: skip

    before, after = counts(bare), counts(shaded)
    assert len([k for k in set(before) & set(after) if after[k] < before[k]]) == 161
    lost = set(bare.kept()) - set(shaded.kept())
    assert len(lost) == 9


def test_nothing_to_read_a_map_over():
    s = boonza.load(DATA / "1TEN.pdb")
    with pytest.raises(ValueError, match="selects no atoms"):
        contact_map(s, "resname XYZ")


# ---------------------------------------------------------------------------
# what a build does with it


def test_a_fold_held_by_contacts_boonza_worked_out(tmp_path):
    """``go=True`` computes the map; a map read from a file makes the same
    wells, so the two ways in meet."""
    computed = martinize(boonza.load(DATA / "1TEN.pdb"), "protein", go=True)
    assert len(computed.go.pairs) == 185
    assert computed.go.sites == 89
    assert computed.go.contact_map.startswith("Go contact map calculated with boonza")

    read = tmp_path / "go_contacts.out"
    read.write_text(computed.go.contact_map)
    again = martinize(boonza.load(DATA / "1TEN.pdb"), "protein", go=read)
    assert again.go.nbparams == computed.go.nbparams
    assert again.go.atomtypes == computed.go.atomtypes
    assert again.go.contact_map == ""  # it was given one, so it wrote none


def test_a_map_boonza_computed_is_martinize2s_map(tmp_path):
    """The whole way through: the wells a computed map makes are the wells
    martinize2's own map makes, which stage A already pinned against
    martinize2's own Go model."""
    given = tmp_path / "theirs.out"
    given.write_text(gzip.decompress(
        (REF / "1TEN.go" / "go_contacts.out.gz").read_bytes()).decode())  # fmt: skip
    theirs = martinize(boonza.load(DATA / "1TEN.pdb"), "protein", go=given)
    ours = martinize(boonza.load(DATA / "1TEN.pdb"), "protein", go=True)
    assert ours.go.nbparams == theirs.go.nbparams


def test_the_map_goes_beside_the_topology(tmp_path):
    """A fold held by a map boonza worked out is a fold nobody else can see
    unless it is written down."""
    from boonza.martini.go import CONTACTS_OUT

    m = martinize(boonza.load(DATA / "1TEN.pdb"), "protein", go=True)
    m.save(tmp_path, martini_itp="martini.itp")
    assert (tmp_path / CONTACTS_OUT).read_text() == m.go.contact_map
    assert read_contact_map(tmp_path / CONTACTS_OUT)  # and it reads back


def test_the_format_is_the_one_read_back(tmp_path):
    """Every row eighteen columns, starting with R, which is how the RCSU
    program says 'this is a row' and how boonza finds one."""
    text = contact_map(_input("1TEN", tmp_path)).text()
    rows = [line for line in text.splitlines() if line.startswith("R ")]
    assert len(rows) == 777
    assert {len(line.split()) for line in rows} == {18}
    path = tmp_path / "written.out"
    path.write_text(text)
    assert len(read_contact_map(path)) == 701


def test_the_settings_need_no_map(tmp_path):
    """``--structure-bias go`` on its own holds a fold: the structure answers
    for itself, and `--go-map` is for a map from somewhere else."""
    from boonza.md.config import go_options, parse_arguments

    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--structure-bias", "go"])  # fmt: skip
    assert go_options(args)["go"] is True
    given = tmp_path / "m.out"
    given.write_text("")
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--structure-bias", "go", "--go-map", str(given)])  # fmt: skip
    assert go_options(args)["go"] == str(given)


def test_it_is_quick_enough_to_do_every_time(tmp_path):
    """The reason it can be the default way in: a hundredth of a second on a
    small protein, where vermouth's own implementation takes about half a
    second.  Generous here, since a loaded machine is still a machine."""
    import time

    s = _input("1HHO", tmp_path)
    contact_map(s)  # the kernels compile on the first call
    start = time.perf_counter()
    contact_map(s)
    assert time.perf_counter() - start < 2.0


def test_the_water_radius_and_the_overlap_factor():
    """The two numbers the method turns on, as the RCSU program sets them."""
    from boonza.martini.rcsu import ALPHA

    assert (WATER_RADIUS, ALPHA) == (2.80, 1.24)
