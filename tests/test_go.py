"""GoMartini: a fold held by native contacts, against martinize2 and GROMACS.

The reference in tests/data/martini/1TEN.go was written by martinize2 with
``-go``, the contact map it computed beside it; tests/data/martini/regenerate.py
remakes both.  The energies are GROMACS 2024.6's, so these tests pin boonza to
GROMACS without needing it installed.
"""

import gzip
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.martini import martinize
from boonza.martini.go import (
    GO_EPS,
    GO_LOW,
    GO_RES_DIST,
    GO_UP,
    go_pairs,
    read_contact_map,
)
from boonza.md.config import DEFAULTS, go_options, parse_arguments

DATA = Path(__file__).parent / "data"
REF = DATA / "martini" / "1TEN.go"
STRUCTURE = DATA / "1TEN.pdb"
#: GROMACS's settings for Martini, as tests/test_martini.py uses them.
MARTINI = {
    "nonbonded_method": "CutoffPeriodic",
    "cutoff": 11.0,
    "dispersion_correction": False,
    "epsilon_r": 15.0,
    "epsilon_rf": 0.0,
    "lj_shift": True,
}


def _ref(name: str) -> str:
    return gzip.decompress((REF / f"{name}.gz").read_bytes()).decode()


@pytest.fixture
def contact_map(tmp_path) -> Path:
    """martinize2's own contact map for 1TEN, as a file to read."""
    path = tmp_path / "go_contacts.out"
    path.write_bytes(gzip.decompress((REF / "go_contacts.out.gz").read_bytes()))
    return path


@pytest.fixture
def built(contact_map):
    """1TEN with its fold held by the contacts of that map."""
    return martinize(boonza.load(STRUCTURE), "protein", go=contact_map)


def _wells(text: str, moltype: str) -> dict:
    """``{(site, site): (funct, sigma, epsilon)}`` from a go_nbparams.itp.

    The site types are named after the molecule, which the two tools name
    differently (martinize2 ``molecule``, boonza ``molecule_0``), so the
    prefix comes off and the pair is kept by the serials it ends with.
    """
    out = {}
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].split()
        if len(line) != 5:
            continue
        i, j, funct, sigma, epsilon = line
        pair = tuple(sorted(int(t.removeprefix(f"{moltype}_")) for t in (i, j)))
        out[pair] = (funct, float(sigma), float(epsilon))
    return out


def _exclusion_pairs(text: str) -> set:
    """Every excluded pair of an .itp, unordered: a row names one atom and the
    atoms it is excluded from, and either end may be the one that names it."""
    out, section = set(), None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
        elif line and section == "exclusions":
            first, *rest = line.split()
            out |= {frozenset((int(first), int(other))) for other in rest}
    return out


def _types(text: str, moltype: str) -> set:
    """The serials of the virtual sites a go_atomtypes.itp declares."""
    return {
        int(line.split()[0].removeprefix(f"{moltype}_"))
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("[", ";"))
    }


# ---------------------------------------------------------------------------
# the map


def test_the_contact_map_is_read_as_the_rcsu_program_writes_it(contact_map):
    """Every row that is an overlap, or that the RCSU rules call a contact
    anyway -- the filter the original script applies -- by the residue numbers
    the structure itself uses."""
    rows = [line for line in contact_map.read_text().splitlines() if line.startswith("R ")]
    contacts = read_contact_map(contact_map)
    assert len(rows) == 777  # every pair the program examined
    assert len(contacts) == 701  # the ones it calls contacts
    resids = {r for a, _, b, _ in ((c[0], c[1], c[2], c[3]) for c in contacts) for r in (a, b)}
    assert min(resids) == 803 and max(resids) == 891  # 1TEN's own numbering
    assert {c[1] for c in contacts} == {"A"}


def test_a_file_that_is_not_a_contact_map_is_refused(tmp_path):
    bad = tmp_path / "nope.out"
    bad.write_text("R 1 2 3\nnot a contact map at all\n")
    with pytest.raises(ValueError, match="18-column"):
        read_contact_map(bad)


def test_the_map_is_read_both_ways_round(contact_map):
    """The RCSU rules are asked of each residue's surface separately and the two
    can disagree, so a map holds each pair twice when both agree.  On 1TEN seven
    of 354 pairs are named one way only."""
    contacts = read_contact_map(contact_map)
    ordered = {((a, ca), (b, cb)) for a, ca, b, cb in contacts}
    one_way = {p for p in ordered if (p[1], p[0]) not in ordered}
    assert len(ordered) == 701
    assert len(one_way) == 7
    assert len(ordered) - len(one_way) == 694 == 2 * 347  # and 347 named both ways


# ---------------------------------------------------------------------------
# the wells, against martinize2


def test_the_go_wells_are_martinize2s(built):
    """Every well martinize2 puts on 1TEN, where it puts it and how deep.

    boonza reads the contacts from the map martinize2 computed, so what is
    compared is the model built from them: which pairs become wells, and the
    sigma and epsilon of each.
    """
    ours = _wells(built.go.nbparams, "molecule_0")
    theirs = _wells(_ref("go_nbparams.itp"), "molecule")
    assert len(ours) == 185
    assert set(ours) == set(theirs)
    for pair, (funct, sigma, epsilon) in ours.items():
        ref_funct, ref_sigma, ref_epsilon = theirs[pair]
        assert funct == ref_funct == "1"
        assert sigma == pytest.approx(ref_sigma, abs=1e-8)
        assert epsilon == pytest.approx(ref_epsilon, abs=1e-8) == pytest.approx(GO_EPS)


def test_every_residue_gets_a_site_whether_a_well_reaches_it_or_not(built):
    """As martinize2 does: a type with no pair never interacts, and numbering
    the sites by residue keeps a map readable against the structure."""
    ours = _types(built.go.atomtypes, "molecule_0")
    assert ours == _types(_ref("go_atomtypes.itp"), "molecule")
    assert ours == set(range(1, 90))  # 89 residues, one site each
    assert built.go.sites == 89
    reached = {s for pair in _wells(built.go.nbparams, "molecule_0") for s in pair}
    assert reached < ours  # not every site is in a well


def test_the_go_topology_is_martinize2s(built, tmp_path):
    """Beads, virtual sites and exclusions, term for term."""
    from test_martinize import _parse_itp  # tests/ is on the path, as conftest puts it

    # the site types carry the molecule's name, which the two tools spell
    # differently; nothing else in the topology mentions it
    ours_text = built.itp(0).replace("molecule_0_", "molecule_")
    ref_text = _ref("molecule.itp")
    ours_atoms, ours = _parse_itp(ours_text)
    ref_atoms, theirs = _parse_itp(ref_text)
    assert ours_atoms == ref_atoms
    assert {k: v for k, v in ours.items() if k != "exclusions"} == {
        k: v for k, v in theirs.items() if k != "exclusions"
    }
    # 89 Go sites, each the centre of the one bead it stands for, beside the
    # side-chain site Martini 3 builds for a tryptophan
    sites = [key for (key, _), n in ours["virtual_sitesn"].items() for _ in range(n)]
    assert sum(1 for _, funct, parents in sites if (funct, len(parents)) == ("1", 1)) == 89
    assert len(sites) == 90
    # the beads a well joins keep their own nonbonded terms out of its way.
    # Which end of a pair names the other is the writer's choice, so the pairs
    # are compared and not the rows: boonza writes the lower bead first.
    assert _exclusion_pairs(ours_text) == _exclusion_pairs(ref_text)
    assert len(_exclusion_pairs(ours_text)) == 242  # 185 wells, 57 the chain's own


def test_the_topology_switches_the_virtual_sites_on(built):
    """GO_VIRT is what Martini's own parameter file reads to let a residue carry
    one, and the two Go files have to come before the molecule that uses their
    types."""
    top = built.top(martini_itp="martini.itp").splitlines()
    assert top[0] == "#define GO_VIRT"
    assert top.index('#include "go_atomtypes.itp"') < top.index('#include "molecule_0.itp"')
    assert top.index('#include "go_nbparams.itp"') < top.index('#include "molecule_0.itp"')


def test_the_go_files_are_written_beside_the_topology(built, tmp_path):
    built.save(tmp_path, martini_itp="martini.itp")
    assert (tmp_path / "go_atomtypes.itp").read_text() == built.go.atomtypes
    assert (tmp_path / "go_nbparams.itp").read_text() == built.go.nbparams


# ---------------------------------------------------------------------------
# what makes a well


def test_a_well_sits_where_the_pair_is(built):
    """A Lennard-Jones well is deepest at 2^(1/6) sigma, which is where the two
    backbone beads are: the fold is a minimum, not a constraint."""
    xyz = np.concatenate([np.asarray(m.positions, float) for m in built.molecules])
    for pair in built.go.pairs:
        d = float(np.linalg.norm(xyz[pair.i] - xyz[pair.j]))
        assert d == pytest.approx(pair.distance, abs=1e-9)
        assert pair.sigma * 2.0 ** (1 / 6) == pytest.approx(d, abs=1e-9)
        assert GO_LOW < d < GO_UP


def test_a_pair_the_bonded_terms_already_hold_is_no_well(built):
    """What the chain holds is not the Go model's business, so residues closer
    than go_res_dist along it are skipped however well they touch."""
    mol = built.molecules[0]
    for pair in built.go.pairs:
        i, j = (int(mol.nodes[k]["resid"]) for k in (pair.i, pair.j))
        assert abs(i - j) > GO_RES_DIST


def test_the_window_and_the_reach_are_the_models(contact_map, built):
    """Widening the window takes in more pairs and narrowing it fewer, and the
    defaults are martinize2's."""
    mol = built.molecules[0]
    contacts = read_contact_map(contact_map)
    assert len(go_pairs(mol, contacts)) == 185  # the defaults
    assert len(go_pairs(mol, contacts, up=0.8)) == 146
    assert len(go_pairs(mol, contacts, low=0.6)) == 97
    assert len(go_pairs(mol, contacts, up=1.4)) == 194
    assert len(go_pairs(mol, contacts, res_dist=10)) == 151
    # and the depth is the one number the structure does not decide
    assert {p.epsilon for p in go_pairs(mol, contacts, eps=5.0)} == {5.0}


def test_a_pair_the_map_names_one_way_only_is_no_well(built, contact_map):
    """A contact is a statement about a pair, so one written only one way is one
    the map is unsure of.  Keeping those seven would make 189 wells, not 185."""
    mol = built.molecules[0]
    contacts = read_contact_map(contact_map)
    both = {((a, ca), (b, cb)) for a, ca, b, cb in contacts}
    doubled = [*contacts, *[(b, cb, a, ca) for (a, ca), (b, cb) in both]]
    assert len(go_pairs(mol, contacts)) == 185
    assert len(go_pairs(mol, doubled)) == 189


def test_a_map_for_another_structure_is_refused(tmp_path):
    """A map is written against the structure it was computed from, so one whose
    residues this structure does not have is a mistake worth saying."""
    other = tmp_path / "elsewhere.out"
    rows = [
        f"R {k:6d} {k:5d}  ALA A {4000 + k:4d}    {k + 1:5d}  ALA A {4100 + k:4d}    "
        f"   5.0000     1 1 1 1     1       1     0"
        for k in range(1, 4)
    ]
    other.write_text("\n".join(rows) + "\n")
    assert len(read_contact_map(other)) == 3  # it reads as a map
    with pytest.raises(ValueError, match="its own numbering"):
        martinize(boonza.load(STRUCTURE), "protein", go=other)


# ---------------------------------------------------------------------------
# the system, and its energy


def test_the_system_carries_every_well(built):
    """The wells reach OpenMM as pair overrides between the site types, and each
    site is placed on the bead it stands for."""
    s = built.system()
    assert s.natoms == 287
    sites = s.tables["virtual_lc1"].atoms
    assert len(sites) == 89
    xyz = np.asarray(s.positions, float)
    for site, bead in sites.tolist():
        assert np.allclose(xyz[site], xyz[bead])
        assert s.atoms["mass"][site] == 0.0
        assert s.atoms["charge"][site] == 0.0
    nb = s.tables["nonbonded"]
    name = {k: str(nb.params.row(k)["type"]) for k in range(nb.params.nparams)}
    ours = {}
    for (a, b), params in nb.overrides.items():
        if not (name[a].startswith("molecule_0_") and name[b].startswith("molecule_0_")):
            continue  # a Martini type override, not a well
        pair = tuple(sorted(int(name[k].removeprefix("molecule_0_")) for k in (a, b)))
        ours[pair] = (params["sigma"] * 0.1, params["epsilon"] * 4.184)  # A, kcal to nm, kJ
    wells = _wells(built.go.nbparams, "molecule_0")
    assert set(ours) == set(wells)
    assert len(ours) == 185
    for pair, (sigma, epsilon) in ours.items():
        assert sigma == pytest.approx(wells[pair][1], abs=1e-8)
        assert epsilon == pytest.approx(wells[pair][2], abs=1e-8)


def test_a_coincident_site_is_excluded_from_its_bead(built):
    """A site sitting on its bead is at r = 0 from it, where OpenMM's
    Lennard-Jones is 1/0 and the energy comes back NaN.  Neither tool's
    topology excludes the pair -- GROMACS steps around the singularity -- and
    the pair is worth nothing either way, so boonza excludes it on the way into
    OpenMM and reaches GROMACS's energy all the same.
    """
    mm = pytest.importorskip("openmm")
    from boonza.martini import OPENMM_OPTIONS

    s = built.system()
    _, system, _ = boonza.to_openmm(s, **OPENMM_OPTIONS)
    force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    exceptions = {}
    for k in range(force.getNumExceptions()):
        i, j, q, _, epsilon = force.getExceptionParameters(k)
        exceptions[min(i, j), max(i, j)] = (q._value, epsilon._value)
    for site, bead in s.tables["virtual_lc1"].atoms.tolist():
        assert exceptions[min(site, bead), max(site, bead)] == (0.0, 0.0)


def test_the_energy_is_gromacs(built, tmp_path):
    """GROMACS 2024.6 on boonza's own Go topology and coordinates, a 0-step md
    run under Martini's reaction field and shifted Lennard-Jones: bond 680.492,
    G96 angle 103.126, restricted angles 588.945, proper dihedrals 0.0242159,
    improper dihedrals 0.0616396, LJ (SR) -1960.48, Coulomb (SR) -195.948,
    potential -783.782 kJ/mol.

    The 185 wells are in LJ (SR): emptying go_nbparams.itp leaves -550.906
    there, so they are worth -1409.58 kJ/mol at the input structure.  Each sits
    at its minimum, which is 185 * 9.414 = 1741.6; the rest is the shift that
    takes every pair to zero at the cutoff.
    """
    pytest.importorskip("openmm")
    built.save(tmp_path)  # including the Martini parameter file boonza carries
    s = boonza.load(tmp_path / "topol.top", coordinates=tmp_path / "cg.gro")
    assert s.natoms == 287
    e = {k: v * 4.184 for k, v in boonza.openmm_energies(s, **MARTINI).items()}
    assert e["stretch_harm"] == pytest.approx(680.492, abs=0.01)
    assert e["angle_cosine_harm"] == pytest.approx(103.126, abs=0.01)
    assert e["angle_restricted"] == pytest.approx(588.945, abs=0.01)
    assert e["dihedral_trig"] == pytest.approx(0.0242159, abs=1e-4)
    assert e["improper_harm"] == pytest.approx(0.0616396, abs=1e-4)
    assert e["nonbonded_vdw"] == pytest.approx(-1960.48, abs=0.01)
    coulomb = sum(
        e.get(k, 0.0) for k in ("nonbonded", "nonbonded_rf_exclusions", "nonbonded_rf_self")
    )
    assert coulomb == pytest.approx(-195.948, abs=0.01)
    assert e["total"] == pytest.approx(-783.782, abs=0.01)


# ---------------------------------------------------------------------------
# the settings


def test_a_go_model_is_what_the_elastic_network_would_be(contact_map):
    """A fold is held one way or the other, so asking for a Go model turns the
    network off rather than stiffening the protein twice."""
    args = parse_arguments([str(STRUCTURE), "--model", "martini3",
                            "--structure-bias", "go", "--go-map", str(contact_map)])  # fmt: skip
    assert args.structure_bias == "go"
    assert args.elastic is False
    assert go_options(args) == {"go": str(contact_map), "go_eps": GO_EPS, "go_low": GO_LOW,
                                "go_up": GO_UP, "go_res_dist": GO_RES_DIST}  # fmt: skip
    # and an elastic network is what a Martini run holds a fold with otherwise
    elastic = parse_arguments([str(STRUCTURE), "--model", "martini3"])
    assert elastic.structure_bias == "elastic" and elastic.elastic is True
    assert go_options(elastic) == {}


def test_asking_for_both_ways_of_holding_a_fold_is_an_error(capsys, contact_map):
    with pytest.raises(SystemExit):
        parse_arguments([str(STRUCTURE), "--model", "martini3", "--elastic",
                         "--structure-bias", "go", "--go-map", str(contact_map)])  # fmt: skip
    assert "instead of an elastic network" in capsys.readouterr().err


def test_a_go_model_needs_its_contacts(capsys):
    with pytest.raises(SystemExit):
        parse_arguments([str(STRUCTURE), "--model", "martini3", "--structure-bias", "go"])
    assert "needs the contacts to hold" in capsys.readouterr().err


def test_a_map_on_its_own_does_nothing(capsys, contact_map):
    with pytest.raises(SystemExit):
        parse_arguments([str(STRUCTURE), "--model", "martini3", "--go-map", str(contact_map)])
    assert "structure_bias = 'go'" in capsys.readouterr().err


@pytest.mark.parametrize("model", ["aa", "sirah"])
def test_a_go_model_is_martinis(capsys, contact_map, model):
    """GoMartini is Martini's; SIRAH holds its fold with springs of its own, and
    an all-atom run holds its own."""
    with pytest.raises(SystemExit):
        parse_arguments([str(STRUCTURE), "--model", model,
                         "--structure-bias", "go", "--go-map", str(contact_map)])  # fmt: skip
    assert "Martini model" in capsys.readouterr().err


def test_the_window_has_to_have_room_for_a_well(capsys, contact_map):
    with pytest.raises(SystemExit):
        parse_arguments([str(STRUCTURE), "--model", "martini3", "--structure-bias", "go",
                         "--go-map", str(contact_map), "--go-low", "1.5"])  # fmt: skip
    assert "under 'go_up'" in capsys.readouterr().err


def test_the_defaults_are_martinize2s():
    """-go-eps, -go-low, -go-up and -go-res-dist, as martinize2 names them."""
    assert (GO_EPS, GO_LOW, GO_UP, GO_RES_DIST) == (9.414, 0.3, 1.1, 3)
    assert DEFAULTS["structure_bias"] == "elastic"
    assert DEFAULTS["go_map"] is None
    for key, value in (("go_eps", GO_EPS), ("go_low", GO_LOW), ("go_up", GO_UP),
                       ("go_res_dist", GO_RES_DIST)):  # fmt: skip
        assert DEFAULTS[key] == value
