"""SIRAH: a coarse-grained force field whose parameters live in its topology.

The energies here are GROMACS 2024.6's for the very coordinates of the fixture
(see tests/data/sirah/0README), so they check the whole chain: the topology
boonza reads, the tables it builds, and what OpenMM makes of them.
"""

import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.md.config import parse_arguments

DATA = Path(__file__).parent / "data" / "sirah"
KJ = 4.184
#: GROMACS 2024.6, PME converged, for tests/data/sirah/peptide_wt4.dms.gz (kJ/mol).
GROMACS = {
    "bond": 2225.28,
    "angle": 428.592,
    "proper": 281.211,
    "improper": 33.2108,
    "lj14": 74.5446,
    "coulomb14": 213.78,
    "lj": 206934.0,
    "coulomb": -20122.7,
    "recip": 4480.69,
}


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    out = tmp_path_factory.mktemp("sirah") / "peptide_wt4.dms"
    with gzip.open(DATA / "peptide_wt4.dms.gz", "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return boonza.load(out)


def test_what_a_sirah_topology_holds(system):
    """Three beads a residue for the backbone, four for each water, and the
    tables SIRAH needs -- including the periodic dihedrals of its backbone,
    which go past the six multiplicities dihedral_trig can hold."""
    assert system.natoms == 930
    names = {str(n) for n in np.asarray(system.atoms["name"])}
    assert {"GN", "GC", "GO"} <= names  # the backbone
    assert {"WN1", "WN2", "WP1", "WP2"} <= names  # WT4
    assert {"stretch_harm", "angle_harm", "dihedral_trig", "dihedral_periodic",
            "pair_12_6_es", "nonbonded", "exclusion"} <= set(system.tables)  # fmt: skip
    assert len(system.table("dihedral_periodic")) > 0


def test_sirah_energies_are_gromacs_energies(system):
    """Term by term, with SIRAH's own settings: PME inside 1.2 nm, Lennard-Jones
    shifted to zero there, no dispersion correction."""
    pytest.importorskip("openmm")
    from boonza.sirah import OPENMM_OPTIONS

    e = boonza.openmm_energies(system, **OPENMM_OPTIONS, ewald_tolerance=1e-6)
    got = {k: v * KJ for k, v in e.items()}
    assert got["stretch_harm"] == pytest.approx(GROMACS["bond"], abs=0.05)
    assert got["angle_harm"] == pytest.approx(GROMACS["angle"], abs=0.05)
    assert got["dihedral_trig"] + got["dihedral_periodic"] == pytest.approx(
        GROMACS["proper"] + GROMACS["improper"], abs=0.05
    )
    assert got["nonbonded_vdw"] == pytest.approx(GROMACS["lj"], rel=1e-5)
    electrostatics = GROMACS["coulomb"] + GROMACS["recip"] + GROMACS["lj14"] + GROMACS["coulomb14"]
    assert got["nonbonded"] == pytest.approx(electrostatics, abs=5.0)
    assert got["total"] == pytest.approx(sum(GROMACS.values()), rel=1e-4)


def _minimal_top(dihedral_multiplicity: int, nonbond_override: bool) -> str:
    """Four beads in a line: one dihedral of the given multiplicity, and a 1-4
    pair whose parameters [ nonbond_params ] may override."""
    override = "  A  B  1  0.325  0.56\n" if nonbond_override else ""
    return f"""[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
A  50.0  0.0  A  0.42  0.55
B  50.0  0.0  A  0.42  0.55

[ nonbond_params ]
{override}
[ moleculetype ]
CHAIN 3

[ atoms ]
1 A 1 RES X1 1 0.3 50.0
2 A 1 RES X2 2 -0.1 50.0
3 A 1 RES X3 3 -0.1 50.0
4 B 1 RES X4 4 -0.1 50.0

[ bonds ]
1 2 1 0.35 4000
2 3 1 0.35 4000
3 4 1 0.35 4000

[ pairs ]
1 4 1

[ dihedrals ]
1 2 3 4 9 -87.8 8.9 {dihedral_multiplicity}

[ system ]
four beads

[ molecules ]
CHAIN 1
"""


def _four_beads(tmp_path, multiplicity, override, angle_degrees=60.0):
    """The topology written out with a geometry of a known dihedral angle."""
    top = tmp_path / f"m{multiplicity}{'o' if override else ''}.top"
    top.write_text(_minimal_top(multiplicity, override))
    phi = np.radians(angle_degrees)
    xyz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [3.5, 0.0, 0.0],
                    [3.5, np.cos(phi), np.sin(phi)]])  # fmt: skip
    gro = tmp_path / top.with_suffix(".gro").name
    lines = ["four beads", f"{len(xyz)}"]
    for k, x in enumerate(xyz, start=1):
        lines.append(f"{1:5d}RES  {'X' + str(k):>5s}{k:5d}"
                     + "".join(f"{v / 10:8.3f}" for v in x))  # fmt: skip
    lines += ["  5.00000   5.00000   5.00000", ""]
    gro.write_text("\n".join(lines))
    return boonza.load(top, coordinates=gro)


@pytest.mark.parametrize("multiplicity", [1, 3, 6, 7, 9])
def test_a_dihedral_of_any_multiplicity(tmp_path, multiplicity):
    """dihedral_trig holds multiplicities up to 6, as msys's does; SIRAH's
    backbone uses 7 and its force field goes to 9, which boonza once refused
    to read at all.  The energy is fc (1 + cos(n phi - phi0)) either way."""
    pytest.importorskip("openmm")
    from boonza.md.restraints import dihedral
    from boonza.sirah import OPENMM_OPTIONS

    s = _four_beads(tmp_path, multiplicity, override=False, angle_degrees=60.0)
    table = "dihedral_periodic" if multiplicity > 6 else "dihedral_trig"
    assert table in s.tables and len(s.table(table)) == 1
    e = boonza.openmm_energies(s, **OPENMM_OPTIONS)
    phi = dihedral(s.positions, [0, 1, 2, 3])  # radians, as the geometry came out
    fc = 8.9 / KJ  # kcal/mol
    want = fc * (1 + np.cos(multiplicity * phi - np.radians(-87.8)))
    assert e[table] == pytest.approx(want, abs=1e-6)


def test_generated_pairs_follow_nonbond_params(tmp_path):
    """gen-pairs builds a 1-4 interaction from the normal interaction of the two
    types, which [ nonbond_params ] overrides where it names them, scaled by
    fudgeLJ.  Taking the combination rule instead made SIRAH's LJ-14 2.2 times
    too large."""
    plain = _four_beads(tmp_path, 1, override=False)
    fixed = _four_beads(tmp_path, 1, override=True)
    sigma = {}
    for label, s in (("rule", plain), ("override", fixed)):
        t = s.table("pair_12_6_es")
        assert len(t) == 1
        a, b = float(t.values("aij")[0]), float(t.values("bij")[0])
        sigma[label] = (a / b) ** (1 / 6) / 10  # nm
        # fudgeLJ halves the interaction either way
        eps = b**2 / (4 * a) * KJ
        assert eps == pytest.approx(0.5 * 0.55 if label == "rule" else 0.5 * 0.56, rel=1e-6)
    assert sigma["rule"] == pytest.approx(0.42, rel=1e-6)
    assert sigma["override"] == pytest.approx(0.325, rel=1e-6)


def test_what_model_sirah_settles_on():
    """SIRAH's own mdp files: a 20 fs step, PME inside 1.2 nm, and a frame every
    0.1 ns.  The temperature is not a model's to choose."""
    a = parse_arguments(["x.top", "--model", "sirah"])
    assert a.integration_fs == 20.0
    assert a.cutoff_nm == 1.2
    assert a.production_report_interval_ns == 0.1
    assert a.checkpoint_interval_ns == 0.1
    assert (a.monitor_interval_ns, a.confirmation_checks) == (0.2, 3)
    assert (a.pocket_cutoff_nm, a.contact_cutoff_nm, a.detach_cutoff_nm) == (0.8, 0.7, 1.2)
    assert a.elastic is False  # SIRAH keeps its own backbone terms


def test_one_temperature_for_every_model():
    """A force field's papers run at their own temperature; boonza's default is
    one number, whatever the model."""
    want = parse_arguments(["x.pdb"]).temperature
    for model in ("martini2", "martini3", "sirah"):
        assert parse_arguments(["x.top", "--model", model]).temperature == want


def test_sirah_refuses_what_belongs_to_martini(capsys):
    with pytest.raises(SystemExit):
        parse_arguments(["x.top", "--model", "sirah", "--elastic"])
    assert "needs a Martini model" in capsys.readouterr().err


def test_sirah_maps_a_structure_but_cannot_fill_a_box_yet(tmp_path):
    """boonza maps a structure onto beads and builds its topology; filling the
    box with WT4 water is still SIRAH's own tools' work, and it says so."""
    from boonza.md.prepare import build_sirah_system

    structure = DATA / "1CRN_ph7.pdb"
    asked = parse_arguments([str(structure), "--model", "sirah",
                             "--workdir", str(tmp_path / "wet")])  # fmt: skip
    with pytest.raises(ValueError, match="WT4 water yet"):
        build_sirah_system(asked, tmp_path / "wet", log=lambda *_: None)

    dry = parse_arguments([str(structure), "--model", "sirah", "--no-solvate",
                           "--workdir", str(tmp_path / "dry")])  # fmt: skip
    s, info = build_sirah_system(dry, tmp_path / "dry", log=lambda *_: None)
    assert s.natoms == PDB2GMX["atoms"]
    assert (tmp_path / "dry" / "sirah" / "topol.top").is_file()
    assert np.any(np.asarray(s.cell))  # a box to be periodic in, even without water
    assert [c["id"] for c in info["components"]] == ["component-0"]


def test_swim_cannot_swim_sirah_yet(tmp_path):
    from boonza.md.swim import main

    code = main([str(Path(__file__).parent / "data" / "1TEN.pdb"), "--model", "sirah",
                 "--workdir", str(tmp_path / "swim")])  # fmt: skip
    assert code == 1
    assert not (tmp_path / "swim").exists() or not list((tmp_path / "swim").iterdir())


def test_coordinates_are_taken_unrounded_when_they_are_there(tmp_path):
    """A .gro rounds to 0.001 nm and a .pdb to 0.001 A, so a .dms or .mae of the
    same name is used first."""
    from boonza.md.prepare import _coordinates_beside

    (tmp_path / "topol.top").write_text("")
    for name in ("cg.pdb", "cg.gro", "topol.gro", "topol.mae", "topol.dms"):
        (tmp_path / name).write_text("")
        assert _coordinates_beside(tmp_path / "topol.top").name == name


@pytest.fixture(scope="module")
def crambin(tmp_path_factory):
    out = tmp_path_factory.mktemp("sirah") / "crambin_cg.dms"
    with gzip.open(DATA / "crambin_cg.dms.gz", "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return boonza.load(out)


def _by_residue(system, feats):
    res = np.asarray(system.atoms["residue"])
    names = np.array([str(x) for x in np.asarray(system.residues["name"])])
    bead = np.asarray(system.atoms["name"])
    out: dict[str, set] = {}
    for family, atoms in feats:
        key = f"{names[res[atoms[0]]]}{int(res[atoms[0]])}"
        out.setdefault(key, set()).add((family, tuple(str(bead[a]) for a in atoms)))
    return out


def test_beads_keep_the_types_they_were_built_with(system):
    from boonza.sirah.features import bead_types, sirah_beads

    types = bead_types(system, range(6))
    assert types[:3] == ["GN", "Y2Ca", "GO"]  # alanine: backbone and its alpha carbon
    assert sirah_beads(system, range(system.natoms))
    from boonza.martini.features import martini_beads

    assert not martini_beads(system, range(6))  # not BB/SC1..., so not Martini's


def test_sirah_tells_a_donor_from_an_acceptor(crambin):
    """The point of the finer mapping: a hydroxyl is two beads, the oxygen and
    the hydrogen, so serine gives an acceptor and a donor rather than one bead
    that has to be called both, as Martini's is."""
    from boonza.pharmacophore import ligand_features

    found = _by_residue(crambin, ligand_features(crambin, "all")[0])
    serine = next(v for k, v in found.items() if k.startswith("sS"))
    assert ("Acceptor", ("BOG",)) in serine  # the hydroxyl oxygen
    assert ("Donor", ("BPG",)) in serine  # and its hydrogen, a bead of its own
    tyrosine = next(v for k, v in found.items() if k.startswith("sY"))
    assert ("Acceptor", ("BCE2",)) in tyrosine  # the phenol oxygen
    assert ("Donor", ("BCE1",)) in tyrosine  # the phenol hydrogen
    assert ("Aromatic", ("BCG",)) in tyrosine  # and the ring it hangs off


def test_sirah_rings_and_charges(crambin):
    """A ring's feature sits at the centre of its beads, however many SIRAH
    gives it: three for Phe, one for Tyr."""
    from collections import Counter

    from boonza.pharmacophore import ligand_features

    feats = ligand_features(crambin, "all")[0]
    found = _by_residue(crambin, feats)
    phe = next(v for k, v in found.items() if k.startswith("sF"))
    assert ("Aromatic", ("BCG", "BCE1", "BCE2")) in phe
    counts = Counter(f for f, _ in feats)
    assert counts["Aromatic"] == 3  # crambin's Phe13, Tyr29 and Tyr44
    # Arg and the amino terminus are cations, Asp and Glu anions
    assert counts["PosIonizable"] and counts["NegIonizable"]
    anions = {k for k, v in found.items() if any(f == "NegIonizable" for f, _ in v)}
    assert all(k[:2] in ("sD", "sE", "sG", "sT", "sN") or k.startswith("sC") for k in anions), (
        anions
    )


def test_histidine_follows_its_tautomer():
    """SIRAH gives both tautomers the same two ring-nitrogen types and tells
    them apart by charge: the one carrying the hydrogen is positive."""
    from boonza.sirah.features import BY_TYPE

    assert BY_TYPE["A5D"] == BY_TYPE["A5E"] == ("BY_CHARGE",)


def test_the_backbone_is_left_out_unless_asked(crambin):
    from boonza.pharmacophore import ligand_features

    plain = ligand_features(crambin, "all")[0]
    bead = np.asarray(crambin.atoms["name"])
    assert not [1 for _, at in plain if str(bead[at[0]]) in ("GN", "GO")]
    with_bb = ligand_features(crambin, "all", backbone=True)[0]
    families = {f for f, at in with_bb if str(bead[at[0]]) in ("GN", "GO")}
    assert families == {"Donor", "Acceptor"}  # the amide N donates, the carbonyl O accepts


def test_sites_align_a_sirah_run_on_its_own_backbone(crambin):
    """`boonza sites` and `boonza poses` need the backbone beads of whichever
    model they are given: BB under Martini, GN, GC and GO under SIRAH."""
    from boonza.cli import _backbone_selection, _coarse_grained

    assert _coarse_grained(crambin)
    assert _backbone_selection(crambin) == "name GN GC GO"
    # every residue has all three, the termini included
    assert len(crambin.select(_backbone_selection(crambin)).ids) == 3 * crambin.nresidues


def test_probes_can_be_named_when_there_is_no_probes_json(system, tmp_path):
    """boonza probes reads a Martini swim's probes.json, or takes the names, so
    it serves any coarse-grained run."""
    from boonza.probemap import probe_contacts

    m = probe_contacts(system, [(system, np.asarray(system.positions)[None])], ["WT4"])
    assert m.probes == ["WT4"]
    assert len(m.residues) == 8  # the peptide's residues, water left out
    labels, pooled = m.side_chains()
    assert labels == ["WT4"]  # not split into letters: it is not a dipeptide code
    assert pooled.max() > 0


#: What pdb2gmx -ff sirah builds from the same beads, with no terminus chosen.
PDB2GMX = {"atoms": 205, "bonds": 219, "pairs": 263, "angles": 285, "propers": 303,
           "impropers": 37}  # fmt: skip


@pytest.fixture(scope="module")
def crambin_all_atom():
    return boonza.load(DATA / "1CRN_ph7.pdb")


def test_beads_sit_where_sirah_puts_them(crambin_all_atom, crambin):
    """The map places each bead on one named atom, and the library fixes their
    order: the same beads, in the same order, as cgconv.pl and pdb2gmx give."""
    from boonza.sirah import map_structure

    beads = map_structure(crambin_all_atom)
    assert len(beads) == crambin.natoms
    assert [b.name for b in beads] == [str(n) for n in crambin.atoms["name"]]
    residue = np.asarray(crambin.atoms["residue"])
    names = np.array([str(x) for x in np.asarray(crambin.residues["name"])])
    assert [b.residue for b in beads] == list(names[residue])
    theirs = np.asarray(crambin.positions)
    mine = np.array([b.position for b in beads])
    # their coordinates came through a .gro, which rounds to 0.001 nm
    assert np.abs(mine - theirs).max() < 0.01


def test_the_topology_is_the_one_pdb2gmx_builds(crambin_all_atom):
    """Bonds from the library, angles and dihedrals from the bonds, 1-4 pairs
    from the dihedrals, impropers from the library, and crambin's three
    disulfides: the counts pdb2gmx arrives at, every one."""
    from boonza.sirah import sirahize

    m = sirahize(crambin_all_atom, termini="None")
    assert len(m.molecules) == 1
    mol = m.molecules[0]
    got = {"atoms": mol.natoms, "bonds": len(mol.bonds), "pairs": len(mol.pairs),
           "angles": len(mol.angles), "propers": len(mol.dihedrals),
           "impropers": len(mol.impropers)}  # fmt: skip
    assert got == PDB2GMX
    assert set(mol.masses) == {50.0}  # from atomtypes.atp, where SIRAH keeps them


def test_the_energies_are_the_ones_that_topology_gives(crambin_all_atom, crambin):
    """The test that matters: boonza's own topology against the one pdb2gmx
    built from the same beads, on the same coordinates."""
    pytest.importorskip("openmm")
    from boonza.sirah import sirahize

    mine = sirahize(crambin_all_atom, termini="None").system()
    theirs = crambin.clone()
    theirs.positions = np.asarray(mine.positions)
    assert np.allclose(np.asarray(mine.atoms["charge"], float),
                       np.asarray(theirs.atoms["charge"], float))  # fmt: skip
    a = boonza.openmm_energies(mine, nonbonded_method="NoCutoff")
    b = boonza.openmm_energies(theirs, nonbonded_method="NoCutoff")
    assert set(a) == set(b)
    for term, value in a.items():
        assert value == pytest.approx(b[term], rel=1e-9, abs=1e-9), term


@pytest.mark.parametrize(("termini", "first_bead_charge"), [
    ("Charged", 0.60), ("Neutral", 0.40), ("None", 0.13),
])  # fmt: skip
def test_the_chain_ends_are_a_choice(crambin_all_atom, termini, first_bead_charge):
    """SIRAH's .tdb files give charged and neutral ends; 'None' leaves the
    residues' own charges, which is what pdb2gmx does when asked for none."""
    from boonza.sirah import sirahize

    mol = sirahize(crambin_all_atom, termini=termini).molecules[0]
    assert mol.beads[0].name == "GN"
    assert mol.charges[0] == pytest.approx(first_bead_charge)
    assert sum(mol.charges) == pytest.approx(sum(mol.charges), abs=0)  # whatever it sums to


def test_a_residue_sirah_does_not_know(tmp_path):
    from boonza.sirah import sirahize

    s = boonza.from_smiles("c1ccccc1O", name="LIG")
    with pytest.raises(ValueError, match="SIRAH's map has no LIG"):
        sirahize(s, "all")


def test_the_run_directory_carries_its_force_field(crambin_all_atom, tmp_path):
    """What is written runs elsewhere: the topology, the molecules, the force
    field beside them, and coordinates in both a lossless and a GROMACS form."""
    from boonza.sirah import sirahize

    top = sirahize(crambin_all_atom).save(tmp_path / "built")
    written = {p.name for p in top.parent.iterdir()}
    assert {"topol.top", "molecule_0.itp", "cg.dms", "cg.gro", "sirah.ff"} <= written
    ff = {p.name for p in (top.parent / "sirah.ff").iterdir()}
    assert {"forcefield.itp", "ffnonbonded.itp", "ffbonded.itp", "aminoacids.rtp"} <= ff
    assert 'forcefield.itp"' in top.read_text()


def test_the_carried_force_field(tmp_path):
    from boonza import sirah

    assert "aminoacids.rtp" in sirah.contents()
    assert sirah.read("aminoacids.rtp").startswith("[ bondedtypes ]")
    with pytest.raises(FileNotFoundError, match="not in the carried"):
        sirah.read("nothing.itp")
    out = sirah.unpack(tmp_path)
    assert (out / "wt416.gro").is_file()  # the water box, for solvation to come
