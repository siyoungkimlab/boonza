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


def test_a_run_can_be_built_dry(tmp_path):
    """Without water the beads still get a periodic box, since the run needs
    one; a membrane is not something SIRAH does here."""
    from boonza.md.prepare import build_sirah_system

    dry = parse_arguments([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah", "--no-solvate",
                           "--workdir", str(tmp_path / "dry")])  # fmt: skip
    s, info = build_sirah_system(dry, tmp_path / "dry", log=lambda *_: None)
    assert s.natoms == PDB2GMX["atoms"]  # the protein alone
    assert np.any(np.asarray(s.cell))
    assert [c["id"] for c in info["components"]] == ["component-0"]

    with pytest.raises(SystemExit):  # a bilayer is Martini's, and is refused up front
        parse_arguments([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah",
                         "--solvate", "membrane", "--upper", "POPC",
                         "--workdir", str(tmp_path / "m")])  # fmt: skip


def test_swim_refuses_a_ligand_library_in_sirah(tmp_path):
    """SIRAH swims dipeptide probes, as Martini does: an arbitrary small molecule
    is not something either model parameterizes."""
    from boonza.md.swim import main

    code = main([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah", "--ligands", "library.sdf",
                 "--workdir", str(tmp_path / "swim")])  # fmt: skip
    assert code == 1


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
    # what the topology includes and what those include in turn, and no more:
    # the release's libraries, boxes, maps and documentation stay in the archive
    assert ff == {"forcefield.itp", "ffnonbonded.itp", "ffbonded.itp", "lipid_ffbonded.itp",
                  "solv.itp", "wls.itp", "wt4.itp", "sirah_ions.itp"}  # fmt: skip
    assert 'forcefield.itp"' in top.read_text()


def test_the_carried_force_field(tmp_path):
    from boonza import sirah

    assert "aminoacids.rtp" in sirah.contents()
    assert sirah.read("aminoacids.rtp").startswith("[ bondedtypes ]")
    with pytest.raises(FileNotFoundError, match="not in the carried"):
        sirah.read("nothing.itp")
    # what the archive holds is the force field, not what a file browser left
    assert not [f for f in sirah.contents() if Path(f).name.startswith(".")]
    out = sirah.unpack(tmp_path)
    assert (out / "forcefield.itp").is_file()
    for name in ("aminoacids.rtp", "wt416.gro", "sirah_prot.map", "0README"):
        assert not (out / name).exists(), name  # read from the archive, not from here
    everything = sirah.unpack(tmp_path / "all", everything=True)
    assert (everything / "aminoacids.rtp").is_file()  # for pdb2gmx beside it
    assert (everything / "wt416.gro").is_file() and (everything / "0README").is_file()


def test_water_fills_the_box_as_sirah_equilibrated_it(crambin_all_atom):
    """WT4 comes from the force field's own box, tiled whole so the water meets
    itself as it was equilibrated; only the solute displaces any."""
    from boonza.sirah.build import EQUILIBRIUM_DENSITY, TILE_DENSITY, sirahize, solvate

    m = sirahize(crambin_all_atom)
    wet = solvate(m, padding=10.0, salt=0.15)
    counts = dict(wet.solvent)
    box = np.diag(np.asarray(wet.cell)) / 10.0
    volume = float(np.prod(box))
    assert counts["WT4"] > 300
    # the box grew to whole 1.72 nm tiles, so nothing is lost at the faces
    assert np.allclose(box / 1.72, np.round(box / 1.72), atol=1e-6)
    density = counts["WT4"] / volume
    assert 0.85 * EQUILIBRIUM_DENSITY < density < TILE_DENSITY
    # cut mid-tile instead and a slab of water goes missing at every face
    loose = solvate(m, padding=10.0, salt=0.15, fit=False)
    thinner = dict(loose.solvent)["WT4"] / float(np.prod(np.diag(np.asarray(loose.cell)) / 10))
    assert thinner < density


def test_the_ions_neutralize_and_salt_the_box(crambin_all_atom):
    """Counter-ions first, then pairs: one pair for every 34 waters is SIRAH's
    own reckoning of 0.15 M."""
    from boonza.sirah.build import sirahize, solvate

    m = sirahize(crambin_all_atom, termini="Charged")
    charge = sum(c for mol in m.molecules for c in mol.charges)
    wet = solvate(m, padding=10.0, salt=0.15)
    counts = dict(wet.solvent)
    assert round(charge + counts["NaW"] - counts["ClW"], 6) == 0  # the box is neutral
    pairs = min(counts["NaW"], counts["ClW"])
    assert pairs == pytest.approx(counts["WT4"] / 34, rel=0.25)

    salty = dict(solvate(m, padding=10.0, salt=0.30).solvent)
    assert min(salty["NaW"], salty["ClW"]) > pairs  # more salt, more pairs
    none = dict(solvate(m, padding=10.0, salt=0.0).solvent)
    assert min(none["NaW"], none["ClW"]) == 0  # only what neutralizes


def test_water_keeps_clear_of_the_solute(crambin_all_atom):
    from boonza.sirah.build import WATER_CLASH, sirahize, solvate
    from boonza.spatial import min_dist2

    m = sirahize(crambin_all_atom)
    wet = solvate(m, padding=10.0, salt=0.15)
    solute = (
        wet.nbeads
        - 4 * dict(wet.solvent)["WT4"]
        - dict(wet.solvent)["NaW"]
        - dict(wet.solvent)["ClW"]
    )
    xyz = np.asarray(wet.positions)
    cell = np.asarray(wet.cell)
    apart = np.sqrt(min_dist2(xyz[solute:], xyz[:solute], 2 * WATER_CLASH, cell=cell).min())
    assert apart >= WATER_CLASH - 1e-6


def test_a_solvated_run_is_built_and_runnable(crambin_all_atom, tmp_path):
    """`boonza md --model sirah protein.pdb` maps, fills the box and runs."""
    from boonza.md.prepare import build_sirah_system

    args = parse_arguments([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah", "--gromacs",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    s, info = build_sirah_system(args, tmp_path, log=lambda *_: None)
    names = {str(n) for n in s.residues["name"]}
    assert {"WT4", "NaW", "ClW"} <= names
    assert s.natoms > 1000 and np.any(np.asarray(s.cell))
    assert (tmp_path / "sirah" / "topol.top").is_file()
    assert "solv.itp" in (tmp_path / "sirah" / "topol.top").read_text()


#: What pdb2gmx -ff sirah builds from the beads of tests/data/sirah/dna_duplex.pdb.
PDB2GMX_DNA = {"atoms": 238, "bonds": 276, "pairs": 342, "angles": 430, "propers": 580,
               "impropers": 0}  # fmt: skip


@pytest.fixture(scope="module")
def dna():
    return boonza.load(DATA / "dna_duplex.pdb")


def test_dna_maps_and_builds_as_pdb2gmx_does(dna):
    """A 20-mer duplex: the same beads in the same order, the same topology, and
    each strand its own molecule."""
    from boonza.sirah import map_structure, sirahize

    beads = map_structure(dna, "all")
    assert len(beads) == PDB2GMX_DNA["atoms"]
    m = sirahize(dna, "all", termini="None")
    assert len(m.molecules) == 2  # two strands
    got = {
        "atoms": sum(x.natoms for x in m.molecules),
        "bonds": sum(len(x.bonds) for x in m.molecules),
        "pairs": sum(len(x.pairs) for x in m.molecules),
        "angles": sum(len(x.angles) for x in m.molecules),
        "propers": sum(len(x.dihedrals) for x in m.molecules),
        "impropers": sum(len(x.impropers) for x in m.molecules),
    }
    assert got == PDB2GMX_DNA


def test_dna_energies_match_that_topology(dna, tmp_path):
    """Against the system pdb2gmx built from the same beads, term for term."""
    pytest.importorskip("openmm")
    import gzip
    import shutil

    from boonza.sirah import sirahize

    out = tmp_path / "dna_cg.dms"
    with gzip.open(DATA / "dna_cg.dms.gz", "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    theirs = boonza.load(out)
    mine = sirahize(dna, "all", termini="None").system()
    assert np.allclose(np.asarray(mine.atoms["charge"], float),
                       np.asarray(theirs.atoms["charge"], float))  # fmt: skip
    a = boonza.openmm_energies(mine, nonbonded_method="NoCutoff")
    b = boonza.openmm_energies(theirs, nonbonded_method="NoCutoff")
    for term, value in a.items():
        assert value == pytest.approx(b[term], rel=1e-9, abs=1e-9), term


def test_a_nucleotide_at_a_strand_end_is_its_own_residue(dna):
    """DAX in the middle, AX5 at the 5' end, AX3 at the 3': the .r2b table, the
    way pdb2gmx picks a building block by position."""
    from boonza.sirah import map_structure
    from boonza.sirah.build import read_variants

    variants = read_variants()
    assert variants["DAX"] == {"main": "DAX", "5": "AX5", "3": "AX3"}
    beads = map_structure(dna, "all")
    from boonza.sirah.build import _at_the_ends, read_residues

    library, _ = read_residues()
    _at_the_ends(beads, library)
    ends = {b.residue for b in beads if b.resid in (1, 20, 21, 40)}
    assert any(e.endswith(("3", "5")) for e in ends)
    middle = {b.residue for b in beads if b.resid == 10}
    assert all(e.startswith("D") for e in middle)  # DAX, DTX, DGX or DCX


def test_the_sugar_bead_is_renamed_as_sirah_renames_it():
    """The map calls it C1X, for the C1' it sits on, and the library calls it
    O3'; SIRAH's .arn file is what reconciles them."""
    from boonza.sirah.build import read_map, read_renames, read_residues

    assert read_renames() == {"C1X": "O3'"}
    library, _ = read_residues()
    assert "O3'" in {a[0] for a in library["DCX"].atoms}
    assert "C1X" in {bead for bead, _ in read_map()["DC"].beads}


def test_every_library_is_read_together():
    """Proteins, DNA and ions map alike, from the maps boonza carries."""
    from boonza.sirah.build import read_map, read_residues

    mapping = read_map()
    assert {"ALA", "DA", "DT", "DG", "DC"} <= set(mapping)
    library, bonded = read_residues()
    assert {"sA", "DAX", "NaW"} <= set(library)
    assert bonded.nrexcl == 3


def test_a_sirah_run_writes_a_view_with_ca(crambin_all_atom, tmp_path):
    """SIRAH's GC bead sits on the alpha carbon itself, so naming it CA in the
    file meant for viewing is what the bead is; a viewer traces a chain through
    it, and through GC draws beads and no more."""
    from boonza.md.prepare import build_sirah_system

    args = parse_arguments([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah", "--no-solvate",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    s, _ = build_sirah_system(args, tmp_path, log=lambda *_: None)
    view = boonza.load(tmp_path / "view.dms")
    assert (tmp_path / "view.mae").is_file()
    assert view.natoms == s.natoms and view.nbonds == s.nbonds  # nothing dropped, only renamed
    assert len(view.select("name CA").ids) == len(s.select("name GC").ids) > 40
    assert not len(view.select("name GC").ids)
    assert not len(s.select("name CA").ids)  # the run's own record keeps SIRAH's names
    rest = [str(n) for n in s.atoms["name"] if str(n) != "GC"]
    assert [str(n) for n in view.atoms["name"] if str(n) != "CA"] == rest


def test_sirah_viewing_can_keep_its_own_names(crambin_all_atom):
    from boonza.sirah import sirahize

    m = sirahize(crambin_all_atom)
    built = m.system()
    assert len(m.for_viewing(built, backbone_as_ca=False).select("name GC").ids) == len(
        built.select("name GC").ids
    )


# ---- probes swimming in SIRAH -------------------------------------------------------


def test_the_sirah_probe_library():
    """The same 105 dipeptides as Martini's, mapped onto SIRAH beads, so a
    surface mapped in one model reads against the other."""
    from boonza.martini.probes import probe_charge as martini_charge
    from boonza.sirah.probes import PROBE_RESIDUES, probe, probe_charge, probe_sequences

    sequences = probe_sequences()
    assert len(sequences) == 105 == len(PROBE_RESIDUES) * (len(PROBE_RESIDUES) + 1) // 2
    for sequence in sequences:
        assert probe_charge(sequence) == pytest.approx(martini_charge(sequence))
    ek = probe("EK")
    assert len(ek.molecules) == 1 and ek.molecules[0].name == "probe_EK"
    assert {b.residue for b in ek.molecules[0].beads} == {"EK"}  # the probe's own name
    assert probe_charge("EK") == pytest.approx(0.0)  # Glu -1 and Lys +1
    assert probe_charge("RR") == pytest.approx(2.0)
    assert probe_charge("HH") == pytest.approx(0.0)  # histidine is neutral at pH 7
    assert np.allclose(ek.positions.mean(0), 0, atol=1e-6)  # centered, for placing
    assert probe("EK") is not probe("EK")  # a copy each time, for the caller to move


def test_a_probe_is_built_from_conventional_names():
    """SIRAH maps beads onto named atoms -- serine's HG, tryptophan's HE1 -- so
    every probe maps only because the builder names them as a force field does."""
    from boonza.sirah.probes import probe

    for sequence in ("SS", "TT", "WW", "YY", "HH"):
        beads = probe(sequence).molecules[0].beads
        assert len(beads) > 6, sequence  # a backbone bead each, and side chains


def test_a_prepared_sirah_swim(tmp_path):
    """`boonza swim --model sirah`: one directory per group, its probes in a box
    of WT4 water, and settings `boonza md` accepts."""
    import json

    from boonza.md.cgswim import prepare

    args = parse_arguments([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah", "--gromacs",
                            "--workdir", str(tmp_path / "swim"), "--padding-nm", "1.0",
                            "--production-ns", "10"])  # fmt: skip
    sims = prepare(args, ["EK", "LL", "RR"], types=2, copies=2, log=lambda *_: None)
    assert len(sims) == 2
    d = sims[0]
    written = json.loads((d / "probes.json").read_text())
    assert written["copies"] == 2 and written["align"] == "name GC"
    top = (d / "sirah" / "topol.top").read_text()
    for name in written["probes"]:
        # the molecule is probe_EK, so that a probe called KW cannot be taken
        # for SIRAH's potassium, while its beads are the residue EK
        assert f"probe_{name} 2" in top and f'#include "probe_{name}.itp"' in top
    assert "WT4" in top and (d / "sirah" / "sirah.ff" / "forcefield.itp").is_file()
    settings = parse_arguments(["--config", str(d / "md.toml")])
    assert settings.model == "sirah" and settings.solvate == "none"
    assert settings.integration_fs == 20.0  # SIRAH's own step, not Martini's
    assert settings.repulsion_selection == "resname " + " ".join(written["probes"])
    assert not settings.elastic  # SIRAH holds its backbone itself
    # the run reads cg.dms, which carries every parameter; topol.top and the
    # force field beside it are what GROMACS would read
    assert Path(settings.input_structure) == (d / "sirah" / "cg.dms").resolve()


def test_a_sirah_swim_is_neutral_and_runs(crambin_all_atom, tmp_path):
    from boonza.md.cgswim import build_sirah
    from boonza.sirah import OPENMM_OPTIONS, sirahize
    from boonza.sirah.probes import probe

    pytest.importorskip("openmm")
    protein = sirahize(crambin_all_atom)
    probes = [probe("RR"), probe("EE")]
    box = np.full(3, float(np.ptp(protein.positions, axis=0).max()) + 20.0)
    m = build_sirah(protein, probes, 2, box, np.random.default_rng(0), salt=0.15)
    assert m.molecule_copies == [1, 2, 2]
    charge = sum(c * sum(mol.charges)
                 for mol, c in zip(m.molecules, m.molecule_copies, strict=True))  # fmt: skip
    (_, _), (_, na), (_, cl) = m.solvent
    assert charge + na - cl == pytest.approx(0)
    s = m.system()
    assert s.natoms == m.nbeads
    assert {"RR", "EE"} <= {str(n) for n in s.residues["name"]}
    assert np.isfinite(boonza.openmm_energies(s, **OPENMM_OPTIONS)["total"])


def test_a_swim_run_finds_the_view_beside_its_topology(tmp_path):
    """The run reads a topology, which carries no view; swim writes one where it
    builds, and the run copies it in so a viewer has the beads to look at."""
    from boonza.md.cgswim import prepare
    from boonza.md.prepare import build_sirah_system

    args = parse_arguments([str(DATA / "1CRN_ph7.pdb"), "--model", "sirah", "--padding-nm", "1.0",
                            "--workdir", str(tmp_path / "swim")])  # fmt: skip
    (d,) = prepare(args, ["EK"], types=1, copies=1, log=lambda *_: None)
    assert (d / "sirah" / "view.dms").is_file() and (d / "sirah" / "view.mae").is_file()
    run = parse_arguments(["--config", str(d / "md.toml")])
    s, _ = build_sirah_system(run, tmp_path, log=lambda *_: None)
    view = boonza.load(tmp_path / "view.dms")
    assert view.natoms == s.natoms
    assert len(view.select("name CA").ids) == len(s.select("name GC").ids) > 40


def test_sirahs_solvent_is_not_a_target_of_the_probes():
    """WT4 is water, though no oxygen and two hydrogens say so; a probe map
    counts the residues of the protein, not the box it swims in."""
    from boonza.probemap import SOLVENT_NAMES

    assert {"WT4", "NaW", "ClW", "W", "ION"} <= set(SOLVENT_NAMES)


def test_a_built_system_runs_without_its_topology(crambin_all_atom, tmp_path):
    """cg.dms carries every parameter the topology gave it, so a run needs
    neither the topology nor the force field files beside it."""
    from boonza.md.prepare import build_sirah_system
    from boonza.sirah import sirahize
    from boonza.sirah.build import solvate

    built = solvate(sirahize(crambin_all_atom), padding=10.0, seed=0)
    built.save(tmp_path / "built")
    for gone in [tmp_path / "built" / "topol.top", *(tmp_path / "built" / "sirah.ff").iterdir()]:
        gone.unlink()
    args = parse_arguments([str(tmp_path / "built" / "cg.dms"), "--model", "sirah",
                            "--no-solvate", "--workdir", str(tmp_path / "run")])  # fmt: skip
    s, info = build_sirah_system(args, tmp_path / "run", log=lambda *_: None)
    assert s.natoms == built.nbeads
    assert {"WT4", "NaW", "ClW"} <= {str(n) for n in s.residues["name"]}
    assert "dihedral_periodic" in s.tables  # the parameters came with the file
    pytest.importorskip("openmm")
    from boonza.sirah import OPENMM_OPTIONS

    assert np.isfinite(boonza.openmm_energies(s, **OPENMM_OPTIONS)["total"])


def test_the_file_for_viewing_is_not_a_file_to_run(crambin_all_atom, tmp_path):
    """Its backbone bead is named CA, and a Martini view has no elastic network:
    running one would quietly let a fold go."""
    from boonza.md.prepare import build_sirah_system
    from boonza.sirah import sirahize

    m = sirahize(crambin_all_atom)
    boonza.save(m.for_viewing(), tmp_path / "view.dms")
    args = parse_arguments([str(tmp_path / "view.dms"), "--model", "sirah", "--no-solvate",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    with pytest.raises(ValueError, match="written for viewing"):
        build_sirah_system(args, tmp_path / "run", log=lambda *_: None)
