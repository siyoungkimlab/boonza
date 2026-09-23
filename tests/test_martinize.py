"""Martinizing proteins: boonza's topologies against martinize2's own.

The references in tests/data/martini were written by martinize2 (vermouth);
tests/data/martini/regenerate.py remakes them.
"""

import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.cli import main
from boonza.martini import OPENMM_OPTIONS, convert_dssp_to_martini, equilibrate, martinize, solvate
from boonza.martini.links import match_order

DATA = Path(__file__).parent / "data"
REF = DATA / "martini"
CASES = {  # name: (input, elastic network)
    "1TEN": (DATA / "1TEN.pdb", True),
    "2TRX": (DATA / "2TRX.pdb", False),
    "1HHO": (DATA / "1HHO.pdb", True),
    "5PTI": (REF / "5PTI_protein.pdb.gz", True),
}
NATOMS = {"bonds": 2, "constraints": 2, "angles": 3, "dihedrals": 4}


def _load(path: Path, tmp_path):
    if path.suffix == ".gz":
        plain = tmp_path / path.stem
        plain.write_bytes(gzip.decompress(path.read_bytes()))
        path = plain
    return boonza.load(path)


def _parse_itp(text: str):
    """Atoms, and each section as a multiset of (atoms, parameters, #ifdef context)."""
    atoms, terms = [], defaultdict(Counter)
    section, context = None, []
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("#ifdef", "#ifndef")):
            context.append(line)
        elif line.startswith("#endif"):
            context.pop()
        elif line.startswith("["):
            section = line.strip("[] ")
        elif section == "atoms":
            t = line.split()
            atoms.append((t[1], t[3], t[4], round(float(t[6]), 4),
                          float(t[7]) if len(t) > 7 else None))  # fmt: skip
        elif section in ("virtual_sitesn", "exclusions"):
            t = line.split()
            key = (t[0], t[1], tuple(sorted(t[2:]))) if section == "virtual_sitesn" else (
                t[0], tuple(sorted(t[1:])))  # fmt: skip
            terms[section][(key, tuple(context))] += 1
        elif section in NATOMS:
            t = line.split()
            n = NATOMS[section]
            idx = tuple(t[:n])
            if n == 2 and int(idx[0]) > int(idx[1]):
                idx = idx[::-1]
            terms[section][(idx, tuple(round(float(v), 3) for v in t[n:]), tuple(context))] += 1
    return atoms, terms


def _reference(name):
    d = REF / name
    top = gzip.decompress((d / "topol.top.gz").read_bytes()).decode()
    molecules, section = [], None
    for raw in top.splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
        elif section == "molecules" and line:
            mol, count = line.split()
            molecules += [mol] * int(count)
    itps = [gzip.decompress((d / f"{m}.itp.gz").read_bytes()).decode() for m in molecules]
    return itps, (d / "ss.txt").read_text().strip()


@pytest.mark.parametrize("name", CASES)
def test_the_topology_is_martinize2s(name, tmp_path):
    """Every bead (type, charge, mass) and every term, #ifdef blocks included,
    and the bead positions, against martinize2 on the same structure."""
    source, elastic = CASES[name]
    itps, ss = _reference(name)
    m = martinize(_load(source, tmp_path), ss=ss, elastic=elastic)
    assert len(m.molecules) == len(itps)
    for k, ref in enumerate(itps):
        ours_atoms, ours = _parse_itp(m.itp(k))
        ref_atoms, theirs = _parse_itp(ref)
        assert ours_atoms == ref_atoms
        assert dict(ours) == dict(theirs)
    cg = gzip.decompress((REF / name / "cg.pdb.gz").read_bytes()).decode()
    xyz = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    for line in cg.splitlines() if line.startswith("ATOM")])  # fmt: skip
    assert np.abs(m.positions - xyz).max() < 1e-3  # martinize2 writes 3 decimals


def test_the_secondary_structure_defaults_to_boonzas_dssp():
    s = boonza.load(DATA / "1HHO.pdb")
    m = martinize(s)
    assert m.ss == "".join("C" if c in ("NA", " ") else c
                           for c in boonza.dssp(s)[0][: len(m.ss)])  # fmt: skip
    helix = [n["cgsecstruct"] for n in m.molecules[0].nodes if n["atomname"] == "BB"]
    assert {"1", "2", "H"} <= set(helix)  # helix starts, ends and cores


@pytest.mark.parametrize(("dssp", "martini"), [  # as vermouth converts them
    ("CHHHHHHHC", "C1113222C"),
    ("HHHHHHHHHHCCHHHHH", "1111HH2222CC13332"),  # long helices keep a core
    ("HHHH", "3333"),  # short helices are all "3", even at the chain ends
    ("EEBGITS ", "EEE33TSC"),  # B is extended, G and I helical, blank coil
])  # fmt: skip
def test_dssp_to_martini(dssp, martini):
    assert convert_dssp_to_martini(dssp) == martini


@pytest.mark.parametrize(("o1", "r1", "o2", "r2", "ok"), [
    (0, 5, 1, 6, True), (0, 5, 1, 7, False),        # numbers: exact offsets
    (0, 5, ">", 9, True), (0, 5, ">", 3, False),    # > : later residue
    (">", 9, ">>", 12, True), (">", 9, ">>", 7, False),
    ("*", 4, 0, 4, False), ("*", 4, "*", 4, True),  # * : another residue
])  # fmt: skip
def test_link_orders(o1, r1, o2, r2, ok):
    assert match_order(o1, r1, o2, r2) is ok


def _beads(m, resname, name):
    return [n for mol in m.molecules for n in mol.nodes
            if n["resname"] == resname and n["atomname"] == name]  # fmt: skip


def test_hydrogens_decide_protonation():
    """boonza's peptides carry neutral side chains; hydrogens are recognized by
    the atom they are bonded to, whatever they are named (here H1, H2, ...)."""
    s = boonza.peptide("AKDHA")
    m = martinize(s, ss="CCCCC")
    lys, asp = _beads(m, "LYS", "SC2")[0], _beads(m, "ASP", "SC1")[0]
    assert (lys["atype"], lys["charge"]) == ("SN6d", 0.0)  # two amine hydrogens
    assert (asp["atype"], asp["charge"]) == ("P2", 0.0)  # a carboxyl hydrogen
    his = [n["atype"] for n in m.molecules[0].nodes if n["resname"] == "HIS"]
    assert his == ["P2", "TC4", "TN6d", "TN5a"]  # HE2 only: HIS as it is
    ends = [n for n in m.molecules[0].nodes if n["atomname"] == "BB"]
    assert (ends[0]["charge"], ends[-1]["charge"]) == (1.0, -1.0)  # charged termini

    no_h = s.select("not element H").clone()
    lys = _beads(martinize(no_h, ss="CCCCC"), "LYS", "SC2")[0]
    assert (lys["atype"], lys["charge"]) == ("SQ4p", 1.0)  # no hydrogens: the charged residue


def test_histidine_tautomers(tmp_path):
    """A hydrogen on ND1 alone is the neutral delta tautomer; on both, charged.
    (martinize2 charges both, having first added the HE2 of its template.)"""
    s = boonza.peptide("AHA")
    his = s.select("resname HIS").ids.tolist()
    names = np.asarray(s.atoms["name"])
    pos = s.positions

    def ring_h(nitrogen):
        n = next(i for i in his if names[i] == nitrogen)
        return pos[n]

    ne2 = ring_h("NE2")
    he2 = min((i for i in his if names[i].startswith("H")),
              key=lambda i: np.linalg.norm(pos[i] - ne2))  # fmt: skip
    nd1 = ring_h("ND1")
    cg = pos[next(i for i in his if names[i] == "CG")]
    ce1 = pos[next(i for i in his if names[i] == "CE1")]
    out = nd1 + (2 * nd1 - cg - ce1) / np.linalg.norm(2 * nd1 - cg - ce1)  # 1 Å out of the ring
    both = boonza.System.from_arrays(
        np.vstack([pos, out]), names=[*names, "HD1"],
        resnames=[*np.asarray(s.residues["name"])[np.asarray(s.atoms["residue"])], "HIS"],
        resids=[*np.asarray(s.residues["resid"])[np.asarray(s.atoms["residue"])], 2],
        elements=[*[boonza.elements.symbol(z) for z in np.asarray(s.atoms["anum"])], "H"],
    )  # fmt: skip
    hp = [n["atype"] for n in martinize(both, "all", ss="CCC").molecules[0].nodes
          if n["resname"] == "HIS"]  # fmt: skip
    assert hp == ["P2", "TC4", "TP1dq", "TP1dq"]
    delta = both.select(f"not index {he2}").clone()
    hd = [n["atype"] for n in martinize(delta, "all", ss="CCC").molecules[0].nodes
          if n["resname"] == "HIS"]  # fmt: skip
    assert hd == ["P2", "TC4", "TN5a", "TN6a"]


def test_neutral_termini_and_no_disulfides():
    s = boonza.load(DATA / "2TRX.pdb")
    m = martinize(s, "protein and chain A", ss=None, neutral_termini=True)
    bb = [n for n in m.molecules[0].nodes if n["atomname"] == "BB"]
    assert (bb[0]["atype"], bb[0]["charge"], bb[-1]["atype"], bb[-1]["charge"]) == (
        "P6", 0.0, "P6", 0.0)  # fmt: skip
    bridge = [t for t in m.molecules[0].interactions["constraints"]
              if t.meta.get("comment") == "Disulfide bridge"]  # fmt: skip
    assert len(bridge) == 1
    none = martinize(s, "protein and chain A", cys="none")
    assert not [t for t in none.molecules[0].interactions["constraints"]
                if t.meta.get("comment") == "Disulfide bridge"]  # fmt: skip


def test_what_it_refuses():
    s = boonza.load(DATA / "1HHO.pdb")
    with pytest.raises(ValueError, match="not Martini 3 protein residues: HEM"):
        martinize(s, "protein or resname HEM")
    lys = s.select("chain A and resname LYS").ids[0]
    residue = int(np.asarray(s.atoms["residue"])[lys])
    side = [i for i in s.select("chain A").ids.tolist()
            if int(np.asarray(s.atoms["residue"])[i]) == residue
            and str(np.asarray(s.atoms["name"])[i]) in ("CE", "NZ")]  # fmt: skip
    cut = s.select(f"protein and not index {' '.join(map(str, side))}").clone()
    with pytest.raises(ValueError, match=r"1 beads have no atoms.*: LYS A\d+ \(SC2\)"):
        martinize(cut)


def _toy_martini_itp(path, types):
    """Stand-in Martini parameters: every bead type with one LJ pair form."""
    lines = ["[ defaults ]", "1 2", "", "[ atomtypes ]"]
    lines += [f"{t} 72.0 0.000 A 0.0 0.0" for t in sorted(types)]
    lines += ["", "[ nonbond_params ]"]
    ts = sorted(types)
    lines += [f"{a} {b} 1 0.47 2.0" for i, a in enumerate(ts) for b in ts[i:]]
    path.write_text("\n".join(lines) + "\n")


def test_the_beads_run_in_openmm(tmp_path):
    """From atoms to an OpenMM system: bonded terms, constraints, virtual sites."""
    pytest.importorskip("openmm")
    s = boonza.load(DATA / "1HHO.pdb")
    m = martinize(s, "protein and chain A", elastic=True)
    types = {n["atype"] for mol in m.molecules for n in mol.nodes}
    _toy_martini_itp(tmp_path / "toy.itp", types)
    cg = m.system(tmp_path / "toy.itp")
    assert cg.natoms == m.nbeads
    assert np.allclose(cg.positions, m.positions, atol=6e-3)  # .gro keeps 3 decimals of nm
    assert "virtual_lc4" in cg.tables and "angle_restricted" in cg.tables
    energies = boonza.openmm_energies(cg, **OPENMM_OPTIONS)
    assert np.isfinite(energies["total"])
    saved = m.save(tmp_path / "out", martini_itp="martini_v3.0.0.itp")
    assert '#include "martini_v3.0.0.itp"' in saved.read_text()
    assert (tmp_path / "out" / "molecule_0.itp").exists()
    assert (tmp_path / "out" / "cg.gro").exists()


def test_the_command_line(tmp_path, capsys):
    out = tmp_path / "cg"
    assert main(["martinize", str(DATA / "1HHO.pdb"), str(out), "--elastic"]) == 0
    assert "2 molecules, 680 beads" in capsys.readouterr().out
    assert {p.name for p in out.iterdir()} == {"topol.top", "molecule_0.itp", "molecule_1.itp",
                                               "cg.gro"}  # fmt: skip
    rubber = [line for line in (out / "molecule_0.itp").read_text().splitlines()
              if line.endswith(" 700")]  # fmt: skip
    assert rubber


@pytest.fixture(scope="module")
def solvated():
    m = martinize(boonza.load(DATA / "1HHO.pdb"), "protein and chain A", elastic=True)
    return m, solvate(m, padding=10.0, salt=0.15)


def test_solvate(solvated):
    """Water fills the box without touching the protein, and ions neutralize it
    and bring NaCl to 0.15 M (counted against the 4 waters each bead is)."""
    from boonza.spatial import min_dist2, pairs_within

    dry, wet = solvated
    (w, nw), (na, n_na), (cl, n_cl) = wet.solvent
    assert (w, na, cl) == ("W", "NA", "CL")
    assert wet.nbeads == dry.nbeads + nw + n_na + n_cl
    charge = sum(n["charge"] for mol in wet.molecules for n in mol.nodes)
    assert charge + n_na - n_cl == pytest.approx(0)
    pairs = min(n_na, n_cl)
    assert pairs == int(0.15 / 55.345 * (4 * (nw + n_na + n_cl) - abs(round(charge))))
    box = np.diag(wet.cell)
    extent = dry.positions.max(0) - dry.positions.min(0)
    assert box == pytest.approx(np.full(3, extent.max() + 20.0))
    x = wet.positions
    assert (x >= 0).all() and (x <= box).all()
    protein, water = x[: dry.nbeads], x[dry.nbeads :]
    assert (min_dist2(water, protein, 4.2, cell=wet.cell) > 4.2**2 - 1e-3).all()
    i, j, d2 = pairs_within(water, 3.5, cell=wet.cell)
    assert len(i) == 0  # not even across the box's faces
    # the proteins moved, not reshaped
    assert np.allclose(protein - protein.mean(0), dry.positions - dry.positions.mean(0))
    density = nw / (np.prod(box) / 1000)  # beads per nm^3 of box
    assert 6.0 < density < 8.3  # bulk is 8.2; the protein takes the rest
    with pytest.raises(ValueError, match="already solvated"):
        solvate(wet)


def test_solvated_beads_run(solvated, tmp_path):
    pytest.importorskip("openmm")
    import openmm as mm
    from openmm import app

    _, wet = solvated
    types = {n["atype"] for mol in wet.molecules for n in mol.nodes} | {"W", "TQ5"}
    _toy_martini_itp(tmp_path / "toy.itp", types)
    s = wet.system(tmp_path / "toy.itp")
    names = np.asarray(s.residues["name"])
    assert (names == "W").sum() == wet.solvent[0][1]
    top, system, pos = boonza.to_openmm(s, **OPENMM_OPTIONS)
    integrator = mm.LangevinMiddleIntegrator(310, 1.0, 0.002)
    sim = app.Simulation(top, system, integrator, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(pos)
    equilibrate(sim, steps=10)
    assert integrator.getStepSize().value_in_unit(mm.unit.picosecond) == pytest.approx(0.020)
    sim.step(10)
    assert np.isfinite(sim.context.getState(getEnergy=True).getPotentialEnergy()._value)


TOY_LIPIDS = """[ moleculetype ]
TLP 1
[ atoms ]
1 Q5 1 TLP HD 1 -1.0
2 N4a 1 TLP GL1 2 0
3 N4a 1 TLP GL2 3 0
4 C1 1 TLP C1A 4 0
5 C1 1 TLP C2A 5 0
6 C1 1 TLP C1B 6 0
7 C1 1 TLP C2B 7 0
[ bonds ]
1 2 1 0.47 1250
2 3 1 0.37 1250
2 4 1 0.47 1250
4 5 1 0.47 1250
3 6 1 0.47 1250
6 7 1 0.47 1250
[ angles ]
2 4 5 2 180.0 35.0
3 6 7 2 180.0 35.0

[ moleculetype ]
STR 1
[ atoms ]
1 P1 1 STR OH 1 0 0.0
2 SC3 1 STR R1 2 0 72.0
3 SC3 1 STR R2 3 0 72.0
4 C2 1 STR C1 4 0 72.0
5 C2 1 STR C2 5 0 72.0
[ bonds ]
4 5 1 0.44 5000
#ifdef FLEXIBLE
2 3 1 0.35 100000
#else
[ constraints ]
4 3 1 0.75
4 2 1 0.78
3 2 1 0.35
#endif
[ virtual_sites3 ]
1 4 3 2 4 1.09 0.36 0.23
[ exclusions ]
1 2 3 4 5
"""


@pytest.fixture(scope="module")
def toy_lipids(tmp_path_factory):
    path = tmp_path_factory.mktemp("lipids") / "toy_lipids.itp"
    path.write_text(TOY_LIPIDS)
    return path


def test_lipid_templates(toy_lipids):
    """Straight templates from the topology: constraints at their lengths,
    virtual sites where their parents put them, long axis along z, head up."""
    from boonza.martini import lipid_templates

    t = lipid_templates([toy_lipids])
    lipid, sterol = t["TLP"], t["STR"]
    assert lipid.charge == -1.0 and sterol.charge == 0.0
    x = lipid.xyz
    assert x[0, 2] == x[:, 2].max() and x[:, 2].min() == 0.0  # head up
    assert abs(x[4, 0] - x[6, 0]) > 2.0  # the tails side by side
    for i, j in ((1, 3), (3, 4), (2, 5), (5, 6)):  # bonded beads about a bond apart
        assert 3.0 < np.linalg.norm(x[i] - x[j]) < 6.0
    y = sterol.xyz
    for i, j, d in ((3, 2, 7.5), (3, 1, 7.8), (2, 1, 3.5)):
        assert np.linalg.norm(y[i] - y[j]) == pytest.approx(d, abs=1e-3)
    rij, rik = y[2] - y[3], y[1] - y[3]
    assert y[0] == pytest.approx(y[3] + 1.09 * rij + 0.36 * rik + 0.023 * np.cross(rij, rik))


def test_bilayer(toy_lipids, tmp_path):
    from boonza.martini import bilayer
    from boonza.spatial import pairs_within

    m = bilayer([toy_lipids], {"TLP": 3, "STR": 1}, {"TLP": 1}, size=60.0,
                area_per_lipid=60.0, water=20.0, salt=0.1)  # fmt: skip
    groups = [(n, c) for n, c, _ in m.lipids]
    upper = dict(groups[:2])
    per_leaflet = sum(upper.values())  # a lattice near 60 Å² a lipid
    assert 60 * 60 / per_leaflet == pytest.approx(60.0, rel=0.1)
    assert upper["TLP"] == 3 * upper["STR"]
    assert groups[2:] == [("TLP", per_leaflet)]
    (w, nw), (na, n_na), (cl, n_cl) = m.solvent
    lipid_charge = -(upper["TLP"] + per_leaflet)
    assert lipid_charge + n_na - n_cl == 0
    box = np.diag(m.cell)
    assert box[:2] == pytest.approx([60.0, 60.0])
    nlip = sum(c * len(b) for _, c, b in m.lipids)
    lipids, water = m.positions[:nlip], m.positions[nlip:]
    top, bottom = lipids[:, 2].max(), lipids[:, 2].min()
    assert not ((water[:, 2] < top - 2) & (water[:, 2] > bottom + 2)).any()  # none inside
    assert box[2] == pytest.approx(top - bottom + 40.0, abs=2.0)  # water on both sides
    # lipids turned to clear each other: no two molecules' beads overlap
    mol = np.repeat(np.arange(sum(c for _, c, _ in m.lipids)),
                    [len(b) for _, c, b in m.lipids for _ in range(c)])  # fmt: skip
    i, j, _ = pairs_within(lipids, 2.0, cell=m.cell)
    assert not (mol[i] != mol[j]).any()
    text = m.top("martini.itp")
    assert f'#include "{toy_lipids.resolve()}"' in text
    assert f"TLP {upper['TLP']}\nSTR {upper['STR']}\nTLP {per_leaflet}\nW " in text


def test_bilayer_around_a_protein(toy_lipids, tmp_path):
    """Lipids on the protein are left out, and the box's height takes in the
    protein; with ``protein_origin`` its z = 0 is the midplane."""
    from boonza.martini import bilayer
    from boonza.spatial import min_dist2

    p = martinize(boonza.peptide("AAAAAAAAAAAAAAAAAAAA"), ss="H" * 20)
    x = p.positions
    axis = np.linalg.svd(x - x.mean(0))[2][0]
    p.positions = x @ _to_z(axis).T  # the helix across the membrane
    free = bilayer([toy_lipids], {"TLP": 1}, size=60.0, water=10.0)
    m = bilayer([toy_lipids], {"TLP": 1}, size=60.0, water=10.0, protein=p)
    assert sum(c for _, c, _ in m.lipids) < sum(c for _, c, _ in free.lipids)
    prot = m.positions[: p.nbeads]
    lipids = m.positions[p.nbeads : p.nbeads + sum(c * len(b) for _, c, b in m.lipids)]
    assert (min_dist2(lipids, prot, 4.5, cell=m.cell) >= 4.5**2 - 1e-3).all()
    mid = np.diag(m.cell)[2] / 2
    assert prot.mean(0)[2] == pytest.approx(mid)
    assert prot[:, 2].min() > 0 and prot[:, 2].max() < np.diag(m.cell)[2]
    shifted = bilayer([toy_lipids], {"TLP": 1}, size=60.0, water=10.0, protein=p,
                      protein_origin=True)  # fmt: skip
    z0 = shifted.positions[: p.nbeads, 2] - p.positions[:, 2]
    assert z0 == pytest.approx(np.full(p.nbeads, np.diag(shifted.cell)[2] / 2))


def _to_z(axis):
    """A rotation taking ``axis`` to z."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(axis, z), float(axis @ z)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def test_bilayer_runs(toy_lipids, tmp_path):
    pytest.importorskip("openmm")
    from boonza.martini import bilayer

    m = bilayer([toy_lipids], {"TLP": 1, "STR": 1}, size=50.0, water=15.0)
    _toy_martini_itp(tmp_path / "toy.itp", {"Q5", "N4a", "C1", "P1", "SC3", "C2", "W", "TQ5"})
    s = m.system(tmp_path / "toy.itp")
    assert "virtual_out3" in s.tables and "constraint_ah1" in s.tables
    e = boonza.openmm_energies(s, **OPENMM_OPTIONS)
    assert np.isfinite(e["total"])


def test_bilayer_command_line(toy_lipids, tmp_path, capsys):
    out = tmp_path / "memb"
    args = ["bilayer", str(out), "--lipid-itp", str(toy_lipids), "--upper", "TLP:3,STR:1",
            "--lower", "TLP", "--size", "40", "--water", "15"]  # fmt: skip
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "TLP" in printed and "STR" in printed and " W" in printed
    assert {p.name for p in out.iterdir()} == {"topol.top", "solvent.itp", "cg.gro"}


def test_the_parameters_come_with_boonza(tmp_path):
    """The Martini 3 files are carried, so martinizing needs no download."""
    from boonza.martini import LIPIDS, NONBONDED, parameters

    every = parameters()
    assert len(every) >= 7 and all(p.is_file() for p in every)
    assert parameters(NONBONDED)[0].name == NONBONDED
    assert [p.name for p in parameters(*LIPIDS)] == list(LIPIDS)
    assert "10.1038" in (parameters(NONBONDED)[0].parent / "README.md").read_text()
    with pytest.raises(FileNotFoundError, match="does not carry"):
        parameters("martini_v2.itp")


def test_martinizing_needs_no_parameter_file(tmp_path):
    """system() with nothing given is system() pointed at the file it carries."""
    from boonza.martini import NONBONDED, parameters

    m = martinize(boonza.load(DATA / "1HHO.pdb"), "protein and chain A", elastic=True)
    carried = m.system()
    named = m.system(parameters(NONBONDED)[0])
    assert carried.natoms == named.natoms
    assert set(carried.tables) == set(named.tables)
    for name, table in carried.tables.items():
        assert len(table) == len(named.tables[name])

    top = Path(m.save(tmp_path / "cg"))  # and what it writes can be read by GROMACS
    included = [x.split('"')[1] for x in top.read_text().splitlines() if x.startswith("#include")]
    assert Path(included[0]).is_file() and Path(included[0]).name == NONBONDED


def test_martini_2_membranes_are_solvated_with_martini_2_water(tmp_path):
    """solvate() wrote Martini 3 bead types whatever the system was, so a
    Martini 2 membrane came out with W and TQ5 beads its parameters do not
    define.  Now the version decides: Martini 3 gets the solvent.itp boonza
    writes, Martini 2 its own upstream water (in martini_v2.2.itp) and ions.
    """
    from boonza.martini import IONS_FOR, LIPIDS_FOR, NONBONDED_FOR, bilayer, parameters

    itps = [str(p) for p in parameters(*LIPIDS_FOR[2])]
    m = bilayer(itps, {"DPPC": 1}, size=60.0, martini=2, salt=0.15)
    assert m.martini == 2
    assert dict(m.solvent)["W"] > 0 and dict(m.solvent)["NA"] > 0

    top = Path(m.save(tmp_path / "cg")).read_text()
    included = [x.split('"')[1] for x in top.splitlines() if x.startswith("#include")]
    assert Path(included[0]).name == NONBONDED_FOR[2]
    assert Path(included[-1]).name == IONS_FOR[2]
    # Martini 2 defines W itself: writing one would be a duplicate moleculetype
    assert "solvent.itp" not in top
    assert not (tmp_path / "cg" / "solvent.itp").exists()
    assert all(Path(p).is_file() for p in included)

    s = m.system()  # and it loads, with Martini 2's bead types
    types = {a["type"] for a in s.atoms}
    assert {"P4", "Qd", "Qa"} <= types and "TQ5" not in types


def test_the_two_martinis_are_not_mixed(tmp_path):
    """A Martini 3 protein in a Martini 2 membrane is refused: the bead types
    share names between the versions and mean different things."""
    from boonza.martini import LIPIDS_FOR, bilayer, parameters

    protein = martinize(boonza.load(DATA / "1HHO.pdb"), "protein and chain A")
    assert protein.martini == 3
    itps = [str(p) for p in parameters(*LIPIDS_FOR[2])]
    with pytest.raises(ValueError, match="cannot be mixed"):
        bilayer(itps, {"DPPC": 1}, size=80.0, martini=2, protein=protein)
    with pytest.raises(ValueError, match="martini must be one of"):
        bilayer(itps, {"DPPC": 1}, size=60.0, martini=4)
