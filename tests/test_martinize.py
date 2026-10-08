"""Martinizing proteins: boonza's topologies against martinize2's own.

The references in tests/data/martini were written by martinize2 (vermouth);
tests/data/martini/regenerate.py remakes them.
"""

import gzip
import warnings
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


#: The cases regenerate.py also writes as Martini 2.2, and what that force field
#: calls a histidine.
MARTINI22 = ("2TRX",)
CHARMM_NAMES = {"HIS": "HSD", "HID": "HSD", "HIE": "HSE", "HIP": "HSP"}


@pytest.mark.parametrize("name", MARTINI22)
def test_the_martini2_topology_is_martinize2s(name, tmp_path):
    """The same comparison for Martini 2.2, which boonza reads from vermouth's
    own files as it reads Martini 3's: every bead and every term.

    Its residues are CHARMM's, so the structure is renamed before martinize2 sees
    it.  boonza renames them itself -- give martinize2 a residue called HIS and it
    builds the HSD block but leaves the name, which martini22's protein_resnames
    macro does not list, so every link skips that residue and the backbone comes
    out severed there.
    """
    source, elastic = CASES[name]
    itps, ss = _reference(f"{name}.martini22")
    s = _load(source, tmp_path).select("protein").clone()
    for r in range(s.nresidues):
        want = CHARMM_NAMES.get(str(s.residues["name"][r]).strip())
        if want:
            s.residue(r).name = want
    m = martinize(s, ss=ss, elastic=elastic, forcefield="martini22")
    assert m.martini == 2  # so the topology includes Martini 2's parameter file
    assert len(m.molecules) == len(itps)
    for k, ref in enumerate(itps):
        ours_atoms, ours = _parse_itp(m.itp(k))
        ref_atoms, theirs = _parse_itp(ref)
        assert ours_atoms == ref_atoms
        assert dict(ours) == dict(theirs)
    # which atoms make which bead is the version's own: Martini 2.2 cuts a
    # tryptophan's two rings along another line and reads a phenylalanine's in
    # another order, so a bead mapped by Martini 3's rules lands in the wrong
    # place and the ring constraints it is then held by cannot be satisfied
    assert np.abs(m.positions - _reference_beads(f"{name}.martini22")).max() < 1e-3


def test_martini2_keeps_a_terminus_a_protonation_cost_martinize2(tmp_path):
    """A terminus and a residue's protonation have nothing to do with each other,
    so one going unmapped does not take the other with it.

    martinize2 gathers the modifications it found and drops the set when any of
    them has no mapping for the force field: on a protein with a protonated
    aspartate under Martini 2.2 -- which has no mapping for one -- its
    N-terminus comes out uncharged.  boonza applies what it can and says what it
    could not.
    """
    s = _load(CASES["1TEN"][0], tmp_path).select("protein").clone()
    with pytest.warns(UserWarning, match="ASP-HD2"):
        m = martinize(s, forcefield="martini22")
    first = m.molecules[0].nodes[0]
    assert first["atomname"] == "BB"
    assert first["atype"] == "Qd" and float(first["charge"]) == 1.0  # the terminus is charged
    last = [n for n in m.molecules[0].nodes if n["atomname"] == "BB"][-1]
    assert last["atype"] == "Qa" and float(last["charge"]) == -1.0


def test_martini2_keeps_a_histidine_in_the_chain(tmp_path):
    """boonza gives a residue the name the force field knows it by, so the links
    find it: without that the backbone has a gap at every histidine, which is
    what martinize2 hands back for a structure named the Amber way."""
    s = _load(CASES["2TRX"][0], tmp_path).select("protein").clone()
    his = [r for r in range(s.nresidues) if str(s.residues["name"][r]).strip() == "HIS"]
    assert his  # the case is only a case if the protein has one
    named = s.clone()
    for r in his:
        named.residue(r).name = "HSD"
    plain, charmm = (martinize(x, forcefield="martini22") for x in (s, named))
    assert [n["atype"] for n in plain.molecules[0].nodes] == \
           [n["atype"] for n in charmm.molecules[0].nodes]  # fmt: skip
    for mine, theirs in zip(plain.molecules, charmm.molecules, strict=True):
        assert len(mine.interactions["bonds"]) == len(theirs.interactions["bonds"])
        assert len(mine.interactions["angles"]) == len(theirs.interactions["angles"])
    # and the backbone really is continuous: every residue's BB bonded to the next
    mol = plain.molecules[0]
    bb = [k for k, n in enumerate(mol.nodes) if n["atomname"] == "BB"]
    joined = {frozenset(t.atoms) for kind in ("bonds", "constraints")
              for t in mol.interactions.get(kind, [])}  # fmt: skip
    assert all(frozenset((a, b)) in joined for a, b in zip(bb, bb[1:], strict=False))


def _reference_beads(name) -> np.ndarray:
    """Where martinize2 put the beads of that reference run."""
    cg = gzip.decompress((REF / name / "cg.pdb.gz").read_bytes()).decode()
    return np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                     for line in cg.splitlines() if line.startswith("ATOM")])  # fmt: skip


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
    assert np.abs(m.positions - _reference_beads(name)).max() < 1e-3  # martinize2 writes 3 decimals


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
    with pytest.raises(ValueError, match="not martini3001 protein residues: HEM"):
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
                                               "cg.gro", "cg.dms"}  # fmt: skip
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


def test_a_martini2_protein_runs_on_its_own_parameters(tmp_path):
    """Martini 2.2 as it ships, solvated, minimized and stepped.

    The toy parameters the other runs use would not catch what this does: a bead
    mapped by the wrong version's rules lands where the ring constraints it is
    held by cannot be satisfied, and the constraint solver walks the coordinates
    to NaN -- which a run reports, steps later, as a particle coordinate being
    NaN.  Martini 2's own settings are Martini 3's: reaction field at 1.1 nm.
    """
    pytest.importorskip("openmm")
    import openmm as mm
    from openmm import app

    from boonza.martini import NONBONDED_FOR, parameters

    s = boonza.load(DATA / "2TRX.pdb").select("protein and chain A").clone()
    for r in range(s.nresidues):
        want = CHARMM_NAMES.get(str(s.residues["name"][r]).strip())
        if want:
            s.residue(r).name = want
    m = solvate(martinize(s, forcefield="martini22", elastic=True), padding=6.0)
    cg = m.system(parameters(NONBONDED_FOR[2])[0])
    rings = {t for name in cg.table_names if name.startswith("constraint")
             for k in range(cg.table(name).nterms)
             for t in [tuple(int(a) for a in cg.table(name).term(k).atoms)]}  # fmt: skip
    assert rings  # the case is only a case if something is constrained
    top, system, pos = boonza.to_openmm(cg, **OPENMM_OPTIONS)
    sim = app.Simulation(top, system, mm.LangevinMiddleIntegrator(300, 1.0, 0.002),
                         mm.Platform.getPlatformByName("CPU"))  # fmt: skip
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=50)
    equilibrate(sim, steps=5)
    sim.step(5)
    state = sim.context.getState(getEnergy=True, getPositions=True)
    assert np.isfinite(state.getPotentialEnergy()._value)
    assert np.isfinite(state.getPositions(asNumpy=True)._value).all()


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


def test_a_bilayer_is_built_by_boonza_md(toy_lipids, tmp_path):
    """`boonza bilayer` built a membrane and stopped there; `boonza md
    --solvate membrane` builds the same one and runs it, so everything the
    command took has to arrive through the settings."""
    from boonza.md.config import parse_arguments
    from boonza.md.prepare import build_martini_system

    args = parse_arguments(
        ["--model", "martini3", "--solvate", "membrane", "--upper", "TLP:3,STR:1",
         "--lower", "TLP", "--size-nm", "4", "--water-nm", "1.5", "--gromacs",
         "--lipid-itp", str(toy_lipids), "--workdir", str(tmp_path / "run")]
    )  # fmt: skip
    s, _ = build_martini_system(args, tmp_path, log=lambda *_: None)
    names = set(s.residues["name"].tolist())
    assert {"TLP", "STR", "W"} <= names
    # --gromacs: the bilayer in GROMACS's form as well as in the cg.dms a run reads
    written = {p.name for p in (tmp_path / "martini").iterdir()}
    assert written == {"topol.top", "solvent.itp", "cg.gro", "cg.dms"}


def test_a_rectangular_bilayer_keeps_both_edges(toy_lipids, tmp_path):
    """--size-nm takes x and y, as the removed command's --size did."""
    from boonza.md.config import parse_arguments
    from boonza.md.prepare import build_martini_system

    args = parse_arguments(
        ["--model", "martini3", "--solvate", "membrane", "--upper", "TLP",
         "--size-nm", "5", "3.5", "--lipid-itp", str(toy_lipids),
         "--workdir", str(tmp_path / "run")]
    )  # fmt: skip
    s, _ = build_martini_system(args, tmp_path, log=lambda *_: None)
    x, y = np.diag(np.asarray(s.cell, float))[:2]
    assert (round(x, 1), round(y, 1)) == (50.0, 35.0)


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


def test_a_mae_carries_martinis_virtual_site_but_not_its_angles(tmp_path):
    """The .mae format holds a 4-point virtual site (ffio_virtuals funct lc4),
    which a tryptophan bead needs; Martini's two angle forms have no funct there,
    and saying so beats "Failed to process dms table"."""
    import warnings

    from boonza.martini import martinize

    s = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein")).system()
    assert len(s.table("virtual_lc4")) and len(s.table("angle_restricted"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boonza.save(s, tmp_path / "beads.mae")
    said = " ".join(str(w.message) for w in caught)
    assert "angle_restricted" in said and "angle_cosine_harm" in said
    assert "virtual_lc4" not in said  # carried, not dropped
    back = boonza.load(tmp_path / "beads.mae")
    assert len(back.table("virtual_lc4")) == len(s.table("virtual_lc4"))
    for prop in ("c1", "c2", "c3"):
        assert np.allclose(back.table("virtual_lc4").values(prop),
                           s.table("virtual_lc4").values(prop))  # fmt: skip


def test_a_cofactor_can_be_held_as_inert_beads():
    """A heme is no residue of Martini's, and mapping it as one bead per heavy atom
    holds the room it takes without claiming to know its chemistry.

    The point is the volume: probes must not enter space the cofactor occupies.
    Nothing else is asserted about it -- no charge, no bead but the plainest
    apolar one -- and every pair inside it is excluded, since beads 1.5 A apart
    and 3.4 A wide would otherwise fly apart on the first step.
    """
    from boonza.martini.build import COFACTOR_BEAD

    s = boonza.load(DATA / "1HHO.pdb")
    sel = "(protein or resname HEM) and chain A"
    with pytest.raises(ValueError, match="cofactors=True"):
        martinize(s, sel)  # the refusal says what to do about it
    m = martinize(s, sel, cofactors=True, elastic=True)
    mol = next(x for x in m.molecules
               if any(n.get("cofactor") for n in x.nodes))  # fmt: skip
    beads = [k for k, n in enumerate(mol.nodes) if n.get("cofactor")]
    heme = s.select("resname HEM and chain A and not element H").ids
    assert len(beads) == len(heme)  # one bead per heavy atom, iron included
    assert {mol.nodes[k]["atype"] for k in beads} == {COFACTOR_BEAD["martini3001"]}
    assert all(float(mol.nodes[k]["charge"]) == 0.0 for k in beads)
    # each bead carries its own atom's mass, the iron's included: 43 beads of the
    # default 72 would make a heme three times the weight it is
    masses = {round(float(mol.nodes[k]["mass"]), 3) for k in beads}
    assert {12.011, 14.007, 15.999, 55.845} <= masses
    # every pair inside it, counted apart from the exclusions the protein's own
    # blocks carry (Martini 3's tryptophan has a virtual site with its own)
    inside = {frozenset(t.atoms) for t in mol.interactions.get("exclusions", [])
              if set(t.atoms) <= set(beads)}  # fmt: skip
    assert len(inside) == len(beads) * (len(beads) - 1) // 2
    held = [t for t in mol.interactions.get("bonds", [])
            if t.meta.get("comment") == "cofactor held"]  # fmt: skip
    assert held  # and bands to whatever protein beads coordinate it
    assert all(len({*t.atoms} & {*beads}) == 1 for t in held)


def test_a_residue_of_the_chain_is_not_a_cofactor():
    """What the force field has no block for is not therefore a cofactor: a
    D-amino acid or a modified residue is part of the chain, and mapping it inert
    would throw away a side chain in the middle of a protein."""
    s = boonza.load(DATA / "2TRX.pdb").select("protein and chain A").clone()
    r = next(k for k in range(s.nresidues) if str(s.residues["name"][k]).strip() == "GLU")
    s.residue(r).name = "DGL"  # a D-glutamate, as a crystal structure writes one
    assert len(s.select("resname DGL").ids)  # still read as protein: it has a backbone
    with pytest.raises(ValueError, match="part of a chain rather than cofactors: DGL"):
        martinize(s, "protein or resname DGL", cofactors=True)


def test_a_cofactor_out_in_the_open_is_warned_about():
    """The inert bead is honest while the cofactor is buried, nothing being able to
    reach it and read as apolar what is really a charge or a phosphate.  One in
    solvent can be reached, so it is said rather than assumed."""
    s = boonza.load(DATA / "1HHO.pdb").select("(protein or resname HEM) and chain A").clone()
    heme = s.select("resname HEM").ids
    xyz = np.asarray(s.positions).copy()
    xyz[heme] += 40.0  # out into the solvent, away from everything
    s.positions = xyz
    with pytest.warns(UserWarning, match="not buried"):
        martinize(s, "protein or resname HEM", cofactors=True)


def test_a_cofactor_is_banded_densely_because_sparsely_it_comes_apart():
    """How many bands a cofactor needs, which measurement settled rather than
    reasoning: three points fix a rigid body, and these bands are soft.

    Banded to two neighbours each and tethered twice, a heme strays 12 A over
    5 ps -- and its own shape drifts just as far, which is the tell: a body of n
    beads wants about 3n-6 independent bands to be rigid, 123 for a heme's 43,
    where two per bead gives 56 and many are redundant along a ring.  So the
    bands are dense by default, and the knobs are there to be measured with.
    """
    from boonza.cofactors import COFACTOR_NEIGHBOURS, COFACTOR_REACH, COFACTOR_TETHERS

    assert (COFACTOR_NEIGHBOURS, COFACTOR_TETHERS) == (None, None)  # dense, both ways
    s = boonza.load(DATA / "1HHO.pdb")
    sel = "(protein or resname HEM) and chain A"

    def bands(**kw):
        m = martinize(s, sel, cofactors=True, elastic=True, **kw)
        mol = next(x for x in m.molecules if any(n.get("cofactor") for n in x.nodes))
        kinds = [t.meta.get("comment") for t in mol.interactions["bonds"]]
        return kinds.count("cofactor shape"), kinds.count("cofactor held")

    shape, held = bands()
    sparse_shape, sparse_held = bands(cofactor_neighbours=2, cofactor_tethers=2)
    assert shape > sparse_shape and held > sparse_held
    # and the reach is the lever on how firmly it is held, so it has to bite
    assert bands(cofactor_reach=6.0)[1] < held < bands(cofactor_reach=20.0)[1]
    assert COFACTOR_REACH == 12.0  # where the returns fall off; see the measurements


def test_a_coordinated_ion_of_sirahs_own_is_bonded_to_what_holds_it():
    """SIRAH maps a zinc to an ion of its own -- real parameters, better than any
    stand-in -- but an ion is one bead with no bonded terms, so a structural zinc
    comes out as a bead that diffuses away and a zinc finger comes apart.

    Where the structure shows coordination the ion is bonded to what holds it,
    keeping SIRAH's own type and charge.  An ion in solvent has nothing that
    close and stays the free ion it is.
    """
    pytest.importorskip("boonza.sirah")
    from boonza.sirah import sirahize
    from boonza.sirah.build import read_map

    assert read_map().get("ZN")  # SIRAH does know a zinc, which is the point
    s = boonza.load(DATA / "1HHO.pdb").select("protein and chain A").clone()
    out = sirahize(s, "protein")  # no ion here: nothing to bond, nothing to break
    assert all(m.natoms > 1 or m.bonds for m in out.molecules)


@pytest.mark.parametrize(("forcefield", "bead"), [("martini3001", "SD"), ("martini22", "Qd")])
def test_an_ion_martini_has_gets_that_ion_and_stays_free(forcefield, bead, tmp_path):
    """A cofactor of one atom that Martini has an ion for deserves that ion.

    The inert bead claims nothing, which is right for a heme and wasteful for a
    calcium: Martini has one, charge and all, and its bead type is in the
    parameter file the topology already includes.  So the ion is borrowed.

    It is not banded.  One bead has no shape to hold, so a band could only tether
    it to what the crystal happened to catch it near, and a calcium or a magnesium
    sitting in a site is free to leave -- which is what it does.  A zinc is the
    exception and keeps its bands: it holds a zinc finger's loops together and
    nothing else in the model says so.  SIRAH draws the same line.
    """
    from boonza.martini.build import COFACTOR_BEAD, ion_beads

    version = 2 if forcefield == "martini22" else 3
    assert ion_beads(version)["CA"][:2] == (bead, 2.0)
    assert "ZN" not in ion_beads(version)  # Martini has no zinc, in either version

    s = boonza.load(DATA / "1HHO.pdb").select("protein and chain A").clone()
    # a calcium of its own, put where a backbone bead will coordinate it
    xyz = np.asarray(s.positions)
    ca = s.select("name CA and resid 20").ids
    s.append(boonza.System.from_arrays(xyz[ca] + [2.2, 0.0, 0.0], names=["CA"], anum=[20],
                                       resnames=["CA"], resids=[900], chains=["A"]))  # fmt: skip
    m = martinize(s, "protein or resname CA", forcefield=forcefield, cofactors=True)
    node = next(n for mol in m.molecules for n in mol.nodes if n.get("cofactor"))
    assert node["atype"] == bead and float(node["charge"]) == 2.0 and node["ion"]
    assert float(node["mass"]) == pytest.approx(40.078)  # the ion's own, not a bead's
    mol = next(x for x in m.molecules if any(n.get("cofactor") for n in x.nodes))
    k = next(i for i, n in enumerate(mol.nodes) if n.get("cofactor"))
    assert not [t for t in mol.interactions["bonds"] if k in t.atoms]  # and free
    assert COFACTOR_BEAD[forcefield] != bead  # the inert bead is a different thing

    # a zinc, which Martini has no ion for, keeps the inert bead and is banded to
    # what coordinates it: that crosslink is the whole of what it is there for
    zinc = s.clone()
    where = next(i for i, n in enumerate(zinc.residues["name"]) if str(n).strip() == "CA")
    zinc.residue(where).name = "ZN"
    for a in zinc.residue(where).atoms:
        a.name, a.anum = "ZN", 30
    z = martinize(zinc, "protein or resname ZN", forcefield=forcefield, cofactors=True)
    mol = next(x for x in z.molecules if any(n.get("cofactor") for n in x.nodes))
    k = next(i for i, n in enumerate(mol.nodes) if n.get("cofactor"))
    assert not mol.nodes[k]["ion"] and float(mol.nodes[k]["charge"]) == 0.0
    assert [t for t in mol.interactions["bonds"] if k in t.atoms]  # held by what holds it


@pytest.mark.parametrize("name,parent", [("ASH", "ASP"), ("ASPP", "ASP"),
                                         ("GLH", "GLU"), ("GLUP", "GLU"),
                                         ("LYN", "LYS")])  # fmt: skip
def test_martini2_builds_a_neutral_residue_from_the_charged_one(name, parent, tmp_path):
    """One chemistry spelled two ways does one thing, and the charge is the thing.

    Martini 2.2's neutral acids and neutral lysine are residues of its own
    (ASP0, GLU0, LSN) that no mapping reaches from an all-atom structure, not in
    vermouth either, so arriving *named* ASH asked for the neutral block and died
    on its missing mapping -- while the same residue arriving named ASP with the
    proton still on it only warned.  It now builds the charged residue, whose
    parameters exist, and sets the side chain's charge to zero, which is what a
    neutral residue differs in.  martinize2 leaves it charged; this says what it
    did instead.
    """
    s = _load(CASES["2TRX"][0], tmp_path).select("protein and chain A").clone()
    where = next(r for r in range(s.nresidues)
                 if str(s.residues["name"][r]).strip() == parent)  # fmt: skip
    plain = martinize(s.clone(), forcefield="martini22")
    renamed = s.clone()
    renamed.residue(where).name = name
    with pytest.warns(UserWarning, match=f"{name} built from {parent}"):
        got = martinize(renamed, forcefield="martini22")

    def beads(m):
        return [(n["atomname"], n["atype"], float(n["charge"]))
                for n in m.molecules[0].nodes if n.get("resid") == where + 1]  # fmt: skip

    mine, theirs = beads(got), beads(plain)
    assert [(n, a) for n, a, _ in mine] == [(n, a) for n, a, _ in theirs]  # same beads
    assert abs(sum(q for _, _, q in theirs)) == 1.0  # the charged one is charged
    assert sum(q for _, _, q in mine) == 0.0  # and this one is not
    assert all(q == 0.0 for n, _, q in mine if n.startswith("SC"))


@pytest.mark.parametrize("name", ["HIP", "HSP"])
def test_martini2_has_a_charged_histidine_of_its_own(name, tmp_path):
    """Not every protonation needs standing in for: Martini 2.2 ships HSP, a
    residue with its own mapping, so a protonated histidine is built as itself
    and carries its charge.  Nothing is warned about and nothing is zeroed."""
    s = _load(CASES["2TRX"][0], tmp_path).select("protein and chain A").clone()
    where = next(r for r in range(s.nresidues)
                 if str(s.residues["name"][r]).strip() == "HIS")  # fmt: skip
    s.residue(where).name = name
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # it is not a compromise
        m = martinize(s, forcefield="martini22")
    beads = [(n["atomname"], float(n["charge"])) for n in m.molecules[0].nodes
             if n.get("resid") == where + 1]  # fmt: skip
    assert sum(q for _, q in beads) == pytest.approx(1.0)


def test_martini3_keeps_a_protonated_residue_neutral(tmp_path):
    """What Martini 2.2 cannot do, Martini 3 can: it ships the mappings for the
    neutral acids under both conventions' names, so the charged bead gives way to
    a polar one and nothing is warned about."""
    s = _load(CASES["2TRX"][0], tmp_path).select("protein and chain A").clone()
    for name, parent in (("ASH", "ASP"), ("GLH", "GLU"), ("LYN", "LYS")):
        where = next(r for r in range(s.nresidues)
                     if str(s.residues["name"][r]).strip() == parent)  # fmt: skip
        x = s.clone()
        x.residue(where).name = name
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # neutral is not a compromise here
            m = martinize(x, forcefield="martini3001")
        q = sum(float(n["charge"]) for n in m.molecules[0].nodes
                if n.get("resid") == where + 1)  # fmt: skip
        assert q == pytest.approx(0.0), f"{name} came out charged under Martini 3"


def test_sirah_has_a_residue_of_its_own_for_a_neutral_acid():
    """SIRAH maps ASH and GLH onto sDh and sEh, its own neutral residues, where
    ASP and GLU give sD and sE: the proton is kept as a plus and minus across the
    two oxygens rather than a charge of minus one over the pair.  Neutral lysine
    is the exception -- its map sends LYN to sK, the charged one."""
    from boonza.sirah.build import read_map, read_residues

    amap, (library, _) = read_map(), read_residues()
    assert (amap["ASP"].residue, amap["ASH"].residue) == ("sD", "sDh")
    assert (amap["GLU"].residue, amap["GLH"].residue) == ("sE", "sEh")
    assert amap["LYN"].residue == amap["LYS"].residue == "sK"  # SIRAH's own choice
    for charged, neutral in (("sD", "sDh"), ("sE", "sEh")):
        q = {k: sum(a[2] for a in library[k].atoms) for k in (charged, neutral)}
        assert q[charged] == pytest.approx(-1.0, abs=0.01)
        assert q[neutral] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("resname,proton", [("ASP", "HD2"), ("GLU", "HE2")])
def test_martini2_reads_a_proton_rather_than_a_name(resname, proton, tmp_path):
    """The same acid arriving named ASP with its proton on comes out neutral too.

    Spelling it ASH says the same thing about the chemistry, and the two used to
    disagree by a whole charge: ASH was zeroed, ASP-with-a-proton stayed at -1.
    The proton is what is read, by element and distance -- it is written HD2,
    HD1 or 2HD depending on who made the file, and the oxygens it sits on are
    not always OD1 and OD2 either.
    """
    s = _load(CASES["2TRX"][0], tmp_path).select("protein and chain A").clone()
    where = next(r for r in range(s.nresidues)
                 if str(s.residues["name"][r]).strip() == resname)  # fmt: skip

    def side_chain(m):
        mine = [n for n in m.molecules[0].nodes if n.get("resid") == where + 1]
        return sum(float(n["charge"]) for n in mine
                   if str(n["atomname"]).startswith("SC"))  # fmt: skip

    charged = martinize(s.clone(), forcefield="martini22")
    assert abs(side_chain(charged)) == 1.0  # with no proton on it, it is charged

    # put the acidic hydrogen on one of the side chain's oxygens, as a structure
    # prepared at low pH has it, and leave the residue named as it was
    held = s.clone()
    res = held.residue(where)
    oxygen = [a for a in res.atoms if a.element == "O" and a.name.strip() not in ("O", "OXT")][0]
    added = held.residue(where).add_atom()
    added.name, added.anum = proton, 1  # the element follows the atomic number
    added.pos = np.asarray(oxygen.pos) + np.array([0.0, 0.0, 1.0])  # 1 A away, as a bond is
    assert side_chain(martinize(held, forcefield="martini22")) == 0.0


@pytest.mark.parametrize(
    "ion,anum,bead,charge,bonded",
    [("ZN", 30, "ZnX", 1.0, True), ("CA", 20, "CaX", 2.0, False), ("MG", 12, "MgX", 2.0, False)],
)
def test_sirah_keeps_an_ion_free_unless_it_is_what_holds_a_fold(ion, anum, bead, charge, bonded):
    """SIRAH ships CaX, MgX and ZnX -- real parameters, +2, +2 and +1 -- and none
    of them carries a bonded term.

    A zinc is bonded to what coordinates it, being what holds a zinc finger's
    loops together and said nowhere else in the model; a calcium and a magnesium
    are left the free ions they are, since bonding one pins it where the crystal
    happened to catch it.  Either way no angle or dihedral runs through an ion:
    SIRAH has no type for a GC-GO-CaX, and writing one would be inventing
    geometry.  Before this, any of the three stopped the build.
    """
    pytest.importorskip("boonza.sirah")
    from boonza.sirah import sirahize

    s = boonza.load(DATA / "1HHO.pdb").select("protein and chain A").clone()
    xyz = np.asarray(s.positions)
    near = s.select("name CA and resid 20").ids
    s.append(boonza.System.from_arrays(xyz[near] + [2.2, 0.0, 0.0], names=[ion], anum=[anum],
                                       resnames=[ion], resids=[900], chains=["A"]))  # fmt: skip
    built = sirahize(s, f"protein or resname {ion}", log=None).system()  # used to raise

    ids = np.asarray(built.select(f"resname {ion} CaX MgX ZnX").ids, np.int64)
    kinds = {str(x) for x in np.asarray(
        [str(v) for v in built.table("nonbonded").values("type")])[ids]}  # fmt: skip
    assert kinds == {bead}
    assert set(np.asarray(built.atoms["charge"], float)[ids].tolist()) == {charge}

    def touching(table):
        return sum(1 for i in range(table.nterms)
                   if set(int(a) for a in table.term(i).atoms) & set(ids.tolist()))  # fmt: skip

    assert (touching(built.table("stretch_harm")) > 0) is bonded
    for kind in ("angle_harm", "dihedral_trig"):
        if kind in built.table_names:
            assert touching(built.table(kind)) == 0  # nothing derived runs through it


def test_sirah_tells_60a_from_60():
    """A residue is known by where it sits in the chain, not by its number.

    A structure numbered as a chymotrypsin is -- 60, 60A, 60B, 60C, 60D, 61 --
    has neither one residue per number nor the next one a number higher, so
    bonding by number wires a side chain to whatever else happened to be called
    60.  SIRAH then asks for a bond type between two beads that are never
    bonded (C6O-P3Cn, an aspartate's oxygen to an asparagine's side chain) and
    the build stops; where a type happens to exist it would build quietly, with
    the wrong connectivity.
    """
    pytest.importorskip("boonza.sirah")
    from boonza.sirah import sirahize

    s = boonza.load(DATA / "1HHO.pdb").select("protein and chain A").clone()
    # renumber as a chymotrypsin does: 59, 60, 60A, 60B, 61 ...
    where = [r for r in range(s.nresidues) if 60 <= int(s.residues["resid"][r]) <= 62]
    for code, r in zip(("", "A", "B"), where, strict=False):
        res = s.residue(r)
        res.resid, res.insertion = 60, code
    assert sum(1 for r in range(s.nresidues) if int(s.residues["resid"][r]) == 60) == 3

    out = sirahize(s, "protein", log=None)
    built = out.system()  # used to raise: no [ bondtypes ] entry for C6O-P3Cn
    assert built.natoms > 0

    # the three are three residues in the built system too, each under its own
    # insertion code -- a GROMACS topology has no column for one
    sixty = [r for r in range(built.nresidues) if int(built.residues["resid"][r]) == 60]
    assert [str(built.residues["insertion"][r]).strip() for r in sixty] == ["", "A", "B"]

    # and they are joined in a line, not to each other's side chains: every
    # bond from one of them to another is backbone to backbone
    of_residue = {i: int(built.atom(i).residue.id) for i in range(built.natoms)
                  if int(built.atom(i).residue.id) in sixty}  # fmt: skip
    names = {i: built.atom(i).name.strip() for i in of_residue}
    bonds = built.table("stretch_harm")
    between = [t for i in range(bonds.nterms)
               for t in [bonds.term(i)]
               if all(int(a) in of_residue for a in t.atoms)
               and len({of_residue[int(a)] for a in t.atoms}) > 1]  # fmt: skip
    assert between, "the three should be bonded along the backbone"
    for t in between:
        assert all(names[int(a)] in ("GN", "GC", "GO") for a in t.atoms)


@pytest.mark.parametrize("model", ["martini22", "martini3001", "sirah"])
def test_a_built_system_keeps_the_numbering_it_was_given(model):
    """A GROMACS topology numbers its residues from 1 in each molecule and has
    no column for an insertion code, so a chain numbered as a chymotrypsin is
    -- 60, 60A, 60B, 61 -- came back as residues all called 60, several
    backbone beads to each, and a chain broken into molecules came back
    numbered from the end of the one before it.  A viewer then draws one
    residue where there are four, and knots a trace through them.
    """
    held = boonza.load(DATA / "1HHO.pdb").select("protein").clone()
    where = [r for r in range(held.nresidues)
             if str(held.residue(r).chain.name).strip() == "A"
             and 60 <= int(held.residues["resid"][r]) <= 62]  # fmt: skip
    for code, r in zip(("", "A", "B"), where, strict=True):
        held.residue(r).resid, held.residue(r).insertion = 60, code
    want = [(str(held.residue(r).chain.name).strip(),
             int(held.residues["resid"][r]),
             str(held.residues["insertion"][r]).strip())
            for r in range(held.nresidues)]  # fmt: skip

    if model == "sirah":
        pytest.importorskip("boonza.sirah")
        from boonza.sirah import sirahize

        built = sirahize(held, "protein", log=None).system()
    else:
        built = martinize(held, forcefield=model).system()

    got = [(str(built.chains["name"][built.residues["chain"][r]]).strip(),
            int(built.residues["resid"][r]),
            str(built.residues["insertion"][r]).strip())
           for r in range(built.nresidues)]  # fmt: skip
    assert got == want
    assert len(set(got)) == len(got)  # and no two residues under one number
