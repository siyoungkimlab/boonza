"""Dipeptide probes and the Martini side of ``boonza swim``."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.martini.probes import LETTERS, PROBE_RESIDUES, probe, probe_charge, probe_sequences
from boonza.md.cgswim import build, groups_of, place, prepare
from boonza.md.config import parse_arguments

DATA = Path(__file__).parent / "data"


def test_the_probe_library():
    """Every pair of the 14 residues, once; the small and the doubled-up ones
    (Ala, Gly, Val, Asp, Asn) and the reactive ones (Cys) are left out."""
    seqs = probe_sequences()
    assert len(seqs) == 105 == 14 * 15 // 2
    assert len(set(seqs)) == len(seqs)
    assert not {"A", "G", "V", "D", "N", "C", "U"} & set("".join(seqs))
    assert set("".join(seqs)) == {LETTERS[r] for r in PROBE_RESIDUES}
    assert "RE" in seqs and "ER" not in seqs  # XY and YX are one probe


@pytest.mark.parametrize(("sequence", "charge"), [
    ("EK", 0.0), ("RR", 2.0), ("EE", -2.0), ("HH", 0.0), ("WY", 0.0), ("RE", 0.0),
])  # fmt: skip
def test_probes_carry_the_charges_of_ph_7(sequence, charge):
    """Arg and Lys +1, Glu -1, His neutral: the ends are capped, so only the
    side chains count (boonza's own peptides are built neutral, which is why
    the probe is mapped from the heavy atoms)."""
    assert probe_charge(sequence) == charge
    ends = [n for n in probe(sequence).molecules[0].nodes if n["atomname"] == "BB"]
    assert [(n["atype"], n["charge"]) for n in ends] == [("P6", 0.0)] * 2


def test_probes_are_free_to_bend():
    """No secondary structure, no side-chain corrections, no elastic network:
    a probe is a small molecule, not a piece of folded protein."""
    m = probe("WY")
    mol = m.molecules[0]
    assert not any("cgsecstruct" in n for n in mol.nodes)
    assert not [t for t in mol.interactions["dihedrals"]
                if t.meta.get("group", "").startswith("SC-BB-BB-SC")]  # fmt: skip
    assert not [t for t in mol.interactions["bonds"] if t.meta.get("group") == "Rubber band"]
    assert {n["resname"] for n in mol.nodes} == {"WY"}  # its own name, not TRP/TYR
    assert m.names == ["WY"]


def test_probes_are_built_once():
    assert probe("EK") is not probe("EK")  # a copy the caller may move
    assert probe("EK").positions == pytest.approx(probe("EK").positions)


def test_probes_are_dealt_round_robin():
    groups = groups_of(probe_sequences(), 10)
    assert len(groups) == 10  # 105 probes, about 10 a simulation
    assert sorted(s for g in groups for s in g) == sorted(probe_sequences())
    assert {len(g) for g in groups} <= {10, 11}
    # every simulation holds a spread of side chains, not RR RQ RE ... together
    assert min(len(set("".join(g))) for g in groups) >= 8


def test_probes_are_placed_clear_of_everything():
    from boonza.spatial import min_dist2

    box = np.full(3, 60.0)
    protein = np.array([[30.0, 30.0, 30.0]])
    probes = [probe("EK"), probe("LL")]
    rng = np.random.default_rng(0)
    placed = place(protein, probes, 3, box, rng, clearance=5.0)
    assert [len(p) for p in placed] == [3 * probes[0].nbeads, 3 * probes[1].nbeads]
    molecules = [chunk for p, m in zip(placed, probes, strict=True)
                 for chunk in np.split(p, 3)]  # every copy of every probe  # fmt: skip
    for k, mine in enumerate(molecules):
        others = np.vstack([protein, *molecules[:k], *molecules[k + 1 :]])
        assert (min_dist2(mine, others, 5.0, cell=np.diag(box)) >= 25.0 - 1e-6).all()


def test_a_prepared_simulation(tmp_path):
    """One directory per group: the built topology, settings `boonza md` accepts,
    and the probes written down for the analysis."""
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--workdir", str(tmp_path / "swim"), "--production-ns", "10",
                            "--seed", "1"])  # fmt: skip
    sims = prepare(args, ["EK", "LL", "RR"], types=2, copies=2, log=lambda *_: None)
    assert len(sims) == 2  # 3 probes, about 2 a simulation
    d = sims[0]
    assert (d / "probes.json").is_file() and (d / "martini" / "topol.top").is_file()
    import json

    written = json.loads((d / "probes.json").read_text())
    assert written["copies"] == 2 and written["align"] == "name BB"
    top = (d / "martini" / "topol.top").read_text()
    for name in written["probes"]:
        assert f"{name} 2" in top  # two copies of each probe
        assert f'#include "{name}.itp"' in top
    settings = parse_arguments(["--config", str(d / "md.toml")])
    assert settings.model == "martini3" and settings.solvate == "none"
    assert settings.repulsion_selection == "resname " + " ".join(written["probes"])
    assert settings.production_ns == 10.0
    assert not settings.forcefields  # nothing all-atom came along
    assert Path(settings.input_structure) == (d / "martini" / "topol.top").resolve()
    assert (tmp_path / "swim" / "assignment.csv").is_file()
    assert (tmp_path / "swim" / "simulations.txt").read_text().count("boonza md") == 2


def test_the_built_system_is_neutral_and_runs(tmp_path):
    """The protein, its probes and water in one box, with the ions the charges ask for."""
    from boonza.martini import OPENMM_OPTIONS, martinize

    pytest.importorskip("openmm")
    protein = martinize(boonza.load(DATA / "1TEN.pdb"), elastic=True)
    probes = [probe("RR"), probe("EE")]
    box = np.full(3, float(np.ptp(protein.positions, axis=0).max()) + 20.0)
    m = build(protein, probes, 2, box, np.random.default_rng(0), salt=0.15)
    counts = dict(zip(m.names, m.molecule_copies, strict=True))
    assert counts["RR"] == counts["EE"] == 2
    charge = sum(c * sum(float(n["charge"]) for n in mol.nodes)
                 for mol, c in zip(m.molecules, m.molecule_copies, strict=True))  # fmt: skip
    (_, _), (_, na), (_, cl) = m.solvent
    assert charge + na - cl == pytest.approx(0)
    s = m.system()
    assert s.natoms == m.nbeads
    assert {"RR", "EE"} <= {str(n) for n in s.residues["name"]}
    assert np.isfinite(boonza.openmm_energies(s, **OPENMM_OPTIONS)["total"])


def test_the_analysis_knows_a_coarse_grained_run(tmp_path):
    """`boonza sites` aligns on BB beads and takes the probes from the run."""
    from boonza.cli import DEFAULT_ALIGN, _cg_selections, _coarse_grained
    from boonza.symmetry import DEFAULT_LIGAND

    aa = boonza.load(DATA / "1TEN.pdb")
    assert not _coarse_grained(aa)
    from boonza.martini import martinize

    cg = martinize(aa).system()
    assert _coarse_grained(cg)
    (tmp_path / "probes.json").write_text('{"probes": ["EK", "LL"]}')
    run = tmp_path / "md"
    run.mkdir()

    class Args:
        alignsel = DEFAULT_ALIGN
        ligandsel = DEFAULT_LIGAND

    args = Args()
    _cg_selections(args, cg, [str(run)])  # probes.json sits beside the run
    assert args.alignsel == "name BB"
    assert args.ligandsel == "resname EK LL"
    unchanged = Args()
    _cg_selections(unchanged, aa, [str(run)])  # all-atom: left alone
    assert unchanged.alignsel == DEFAULT_ALIGN and unchanged.ligandsel == DEFAULT_LIGAND


def test_swim_refuses_a_ligand_library_in_martini(tmp_path):
    from boonza.md.swim import main

    code = main([str(DATA / "1TEN.pdb"), "--model", "martini3", "--ligands", "library.sdf",
                 "--workdir", str(tmp_path / "swim")])  # fmt: skip
    assert code == 1
