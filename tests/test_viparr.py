"""viparr force fields: reading, matching and parameterization.

Synthetic force fields test the rules (template matching across residues,
exact before wildcard parameters, multi-term dihedrals, virtual sites,
first-match priority, merging).  With viparr-ffpublic (``$VIPARR_FFPATH``,
default ~/viparr-ffpublic/ff) the public force fields are checked against
OpenMM's own Amber and CHARMM force fields; with a viparr build
(``$BOONZA_VIPARR``: a command that runs the viparr CLI) whole systems are
compared term by term with viparr's output.
"""

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza import viparr

# ---- a small force field written on the fly ---------------------------------------


def _write_ff(d: Path, rules=None, templates=None, **tables) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rules = rules or {"vdw_func": "LJ12_6_sig_epsilon", "vdw_comb_rule": "ARITHMETIC/GEOMETRIC",
                      "exclusions": 4, "es_scale": [0.0, 0.0, 0.5], "lj_scale": [0.0, 0.0, 0.5],
                      "plugins": ["bonds", "angles", "propers", "vdw1", "exclusions",
                                  "mass"]}  # fmt: skip
    (d / "rules").write_text(json.dumps(rules))
    (d / "templates").write_text(json.dumps(templates or {}))
    for name, rows in tables.items():
        (d / name).write_text(json.dumps(rows))
    return d


def _rows(spec, keys):
    return [{"type": t, "params": dict(zip(keys, v, strict=True))} for t, *v in spec]


HALF = {  # CH3-CH2- : half a butane, bonded to the other half through $1
    "atoms": [["C1", 6, -0.3, ["CT3"]], ["H11", 1, 0.1, ["HC"]], ["H12", 1, 0.1, ["HC"]],
              ["H13", 1, 0.1, ["HC"]], ["C2", 6, -0.2, ["CT2"]], ["H21", 1, 0.1, ["HC"]],
              ["H22", 1, 0.1, ["HC"]]],
    "bonds": [["C1", "H11"], ["C1", "H12"], ["C1", "H13"], ["C1", "C2"], ["C2", "H21"],
              ["C2", "H22"], ["C2", "$1"]],
}  # fmt: skip
WAT = {"atoms": [["O", 8, -0.8, ["OW"]], ["H1", 1, 0.4, ["HW"]], ["H2", 1, 0.4, ["HW"]]],
       "bonds": [["O", "H1"], ["O", "H2"]]}  # fmt: skip
TYPES = ["CT3", "CT2", "HC", "OW", "HW", "Vs"]


def _tables(**over):
    t = {
        "stretch_harm": _rows([("CT3 CT2", 1.53, 222.0), ("CT2 CT2", 1.53, 222.5),
                               ("CT3 HC", 1.11, 322.0), ("CT2 HC", 1.11, 309.0),
                               ("OW HW", 0.9572, 450.0)], ["r0", "fc"]),
        "angle_harm": _rows([("HC CT3 HC", 108.4, 35.5), ("HC CT3 CT2", 110.1, 34.6),
                             ("CT3 CT2 HC", 110.1, 34.6), ("CT3 CT2 CT2", 114.0, 58.35),
                             ("HC CT2 HC", 109.0, 35.5), ("HC CT2 CT2", 110.1, 26.5),
                             ("HW OW HW", 104.52, 55.0)], ["theta0", "fc"]),
        "dihedral_trig": _rows([("CT3 CT2 CT2 CT3", 0.0, 0.0, 0.5, 0.0, 0.0),
                                ("CT3 CT2 CT2 CT3", 0.0, 0.0, 0.0, 0.0, 0.3),
                                ("* CT3 CT2 *", 0.0, 0.0, 0.0, 0.0, 0.16),
                                ("* CT2 CT2 *", 0.0, 0.0, 0.0, 0.0, 0.19)],
                               ["phi0", "fc0", "fc1", "fc2", "fc3"]),
        "vdw1": _rows([(t, 3.0 + k / 10, 0.1 + k / 100) for k, t in enumerate(TYPES)],
                      ["sigma", "epsilon"]),
        "mass": _rows([(t, m) for t, m in zip(TYPES, [12.011, 12.011, 1.008, 15.999, 1.008, 0.0],
                                              strict=True)], ["amu"]),
    }  # fmt: skip
    t.update(over)
    return t


def _butane_water():
    s = boonza.System("test")
    ch = s.add_chain(name="A")
    xyz = iter(np.random.default_rng(0).normal(0, 3, (17, 3)))
    ids = {}
    for resid, atoms in ((1, ["C1", "H11", "H12", "H13", "C2", "H21", "H22"]),
                         (2, ["H33", "C4", "H31", "H32", "C3", "H41", "H42"])):  # fmt: skip
        r = s.add_residue(ch, name="BUT", resid=resid)
        for nm in atoms:
            ids[(resid, nm)] = s.add_atom(
                r, name=nm, anum=1 if nm[0] == "H" else 6, pos=next(xyz)
            ).id
    for a, b in [
        ("C1", "H11"),
        ("C1", "H12"),
        ("C1", "H13"),
        ("C1", "C2"),
        ("C2", "H21"),
        ("C2", "H22"),
    ]:
        s.add_bond(ids[(1, a)], ids[(1, b)])
    for a, b in [
        ("C3", "C4"),
        ("C4", "H41"),
        ("C4", "H42"),
        ("C4", "H33"),
        ("C3", "H31"),
        ("C3", "H32"),
    ]:
        s.add_bond(ids[(2, a)], ids[(2, b)])
    s.add_bond(ids[(1, "C2")], ids[(2, "C3")])
    r = s.add_residue(ch, name="HOH", resid=3)
    o = s.add_atom(r, name="OW", anum=8, pos=next(xyz)).id
    for nm in ("HW1", "HW2"):
        s.add_bond(o, s.add_atom(r, name=nm, anum=1, pos=next(xyz)).id)
    return s, ids


def _ff(tmp_path, name="ff", templates=None, rules=None, **over):
    tpl = {"BUT": HALF, "WAT": WAT} if templates is None else templates
    return viparr.load_forcefield(_write_ff(tmp_path / name, rules, tpl, **_tables(**over)))


def _terms(s, table):
    t = s.table(table)
    return [
        (tuple(a), t.params.row(p))
        for a, p in zip(t.atoms.tolist(), t.param_ids.tolist(), strict=True)
    ]


def test_templates_match_by_graph_across_residues(tmp_path):
    s, ids = _butane_water()
    p = viparr.parameterize(s, [_ff(tmp_path)])
    assert p.natoms == s.natoms
    # charges come from the templates, whatever the atoms are called
    assert p.atoms["charge"][ids[(2, "C4")]] == -0.3 and p.atoms["charge"][ids[(2, "C3")]] == -0.2
    assert len(p.table("stretch_harm")) == s.nbonds
    assert len(p.table("angle_harm")) == 6 * 4 + 1  # four neighbors at each carbon, and water
    # 27 butane dihedrals, the central C-C-C-C one with the two rows of its exact type
    dih = _terms(p, "dihedral_trig")
    assert len(dih) == 28
    c1, c2, c3, c4 = (ids[k] for k in ((1, "C1"), (1, "C2"), (2, "C3"), (2, "C4")))
    central = [r for a, r in dih if set(a) == {c1, c2, c3, c4}]
    assert sorted((r["fc1"], r["fc3"]) for r in central) == [(0.0, 0.3), (0.5, 0.0)]
    # other dihedrals fall back to the first wildcard row that fits
    types = {tuple(a): r["type"] for a, r in dih}
    h11, h31 = ids[(1, "H11")], ids[(2, "H31")]
    assert types[next(a for a in types if a[0] == h11 and c3 in a)] == "* CT3 CT2 *"
    assert "* CT2 CT2 *" in {types[a] for a in types if h31 in a and ids[(1, "H21")] in a}


def test_exclusions_pairs_and_masses(tmp_path):
    s, ids = _butane_water()
    p = viparr.parameterize(s, [_ff(tmp_path)])
    excl = {tuple(a) for a in p.table("exclusion").atoms.tolist()}
    assert len(excl) == 13 + 24 + 27 + 3  # butane 1-2, 1-3, 1-4 and water
    pairs = _terms(p, "pair_12_6_es")
    assert len(pairs) == 27  # only 1-4 pairs are scaled (0.5), not excluded
    q = p.atoms["charge"]
    c1, c4 = ids[(1, "C1")], ids[(2, "C4")]
    (row,) = [r for a, r in pairs if set(a) == {c1, c4}]
    assert row["qij"] == pytest.approx(0.5 * q[c1] * q[c4])
    sig, eps = 3.0, 0.1  # CT3 with CT3
    assert row["aij"] == pytest.approx(0.5 * 4 * eps * sig**12)
    assert set(p.atoms["mass"][p.atoms["anum"] == 6].tolist()) == {12.011}
    assert (
        p.nonbonded_info.vdw_funct == "vdw_12_6"
        and p.nonbonded_info.vdw_rule == "arithmetic/geometric"
    )


def test_first_force_field_wins(tmp_path):
    s, _ = _butane_water()
    wat2 = {
        **WAT,
        "atoms": [["O", 8, -0.834, ["OW"]], ["H1", 1, 0.417, ["HW"]], ["H2", 1, 0.417, ["HW"]]],
    }
    a = _ff(tmp_path, "a")
    b = _ff(tmp_path, "b", templates={"TIP": wat2})
    with pytest.warns(viparr.ViparrWarning, match="first match takes precedence"):
        p = viparr.parameterize(s, [a, b])
    assert p.atoms["charge"][-3] == -0.8
    p = viparr.parameterize(s, [b, a])  # butane only matches a, water now takes b's charges
    assert p.atoms["charge"][-3] == -0.834


def test_one_force_field_with_two_matching_templates_is_an_error(tmp_path):
    s, _ = _butane_water()
    ff = _ff(tmp_path, templates={"BUT": HALF, "WAT": WAT, "WAT2": WAT})
    with pytest.raises(viparr.ViparrError, match="Multiple templates"):
        viparr.parameterize(s, [ff])


def test_unmatched_molecules_and_missing_parameters(tmp_path):
    s, _ = _butane_water()
    with pytest.raises(viparr.ViparrError, match="No force field could parameterize fragment 1"):
        viparr.parameterize(s, [_ff(tmp_path, templates={"BUT": HALF})])
    missing = _tables()["stretch_harm"][:-1]  # no O-H bond
    ff = _ff(tmp_path, "m", stretch_harm=missing)
    with pytest.raises(viparr.ViparrError, match="No match found for table 'stretch_harm'"):
        viparr.parameterize(s, [ff])
    with pytest.warns(viparr.ViparrWarning, match="stretch_harm"):
        p = viparr.parameterize(s, [ff], fatal=False)
    assert len(p.table("stretch_harm")) == s.nbonds - 2


def test_merge_replaces_templates_and_types(tmp_path):
    base = _ff(tmp_path, "base")
    wat2 = {
        **WAT,
        "atoms": [["O", 8, -1.0, ["OW"]], ["H1", 1, 0.5, ["HW"]], ["H2", 1, 0.5, ["HW"]]],
    }
    rows = _rows([("OW HW", 1.0, 500.0), ("OW OW", 2.0, 1.0)], ["r0", "fc"])
    patch = _write_ff(tmp_path / "patch", {}, {"WAT": wat2}, stretch_harm=rows)
    (patch / "rules").unlink()
    merged = viparr.merge_forcefields(base, patch)
    assert [t.charge for t in merged.templates if t.name == "WAT"] == [[-1.0, 0.5, 0.5]]
    types = [r.type for r in merged.params["stretch_harm"]]
    assert types[0] == "OW OW" and types.count("OW HW") == 1  # new types first, then base order
    assert next(r for r in merged.params["stretch_harm"] if r.type == "OW HW").params["r0"] == 1.0
    assert len(base.params["stretch_harm"]) == 5  # the base force field is unchanged
    with pytest.raises(viparr.ViparrError, match="append-only"):
        viparr.merge_forcefields(base, patch, append_only=True)


def test_virtual_sites(tmp_path):
    tip4 = {**WAT, "pseudos": [["M", -1.0, ["Vs"], "virtual_lc3", "O", "H1", "H2", "pset0"]],
            "atoms": [["O", 8, 0.0, ["OW"]], ["H1", 1, 0.5, ["HW"]],
                      ["H2", 1, 0.5, ["HW"]]]}  # fmt: skip
    rules = {"vdw_func": "lj12_6_sig_epsilon", "vdw_comb_rule": "arithmetic/geometric",
             "plugins": ["bonds", "angles", "vdw1", "exclusions", "mass", "virtuals"]}  # fmt: skip
    lc3 = [{"type": "OW HW HW pset0", "params": {"c1": 0.1, "c2": 0.1}}]
    ff = viparr.load_forcefield(_write_ff(tmp_path / "t4", rules, {"TIP4": tip4},
                                          **_tables(), virtuals_lc3=lc3))  # fmt: skip
    w = boonza.System("w")
    r = w.add_residue(w.add_chain(), name="HOH")
    o = w.add_atom(r, name="O", anum=8, pos=(0.0, 0.0, 0.0)).id
    for x in (1.0, -1.0):
        w.add_bond(o, w.add_atom(r, name="H", anum=1, pos=(x, 0.5, 0.0)).id)
    p = viparr.parameterize(w, [ff])
    assert p.natoms == 4 and p.atoms["anum"][3] == 0 and p.atoms["name"][3] == "M"
    assert p.atoms["charge"].tolist() == [0.0, 0.5, 0.5, -1.0]
    assert p.table("virtual_lc3").atoms.tolist() == [[3, 0, 1, 2]]
    assert len(p.table("exclusion")) == 6  # every pair of the four particles
    assert viparr.parameterize(w, [ff], reorder_ids=True).atoms["anum"].tolist() == [8, 0, 1, 1]


def test_mirrored_cmap_grid():
    axis = np.arange(-180.0, 180.0, 15.0)
    phi, psi = np.repeat(axis, 24), np.tile(axis, 24)
    grid = np.column_stack([phi, psi, phi + 1000 * psi])
    m = viparr._mirror_cmap(grid)
    wrap = (-phi + 180) % 360 - 180
    wrap2 = (-psi + 180) % 360 - 180
    assert np.array_equal(m[:, 2], wrap + 1000 * wrap2)
    assert np.array_equal(m[:, :2], grid[:, :2])


def test_command_line(tmp_path, capsys):
    from boonza.cli import main

    s, _ = _butane_water()
    boonza.save(s, tmp_path / "in.dms")
    ff = _write_ff(tmp_path / "ffs" / "mine", None, {"BUT": HALF, "WAT": WAT}, **_tables())
    out = tmp_path / "out.dms"
    assert main(["parameterize", str(tmp_path / "in.dms"), str(out), "-f", "mine",
                 "--ffpath", str(ff.parent)]) == 0  # fmt: skip
    assert "dihedral_trig 28" in capsys.readouterr().out
    assert not boonza.diff(boonza.load(out), viparr.parameterize(s, [str(ff)]))


# ---- the public force fields -------------------------------------------------------------

FFPATH = [
    Path(p).expanduser() for p in os.environ.get("VIPARR_FFPATH", "~/viparr-ffpublic/ff").split(":")
]
DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _public(name):
    for d in FFPATH:
        if (d / name).is_dir():
            return viparr.load_forcefield(d / name)
    pytest.skip(f"viparr force field {name} not found (set VIPARR_FFPATH)")


def test_d_residues_get_the_mirrored_cmap():
    ff = _public("aa.charmm.c36m")
    pep = boonza.peptide("AVLSKEF", conformation="helix")
    L = pep.clone(np.argsort(pep.atoms["residue"], kind="stable"))
    D = L.copy()
    D.positions = L.positions * [-1.0, 1.0, 1.0]  # the mirror image: all residues D
    e = {(name, chiral): boonza.openmm_energies(viparr.parameterize(s, [ff], cmap_chirality=chiral))
         for name, s in (("L", L), ("D", D)) for chiral in (True, False)}  # fmt: skip
    cmap = {k: v["torsiontorsion_cmap"] for k, v in e.items()}
    assert cmap[("L", True)] == cmap[("L", False)]
    assert cmap[("D", True)] == pytest.approx(cmap[("L", True)], abs=1e-6)  # mirror symmetry
    assert abs(cmap[("D", False)] - cmap[("L", True)]) > 1.0  # what viparr does
    assert e[("D", True)]["total"] == pytest.approx(e[("L", True)]["total"], abs=1e-4)


def _openmm_protein(xml):
    from openmm import app

    app_data = Path(app.__file__).parent / "data"
    if not (app_data / xml).exists():
        pytest.skip(f"OpenMM has no {xml}")
    pdb = app.PDBFile(str(DATA / "1TEN.pdb"))
    model = app.Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    model.delete([r for r in model.topology.residues()
                  if not {"N", "CA", "C"} <= {a.name for a in r.atoms()}])  # fmt: skip
    ff = app.ForceField(xml)
    model.addHydrogens(ff)
    omm = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                          removeCMMotion=False)  # fmt: skip
    return boonza.from_openmm(model.topology, omm, model.positions)


def _dihedrals(e):
    return e.get("dihedral_trig", 0.0) + e.get("dihedral_trig_constant", 0.0)


def test_ff14sb_matches_openmm_amber14():
    ff = _public("aa.amber.ff14SB")
    ff.rules.es_scale = [0.0, 0.0, 1 / 1.2]  # the file rounds 1/1.2 to 0.8333
    ref = _openmm_protein("amber14-all.xml")
    ours = viparr.parameterize(ref, [ff])
    a, b = boonza.openmm_energies(ref), boonza.openmm_energies(ours)
    for term in ("stretch_harm", "angle_harm"):
        assert b[term] == pytest.approx(a[term], abs=1e-6)
    # impropers on equivalent atoms (Arg NH2, Asn ND2) are listed in another order
    assert _dihedrals(b) == pytest.approx(_dihedrals(a), abs=0.1)
    assert b["nonbonded"] == pytest.approx(a["nonbonded"], abs=0.01)


def test_c36m_matches_openmm_charmm36_2024():
    ff = _public("aa.charmm.c36m")
    ref = _openmm_protein("charmm36_2024.xml")
    ours = viparr.parameterize(ref, [ff])
    a, b = boonza.openmm_energies(ref), boonza.openmm_energies(ours)
    for term in ("stretch_harm", "angle_harm", "torsiontorsion_cmap", "nonbonded_vdw"):
        assert b[term] == pytest.approx(a[term], abs=1e-4)
    assert _dihedrals(b) == pytest.approx(_dihedrals(a), abs=1e-4)
    assert b["improper_harm"] == pytest.approx(a["improper_harm"], abs=0.05)


# ---- viparr itself -------------------------------------------------------------------------

VIPARR = os.environ.get("BOONZA_VIPARR")
VIPARR_TESTS = Path(os.environ.get("BOONZA_VIPARR_TESTS", "~/viparr/test/dms")).expanduser()
AMBER_REST = ["water.tip3p", "ions.amber1jc.tip3p"]
CASES = {
    "c36m": ("ww_solv.dms", ["-f", "aa.charmm.c36m", "-f", "water.tip3p_charmm",
                             "-f", "ions.charmm36"]),
    "ff14sb": ("ww.dms", ["-f", "aa.amber.ff14SB", *sum((["-f", f] for f in AMBER_REST), [])]),
    "ff19sb": ("ww.dms", ["-f", "aa.amber.ff19SB", *sum((["-f", f] for f in AMBER_REST), [])]),
    "des_amber": ("ww.dms", ["-f", "aa.DES-Amber", *sum((["-f", f] for f in AMBER_REST), [])]),
    "merge": ("ww.dms", ["-f", "aa.amber.ff99SB", "-m", "aa.amber.ff99SB-ILDN",
                         *sum((["-f", f] for f in AMBER_REST), [])]),
    "priority": ("ww.dms", ["-f", "aa.amber.ff14SB", "-f", "aa.amber.ff99SB",
                            *sum((["-f", f] for f in AMBER_REST), [])]),
    "lipid_nbfix": ("POPS_ions.dms", ["-f", "lipid.charmm.c36", "-f", "ions.charmm36"]),
    "tip4pew": ("tip4p.dms", ["-f", "water.tip4pew"]),
    "tip5p": ("tip5p.dms", ["-f", "water.tip5p"]),
}  # fmt: skip


def _ours(inp, args):
    ffs = []
    for flag, name in zip(args[::2], args[1::2], strict=True):
        if flag == "-f":
            ffs.append(_public(name))
        else:
            ffs[-1] = viparr.merge_forcefields(ffs[-1], _public(name))
    return viparr.parameterize(boonza.load(inp), ffs, cmap_chirality=False)


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_viparr(tmp_path, case):
    if not VIPARR or not shutil.which(VIPARR.split()[0]):
        pytest.skip("viparr not available; set BOONZA_VIPARR to a command running its CLI")
    name, args = CASES[case]
    inp = VIPARR_TESTS / name
    if not inp.exists():
        pytest.skip(f"{inp} not found (set BOONZA_VIPARR_TESTS)")
    out = tmp_path / "viparr.dms"
    env = {**os.environ, "VIPARR_FFPATH": ":".join(map(str, FFPATH))}
    subprocess.run([*VIPARR.split(), str(inp), str(out), *args, "--without-constraints"],
                   check=True, capture_output=True, env=env)  # fmt: skip
    with pytest.warns() if case == "priority" else _nothing():
        ours = _ours(inp, args)
    assert not boonza.diff(ours, boonza.load(out))
    assert math.isclose(
        ours.atoms["charge"].sum(), boonza.load(out).atoms["charge"].sum(), abs_tol=1e-9
    )


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
