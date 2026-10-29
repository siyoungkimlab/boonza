"""boonza fep's long-range dispersion correction: an end state's coefficient
(boonza.fep.prep.dispersion_coefficient), against a sum over pairs and OpenMM."""

import math

import numpy as np
import pytest

import boonza
from boonza.fep.prep import dispersion_coefficient
from boonza.io.gromacs import load_top

mm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

# two Lennard-Jones types (sigma nm, epsilon kJ/mol), Lorentz-Berthelot, with one
# pair override, so both the combining rule and the overrides are exercised
TOP = """[ defaults ]
1 2
[ atomtypes ]
A 12.0 0.0 A 0.34 0.36
B 16.0 0.0 A 0.30 0.65
[ nonbond_params ]
A B 1 0.33 0.5
[ moleculetype ]
X 1
[ atoms ]
1 A 1 X A 1 0.0
[ moleculetype ]
Y 1
[ atoms ]
1 B 1 Y B 1 0.0
[ system ]
two kinds
[ molecules ]
X {na}
Y {nb}
"""


def _fluid(tmp_path, na, nb, edge=4.0):
    rng = np.random.default_rng(0)
    rows = [
        f"{k + 1:5d}{'X' if k < na else 'Y':<5s}{'A' if k < na else 'B':>5s}{k + 1:5d}"
        + "".join(f"{v:8.3f}" for v in rng.uniform(0, edge, 3))
        for k in range(na + nb)
    ]
    (tmp_path / "t.top").write_text(TOP.format(na=na, nb=nb))
    (tmp_path / "t.gro").write_text(f"t\n{na + nb}\n" + "\n".join(rows)
                                    + f"\n{edge:10.5f}{edge:10.5f}{edge:10.5f}\n")  # fmt: skip
    return load_top(tmp_path / "t.top", tmp_path / "t.gro")


def _brute(s, rc):
    """4π over every pair i < j of A/(9 rc^9) - B/(3 rc^3), one pair at a time (Å, kcal)."""
    nb = s.tables["nonbonded"]
    ids = np.asarray(nb.param_ids)
    ps, pe = np.asarray(nb.params["sigma"]), np.asarray(nb.params["epsilon"])
    over = {(p, q): v for (p, q), v in nb.overrides.items()}
    total = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p, q = sorted((int(ids[i]), int(ids[j])))
            if (p, q) in over:
                sig, eps = over[(p, q)]["sigma"], over[(p, q)]["epsilon"]
            else:
                sig, eps = 0.5 * (ps[p] + ps[q]), math.sqrt(pe[p] * pe[q])
            total += 4 * eps * sig**12 / (9 * rc**9) - 4 * eps * sig**6 / (3 * rc**3)
    return 4 * math.pi * total


def test_against_every_pair(tmp_path):
    s = _fluid(tmp_path, 7, 5)
    assert dispersion_coefficient(s, 10.0) == pytest.approx(_brute(s, 10.0), rel=1e-12)


def test_against_openmm(tmp_path):
    na, nb = 40, 25
    s = _fluid(tmp_path, na, nb)
    _, system, pos = boonza.to_openmm(s, nonbonded_method="CutoffPeriodic", cutoff=10.0,
                                      dispersion_correction=True)  # fmt: skip

    def energy(on):
        for f in system.getForces():
            if isinstance(f, mm.CustomNonbondedForce):
                f.setUseLongRangeCorrection(on)
        ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))  # fmt: skip
        ctx.setPositions(pos)
        return (
            ctx.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalorie_per_mole)
        )

    volume = 40.0**3  # Å^3
    openmm_k = (energy(True) - energy(False)) * volume
    # OpenMM counts an atom with itself among the pairs: the same to within 1/N
    assert dispersion_coefficient(s, 10.0) == pytest.approx(openmm_k, rel=2.0 / (na + nb))


def test_a_vacuum_or_empty_system_has_none(tmp_path):
    s = _fluid(tmp_path, 1, 0)
    assert dispersion_coefficient(s, 10.0) == 0.0  # one atom: no pair at all


def _amber_tools() -> bool:
    from boonza.gaff import AmberToolsError, find_amber_tools

    try:
        find_amber_tools()
    except AmberToolsError:
        return False
    return True


@pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")
def test_a_real_leg_against_openmm(tmp_path):
    """benzene -> toluene in TIP3P with ions, built as prep builds it: dK / V is what
    OpenMM's own correction says on the two end states, which (unlike the hybrid) can
    carry one -- the units, the sign, the combining rule and every table checked."""
    from pathlib import Path

    from rdkit import Chem

    from boonza import gaff, viparr
    from boonza.fep import network
    from boonza.fep import prep as fep_prep
    from boonza.fep.cli import FEP_DEFAULTS
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields

    data = Path(__file__).parent / "data" / "fep" / "benzene_modifications.sdf"
    sdf = tmp_path / "two.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for m in Chem.SDMolSupplier(str(data), removeHs=False):
            if m.GetProp("_Name") in ("benzene", "toluene"):
                w.write(m)
    args = parse_arguments(["--workdir", str(tmp_path / "w")], build_parser("boonza fep prep"))
    for k, v in FEP_DEFAULTS.items():
        setattr(args, k, v)
    mols, ligs, _ = fep_prep.load_ligands(sdf)
    _, ff = forcefields(args)
    both = ligs[0].copy()
    both.append(ligs[1])
    patch = gaff.gaff2_patch(both, ff, charge_geometry="reference")
    host = gaff.host_index(ff)
    ff[host] = viparr.merge_forcefields(ff[host], patch)
    e = network.edges(mols)[0]
    leg = fep_prep.build_leg(None, ligs[e.a], ligs[e.b], e.mapping, ff, args, 1.0,
                             log=lambda *_: None)  # fmt: skip

    def tail(system, s):
        """OpenMM's dispersion correction of an end state (kcal/mol): its energy
        with the correction on less with it off, everything else the same."""

        def energy(on):
            for f in system.getForces():
                if isinstance(f, mm.NonbondedForce):
                    f.setUseDispersionCorrection(on)
                elif isinstance(f, mm.CustomNonbondedForce):
                    f.setUseLongRangeCorrection(on)
            ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                             mm.Platform.getPlatformByName("Reference"))  # fmt: skip
            ctx.setPeriodicBoxVectors(*(np.asarray(s.cell) / 10.0 * unit.nanometer))
            ctx.setPositions(np.asarray(s.positions) / 10.0 * unit.nanometer)
            return (
                ctx.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilocalorie_per_mole)
            )

        return energy(True) - energy(False)

    volume = abs(np.linalg.det(np.asarray(leg.old.cell, float)))  # Å^3, both states' box
    openmm_change = tail(leg.new_system, leg.new) - tail(leg.old_system, leg.old)
    assert leg.dispersion_dK < 0  # a methyl's more Lennard-Jones: more tail
    assert leg.dispersion_dK / volume == pytest.approx(openmm_change, rel=0.01)
