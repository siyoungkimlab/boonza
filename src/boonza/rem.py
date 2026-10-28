"""REM: minimization with softened nonbonded interactions, for structures that clash.

A structure with atoms on top of each other (a ligand placed into a pocket, a
mutation, a backmapped coarse-grained model) has Lennard-Jones and Coulomb
energies that are effectively infinite, and a minimizer started there takes
huge, damaging steps.  REM minimizes first with every nonbonded pair made soft
and bounded, after mstool's REM::

    E(r) = min(rep, LJ) + coul
    rep  = A cos^p(pi r / 2 sigma)            A at r = 0, zero at r = sigma
    coul = C q_i q_j cos^p(pi r / 2 r_c)      bounded, zero at the cutoff

so beyond sigma the pair is the force field's own Lennard-Jones and within it
the repulsion never exceeds A.  Atoms can then slip past each other -- the
atoms around a chiral centre or a double bond too, so a soft minimization can
invert one.  Nothing is held unless named: :func:`hold_torsions` holds the
torsions given (impropers for chirality, torsions across double bonds) at the
angles given while the potentials are soft.  The real force field's
minimization follows (:func:`relax` leaves that to the caller).

    new, report = rem.relax(omm_system, positions_nm)
    new, report = rem.relax(omm_system, positions_nm, torsions=[(4, 0, 5, 6, -34.0)])
"""

from __future__ import annotations

import math

import numpy as np

#: mstool's defaults: the repulsion's height and the Coulomb term's scale (kJ/mol),
#: the cosine's power, and the cutoff (nm) when the system has none
A_KJ = 100.0
C_KJ = 50.0
POWER = 2
CUTOFF_NM = 1.2
#: how hard named torsions are held while the potentials are soft (kJ/mol/rad^2)
HOLD_KJ = 300.0

#: the forces boonza.omm.to_openmm makes for nonbonded terms, replaced by soft ones
_REPLACED = ("nonbonded_vdw", "pair_12_6_es", "nonbonded_rf_exclusions", "nonbonded_rf_self")
# Lennard-Jones is read no nearer than sig/2, where it is far above A (min takes the
# soft wall): two atoms at one point (backmapping makes them) then have finite forces
_SOFT = ("min(rep, lj) + coul; rep = A*cos(pi/2*r/sig)^{p}; "
         "lj = 4*eps*((sig/rr)^12 - (sig/rr)^6); rr = max(r, 0.5*sig); "
         "coul = C*qq*cos(pi/2*r/rc)^{p}")  # fmt: skip


def soften(omm_system, A: float = A_KJ, C: float = C_KJ, cutoff: float | None = None,
           power: int = POWER):  # fmt: skip
    """A copy of ``omm_system`` with its nonbonded interactions soft (the module's
    E(r)); every other force as it was.  ``A`` and ``C`` in kJ/mol; ``cutoff`` in nm
    (default the system's, else :data:`CUTOFF_NM`).  Lennard-Jones comes from the
    NonbondedForce (Lorentz-Berthelot) or, where boonza tabulates it (geometric rules,
    NBFIX, Martini), from those tables; exceptions (1-4 pairs) keep their own.  A
    reaction field's corrections go: a soft pair has no field."""
    import openmm as mm

    out = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(omm_system))
    forces = list(out.getForces())
    nbs = [f for f in forces if isinstance(f, mm.NonbondedForce)]
    if len(nbs) != 1:
        raise ValueError(f"REM needs one NonbondedForce, not {len(nbs)}")
    nb = nbs[0]
    # other CustomNonbondedForces (boonza md's repulsion between ligand copies) are
    # terms of their own, kept: they exclude what the NonbondedForce does, as this does
    tables = [f for f in forces if f.getName() == "nonbonded_vdw"]
    periodic = nb.getNonbondedMethod() not in (nb.NoCutoff, nb.CutoffNonPeriodic)
    if cutoff is None:
        cutoff = (nb.getCutoffDistance().value_in_unit_system(mm.unit.md_unit_system)
                  if nb.getNonbondedMethod() != nb.NoCutoff else CUTOFF_NM)  # fmt: skip
    n = nb.getNumParticles()
    q, sigma, eps = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        qi, si, ei = nb.getParticleParameters(i)
        q[i], sigma[i], eps[i] = (x.value_in_unit_system(mm.unit.md_unit_system)
                                  for x in (qi, si, ei))  # fmt: skip

    def common(f):
        f.addGlobalParameter("A", float(A))
        f.addGlobalParameter("C", float(C))
        f.addGlobalParameter("rc", float(cutoff))
        f.addGlobalParameter("pi", math.pi)

    # every pair the force field does not see as a plain pair: excluded here
    pairs = soft_pairs = None
    if tables:
        lj = tables[0]
        ntypes, _, acoef = lj.getTabulatedFunction(0).getFunctionParameters()
        _, _, bcoef = lj.getTabulatedFunction(1).getFunctionParameters()
        a, b = np.asarray(acoef), np.asarray(bcoef)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.where((a > 0) & (b > 0), (a / b) ** (1 / 6), 0.1)
            ep = np.where((a > 0) & (b > 0), b * b / (4 * a), 0.0)
        pairs = mm.CustomNonbondedForce(_SOFT.format(p=power) + "; qq = q1*q2; "
                                        "sig = sigt(t1, t2); eps = epst(t1, t2)")  # fmt: skip
        pairs.addTabulatedFunction("sigt", mm.Discrete2DFunction(ntypes, ntypes, sig.tolist()))
        pairs.addTabulatedFunction("epst", mm.Discrete2DFunction(ntypes, ntypes, ep.tolist()))
        pairs.addPerParticleParameter("t")
        pairs.addPerParticleParameter("q")
        for i in range(n):
            pairs.addParticle([lj.getParticleParameters(i)[0], float(q[i])])
    else:
        pairs = mm.CustomNonbondedForce(_SOFT.format(p=power) + "; qq = q1*q2; "
                                        "sig = 0.5*(s1 + s2); eps = sqrt(e1*e2)")  # fmt: skip
        for name in ("q", "s", "e"):
            pairs.addPerParticleParameter(name)
        for i in range(n):
            pairs.addParticle([float(q[i]), float(sigma[i]), float(eps[i])])
    common(pairs)
    pairs.setNonbondedMethod(pairs.CutoffPeriodic if periodic else pairs.CutoffNonPeriodic)
    pairs.setCutoffDistance(cutoff)
    pairs.setUseLongRangeCorrection(False)
    pairs.setName("rem_pairs")

    # the exceptions: 1-4 pairs (and others with terms) soft as well, the rest excluded
    soft_pairs = mm.CustomBondForce(_SOFT.format(p=power))
    for name in ("qq", "sig", "eps"):
        soft_pairs.addPerBondParameter(name)
    common(soft_pairs)
    soft_pairs.setUsesPeriodicBoundaryConditions(periodic)
    soft_pairs.setName("rem_exceptions")
    for k in range(nb.getNumExceptions()):
        i, j, qq, s, e = nb.getExceptionParameters(k)
        pairs.addExclusion(i, j)
        qq, s, e = (x.value_in_unit_system(mm.unit.md_unit_system) for x in (qq, s, e))
        if qq or e:
            soft_pairs.addBond(i, j, [float(qq), float(s), float(e)])
    for f in forces:
        if f.getName() == "pair_12_6_es":  # boonza's pairs the NonbondedForce cannot hold
            for k in range(f.getNumBonds()):
                i, j, (a, b, qq) = f.getBondParameters(k)
                s, e = ((a / b) ** (1 / 6), b * b / (4 * a)) if a > 0 and b > 0 else (0.1, 0.0)
                soft_pairs.addBond(i, j, [float(qq), float(s), float(e)])
    for k in reversed(range(out.getNumForces())):  # wrappers differ: by type and name
        f = out.getForce(k)
        if isinstance(f, mm.NonbondedForce) or f.getName() in _REPLACED:
            out.removeForce(k)
    group = min(out.getNumForces(), 31)
    for f in (pairs, soft_pairs):
        f.setForceGroup(group)
        if f is pairs or f.getNumBonds():
            out.addForce(f)
    return out


def _dihedral(pos, a, b, c, d) -> float:
    """The a-b-c-d torsion in radians, as OpenMM measures it."""
    b0, b1, b2 = pos[b] - pos[a], pos[c] - pos[b], pos[d] - pos[c]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1))
    return float(-math.atan2(np.dot(m, n2), np.dot(n1, n2)))


def hold_torsions(torsions, positions_nm, k: float = HOLD_KJ):
    """A force holding each of ``torsions`` -- ``(a, b, c, d)`` at its angle in
    ``positions_nm``, or ``(a, b, c, d, degrees)`` at that angle -- harmonic in the
    angle, ``k`` kJ/mol/rad^2."""
    import openmm as mm

    pos = np.asarray(positions_nm, float)
    f = mm.CustomTorsionForce("0.5*k_hold*d^2; d = min(dt, 2*pi - dt); "
                              f"dt = abs(theta - theta0); pi = {math.pi!r}")  # fmt: skip
    f.addGlobalParameter("k_hold", float(k))
    f.addPerTorsionParameter("theta0")
    for t in torsions:
        a, b, c, d = (int(x) for x in t[:4])
        target = math.radians(float(t[4])) if len(t) > 4 else _dihedral(pos, a, b, c, d)
        f.addTorsion(a, b, c, d, [target])
    f.setName("rem_torsions")
    return f


def _apart(xyz: np.ndarray, within: float = 1e-4, by: float = 0.01) -> np.ndarray:
    """``xyz`` (nm) with atoms at one point (backmapping makes them) moved ``by`` apart
    in a fixed random direction: a pair force at r = 0 has no direction."""
    from scipy.spatial import cKDTree

    pairs = cKDTree(xyz).query_pairs(within, output_type="ndarray")
    if not len(pairs):
        return xyz
    out = xyz.copy()
    step = np.random.default_rng(0).normal(size=(len(pairs), 3))
    out[pairs[:, 1]] += by * step / np.linalg.norm(step, axis=1)[:, None]
    return out


def relax(omm_system, positions_nm, *, torsions=(), hold_kJ: float = HOLD_KJ,
          A: float = A_KJ, C: float = C_KJ, cutoff: float | None = None,
          tolerance: float = 10.0, max_iterations: int = 0, platform=None,
          reporter=None) -> tuple[np.ndarray, dict]:  # fmt: skip
    """Minimize with soft nonbonded interactions (:func:`soften`), the ``torsions``
    named held by ``hold_kJ`` (:func:`hold_torsions`; none by default; turning a
    centre into its mirror image takes more than the default); returns the new positions
    (nm, particles of ``omm_system``) and the soft energies before and after
    (kJ/mol).  The force field's own minimization is the caller's next step.
    ``reporter``: an openmm.MinimizationReporter to follow it."""
    import openmm as mm
    import openmm.unit as u

    soft = soften(omm_system, A=A, C=C, cutoff=cutoff)
    xyz = _apart(np.asarray(positions_nm, float).reshape(-1, 3))
    if len(torsions):
        soft.addForce(hold_torsions(torsions, xyz, hold_kJ))
    integrator = mm.VerletIntegrator(0.001)
    context = (mm.Context(soft, integrator, platform) if platform is not None
               else mm.Context(soft, integrator))  # fmt: skip
    context.setPositions(xyz * u.nanometer)
    context.computeVirtualSites()

    def energy():
        e = context.getState(getEnergy=True).getPotentialEnergy()
        return round(e.value_in_unit(u.kilojoule_per_mole), 1)

    before = energy()
    if reporter is not None:
        mm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations, reporter)
    else:
        mm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
    after = energy()
    state = context.getState(getPositions=True)
    new = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    return new, {"rem_energy_before": before, "rem_energy_after": after}
