"""A hybrid-topology OpenMM system for a relative free energy: ligand A to B.

Two systems that differ only in one ligand -- the environment (protein, water,
ions, cofactors) atom for atom the same -- are joined into one, after perses
and openfe's ``HybridTopologyFactory``.  The atoms the mapping pairs (the core)
are one particle whose parameters move from A's to B's; A's other atoms
(unique old) fade to dummies and B's (unique new) appear from them, at their
own positions.  Particles are A's system in order, then B's unique atoms.

Valence terms: those among core atoms (and the environment) move from A's to
B's with lambda_bonds, _angles, _torsions; those touching a unique atom stay
on throughout, so a dummy keeps its geometry and adds a constant (cancelling
between the complex and solvent legs) to the free energy.

Nonbonded: a NonbondedForce carries all electrostatics (charges move through
parameter offsets) and the environment's Lennard-Jones; a CustomNonbondedForce
carries every Lennard-Jones pair touching the ligand, with Gapsys' softcore
for the unique atoms.  1-4 pairs touching a unique atom move to a
CustomBondForce, softened alike.  A's and B's unique atoms never meet.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

ONE_4PI_EPS0 = 138.935456  # kJ mol^-1 nm e^-2
#: Gapsys' softcore (openfe's default)
SOFTCORE_ALPHA = 0.85

_GAPSYS = (
    "U_sterics = select(step(r - r_LJ), 4*epsilon*x*(x-1.0), U_sterics_quad);"
    "U_sterics_quad = Force*(((r - r_LJ)^2)/2 - (r - r_LJ)) + U_sterics_cut;"
    "U_sterics_cut = 4*epsilon*((sigma/r_LJ)^6)*(((sigma/r_LJ)^6) - 1.0);"
    "Force = -4*epsilon*((-12*sigma^12)/(r_LJ^13) + (6*sigma^6)/(r_LJ^7));"
    "x = (sigma/r)^6;"
    "r_LJ = softcore_alpha*((26/7)*(sigma^6)*lambda_soft)^(1/6);"
)
_PAIRS = (
    "U_sterics;" + _GAPSYS
    + "lambda_soft = new_interaction*(1.0 - lambda_sterics_insert)"
    " + old_interaction*lambda_sterics_delete;"
    "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
    "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
    "lambda_sterics = core_interaction*lambda_sterics_core"
    " + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
    "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);"
    "new_interaction = max(unique_new1, unique_new2);"
    "old_interaction = max(unique_old1, unique_old2);"
    "epsilonA = sqrt(epsilonA1*epsilonA2); epsilonB = sqrt(epsilonB1*epsilonB2);"
    "sigmaA = 0.5*(sigmaA1 + sigmaA2); sigmaB = 0.5*(sigmaB1 + sigmaB2);"
)  # fmt: skip
_EXCEPTIONS = (
    "U_electrostatics + U_sterics;" + _GAPSYS
    + "lambda_soft = new_interaction*(1.0 - lambda_sterics_insert)"
    " + old_interaction*lambda_sterics_delete;"
    "U_electrostatics = (lambda_electrostatics_insert*unique_new"
    " + unique_old*(1 - lambda_electrostatics_delete))*ONE_4PI_EPS0*chargeProd/r;"
    f"ONE_4PI_EPS0 = {ONE_4PI_EPS0!r};"
    "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
    "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
    "lambda_sterics = new_interaction*lambda_sterics_insert"
    " + old_interaction*lambda_sterics_delete;"
    "new_interaction = delta(1-unique_new); old_interaction = delta(1-unique_old);"
)  # fmt: skip

#: forces a system may hold: the ones boonza.omm.to_openmm makes for an Amber
#: all-atom system, and what boonza md adds
_KNOWN = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce",
          "CMAPTorsionForce", "MonteCarloBarostat", "CMMotionRemover",
          "CustomExternalForce"}  # fmt: skip


def _md(x) -> float:
    """A float in OpenMM's units (nm, kJ/mol, e, rad), quantity or not."""
    return float(x.value_in_unit_system(_unit().md_unit_system)) if hasattr(x, "unit") else float(x)


def _unit():
    import openmm.unit as u

    return u


@dataclass
class Hybrid:
    """The hybrid system and its bookkeeping."""

    system: object  # openmm.System
    positions: np.ndarray  # nm, particles of system
    old_to_hybrid: np.ndarray
    new_to_hybrid: np.ndarray
    core: list[int]  # hybrid indices
    unique_old: list[int]
    unique_new: list[int]
    environment: list[int]
    lambda_parameters: tuple = field(default_factory=tuple)

    def set_lambda(self, context, x: float) -> None:
        """Put master lambda ``x`` into ``context`` (one made of this system)."""
        from .lambdas import values

        for name, value in values(x).items():
            context.setParameter(name, value)


def build(old_system, new_system, old_positions, new_positions, core: dict[int, int],
          environment: dict[int, int],
          softcore_alpha: float = SOFTCORE_ALPHA) -> Hybrid:  # fmt: skip
    """The hybrid of ``old_system`` (ligand A's) and ``new_system`` (B's).

    ``core`` maps A's ligand atoms to B's (the atom mapping); ``environment``
    maps every other atom of A's system to B's (identical in parameters).
    Positions in nm.  The old system's unmapped atoms are A's unique atoms,
    the new system's B's.
    """
    return _Builder(old_system, new_system, old_positions, new_positions, core, environment,
                    softcore_alpha).build()  # fmt: skip


class _Builder:
    def __init__(self, old, new, old_pos, new_pos, core, env, alpha):
        import openmm

        self.mm = openmm
        self.old, self.new, self.alpha = old, new, float(alpha)
        n_old, n_new = old.getNumParticles(), new.getNumParticles()
        self.core = {int(a): int(b) for a, b in core.items()}
        self.env = {int(a): int(b) for a, b in env.items()}
        if set(self.core) & set(self.env) or set(self.core.values()) & set(self.env.values()):
            raise ValueError("an atom is in both the core and the environment")
        unique_old = sorted(set(range(n_old)) - set(self.core) - set(self.env))
        mapped_new = set(self.core.values()) | set(self.env.values())
        unique_new = sorted(set(range(n_new)) - mapped_new)
        self.old_to_hybrid = np.arange(n_old)
        self.new_to_hybrid = np.full(n_new, -1, dtype=np.int64)
        for a, b in {**self.core, **self.env}.items():
            self.new_to_hybrid[b] = a
        for k, b in enumerate(unique_new):
            self.new_to_hybrid[b] = n_old + k
        self.n = n_old + len(unique_new)
        self.hybrid_to_old = {h: h for h in range(n_old)}
        self.hybrid_to_new = {int(h): b for b, h in enumerate(self.new_to_hybrid)}
        self.cls = np.zeros(self.n, dtype=np.int8)  # 0 env, 1 core, 2 unique old, 3 unique new
        self.cls[list(self.core)] = 1
        self.cls[unique_old] = 2
        self.cls[n_old:] = 3
        self.core_h = sorted(self.core)
        self.uo_h = unique_old
        self.un_h = list(range(n_old, self.n))
        self.env_h = sorted(self.env)
        pos = np.zeros((self.n, 3))
        pos[:n_old] = np.asarray(old_pos, float).reshape(-1, 3)
        pos[n_old:] = np.asarray(new_pos, float).reshape(-1, 3)[unique_new]
        self.positions = pos
        self.forces_old = self._forces(old, "A")
        self.forces_new = self._forces(new, "B")

    def _forces(self, system, which):
        out = {}
        for f in system.getForces():
            kind = type(f).__name__
            if kind not in _KNOWN:
                raise NotImplementedError(
                    f"ligand {which}'s system has a {kind} ({f.getName()!r}): a hybrid "
                    "cannot take it yet")  # fmt: skip
            if kind == "CustomExternalForce" and f.getEnergyFunction().strip() != "c":
                raise NotImplementedError(f"ligand {which}'s system has a CustomExternalForce "
                                          f"{f.getName()!r} that is not a constant")  # fmt: skip
            if kind in out and kind not in ("PeriodicTorsionForce", "CustomExternalForce"):
                raise NotImplementedError(f"ligand {which}'s system has two {kind}s")
            out.setdefault(kind, []).append(f)
        if "NonbondedForce" not in out:
            raise ValueError(f"ligand {which}'s system has no NonbondedForce")
        return out

    def is_(self, h, kind) -> bool:
        return self.cls[h] == {"env": 0, "core": 1, "uo": 2, "un": 3}[kind]

    # --- particles and constraints -------------------------------------------------

    def build(self) -> Hybrid:
        mm = self.mm
        h = mm.System()
        self.h = h
        for i in range(self.n):
            if self.cls[i] == 3:
                m = _md(self.new.getParticleMass(self.hybrid_to_new[i]))
            elif self.cls[i] == 1:  # the mean of the two (openfe)
                m = 0.5 * (_md(self.old.getParticleMass(i))
                           + _md(self.new.getParticleMass(self.core[i])))  # fmt: skip
            else:
                m = _md(self.old.getParticleMass(i))
            h.addParticle(m)
        h.setDefaultPeriodicBoxVectors(*self.old.getDefaultPeriodicBoxVectors())
        self._constraints()
        for kind in ("MonteCarloBarostat", "CMMotionRemover"):
            for f in self.forces_old.get(kind, []):
                h.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f)))
        self._bonds()
        self._angles()
        self._torsions()
        self._cmap()
        self._constants()
        self._nonbonded()
        from .lambdas import PARAMETERS

        return Hybrid(system=h, positions=self.positions, old_to_hybrid=self.old_to_hybrid,
                      new_to_hybrid=self.new_to_hybrid, core=self.core_h, unique_old=self.uo_h,
                      unique_new=self.un_h, environment=self.env_h,
                      lambda_parameters=PARAMETERS)  # fmt: skip

    def _constraints(self):
        seen = {}
        for system, to_h in ((self.old, self.old_to_hybrid), (self.new, self.new_to_hybrid)):
            for k in range(system.getNumConstraints()):
                i, j, d = system.getConstraintParameters(k)
                a, b = sorted((int(to_h[i]), int(to_h[j])))
                d = _md(d)
                if (a, b) in seen:
                    if abs(seen[a, b] - d) > 1e-6:
                        raise ValueError(f"atoms {a} and {b} are constrained to {seen[a, b]:.4f} "
                                         f"nm in A and {d:.4f} nm in B: leave one of them out "
                                         "of the mapping")  # fmt: skip
                    continue
                seen[a, b] = d
                self.h.addConstraint(a, b, d)

    # --- valence terms --------------------------------------------------------------

    def _changing(self, atoms) -> bool:
        """A term among core atoms (any environment ones too) moves from A to B."""
        c = self.cls[list(atoms)]
        return bool((c == 1).any() and not (c >= 2).any())

    def _bonds(self):
        mm = self.mm
        std = mm.HarmonicBondForce()
        custom = mm.CustomBondForce(
            "0.5*K*(r-length)^2; K = (1-lambda_bonds)*K1 + lambda_bonds*K2;"
            "length = (1-lambda_bonds)*length1 + lambda_bonds*length2")  # fmt: skip
        custom.addGlobalParameter("lambda_bonds", 0.0)
        for p in ("length1", "K1", "length2", "K2"):
            custom.addPerBondParameter(p)
        old = self._terms("HarmonicBondForce", self.old, self.old_to_hybrid, 2)
        new = self._terms("HarmonicBondForce", self.new, self.new_to_hybrid, 2)
        self._valence(std.addBond, custom.addBond, old, new, lambda p: (p[0], p[1]),
                      lambda r, k: (r, 0.0))  # fmt: skip
        self._add(std, "hybrid_bonds")
        self._add(custom, "hybrid_core_bonds")

    def _angles(self):
        mm = self.mm
        std = mm.HarmonicAngleForce()
        custom = mm.CustomAngleForce("0.5*K*(theta-angle)^2; K = (1-lambda_angles)*K1"
                                     " + lambda_angles*K2; angle = (1-lambda_angles)*angle1"
                                     " + lambda_angles*angle2")  # fmt: skip
        custom.addGlobalParameter("lambda_angles", 0.0)
        for p in ("angle1", "K1", "angle2", "K2"):
            custom.addPerAngleParameter(p)
        old = self._terms("HarmonicAngleForce", self.old, self.old_to_hybrid, 3)
        new = self._terms("HarmonicAngleForce", self.new, self.new_to_hybrid, 3)
        self._valence(std.addAngle, custom.addAngle, old, new, lambda p: (p[0], p[1]),
                      lambda a, k: (a, 0.0))  # fmt: skip
        self._add(std, "hybrid_angles")
        self._add(custom, "hybrid_core_angles")

    def _terms(self, kind, system, to_h, n):
        """{sorted hybrid atoms: (atoms in order, [params])} of a force kind."""
        out = {}
        forces = (self.forces_old if system is self.old else self.forces_new).get(kind, [])
        for f in forces:
            what = {2: "Bond", 3: "Angle"}[n]
            get, count = getattr(f, f"get{what}Parameters"), getattr(f, f"getNum{what}s")()
            for k in range(count):
                *atoms, p0, p1 = get(k)
                atoms = tuple(int(to_h[a]) for a in atoms)
                key = atoms if atoms[0] < atoms[-1] else atoms[::-1]
                if key in out:
                    raise NotImplementedError(f"two {kind} terms for atoms {key}")
                out[key] = (atoms, (_md(p0), _md(p1)))
        return out

    def _valence(self, add_std, add_custom, old, new, params, absent):
        for key, (atoms, p) in old.items():
            if self._changing(atoms):
                q = new.get(key, (None, absent(*p)))[1]
                add_custom(*atoms, [p[0], p[1], q[0], q[1]])
            elif not (self.cls[list(atoms)] == 3).any():
                add_std(*atoms, *params(p))
        for key, (atoms, p) in new.items():
            if self._changing(atoms):
                if key not in old:
                    q = absent(*p)
                    add_custom(*atoms, [q[0], q[1], p[0], p[1]])
            elif (self.cls[list(atoms)] == 3).any():
                add_std(*atoms, *params(p))
            elif not (self.cls[list(atoms)] == 2).any() and key not in old:
                raise ValueError(f"ligand B's system has a term for atoms {atoms} that A's lacks "
                                 "although none of them changes")  # fmt: skip

    def _torsions(self):
        mm = self.mm
        std = mm.PeriodicTorsionForce()
        expr = "K*(1+cos(periodicity*theta-phase))"
        old_core = mm.CustomTorsionForce(f"(1-lambda_torsions)*{expr}")
        new_core = mm.CustomTorsionForce(f"lambda_torsions*{expr}")
        for f in (old_core, new_core):
            f.addGlobalParameter("lambda_torsions", 0.0)
            for p in ("periodicity", "phase", "K"):
                f.addPerTorsionParameter(p)
        for system, to_h, core_force, mine in (
            (self.old, self.old_to_hybrid, old_core, 2),
            (self.new, self.new_to_hybrid, new_core, 3),
        ):
            forces = (self.forces_old if system is self.old else self.forces_new).get(
                "PeriodicTorsionForce", [])  # fmt: skip
            for f in forces:
                for k in range(f.getNumTorsions()):
                    *atoms, n, phase, K = f.getTorsionParameters(k)
                    atoms = [int(to_h[a]) for a in atoms]
                    c = self.cls[atoms]
                    if self._changing(atoms):
                        core_force.addTorsion(*atoms, [float(n), _md(phase), _md(K)])
                    elif (c == mine).any() or (mine == 2 and not (c >= 2).any()):
                        # unique atoms' torsions stay on; the environment's from A
                        std.addTorsion(*atoms, int(n), _md(phase), _md(K))
        self._add(std, "hybrid_torsions")
        self._add(old_core, "hybrid_core_torsions_A")
        self._add(new_core, "hybrid_core_torsions_B")

    def _cmap(self):
        mm = self.mm
        fs = self.forces_old.get("CMAPTorsionForce", [])
        if not fs:
            return
        f = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(fs[0]))
        for k in range(f.getNumTorsions()):
            _, *atoms = f.getTorsionParameters(k)
            if (self.cls[atoms] != 0).any():
                raise NotImplementedError("a CMAP term touches the ligand")
        self.h.addForce(f)

    def _constants(self):
        """boonza's constant energies (cos terms written as constants): A's fade to B's."""
        mm = self.mm
        totals = []
        for forces in (self.forces_old, self.forces_new):
            total = 0.0
            for f in forces.get("CustomExternalForce", []):  # its total in global c
                names = [f.getGlobalParameterName(k) for k in range(f.getNumGlobalParameters())]
                total += f.getGlobalParameterDefaultValue(names.index("c")) * f.getNumParticles()
            totals.append(total)
        if not any(totals):
            return
        f = mm.CustomExternalForce("(1-lambda_torsions)*cA + lambda_torsions*cB")
        f.addGlobalParameter("lambda_torsions", 0.0)
        f.addPerParticleParameter("cA")
        f.addPerParticleParameter("cB")
        f.addParticle(0, [totals[0], totals[1]])
        self._add(f, "hybrid_constant")

    def _add(self, force, name):
        names = ("getNumBonds", "getNumAngles", "getNumTorsions", "getNumParticles")
        counts = [getattr(force, m)() for m in names if hasattr(force, m)]
        if counts and not any(counts):
            return
        force.setName(name)
        self.h.addForce(force)

    # --- nonbonded -------------------------------------------------------------------

    def _nonbonded(self):
        mm = self.mm
        old_nb = self.forces_old["NonbondedForce"][0]
        new_nb = self.forces_new["NonbondedForce"][0]
        if old_nb.getUseDispersionCorrection() or new_nb.getUseDispersionCorrection():
            # the ligand's Lennard-Jones leaves the NonbondedForce: its long-range
            # correction would no longer count it (openfe turns it off too)
            raise ValueError("build the systems without a dispersion correction: a hybrid "
                             "cannot carry it")  # fmt: skip
        nb = mm.NonbondedForce()
        method = old_nb.getNonbondedMethod()
        nb.setNonbondedMethod(method)
        nb.setCutoffDistance(old_nb.getCutoffDistance())
        nb.setEwaldErrorTolerance(old_nb.getEwaldErrorTolerance())
        nb.setPMEParameters(*old_nb.getPMEParameters())
        nb.setReactionFieldDielectric(old_nb.getReactionFieldDielectric())
        nb.setUseDispersionCorrection(False)
        nb.setUseSwitchingFunction(old_nb.getUseSwitchingFunction())
        nb.setSwitchingDistance(old_nb.getSwitchingDistance())
        for p in ("lambda_electrostatics_core", "lambda_sterics_core",
                  "lambda_electrostatics_delete", "lambda_electrostatics_insert"):  # fmt: skip
            nb.addGlobalParameter(p, 0.0)
        lj = mm.CustomNonbondedForce(_PAIRS)
        lj.addGlobalParameter("softcore_alpha", self.alpha)
        for p in ("lambda_sterics_core", "lambda_electrostatics_core", "lambda_sterics_insert",
                  "lambda_sterics_delete"):  # fmt: skip
            lj.addGlobalParameter(p, 0.0)
        for p in ("sigmaA", "epsilonA", "sigmaB", "epsilonB", "unique_old", "unique_new"):
            lj.addPerParticleParameter(p)
        if method == nb.NoCutoff:
            lj.setNonbondedMethod(lj.NoCutoff)
        elif method == nb.CutoffNonPeriodic:
            lj.setNonbondedMethod(lj.CutoffNonPeriodic)
        else:
            lj.setNonbondedMethod(lj.CutoffPeriodic)
        lj.setCutoffDistance(old_nb.getCutoffDistance())
        lj.setUseSwitchingFunction(old_nb.getUseSwitchingFunction())
        lj.setSwitchingDistance(old_nb.getSwitchingDistance())
        lj.setUseLongRangeCorrection(False)

        def params(force, i):
            q, s, e = force.getParticleParameters(i)
            return _md(q), _md(s), _md(e)

        for i in range(self.n):
            c = self.cls[i]
            if c == 0:
                q, s, e = params(old_nb, i)
                nb.addParticle(q, s, e)
                lj.addParticle([s, e, s, e, 0, 0])
            elif c == 1:
                q, s, e = params(old_nb, i)
                q2, s2, e2 = params(new_nb, self.core[i])
                nb.addParticle(q, s, 0.0)
                nb.addParticleParameterOffset("lambda_electrostatics_core", i, q2 - q, 0, 0)
                lj.addParticle([s, e, s2, e2, 0, 0])
            elif c == 2:
                q, s, e = params(old_nb, i)
                nb.addParticle(q, s, 0.0)
                nb.addParticleParameterOffset("lambda_electrostatics_delete", i, -q, 0, 0)
                lj.addParticle([s, e, s, 0.0, 1, 0])
            else:
                q, s, e = params(new_nb, self.hybrid_to_new[i])
                nb.addParticle(0.0, s, 0.0)
                nb.addParticleParameterOffset("lambda_electrostatics_insert", i, q, 0, 0)
                lj.addParticle([s, 0.0, s, e, 0, 1])
        self.nb, self.lj = nb, lj
        self.excluded = set()
        self._exceptions(old_nb, new_nb)
        for g1, g2 in ((self.uo_h, self.core_h), (self.uo_h, self.env_h),
                       (self.un_h, self.core_h), (self.un_h, self.env_h),
                       (self.core_h, self.env_h), (self.core_h, self.core_h),
                       (self.un_h, self.un_h), (self.uo_h, self.uo_h)):  # fmt: skip
            if g1 and g2:
                lj.addInteractionGroup(g1, g2)
        self.h.addForce(nb)
        nb.setName("hybrid_nonbonded")
        lj.setName("hybrid_sterics")
        self.h.addForce(lj)
        if self.pairs.getNumBonds():
            self.pairs.setName("hybrid_exceptions")
            self.h.addForce(self.pairs)

    def _exclude(self, i, j, q, s, e):
        key = (min(i, j), max(i, j))
        if key in self.excluded:
            raise ValueError(f"atoms {key} get two exceptions")
        self.excluded.add(key)
        self.lj.addExclusion(i, j)
        return self.nb.addException(i, j, q, s, e)

    def _exceptions(self, old_nb, new_nb):
        mm = self.mm
        pairs = mm.CustomBondForce(_EXCEPTIONS)
        pairs.addGlobalParameter("softcore_alpha", self.alpha)
        for p in ("lambda_electrostatics_insert", "lambda_electrostatics_delete",
                  "lambda_sterics_insert", "lambda_sterics_delete"):  # fmt: skip
            pairs.addGlobalParameter(p, 0.0)
        for p in ("chargeProd", "sigmaA", "epsilonA", "sigmaB", "epsilonB", "unique_old",
                  "unique_new"):  # fmt: skip
            pairs.addPerBondParameter(p)
        self.pairs = pairs

        def exceptions(force, to_h):
            out = {}
            for k in range(force.getNumExceptions()):
                i, j, q, s, e = force.getExceptionParameters(k)
                a, b = int(to_h[i]), int(to_h[j])
                out[min(a, b), max(a, b)] = (_md(q), _md(s), _md(e))
            return out

        old_ex = exceptions(old_nb, self.old_to_hybrid)
        new_ex = exceptions(new_nb, self.new_to_hybrid)

        def mixed(force, i, j):
            qi, si, ei = (_md(x) for x in force.getParticleParameters(i))
            qj, sj, ej = (_md(x) for x in force.getParticleParameters(j))
            return qi * qj, 0.5 * (si + sj), float(np.sqrt(ei * ej))

        for (a, b), (q, s, e) in old_ex.items():
            c = {int(self.cls[a]), int(self.cls[b])}
            if c == {0}:
                self._exclude(a, b, q, s, e)
            elif 2 in c:  # off in the NonbondedForce; softened in the pair force
                self._exclude(a, b, 0.0, s, 0.0)
                if q or e:
                    pairs.addBond(a, b, [q, s, e, s, 0.0, 1, 0])
            else:  # core (with core or environment): A's to B's
                na, nb_ = self.hybrid_to_new[a], self.hybrid_to_new[b]
                q2, s2, e2 = new_ex.get((min(a, b), max(a, b))) or mixed(new_nb, na, nb_)
                k = self._exclude(a, b, q, s, e)
                self.nb.addExceptionParameterOffset("lambda_electrostatics_core", k, q2 - q, 0, 0)
                self.nb.addExceptionParameterOffset("lambda_sterics_core", k, 0, s2 - s, e2 - e)
        for (a, b), (q, s, e) in new_ex.items():
            c = {int(self.cls[a]), int(self.cls[b])}
            if 3 in c:
                self._exclude(a, b, 0.0, s, 0.0)
                if q or e:
                    pairs.addBond(a, b, [q, s, 0.0, s, e, 0, 1])
            elif (min(a, b), max(a, b)) in old_ex:
                continue
            elif c == {0}:
                raise ValueError(f"atoms {a} and {b} of the environment are an exception in B "
                                 "but not in A: the environments must be the same")  # fmt: skip
            else:  # a core pair B excepts and A does not
                q1, s1, e1 = mixed(old_nb, a, b)
                k = self._exclude(a, b, q1, s1, e1)
                self.nb.addExceptionParameterOffset("lambda_electrostatics_core", k, q - q1, 0, 0)
                self.nb.addExceptionParameterOffset("lambda_sterics_core", k, 0, s - s1, e - e1)
        # A's and B's own atoms never meet
        for a, b in itertools.product(self.uo_h, self.un_h):
            self._exclude(a, b, 0.0, 1.0, 0.0)
