"""OpenMM bridge: DMS-style force-field tables to an OpenMM System and back.

    topology, omm_system, positions = boonza.to_openmm(system)
    system = boonza.from_openmm(topology, omm_system, positions)
    boonza.openmm_energies(system)          # kcal/mol per force

Units: boonza uses Å, kcal/mol, degrees and e; OpenMM uses nm, kJ/mol and
radians.  Functional forms follow the DMS specification:

* ``stretch_harm``  fc (r - r0)^2
* ``angle_harm``    fc (theta - theta0)^2, the second atom at the vertex
* ``dihedral_trig`` fc0 + sum_n fc_n cos(n phi - phi0)
* ``improper_harm`` fc (phi - phi0)^2
* ``pair_12_6_es``  a/r^12 - b/r^6 + q/r, added to the nonbonded interaction
  (for excluded pairs this becomes an OpenMM exception)
* ``nonbonded`` ``vdw_12_6`` with ``arithmetic/geometric`` or ``geometric``
  combining and pair overrides (NBFIX), ``exclusion``
* constraints (``constraint_ahN``, ``constraint_hoh``), virtual sites
  (``virtual_lc2``, ``virtual_lc3``, ``virtual_out3``, and ``virtual_fdat3``
  for CHARMM lone pairs, angles in degrees), ``posre_harm``,
  ``torsiontorsion_cmap``
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .system import System

KCAL = 4.184  # kJ per kcal
NM = 0.1  # nm per Å
_ONE_4PI_EPS0 = 138.935456  # kJ nm / (mol e^2)
_KNOWN = {
    "stretch_harm", "angle_harm", "dihedral_trig", "improper_harm", "pair_12_6_es",
    "nonbonded", "exclusion", "constraint_hoh", "virtual_lc2", "virtual_lc3", "virtual_out3",
    "virtual_midpoint", "virtual_fdat3", "posre_harm", "torsiontorsion_cmap",
    *[f"constraint_ah{n}" for n in range(1, 9)],
}  # fmt: skip


def _omm():
    try:
        import openmm
        from openmm import app, unit
    except ImportError as e:
        raise ImportError("the OpenMM bridge needs openmm (conda-forge package openmm)") from e
    return openmm, app, unit


# ---------------------------------------------------------------------------
# boonza -> OpenMM


def to_openmm(system, nonbonded_method: str = "NoCutoff", cutoff: float = 9.0,
              constraints: bool = True, dispersion_correction: bool = True,
              ewald_tolerance: float = 5e-4):  # fmt: skip
    """(Topology, System, positions) for OpenMM; cutoff in Å.

    ``nonbonded_method`` is an OpenMM NonbondedForce method name (NoCutoff,
    CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, LJPME).  With
    ``constraints``, the constraint tables become OpenMM constraints and
    stretch/angle terms marked ``constrained`` are left out.
    """
    mm, app, unit = _omm()
    s = system
    unknown = sorted(set(s.tables) - _KNOWN)
    if unknown:
        raise NotImplementedError(f"tables without an OpenMM translation: {unknown}")
    omm = mm.System()
    anum = s.atoms["anum"]
    mass = s.atoms["mass"]
    for m in mass.tolist():
        omm.addParticle(m)
    if s.cell.any():
        omm.setDefaultPeriodicBoxVectors(*[mm.Vec3(*(row * NM)) for row in s.cell])
    tables = s.tables

    def add(force, name):
        force.setName(name)
        force.setForceGroup(min(omm.getNumForces(), 31))
        omm.addForce(force)

    def skip_constrained(t):
        if constraints and "constrained" in t.term_props:
            return t.values("constrained") != 0
        return np.zeros(len(t), bool)

    if "stretch_harm" in tables:
        t = tables["stretch_harm"]
        f = mm.HarmonicBondForce()
        keep = ~skip_constrained(t)
        for (i, j), r0, fc in zip(t.atoms[keep].tolist(), t.values("r0")[keep].tolist(),
                                  t.values("fc")[keep].tolist(), strict=True):  # fmt: skip
            f.addBond(i, j, r0 * NM, 2 * fc * KCAL / NM**2)
        add(f, "stretch_harm")
    if "angle_harm" in tables:
        t = tables["angle_harm"]
        f = mm.HarmonicAngleForce()
        keep = ~skip_constrained(t)
        for (i, j, k), th, fc in zip(t.atoms[keep].tolist(), t.values("theta0")[keep].tolist(),
                                     t.values("fc")[keep].tolist(), strict=True):  # fmt: skip
            f.addAngle(i, j, k, math.radians(th), 2 * fc * KCAL)
        add(f, "angle_harm")
    if "dihedral_trig" in tables:
        t = tables["dihedral_trig"]
        f = mm.PeriodicTorsionForce()
        fcs = np.column_stack([t.values(f"fc{n}") for n in range(7)])
        phase = np.radians(t.values("phi0"))
        offset = 0.0
        for (a, b, c, d), p, row in zip(t.atoms.tolist(), phase.tolist(), fcs.tolist(),
                                        strict=True):  # fmt: skip
            offset += row[0] - sum(row[1:])
            for n in range(1, 7):
                if row[n] != 0:
                    f.addTorsion(a, b, c, d, n, p, row[n] * KCAL)
        add(f, "dihedral_trig")
        if abs(offset) > 0:  # fc0 - sum(fc_n) is a constant energy per term
            const = mm.CustomExternalForce("c")
            const.addGlobalParameter("c", offset * KCAL)
            const.addParticle(0, [])
            add(const, "dihedral_trig_constant")
    if "improper_harm" in tables:
        t = tables["improper_harm"]
        f = mm.CustomTorsionForce(
            "fc*d^2; d = dp - 6.283185307179586*floor((dp + 3.141592653589793)"
            "/6.283185307179586); dp = theta - phi0"
        )
        f.addPerTorsionParameter("fc")
        f.addPerTorsionParameter("phi0")
        for atoms, fc, phi in zip(t.atoms.tolist(), t.values("fc").tolist(),
                                  t.values("phi0").tolist(), strict=True):  # fmt: skip
            f.addTorsion(*atoms, [fc * KCAL, math.radians(phi)])
        add(f, "improper_harm")
    if constraints:
        _add_constraints(s, omm)
    _add_virtual_sites(s, omm, mm)
    _add_nonbonded(s, omm, mm, add, nonbonded_method, cutoff, dispersion_correction,
                   ewald_tolerance)  # fmt: skip
    if "posre_harm" in tables:
        t = tables["posre_harm"]
        f = mm.CustomExternalForce("0.5*(fcx*(x-x0)^2 + fcy*(y-y0)^2 + fcz*(z-z0)^2)")
        for p in ("fcx", "fcy", "fcz", "x0", "y0", "z0"):
            f.addPerParticleParameter(p)
        cols = [t.values(p) for p in ("fcx", "fcy", "fcz")]
        refs = [t.values(p) for p in ("x0", "y0", "z0")]
        for k, (i,) in enumerate(t.atoms.tolist()):
            f.addParticle(i, [c[k] * KCAL / NM**2 for c in cols] + [r[k] * NM for r in refs])
        add(f, "posre_harm")
    if "torsiontorsion_cmap" in tables:
        _add_cmap(s, omm, mm, add)
    return _topology(s, app, anum), omm, unit.Quantity(s.positions * NM, unit.nanometer)


def _topology(s, app, anum):
    top = app.Topology()
    chains = [top.addChain(str(n)) for n in s.chains["name"].tolist()]
    res_chain = s.residues["chain"].tolist()
    residues = [
        top.addResidue(nm, chains[c], id=str(rid), insertionCode=ins or " ")
        for nm, c, rid, ins in zip(s.residues["name"].tolist(), res_chain,
                                   s.residues["resid"].tolist(), s.residues["insertion"].tolist(),
                                   strict=True)
    ]  # fmt: skip
    atoms = []
    for nm, z, r in zip(s.atoms["name"].tolist(), anum.tolist(), s.atoms["residue"].tolist(),
                        strict=True):  # fmt: skip
        element = app.Element.getByAtomicNumber(z) if z > 0 else None
        atoms.append(top.addAtom(nm, element, residues[r]))
    types = {1: app.Single, 2: app.Double, 3: app.Triple}
    for i, j, o in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), s.bonds["order"].tolist(),
                       strict=True):  # fmt: skip
        top.addBond(atoms[i], atoms[j], type=types.get(o), order=o or None)
    if s.cell.any():
        from openmm import Vec3

        top.setPeriodicBoxVectors([Vec3(*(row * NM)) for row in s.cell])
    return top


def _add_constraints(s, omm):
    for name, t in s.tables.items():
        if not name.startswith("constraint_"):
            continue
        if name == "constraint_hoh":
            for (o, h1, h2), th, r1, r2 in zip(t.atoms.tolist(), t.values("theta").tolist(),
                                               t.values("r1").tolist(), t.values("r2").tolist(),
                                               strict=True):  # fmt: skip
                hh = math.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * math.cos(math.radians(th)))
                omm.addConstraint(o, h1, r1 * NM)
                omm.addConstraint(o, h2, r2 * NM)
                omm.addConstraint(h1, h2, hh * NM)
        else:
            n = t.natoms - 1
            dist = np.column_stack([t.values(f"r{k}") for k in range(1, n + 1)])
            for atoms, row in zip(t.atoms.tolist(), dist.tolist(), strict=True):
                for k in range(n):
                    omm.addConstraint(atoms[0], atoms[k + 1], row[k] * NM)


def _add_virtual_sites(s, omm, mm):
    for name, t in s.tables.items():
        if not name.startswith("virtual_"):
            continue
        c = [t.values(p) for p in t.params.props if p.startswith("c")]
        for k, atoms in enumerate(t.atoms.tolist()):
            v = atoms[0]
            omm.setParticleMass(v, 0.0)
            if name in ("virtual_lc2", "virtual_midpoint"):
                w = c[0][k]
                site = mm.TwoParticleAverageSite(atoms[1], atoms[2], 1 - w, w)
            elif name == "virtual_lc3":
                w1, w2 = c[0][k], c[1][k]
                site = mm.ThreeParticleAverageSite(*atoms[1:4], 1 - w1 - w2, w1, w2)
            elif name == "virtual_out3":  # cross term coefficient is per length
                site = mm.OutOfPlaneSite(atoms[1], atoms[2], atoms[3], c[0][k], c[1][k],
                                         c[2][k] / NM)  # fmt: skip
            elif name == "virtual_fdat3":
                site = _fdat3_site(mm, atoms[1:4], c[0][k], c[1][k], c[2][k])
            else:
                raise NotImplementedError(f"{name} has no OpenMM translation")
            omm.setVirtualSite(v, site)


# fdat3 as an OpenMM local frame: origin i, x along i - j, y along k - j (made
# orthogonal to x), z = x cross y.  With r1 = -x and r2 = y this is the DMS
# r_i + a r1 + b r2 + c r2 x r1, so the local position is (-a, b, c).
_FDAT3_WEIGHTS = ([1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, -1.0, 1.0])


def _fdat3_site(mm, parents, d, theta, phi):
    th, ph = math.radians(theta), math.radians(phi)
    a, b, c = d * math.cos(th), d * math.sin(th) * math.cos(ph), d * math.sin(th) * math.sin(ph)
    return mm.LocalCoordinatesSite([int(p) for p in parents], *_FDAT3_WEIGHTS,
                                   mm.Vec3(-a * NM, b * NM, c * NM))  # fmt: skip


def _pair_terms(s):
    """Excluded pairs and pair_12_6_es terms keyed by (i, j), i < j."""
    excl = set()
    if "exclusion" in s.tables:
        a = np.sort(s.tables["exclusion"].atoms, axis=1)
        excl = set(map(tuple, a.tolist()))
    pairs: dict = {}
    if "pair_12_6_es" in s.tables:
        t = s.tables["pair_12_6_es"]
        vals = np.column_stack([t.values(p) for p in ("aij", "bij", "qij")])
        for (i, j), row in zip(np.sort(t.atoms, axis=1).tolist(), vals.tolist(), strict=True):
            old = pairs.get((i, j), (0.0, 0.0, 0.0))
            pairs[(i, j)] = tuple(o + v for o, v in zip(old, row, strict=True))
    return excl, pairs


def _combine(rule, s1, e1, s2, e2):
    if rule == "geometric":
        return np.sqrt(s1 * s2), np.sqrt(e1 * e2)
    return 0.5 * (s1 + s2), np.sqrt(e1 * e2)


def _add_nonbonded(s, omm, mm, add, method, cutoff, dispersion, tolerance):
    n = s.natoms
    nb = s.tables.get("nonbonded")
    sigma, eps, ptype = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    rule = s.nonbonded_info.vdw_rule.lower() or "arithmetic/geometric"
    if nb is not None:
        funct = s.nonbonded_info.vdw_funct.lower()
        if funct not in ("", "vdw_12_6"):
            raise NotImplementedError(f"nonbonded functional form {funct!r}")
        if rule not in ("arithmetic/geometric", "geometric"):
            raise NotImplementedError(f"combining rule {rule!r}")
        atoms = nb.atoms[:, 0]
        sigma[atoms], eps[atoms] = nb.values("sigma"), nb.values("epsilon")
        ptype[atoms] = nb.param_ids
    custom = nb is not None and (rule == "geometric" or len(nb.overrides) > 0)
    f = mm.NonbondedForce()
    methods = {"NoCutoff": f.NoCutoff, "CutoffNonPeriodic": f.CutoffNonPeriodic,
               "CutoffPeriodic": f.CutoffPeriodic, "Ewald": f.Ewald, "PME": f.PME,
               "LJPME": f.LJPME}  # fmt: skip
    if method not in methods:
        raise ValueError(f"unknown nonbonded method {method!r}")
    f.setNonbondedMethod(methods[method])
    f.setCutoffDistance(cutoff * NM)
    f.setEwaldErrorTolerance(tolerance)
    f.setUseDispersionCorrection(dispersion and not custom)
    charge = s.atoms["charge"].tolist()
    for i in range(n):
        sig = sigma[i] * NM if sigma[i] > 0 else NM
        f.addParticle(charge[i], sig, 0.0 if custom else eps[i] * KCAL)
    excl, pairs = _pair_terms(s)
    extra = []
    for i, j in sorted(excl):
        a, b, q = pairs.pop((i, j), (0.0, 0.0, 0.0))
        if a > 0 and b > 0:
            f.addException(i, j, q, (a / b) ** (1 / 6) * NM, b * b / (4 * a) * KCAL)
        else:
            f.addException(i, j, q, NM, 0.0)
            if a or b:
                extra.append((i, j, a, b, 0.0))
    extra += [(i, j, a, b, q) for (i, j), (a, b, q) in pairs.items()]
    add(f, "nonbonded")
    if custom:
        types = len(nb.params)
        ps, pe = nb.params["sigma"], nb.params["epsilon"]
        ss, ee = _combine(rule, ps[:, None], pe[:, None], ps[None, :], pe[None, :])
        for (p, q), vals in nb.overrides.items():
            ss[p, q] = ss[q, p] = vals["sigma"]
            ee[p, q] = ee[q, p] = vals["epsilon"]
        acoef = (4 * ee * ss**12 * KCAL * NM**12).ravel(order="F").tolist()
        bcoef = (4 * ee * ss**6 * KCAL * NM**6).ravel(order="F").tolist()
        lj = mm.CustomNonbondedForce("acoef(t1, t2)/r^12 - bcoef(t1, t2)/r^6")
        lj.addTabulatedFunction("acoef", mm.Discrete2DFunction(types, types, acoef))
        lj.addTabulatedFunction("bcoef", mm.Discrete2DFunction(types, types, bcoef))
        lj.addPerParticleParameter("t")
        for i in range(n):
            lj.addParticle([float(ptype[i])])
        for i, j in sorted(excl):
            lj.addExclusion(i, j)
        if method == "NoCutoff":
            lj.setNonbondedMethod(lj.NoCutoff)
        elif method == "CutoffNonPeriodic":
            lj.setNonbondedMethod(lj.CutoffNonPeriodic)
        else:
            lj.setNonbondedMethod(lj.CutoffPeriodic)
        lj.setCutoffDistance(cutoff * NM)
        lj.setUseLongRangeCorrection(dispersion and method != "NoCutoff")
        add(lj, "nonbonded_vdw")
    if extra:
        pf = mm.CustomBondForce(f"a/r^12 - b/r^6 + {_ONE_4PI_EPS0}*q/r")
        for p in ("a", "b", "q"):
            pf.addPerBondParameter(p)
        for i, j, a, b, q in extra:
            pf.addBond(i, j, [a * KCAL * NM**12, b * KCAL * NM**6, q])
        add(pf, "pair_12_6_es")


def _add_cmap(s, omm, mm, add):
    t = s.tables["torsiontorsion_cmap"]
    f = mm.CMAPTorsionForce()
    maps: dict[str, int] = {}
    for atoms, cid in zip(t.atoms.tolist(), t.values("cmapid").tolist(), strict=True):
        if cid not in maps:
            grid = s.aux_tables[cid]
            size = int(round(math.sqrt(len(grid))))
            order = np.lexsort((grid["psi"], grid["phi"]))
            e = grid["energy"][order].reshape(size, size)  # [phi index, psi index] from -180
            shift = np.roll(np.roll(e, -size // 2, axis=0), -size // 2, axis=1)  # from 0
            maps[cid] = f.addMap(size, (shift.T.ravel() * KCAL).tolist())
        f.addTorsion(maps[cid], *atoms)
    add(f, "torsiontorsion_cmap")


def openmm_energies(system, positions=None, platform: str = "Reference", **kwargs) -> dict:
    """Energy of each translated force in kcal/mol (and the total)."""
    mm, _, unit = _omm()
    _, omm, pos = to_openmm(system, **kwargs)
    if positions is not None:
        pos = unit.Quantity(np.asarray(positions, dtype=np.float64) * NM, unit.nanometer)
    ctx = mm.Context(omm, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform))
    ctx.setPositions(pos)
    ctx.computeVirtualSites()
    out = {}
    for f in omm.getForces():
        state = ctx.getState(getEnergy=True, groups={f.getForceGroup()})
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / KCAL
        out[f.getName()] = out.get(f.getName(), 0.0) + e
    total = ctx.getState(getEnergy=True).getPotentialEnergy()
    out["total"] = total.value_in_unit(unit.kilojoule_per_mole) / KCAL
    return out


# ---------------------------------------------------------------------------
# OpenMM -> boonza


class _Params:
    """Rows of a ParamTable keyed by their values."""

    def __init__(self, table, props):
        self.table = table
        self.props = list(props)
        for p in self.props:
            if p not in table.params.props:
                table.params.add_prop(p, float)
        self.index: dict = {}

    def __call__(self, values) -> int:
        key = tuple(float(v) for v in values)
        pid = self.index.get(key)
        if pid is None:
            pid = self.index[key] = self.table.params.add_param(
                **dict(zip(self.props, key, strict=True))
            )
        return pid


def from_openmm(topology, omm_system=None, positions=None, ignore_unknown: bool = False) -> System:
    """System from an OpenMM Topology and, optionally, System (positions in nm).

    Without ``omm_system`` only the structure is converted, with masses from
    the elements.  Forces become DMS tables: harmonic bonds (Urey-Bradley
    terms included) and angles, periodic and Ryckaert-Bellemans torsions,
    CMAP, Coulomb and Lennard-Jones with exceptions, and the custom forms
    CHARMM, GROMACS and OpenMM's ForceField use: harmonic impropers,
    position restraints, tabulated Lennard-Jones (NBFIX), per-particle
    geometric Lennard-Jones and 12-6/Coulomb pair bonds.  Custom forces are
    recognized by evaluating them on small test geometries, not by their
    text.  ``ignore_unknown`` skips forces without a DMS equivalent instead
    of raising.
    """
    mm, app, unit = _omm()
    s = System()
    ct = s.add_ct()
    chains = {}
    for chain in topology.chains():
        chains[chain.index] = s.add_chain(ct, name=str(chain.id or ""))
    res_of = {}
    for r in topology.residues():
        try:
            rid = int(r.id)
        except (TypeError, ValueError):
            rid = r.index + 1
        res_of[r.index] = s.add_residue(chains[r.chain.index], name=r.name, resid=rid,
                                        insertion=(r.insertionCode or "").strip()).id  # fmt: skip
    atoms = list(topology.atoms())
    n = len(atoms)
    if omm_system is not None and n != omm_system.getNumParticles():
        raise ValueError("topology and system have different numbers of particles")
    anum = [a.element.atomic_number if a.element is not None else 0 for a in atoms]
    if omm_system is not None:
        mass = [omm_system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)]
    else:
        mass = [0.0 if a.element is None else a.element.mass.value_in_unit(unit.dalton)
                for a in atoms]  # fmt: skip
    s.add_atoms(n, [res_of[a.residue.index] for a in atoms], name=[a.name for a in atoms],
                anum=np.array(anum), mass=np.array(mass))  # fmt: skip
    if positions is not None:
        pos = positions.value_in_unit(unit.nanometer) if hasattr(positions, "value_in_unit") \
            else positions  # fmt: skip
        s.positions = np.asarray(pos, dtype=np.float64) / NM
    box = topology.getPeriodicBoxVectors()
    if box is None and omm_system is not None and omm_system.usesPeriodicBoundaryConditions():
        box = omm_system.getDefaultPeriodicBoxVectors()
    if box is not None:
        s.cell = np.array([v.value_in_unit(unit.nanometer) for v in box]) / NM
    bonds = [(b.atom1.index, b.atom2.index, b.order or 1) for b in topology.bonds()]
    if bonds:
        b = np.array(bonds)
        s.add_bonds(b[:, :2], order=b[:, 2])
    if omm_system is None:
        return s

    nonbonded, custom_lj, constant = None, [], 0.0
    for force in omm_system.getForces():
        kind = type(force).__name__
        try:
            if kind == "RBTorsionForce":
                _rb_from_openmm(s, force, unit)
                continue
            if kind == "CustomTorsionForce":
                _custom_torsion_from_openmm(s, force, mm)
                continue
            if kind == "CustomExternalForce":
                constant += _custom_external_from_openmm(s, force, mm)
                continue
            if kind == "CustomBondForce":
                _custom_bond_from_openmm(s, force, mm)
                continue
        except NotImplementedError:
            if ignore_unknown:
                continue
            raise
        if kind == "CustomNonbondedForce":
            custom_lj.append(force)
        elif kind == "NonbondedForce":
            nonbonded = force
        elif kind == "HarmonicBondForce":
            t = s.add_table_from_schema("stretch_harm")
            pm = _Params(t, ["r0", "fc"])
            for k in range(force.getNumBonds()):
                i, j, r0, kk = force.getBondParameters(k)
                t.add_term([i, j], pm([r0.value_in_unit(unit.nanometer) / NM,
                                       kk.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                                       / 2 / KCAL * NM**2]))  # fmt: skip
        elif kind == "HarmonicAngleForce":
            t = s.add_table_from_schema("angle_harm")
            pm = _Params(t, ["theta0", "fc"])
            for k in range(force.getNumAngles()):
                i, j, kk_, th, kk = force.getAngleParameters(k)
                theta = math.degrees(th.value_in_unit(unit.radian))
                fc = kk.value_in_unit(unit.kilojoule_per_mole / unit.radian**2) / 2 / KCAL
                t.add_term([i, j, kk_], pm([theta, fc]))
        elif kind == "PeriodicTorsionForce":
            t = s.add_table_from_schema("dihedral_trig")
            pm = _Params(t, ["phi0", *[f"fc{k}" for k in range(7)]])
            for k in range(force.getNumTorsions()):
                a, b, c, d, per, phase, kk = force.getTorsionParameters(k)
                if not 1 <= per <= 6:
                    raise NotImplementedError(f"torsion periodicity {per}")
                k_kcal = kk.value_in_unit(unit.kilojoule_per_mole) / KCAL
                fc = [k_kcal] + [k_kcal if m == per else 0.0 for m in range(1, 7)]
                t.add_term([a, b, c, d], pm([math.degrees(phase.value_in_unit(unit.radian)), *fc]))
        elif kind == "CMAPTorsionForce":
            _cmap_from_openmm(s, force)
        elif kind == "CMMotionRemover":
            continue
        elif not ignore_unknown:
            raise NotImplementedError(f"{kind} has no DMS equivalent; pass ignore_unknown=True")
    if nonbonded is not None or custom_lj:
        _nonbonded_from_openmm(s, nonbonded, custom_lj, mm, unit, ignore_unknown)
    if constant:
        _add_constant(s, constant)
    _constraints_from_openmm(s, omm_system, unit)
    _virtual_sites_from_openmm(s, omm_system, mm)
    return s


# Custom forces are identified numerically: each is copied into a tiny test
# system and evaluated at chosen geometries and parameters.

_PROBE_R = np.array([0.25, 0.32, 0.41, 0.55, 0.8, 1.2])  # nm
_PROBE_BOX = np.eye(3) * 20.0  # nm; only for expressions using periodic distances


def _probe_energy(mm, force, positions) -> float:
    """Energy in kJ/mol of ``force`` alone on particles at ``positions`` (nm)."""
    system = mm.System()
    for _ in positions:
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in _PROBE_BOX])
    system.addForce(force)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([mm.Vec3(*map(float, p)) for p in positions])
    energy = ctx.getState(getEnergy=True).getPotentialEnergy()
    return energy.value_in_unit(energy.unit)


def _blank(force, cls, names, add_param):
    """A new ``cls`` with the expression, globals, functions and parameter names of ``force``."""
    new = cls(force.getEnergyFunction())
    for i in range(force.getNumGlobalParameters()):
        new.addGlobalParameter(force.getGlobalParameterName(i),
                               force.getGlobalParameterDefaultValue(i))  # fmt: skip
    for i in range(getattr(force, "getNumTabulatedFunctions", lambda: 0)()):
        new.addTabulatedFunction(force.getTabulatedFunctionName(i),
                                 copy.deepcopy(force.getTabulatedFunction(i)))  # fmt: skip
    if hasattr(force, "usesPeriodicBoundaryConditions") and hasattr(
        new, "setUsesPeriodicBoundaryConditions"
    ):
        new.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    for name in names:
        getattr(new, add_param)(name)
    return new


def _fit_pair(energies, powers=(12, 6, 1)):
    """Coefficients c with E(r) = sum c_k r^-p_k over the probe distances, or None if no fit."""
    X = np.column_stack([_PROBE_R ** -float(p) for p in powers])
    norm = np.linalg.norm(X, axis=0)
    coef = np.linalg.lstsq(X / norm, energies, rcond=None)[0] / norm
    scale = max(np.abs(energies).max(), 1e-12)
    coef[np.abs(X * coef).max(axis=0) <= 1e-10 * scale] = 0.0  # fit noise in absent terms
    if np.abs(X @ coef - energies).max() > 1e-9 * scale:
        return None
    return coef


def _custom_bond_from_openmm(s, force, mm):
    """12-6/Coulomb pair bonds (e.g. CHARMM/GROMACS 1-4 Lennard-Jones) as pair_12_6_es terms."""
    names = [force.getPerBondParameterName(i) for i in range(force.getNumPerBondParameters())]
    expr = force.getEnergyFunction()

    def energies(params):
        out = []
        for r in _PROBE_R:
            f = _blank(force, mm.CustomBondForce, names, "addPerBondParameter")
            f.addBond(0, 1, [float(p) for p in params])
            out.append(_probe_energy(mm, f, [(1.0, 1.0, 1.0), (1.0 + r, 1.0, 1.0)]))
        return np.array(out)

    unsupported = NotImplementedError(f"CustomBondForce {expr!r} is not a 12-6/Coulomb pair form")
    if sorted(names) == ["epsilon", "sigma"]:
        i_s, i_e = names.index("sigma"), names.index("epsilon")
        probe = [0.0, 0.0]
        probe[i_s], probe[i_e] = 0.3, 1.0
        c = _fit_pair(energies(probe))
        if c is None or not np.isclose(c[0] / 0.3**12, -c[1] / 0.3**6, rtol=1e-7) or \
                abs(c[2]) > 1e-9 * abs(c[0]) * 0.3**-11:  # fmt: skip
            raise unsupported
        scale = c[0] / 0.3**12

        def coef(p):
            return scale * p[i_e] * p[i_s] ** 12, -scale * p[i_e] * p[i_s] ** 6, 0.0
    else:
        basis = []
        for k in range(len(names)):
            unit_params = np.zeros(len(names))
            unit_params[k] = 1.0
            c = _fit_pair(energies(unit_params))
            if c is None:
                raise unsupported
            basis.append(c)
        basis = np.array(basis).reshape(len(names), 3)
        test = np.linspace(0.5, 1.5, len(names))
        got = energies(test)  # the energy must be linear in the parameters
        want = np.column_stack([_PROBE_R**-12.0, _PROBE_R**-6.0, 1 / _PROBE_R]) @ (test @ basis)
        if not np.allclose(got, want, rtol=1e-9, atol=1e-9 * np.abs(got).max()):
            raise unsupported

        def coef(p):
            return tuple(np.asarray(p, dtype=np.float64) @ basis)

    if not force.getNumBonds():
        return
    t = s.add_table_from_schema("pair_12_6_es")
    pm = _Params(t, ["aij", "bij", "qij"])
    rows, pids = [], []
    for k in range(force.getNumBonds()):
        i, j, params = force.getBondParameters(k)
        a12, c6, c1 = coef(params)
        rows.append((i, j))
        pids.append(pm([a12 / (KCAL * NM**12), -c6 / (KCAL * NM**6), c1 / _ONE_4PI_EPS0]))
    t.add_terms(np.array(rows), pids)


def _torsion_positions(theta):
    return [(1.1, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.15),
            (1.0 + 0.1 * math.cos(theta), 1.0 + 0.1 * math.sin(theta), 1.15)]  # fmt: skip


def _custom_torsion_from_openmm(s, force, mm):
    """Harmonic impropers, k (theta - theta0)^2 in any scaling, as improper_harm."""
    from .pbc import dihedrals

    names = [force.getPerTorsionParameterName(i) for i in range(force.getNumPerTorsionParameters())]
    lower = [x.lower() for x in names]
    k_name = next((x for x in ("k", "fc", "kf", "kimp") if x in lower), None)
    t_name = next((x for x in ("theta0", "phi0", "psi0", "theta_0", "phi_0", "t0") if x in lower),
                  None)  # fmt: skip
    unsupported = NotImplementedError(
        f"CustomTorsionForce {force.getEnergyFunction()!r} is not a harmonic improper"
    )
    if len(names) != 2 or k_name is None or t_name is None:
        raise unsupported
    i_k, i_t = lower.index(k_name), lower.index(t_name)
    theta0, scales = 0.3, []
    for delta in (0.1, -0.35, 0.6):
        pos = _torsion_positions(theta0 + delta)
        f = _blank(force, mm.CustomTorsionForce, names, "addPerTorsionParameter")
        params = [0.0, 0.0]
        params[i_k], params[i_t] = 1.0, theta0
        f.addTorsion(0, 1, 2, 3, params)
        measured = float(dihedrals(*np.array(pos))[0])
        scales.append(_probe_energy(mm, f, pos) / (measured - theta0) ** 2)
    if scales[0] <= 0 or not np.allclose(scales, scales[0], rtol=1e-6):
        raise unsupported
    if not force.getNumTorsions():
        return
    t = s.add_table_from_schema("improper_harm")
    pm = _Params(t, ["phi0", "fc"])
    for k in range(force.getNumTorsions()):
        a, b, c, d, params = force.getTorsionParameters(k)
        t.add_term([a, b, c, d], pm([math.degrees(params[i_t]), scales[0] * params[i_k] / KCAL]))


def _custom_external_from_openmm(s, force, mm) -> float:
    """Harmonic position restraints as posre_harm; returns any constant energy (kcal/mol)."""
    names = [force.getPerParticleParameterName(i)
             for i in range(force.getNumPerParticleParameters())]  # fmt: skip
    unsupported = NotImplementedError(
        f"CustomExternalForce {force.getEnergyFunction()!r} is not a harmonic restraint"
    )

    def energy(params, where):
        f = _blank(force, mm.CustomExternalForce, names, "addPerParticleParameter")
        f.addParticle(0, [float(p) for p in params])
        return _probe_energy(mm, f, [where])

    if not names:  # a constant, such as boonza's dihedral_trig offset
        e = energy([], (1.0, 1.2, 0.8))
        if not np.isclose(e, energy([], (3.1, 0.4, 2.2)), rtol=1e-12, atol=1e-12):
            raise unsupported
        return e * force.getNumParticles() / KCAL
    if not {"x0", "y0", "z0"} <= set(names):
        raise unsupported
    ref = [names.index(x) for x in ("x0", "y0", "z0")]
    k_names = [x for x in names if x not in ("x0", "y0", "z0")]
    if len(k_names) not in (1, 3):
        raise unsupported
    center = np.array([1.0, 1.0, 1.0])
    coeff = np.zeros((3, len(k_names)))  # energy / (k d^2) per axis and force constant
    for m, kn in enumerate(k_names):
        params = np.zeros(len(names))
        params[ref] = center
        params[names.index(kn)] = 1.0
        for axis in range(3):
            values = []
            for d in (0.05, 0.12):
                values.append(energy(params, center + d * np.eye(3)[axis]) / d**2)
            if not np.isclose(values[0], values[1], rtol=1e-7, atol=1e-12):
                raise unsupported
            coeff[axis, m] = values[0]
        diag = energy(params, center + 0.07) / 0.07**2
        if not np.isclose(diag, coeff[:, m].sum(), rtol=1e-7, atol=1e-12):
            raise unsupported  # cross terms
    if not force.getNumParticles():
        return 0.0
    t = s.add_table_from_schema("posre_harm")
    pm = _Params(t, ["fcx", "fcy", "fcz"])
    k_cols = [names.index(x) for x in k_names]
    for k in range(force.getNumParticles()):
        i, params = force.getParticleParameters(k)
        params = np.asarray(params, dtype=np.float64)
        fc = 2 * coeff @ params[k_cols] * NM**2 / KCAL
        x0, y0, z0 = params[ref] / NM
        t.add_term([i], pm(fc), x0=x0, y0=y0, z0=z0)
    return 0.0


def _add_constant(s, energy):
    """Fold a constant energy (kcal/mol) into a dihedral_trig term's fc0."""
    t = s.tables.get("dihedral_trig")
    if t is not None and len(t):
        atoms = t.atoms[0]
    elif s.natoms >= 4:
        atoms = np.arange(4)
    else:
        raise NotImplementedError("a constant energy needs a dihedral to hold it")
    t = s.add_table_from_schema("dihedral_trig")
    values = {"phi0": 0.0, **{f"fc{n}": 0.0 for n in range(7)}}
    values["fc0"] = energy
    t.add_term(atoms, t.params.add_param(**values))


def _rb_from_openmm(s, force, unit):
    """Ryckaert-Bellemans torsions, sum C_n cos^n(phi - 180), as exact Fourier series."""
    from numpy.polynomial import chebyshev

    t = s.add_table_from_schema("dihedral_trig")
    pm = _Params(t, ["phi0", *[f"fc{k}" for k in range(7)]])
    for k in range(force.getNumTorsions()):
        a, b, c, d, *cs = force.getTorsionParameters(k)
        cs = [x.value_in_unit(unit.kilojoule_per_mole) if hasattr(x, "value_in_unit") else x
              for x in cs]  # fmt: skip
        # cos(phi - 180) = -cos(phi); cos^n in terms of cos(k phi) is a Chebyshev series
        cheb = chebyshev.poly2cheb([cn * (-1) ** n for n, cn in enumerate(cs)])
        fc = np.zeros(7)
        fc[: len(cheb)] = cheb / KCAL
        t.add_term([a, b, c, d], pm([0.0, *fc]))


def _custom_lj(force, mm):
    """Lennard-Jones in a CustomNonbondedForce as (rule, sigma, epsilon, type per atom, overrides).

    Tabulated A/B (or sqrt(A)/B) coefficients by particle type, as CHARMM,
    GROMACS NBFIX and OpenMM's LennardJonesForce write them, become per-type
    parameters with Lorentz-Berthelot combining and overrides where the table
    departs from it.  GROMACS geometric per-particle A1*A2/r^12 - C1*C2/r^6
    becomes the geometric rule.  Sigma in Å, epsilon in kcal/mol.
    """
    names = [force.getPerParticleParameterName(i)
             for i in range(force.getNumPerParticleParameters())]  # fmt: skip
    n = force.getNumParticles()
    params = np.array([force.getParticleParameters(i) for i in range(n)], dtype=np.float64)
    params = params.reshape(n, len(names))
    funcs = {force.getTabulatedFunctionName(i): force.getTabulatedFunction(i)
             for i in range(force.getNumTabulatedFunctions())}  # fmt: skip
    unsupported = NotImplementedError(
        f"CustomNonbondedForce {force.getEnergyFunction()!r} is not a supported Lennard-Jones form"
    )

    def pair(p1, p2):
        out = []
        for r in _PROBE_R:
            f = _blank(force, mm.CustomNonbondedForce, names, "addPerParticleParameter")
            f.addParticle([float(x) for x in p1])
            f.addParticle([float(x) for x in p2])
            out.append(_probe_energy(mm, f, [(1.0, 1.0, 1.0), (1.0 + r, 1.0, 1.0)]))
        return _fit_pair(np.array(out), powers=(12, 6))

    if names == ["type"] and {"acoef", "bcoef"} <= funcs.keys():
        size, _, a_vals = funcs["acoef"].getFunctionParameters()
        _, _, b_vals = funcs["bcoef"].getFunctionParameters()
        acoef = np.asarray(a_vals, dtype=np.float64).reshape(size, size).T  # [type1, type2]
        bcoef = np.asarray(b_vals, dtype=np.float64).reshape(size, size).T
        types = params[:, 0].astype(np.int64)
        diag = np.flatnonzero(np.diag(acoef) > 0)
        if not len(diag):
            raise unsupported
        t0 = int(diag[0])
        c = pair([t0], [t0])
        if c is None or not np.isclose(-c[1], bcoef[t0, t0], rtol=1e-7):
            raise unsupported
        if np.isclose(c[0], acoef[t0, t0], rtol=1e-7):
            A = acoef
        elif np.isclose(c[0], acoef[t0, t0] ** 2, rtol=1e-7):
            A = acoef**2
        else:
            raise unsupported
        B = bcoef
        ad, bd = np.diag(A), np.diag(B)
        ok = (ad > 0) & (bd > 0)
        sig = np.where(ok, (np.where(ok, ad, 1) / np.where(ok, bd, 1)) ** (1 / 6), 0.0)
        eps = np.where(ok, bd**2 / (4 * np.where(ok, ad, 1)), 0.0)
        s_lb = 0.5 * (sig[:, None] + sig[None, :])
        e_lb = np.sqrt(eps[:, None] * eps[None, :])
        a_lb, b_lb = 4 * e_lb * s_lb**12, 4 * e_lb * s_lb**6
        scale_a, scale_b = max(np.abs(A).max(), 1e-300), max(np.abs(B).max(), 1e-300)
        differs = ~(np.isclose(A, a_lb, rtol=1e-7, atol=1e-12 * scale_a)
                    & np.isclose(B, b_lb, rtol=1e-7, atol=1e-12 * scale_b))  # fmt: skip
        overrides = {}
        for i, j in np.argwhere(np.triu(differs)).tolist():
            if A[i, j] > 0 and B[i, j] > 0:
                so, eo = (A[i, j] / B[i, j]) ** (1 / 6), B[i, j] ** 2 / (4 * A[i, j])
            else:
                so, eo = s_lb[i, j], 0.0
            overrides[(i, j)] = (so / NM, eo / KCAL)
        return "arithmetic/geometric", sig / NM, eps / KCAL, types, overrides

    if sorted(names) == ["A", "C"]:
        i_a, i_c = names.index("A"), names.index("C")
        one_a, one_c = np.zeros(2), np.zeros(2)
        one_a[i_a], one_c[i_c] = 1.0, 1.0
        ca, cc = pair(one_a, one_a), pair(one_c, one_c)
        if ca is None or cc is None or not np.allclose([ca[0], ca[1], cc[0], cc[1]],
                                                       [1, 0, 0, -1], atol=1e-9):  # fmt: skip
            raise unsupported
        a, c6 = params[:, i_a], params[:, i_c]
        ok = (a > 0) & (c6 > 0)
        sig = np.where(ok, (np.where(ok, a, 1) / np.where(ok, c6, 1)) ** (1 / 3), 0.0)
        eps = np.where(ok, c6**4 / (4 * np.where(ok, a, 1) ** 2), 0.0)
        table, types = np.unique(np.column_stack([sig, eps]), axis=0, return_inverse=True)
        return "geometric", table[:, 0] / NM, table[:, 1] / KCAL, types.ravel(), {}
    raise unsupported


def _nonbonded_from_openmm(s, force, customs, mm, unit, ignore_unknown=False):
    n = s.natoms
    q, sig, eps = np.zeros(n), np.zeros(n), np.zeros(n)
    if force is not None:
        for i in range(n):
            c, sg, ep = force.getParticleParameters(i)
            q[i] = c.value_in_unit(unit.elementary_charge)
            sig[i] = sg.value_in_unit(unit.nanometer) / NM
            eps[i] = ep.value_in_unit(unit.kilojoule_per_mole) / KCAL
    s.atoms["charge"] = q
    lj = None
    for cf in customs:
        try:
            found = _custom_lj(cf, mm)
        except NotImplementedError:
            if ignore_unknown:
                continue
            raise
        if lj is not None:
            raise NotImplementedError("more than one Lennard-Jones CustomNonbondedForce")
        lj = (found, cf)
    if lj is None:
        t = s.add_nonbonded_from_schema("vdw_12_6", "arithmetic/geometric")
        pm = _Params(t, ["sigma", "epsilon"])
        t.add_terms(
            np.arange(n).reshape(-1, 1), [pm([a, b]) for a, b in zip(sig, eps, strict=True)]
        )
    else:
        if (eps != 0).any():
            raise NotImplementedError(
                "Lennard-Jones in both NonbondedForce and a CustomNonbondedForce"
            )
        (rule, tsig, teps, types, overrides), _ = lj
        t = s.add_nonbonded_from_schema("vdw_12_6", rule)
        pids = t.params.add_params(len(tsig), sigma=tsig, epsilon=teps)
        t.add_terms(np.arange(n).reshape(-1, 1), pids[types])
        for (i, j), (so, eo) in overrides.items():
            t.overrides.set(pids[i], pids[j], sigma=so, epsilon=eo)

    ex_atoms, pair_atoms, pair_ids = {}, [], []
    if force is not None and force.getNumExceptions():
        pairs = s.add_table_from_schema("pair_12_6_es")
        pp = _Params(pairs, ["aij", "bij", "qij"])
        for k in range(force.getNumExceptions()):
            i, j, qq, sg, ep = force.getExceptionParameters(k)
            ex_atoms[(min(i, j), max(i, j))] = None
            qq = qq.value_in_unit(unit.elementary_charge**2)
            sg = sg.value_in_unit(unit.nanometer) / NM
            ep = ep.value_in_unit(unit.kilojoule_per_mole) / KCAL
            if qq != 0 or ep != 0:
                pair_atoms.append((i, j))
                pair_ids.append(pp([4 * ep * sg**12, 4 * ep * sg**6, qq]))
        if pair_atoms:
            pairs.add_terms(np.array(pair_atoms), pair_ids)
    if lj is not None:
        cf = lj[1]
        for k in range(cf.getNumExclusions()):
            i, j = cf.getExclusionParticles(k)
            ex_atoms[(min(i, j), max(i, j))] = None
    if ex_atoms:
        s.add_table_from_schema("exclusion").add_terms(np.array(list(ex_atoms)))


def _cmap_from_openmm(s, force):
    t = s.add_table_from_schema("torsiontorsion_cmap")
    ids = []
    for m in range(force.getNumMaps()):
        size, energy = force.getMapParameters(m)
        e = (
            np.array(
                energy.value_in_unit_system(__import__("openmm").unit.md_unit_system)
                if hasattr(energy, "value_in_unit_system")
                else energy
            )
            .reshape(size, size)
            .T
        )
        grid = np.roll(np.roll(e, size // 2, axis=0), size // 2, axis=1) / KCAL
        from .terms import ParamTable

        aux = ParamTable()
        for p in ("phi", "psi", "energy"):
            aux.add_prop(p, float)
        ang = -180.0 + 360.0 / size * np.arange(size)
        phi, psi = np.meshgrid(ang, ang, indexing="ij")
        aux.add_params(size * size, phi=phi.ravel(), psi=psi.ravel(), energy=grid.ravel())
        name = f"cmap{m + 1}"
        s.aux_tables[name] = aux
        ids.append(t.params.add_param(cmapid=name))
    for k in range(force.getNumTorsions()):
        m, *atoms = force.getTorsionParameters(k)
        t.add_term(atoms, ids[m])


def _constraints_from_openmm(s, omm_system, unit):
    nc = omm_system.getNumConstraints()
    if not nc:
        return
    anum = s.atoms["anum"]
    cons = {}
    for k in range(nc):
        i, j, d = omm_system.getConstraintParameters(k)
        cons[(min(i, j), max(i, j))] = d.value_in_unit(unit.nanometer) / NM
    used = set()
    # rigid waters: three mutual constraints among O, H, H
    by_atom: dict[int, list] = {}
    for i, j in cons:
        by_atom.setdefault(i, []).append(j)
        by_atom.setdefault(j, []).append(i)
    hoh = []
    for o, partners in by_atom.items():
        if anum[o] == 8 and len(partners) == 2:
            h1, h2 = sorted(partners)
            if (h1, h2) in cons and anum[h1] == 1 and anum[h2] == 1:
                r1, r2 = cons[(min(o, h1), max(o, h1))], cons[(min(o, h2), max(o, h2))]
                hh = cons[(h1, h2)]
                theta = math.degrees(math.acos((r1 * r1 + r2 * r2 - hh * hh) / (2 * r1 * r2)))
                hoh.append(((o, h1, h2), (theta, r1, r2)))
                used |= {(min(o, h1), max(o, h1)), (min(o, h2), max(o, h2)), (h1, h2)}
    if hoh:
        t = s.add_table_from_schema("constraint_hoh")
        pm = _Params(t, ["theta", "r1", "r2"])
        for atoms, vals in hoh:
            t.add_term(list(atoms), pm(vals))
    groups: dict[int, list] = {}
    for (i, j), d in cons.items():
        if (i, j) in used:
            continue
        center, other = (i, j) if anum[i] >= anum[j] else (j, i)
        groups.setdefault(center, []).append((other, d))
    for center, partners in sorted(groups.items()):
        partners.sort()
        while partners:
            chunk, partners = partners[:8], partners[8:]
            t = s.add_table_from_schema(f"constraint_ah{len(chunk)}")
            pm = _Params(t, [f"r{k}" for k in range(1, len(chunk) + 1)])
            t.add_term([center] + [o for o, _ in chunk], pm([d for _, d in chunk]))
    constrained = set(cons)
    for name in ("stretch_harm",):
        t = s.tables.get(name)
        if t is not None:
            flags = [int(tuple(sorted(a)) in constrained) for a in t.atoms.tolist()]
            t._t.set("constrained", np.array(flags, np.int64))


def _virtual_sites_from_openmm(s, omm_system, mm):
    tables = {}
    for v in range(omm_system.getNumParticles()):
        if not omm_system.isVirtualSite(v):
            continue
        site = omm_system.getVirtualSite(v)
        kind = type(site).__name__
        parents = [site.getParticle(k) for k in range(site.getNumParticles())]
        if kind == "TwoParticleAverageSite":
            name, vals = "virtual_lc2", [site.getWeight(1)]
        elif kind == "ThreeParticleAverageSite":
            name, vals = "virtual_lc3", [site.getWeight(1), site.getWeight(2)]
        elif kind == "OutOfPlaneSite":
            name, vals = "virtual_out3", [site.getWeight12(), site.getWeight13(),
                                          site.getWeightCross() * NM]  # fmt: skip
        elif kind == "LocalCoordinatesSite":
            weights = [list(site.getOriginWeights()), list(site.getXWeights()),
                       list(site.getYWeights())]  # fmt: skip
            if len(parents) != 3 or not np.allclose(weights, _FDAT3_WEIGHTS):
                raise NotImplementedError(
                    "LocalCoordinatesSite other than a fixed distance-angle-torsion (fdat3) frame"
                )
            local = site.getLocalPosition()
            lx, ly, lz = (x / NM for x in local.value_in_unit(local.unit))
            a, b, c = -lx, ly, lz
            d = math.sqrt(a * a + b * b + c * c)
            name, vals = "virtual_fdat3", [d, math.degrees(math.acos(a / d)) if d else 0.0,
                                           math.degrees(math.atan2(c, b))]  # fmt: skip
        else:
            raise NotImplementedError(f"virtual site type {kind}")
        if name not in tables:
            t = s.add_table_from_schema(name)
            pm = _Params(t, t.params.props)
            tables[name] = (t, pm)
        t, pm = tables[name]
        t.add_term([v, *parents], pm(vals))
