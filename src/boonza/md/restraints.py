"""Backbone phi/psi restraints holding the protein near its input conformation.

The restraint on each torsion is a truncated Fourier well,

    V(theta) = sum_{i=1..6} K (-1)^i / i! [1 + cos(i (theta - theta0 - pi))],

with K = -|strength|, so that it sits on theta0. Each term is one OpenMM
periodic torsion with periodicity i and phase i (theta0 + pi). The force is
part of the OpenMM system, so ``system.xml`` carries it into restarts.
"""

from __future__ import annotations

import csv
import math

import numpy as np

FOURIER_TERMS = 6


def backbone_torsions(s) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    """(residue, "phi" or "psi", atoms) for each backbone torsion whose peptide
    bond exists, so termini and chain breaks are left free."""
    names, res = s.atoms["name"], s.atoms["residue"]
    bb: dict[int, dict[str, int]] = {}
    for a in np.flatnonzero(np.isin(names, ["N", "CA", "C"])).tolist():
        bb.setdefault(int(res[a]), {})[str(names[a])] = a
    out = []
    for r in sorted(bb):
        if len(bb[r]) != 3:
            continue
        n, ca, c = bb[r]["N"], bb[r]["CA"], bb[r]["C"]
        prev = [o for o in s.bonded_atoms(n).tolist() if res[o] != r and names[o] == "C"]
        nxt = [o for o in s.bonded_atoms(c).tolist() if res[o] != r and names[o] == "N"]
        if prev:
            out.append((r, "phi", (prev[0], n, ca, c)))
        if nxt:
            out.append((r, "psi", (n, ca, c, nxt[0])))
    return out


def fourier_terms(strength_kj: float):
    """(periodicity, force constant in kJ/mol) of the well."""
    k = -abs(strength_kj)
    for i in range(1, FOURIER_TERMS + 1):
        yield i, k * (-1) ** i / math.factorial(i)


def dihedral(pos, atoms) -> float:
    """The torsion angle of four atoms (radians)."""
    p0, p1, p2, p3 = (np.asarray(pos[a], dtype=np.float64) for a in atoms)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.atan2(float(np.dot(np.cross(b1, v), w)), float(np.dot(v, w)))


def add_dihedral_restraints(omm_system, s, mode: str, strength_kj: float):
    """Restrain phi/psi of ``s`` (``mode`` "bb" or "ss") to their values in
    its positions; returns (records, description)."""
    import openmm as mm

    torsions = backbone_torsions(s)
    if mode == "ss":
        from ..secondary import dssp

        codes = dssp(s, simplified=True)[0]
        chosen = {r for r, code in enumerate(codes.tolist()) if code in ("H", "E")}
        torsions = [t for t in torsions if t[0] in chosen]
        description = f"{len(chosen)} residues in helices and sheets"
    else:
        description = f"{len({t[0] for t in torsions})} protein residues"
    if not torsions:
        raise ValueError(
            f"dihedral_restraint = '{mode}' found no backbone torsions ({description})"
        )
    force = mm.PeriodicTorsionForce()
    force.setName("DihedralRestraint")
    pos = s.positions
    chains = s.chains["name"]
    records = []
    for r, kind, atoms in torsions:
        ref = dihedral(pos, atoms)
        for n, k in fourier_terms(strength_kj):
            force.addTorsion(*atoms, n, (n * (ref + math.pi)) % (2 * math.pi), k)
        records.append(
            {
                "angle": kind,
                "chain_id": str(chains[s.residues["chain"][r]]),
                "residue_name": str(s.residues["name"][r]),
                "residue_id": int(s.residues["resid"][r]),
                "atom_indices": " ".join(map(str, atoms)),
                "atom_names": " ".join(str(s.atoms["name"][a]) for a in atoms),
                "reference_degrees": round(math.degrees(ref), 3),
            }
        )
    omm_system.addForce(force)
    return records, description


def write_records(path, records) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def plot_well(path, strength_kj: float) -> bool:
    """Plot the well against the equivalent harmonic; False without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    x = np.linspace(-math.pi, math.pi, 721)
    well = sum(k * (1 + np.cos(n * (x - math.pi))) for n, k in fourier_terms(strength_kj))
    well -= well.min()
    harmonic = 0.5 * _curvature(strength_kj) * x**2
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.degrees(x), well, label="restraint")
    ax.plot(np.degrees(x), harmonic, "--", label="harmonic, same curvature")
    ax.set_xlabel("deviation from reference (degrees)")
    ax.set_ylabel("energy (kJ/mol)")
    ax.set_ylim(0, max(float(well.max()) * 1.2, 1e-9))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _curvature(strength_kj: float) -> float:
    """d2V/dtheta2 at the reference: sum of k n^2 cos(n pi) terms."""
    return sum(-k * n * n * math.cos(n * math.pi) for n, k in fourier_terms(strength_kj))
