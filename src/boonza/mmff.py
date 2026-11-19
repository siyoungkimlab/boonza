"""MMFF94 templates for ligands, from RDKit.

A third ligand force field, next to GAFF2 (:mod:`boonza.gaff`) and OpenFF
(:mod:`boonza.openff`): RDKit types each molecule the force fields cannot
parameterize with MMFF94 (or MMFF94s), and :func:`mmff_patch` turns its
parameters into a viparr patch, merged onto an Amber protein force field like
a GAFF2 patch is.  It needs RDKit alone -- no AmberTools, no quantum
calculation -- and takes a few milliseconds a molecule.

Inside a ligand the energy is MMFF94's own, every term, and equals RDKit's:
quartic bond stretches, cubic angle bends (and the cosine form of angles that
are linear at rest), stretch-bend coupling, Wilson out-of-plane bends, three-
term torsions, and between atoms three or more bonds apart the buffered 14-7
van der Waals and buffered Coulomb interactions (1-4 Coulomb at 0.75).  The
harmonic part of a bond or an angle is an ordinary ``stretch_harm`` or
``angle_harm`` term, so hydrogen bonds are constrained as anywhere else; the
rest are the tables :data:`TABLES`.

**Between a ligand and everything else there is no MMFF94**: the protein,
water and ions are Amber's, whose atoms have no 14-7 parameters, and MMFF94
has no rule for meeting them.  boonza's rule, which is its own and nobody's
published force field: plain Coulomb with the MMFF94 charges, and 12-6
Lennard-Jones with each ligand atom's own MMFF94 minimum (``R*``) and well
depth, combined with the other atom's as Amber combines.  A hydrogen that
MMFF94 counts a hydrogen-bond donor gets no Lennard-Jones at all, as a water
or hydroxyl hydrogen has none in Amber: MMFF94 itself shrinks a donor-acceptor
contact to 0.8 of its size, and a 12-6 wall at the unshrunk size would hold
hydrogen bonds half an ångström apart.  Two copies of a ligand meet by the same
rule.  So conformational energies of the ligand are MMFF94's; binding
energies are not anybody's validated model.

Only free molecules of one residue, as with OpenFF: a ligand bonded to the
protein takes GAFF2 (:func:`boonza.openff.ligand_patch` gives both).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

from .system import System
from .viparr import ParamRow, Rules, Template, ViparrError, ViparrForcefield

__all__ = ["VARIANTS", "MMFFError", "mmff_patch"]

#: the MMFF94 force fields RDKit has: "mmff94", and "mmff94s", whose out-of-plane
#: and torsion terms keep delocalized trigonal nitrogens flat (made for minimized
#: geometries; MMFF94 is the one made for dynamics)
VARIANTS = ("mmff94", "mmff94s")
DEFAULT = "mmff94"
#: the tables an MMFF94 ligand has beyond stretch_harm, angle_harm, dihedral_trig
TABLES = ("stretch_anharm", "angle_anharm", "angle_linear", "stretchbend_harm",
          "wilson_harm", "pair_buf14_7_es")  # fmt: skip

# RDKit's constants (Code/ForceField/MMFF), so that energies equal its own
_MDYNE = 143.9325  # mdyn/Å to kcal/(mol Å^2)
_CS = -2.0  # cubic stretch constant, 1/Å
_CB = -0.006981317 * 180.0 / math.pi  # cubic bend constant, 1/rad
_SB = 2.51210 * 180.0 / math.pi  # stretch-bend, to kcal/(mol Å rad)
_OOP = 0.043844 / 2 * (180.0 / math.pi) ** 2  # out-of-plane, to kcal/(mol rad^2)
_ES14 = 0.75
# atom types whose angles are linear at rest (MMFFPROP.PAR's "lin"): CSP, =N=, NR%
_LINEAR = (4, 53, 61)
_TORSION0 = {"phi0": 0.0, **{f"fc{n}": 0.0 for n in range(7)}}


class MMFFError(ViparrError):
    """MMFF94 has no parameters for a molecule, or RDKit could not read it."""


def is_mmff(ligandff) -> bool:
    return str(ligandff) in VARIANTS


def check(ligandff, charge_model: str = "auto", pseudopeptide: str = "off") -> None:
    """Refuse what does not go with an MMFF94 ligand: charges other than its own,
    and a pseudopeptide, whose protein types are GAFF2's to mix."""
    if not is_mmff(ligandff):
        raise ValueError(f"an MMFF94 force field is one of {VARIANTS}, not {ligandff!r}")
    if charge_model not in (None, "auto"):
        raise ValueError(f"charge_model = '{charge_model}' does not go with ligandff = "
                         f"'{ligandff}': MMFF94 has its own charges, and its energy is "
                         "fitted with them")  # fmt: skip
    if (pseudopeptide or "off") != "off":
        raise ValueError(f"pseudopeptide = '{pseudopeptide}' mixes protein types with GAFF2's: "
                         f"it goes with ligandff = 'gaff-2.11', not {ligandff}")  # fmt: skip


def mmff_patch(system: System, forcefields=(), *, ligandff: str = DEFAULT, charges=None,
               path=None, tag=None, bound: str = "raise") -> ViparrForcefield:  # fmt: skip
    """A viparr patch with MMFF94 templates for what ``forcefields`` cannot parameterize.

    Used like :func:`boonza.gaff2_patch`, and merged the same way::

        patch = boonza.mmff_patch(system, ["aa.amber.ff19SB", "water.tip3p"])
        ff = boonza.merge_forcefields("aa.amber.ff19SB", patch)
        out = boonza.parameterize(system, [ff, "water.tip3p"])

    Each group from :func:`boonza.find_unmatched` must be one free molecule in
    one residue; anything bonded to the protein, and any amino acid, raises
    and points at GAFF2 -- or, with ``bound="skip"``, is left without a
    template for GAFF2 to make.  ``ligandff`` is "mmff94" or "mmff94s"
    (:data:`VARIANTS`).  The partial charges are MMFF94's, from the formal
    charges and bond orders of the input: ``charges`` (residue name to net
    charge) can only confirm them.  ``tag`` makes the types (``C1~<tag>``)
    and template names of this patch unique, so patches made separately can
    join one force field.  With no force fields, the patch is a complete
    force field.

    What the energy is, and what it is not, is in :mod:`boonza.mmff`.
    """
    from . import gaff
    from .openff import free_groups, new_names, used_tags

    if bound not in ("raise", "skip"):
        raise ValueError('bound is "raise" or "skip"')
    check(ligandff)
    ffs = gaff._load(forcefields, path)
    b = gaff._Builder(system, ffs, charges or {}, {}, None, None)
    groups, skipped = free_groups(b, bound, "MMFF94")
    used = used_tags(b)
    templates, params = [], {}
    made = 0
    for residues in groups:
        r = residues[0]
        atoms = b.res_atoms[r]
        so_far = ViparrForcefield("made", Rules(), templates, {})
        if templates and b.P._find_uncached(so_far, atoms)[0] is not None:
            continue  # another copy of a molecule already made
        made += 1
        _, f = b._capped(atoms, residues)
        label = f"{b.resnames[r] or 'LIG'}{b.resid[r]}"
        name, types = new_names(b, templates, used, r, ligandff, tag, made)
        tpl, rows = _template(f, name, types, ligandff, label, b.charges.get(b.resnames[r]))
        templates.append(tpl)
        for table, new in rows.items():
            params.setdefault(table, []).extend(new)
    if b.host is None:
        rules = Rules(info=[f"{_title(ligandff)} (RDKit)"],
                      vdw_func="lj12_6_sig_epsilon", vdw_comb_rule="arithmetic/geometric",
                      es_scale=[0.0, 0.0, 1 / 1.2], lj_scale=[0.0, 0.0, 0.5],
                      plugins=["exclusions", "mass", "bonds", "angles", "vdw1",
                               *(["propers"] if params.get("dihedral_trig") else [])])  # fmt: skip
    else:
        gaff._check_host(b.host)
        rules = Rules(es_scale=[], lj_scale=[])
    out = ViparrForcefield("mmff-patch", rules, templates, params)
    out.parents, out.units, out.drawings = [], [], []  # as a GAFF2 patch has them
    out.ligandff, out.skipped, out.charge_model = ligandff, skipped, "mmff94"
    return out


def _title(ligandff: str) -> str:
    return str(ligandff).upper().replace("94S", "94s")


def _properties(f: System, ligandff: str, label: str):
    """``f`` (all hydrogens, bond orders) as an RDKit molecule with its MMFF94
    properties, atoms in order."""
    from rdkit.Chem import AllChem

    from .chem import to_rdkit

    try:
        mol = to_rdkit(f, implicit_hydrogens=False)
    except Exception as e:
        raise MMFFError(f"{label}: RDKit could not read the molecule ({e})") from e
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != f.atoms["anum"].tolist():
        raise MMFFError(f"{label}: RDKit changed the atoms of the molecule")
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise MMFFError(f"{label}: MMFF94 has no atom types for this molecule (it covers "
                        "organic molecules: no metals, no boron or silicon chemistry beyond "
                        "its own); GAFF2 or OpenFF may take it")  # fmt: skip
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=_title(ligandff))
    if props is None:
        raise MMFFError(f"{label}: RDKit could not set up MMFF94 for the molecule")
    return mol, props


def _donor_hydrogens(mol, ligandff: str) -> set[int]:
    """The hydrogens MMFF94 counts hydrogen-bond donors: those whose contact with a
    water oxygen (an acceptor) it shrinks.  Asked of RDKit, which keeps the table."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    n = mol.GetNumAtoms()
    hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    if not hydrogens:
        return set()
    both = Chem.CombineMols(mol, Chem.AddHs(Chem.MolFromSmiles("O")))
    Chem.SanitizeMol(both)
    props = AllChem.MMFFGetMoleculeProperties(both, mmffVariant=_title(ligandff))
    out = set()
    for h in hydrogens:
        v = props.GetMMFFVdWParams(h, n)  # n: the water's oxygen
        if abs(v[0] - v[2]) > 1e-9:
            out.add(h)
    return out


def _template(f: System, name: str, tag: str, ligandff: str, label: str, net):
    """The template and parameter rows of one molecule, a type per atom."""
    from rdkit import Chem

    from .elements import MSYS_MASSES, symbol

    mol, props = _properties(f, ligandff, label)
    n = mol.GetNumAtoms()
    anum = f.atoms["anum"].tolist()
    charge = [float(props.GetMMFFPartialCharge(i)) for i in range(n)]
    total = round(sum(charge))
    if net is not None and int(net) != total:
        raise MMFFError(f"{label}: its formal charges sum to {total:+d}, not the {int(net):+d} "
                        "asked for: MMFF94 charges follow the formal charges, so give them on "
                        "the atoms")  # fmt: skip
    count: dict[str, int] = {}
    types, names = [], []
    for z in anum:
        sym = symbol(z)
        count[sym] = count.get(sym, 0) + 1
        types.append(f"{sym}{count[sym]}~{tag}")
        names.append(f"{sym}{count[sym]}")
    nbrs = [[x.GetIdx() for x in a.GetNeighbors()] for a in mol.GetAtoms()]
    bonds = sorted(tuple(sorted((x.GetBeginAtomIdx(), x.GetEndAtomIdx())))
                   for x in mol.GetBonds())  # fmt: skip
    linear = {i for i in range(n) if props.GetMMFFAtomType(i) in _LINEAR}
    params: dict[str, list[ParamRow]] = {}
    memo = f"{_title(ligandff)} for {name}"

    def add(table, atoms, row):
        params.setdefault(table, []).append(ParamRow(" ".join(types[a] for a in atoms), row, memo))

    def missing(what, atoms):
        return MMFFError(f"{label}: MMFF94 has no {what} parameters for "
                         f"{'-'.join(names[a] for a in atoms)}")  # fmt: skip

    r0: dict[tuple[int, int], float] = {}
    for i, j in bonds:
        p = props.GetMMFFBondStretchParams(mol, i, j)
        if p is None:
            raise missing("bond", (i, j))
        fc = _MDYNE / 2 * p[1]
        r0[(i, j)] = r0[(j, i)] = p[2]
        add("stretch_harm", (i, j), {"r0": p[2], "fc": fc})
        add("stretch_anharm", (i, j), {"r0": p[2], "fc3": fc * _CS,
                                       "fc4": fc * 7 / 12 * _CS * _CS})  # fmt: skip
    for j in range(n):
        for i, k in itertools.combinations(nbrs[j], 2):
            p = props.GetMMFFAngleBendParams(mol, i, j, k)
            if p is None:
                raise missing("angle", (i, j, k))
            if j in linear:
                add("angle_harm", (i, j, k), {"theta0": 180.0, "fc": 0.0})
                add("angle_linear", (i, j, k), {"fc": _MDYNE * p[1]})
                continue
            fc = _MDYNE / 2 * p[1]
            add("angle_harm", (i, j, k), {"theta0": p[2], "fc": fc})
            add("angle_anharm", (i, j, k), {"theta0": p[2], "fc3": fc * _CB})
            sb = props.GetMMFFStretchBendParams(mol, i, j, k)
            if sb is not None and (sb[1] or sb[2]):
                add("stretchbend_harm", (i, j, k),
                    {"r1": r0[(i, j)], "r2": r0[(k, j)], "theta0": p[2],
                     "fc1": _SB * sb[1], "fc2": _SB * sb[2]})  # fmt: skip
        if len(nbrs[j]) == 3:
            i, k, m = nbrs[j]
            koop = props.GetMMFFOopBendParams(mol, i, j, k, m)
            if koop:  # one constant, three angles: each neighbour out of the others' plane
                for a, c, o in ((i, k, m), (i, m, k), (k, m, i)):
                    add("wilson_harm", (j, a, c, o), {"fc": _OOP * koop})
    for j, k in bonds:  # every proper dihedral needs a row, zero where MMFF94 has none
        for i in nbrs[j]:
            for m in nbrs[k]:
                if i in (j, k) or m in (j, k) or i == m:
                    continue
                p = props.GetMMFFTorsionParams(mol, i, j, k, m)
                row = dict(_TORSION0)
                if p is not None:
                    v1, v2, v3 = p[1], p[2], p[3]
                    row.update(fc0=0.5 * (v1 + v2 + v3), fc1=0.5 * v1, fc2=-0.5 * v2,
                               fc3=0.5 * v3)  # fmt: skip
                add("dihedral_trig", (i, j, k, m), row)
    donors = _donor_hydrogens(mol, ligandff)
    mass = MSYS_MASSES[f.atoms["anum"]].tolist()
    for a in range(n):
        v = props.GetMMFFVdWParams(a, a)
        add("vdw1", (a,), {"sigma": v[0] / 2 ** (1 / 6), "epsilon": 0.0 if a in donors else v[1]})
        add("mass", (a,), {"amu": float(mass[a])})
    apart = Chem.GetDistanceMatrix(mol)
    for i in range(n):
        for j in range(i + 1, n):
            if apart[i, j] < 3:
                continue
            v = props.GetMMFFVdWParams(i, j)
            scale = _ES14 if apart[i, j] == 3 else 1.0
            add("pair_buf14_7_es", (i, j), {"rij": v[2], "eij": v[3],
                                            "qij": scale * charge[i] * charge[j]})  # fmt: skip
    tpl = Template(name, names, anum, charge, list(types), list(types), [""] * n, bonds)
    return tpl, params


def patch_directory(workdir) -> Path:
    """Where a run keeps its MMFF94 templates."""
    return Path(workdir) / "mmff_patch"
