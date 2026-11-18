"""OpenFF (SMIRNOFF) templates for ligands, through openmmforcefields.

The other ligand force field, next to GAFF2 (:mod:`boonza.gaff`): each
molecule the force fields cannot parameterize is typed by an OpenFF force
field such as Sage (``openff-2.2.1``) with openmmforcefields'
``SMIRNOFFTemplateGenerator``, and :func:`openff_patch` turns what it makes
into a viparr patch, merged onto an Amber protein force field like a GAFF2
patch is.

Charges are AM1-BCC from AmberTools, computed as boonza's GAFF2 path computes
them (so ``charge_geometry`` means the same) and handed to OpenFF, which
would otherwise compute them itself.  The valence terms and Lennard-Jones are
OpenFF's to the last digit; the 1-4 electrostatics take the host force
field's scale (viparr's Amber files say 0.8333 where Sage says 0.833333),
which is a few thousandths of a kcal/mol in a drug-sized molecule.

Only free molecules of one residue: a ligand bonded to the protein, a
non-standard residue or a pseudopeptide mixes protein types with the ligand's,
which is built on GAFF2.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .system import System
from .viparr import Rules, ViparrError, ViparrForcefield, patch_from_system

__all__ = ["OpenFFError", "find_openff", "openff_patch"]

#: the OpenFF force field taken when none is named: Sage
DEFAULT = "openff-2.2.1"
_NAME = re.compile(r"openff-\d+\.\d+\.\d+")
_PLUGINS = ["exclusions", "mass", "bonds", "angles", "vdw1", "propers", "impropers"]


class OpenFFError(RuntimeError):
    """openmmforcefields or the OpenFF toolkit is missing, or lacks a force field."""


def find_openff(ligandff: str = DEFAULT):
    """openmmforcefields' ``SMIRNOFFTemplateGenerator``, checked to have ``ligandff``."""
    if not _NAME.fullmatch(str(ligandff)):
        raise ValueError(f"an OpenFF force field is named like {DEFAULT}, not {ligandff!r}")
    try:
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    except ImportError as e:
        raise OpenFFError(
            "openmmforcefields (with the OpenFF toolkit) was not found: install it with "
            "'conda install -c conda-forge openmmforcefields' into this environment "
            f"({e})"
        ) from e
    have = [f for f in SMIRNOFFTemplateGenerator.INSTALLED_FORCEFIELDS if _NAME.fullmatch(f)]
    if ligandff not in have:
        raise OpenFFError(f"this openmmforcefields has no {ligandff}: it has "
                          f"{', '.join(sorted(have)) or 'no OpenFF force field'}")  # fmt: skip
    return SMIRNOFFTemplateGenerator


def openff_patch(system: System, forcefields=(), *, ligandff: str = DEFAULT, charges=None,
                 charge_method: str = "bcc", charge_geometry: str = "optimized",
                 amberhome=None, path=None, tag=None) -> ViparrForcefield:  # fmt: skip
    """A viparr patch with OpenFF templates for what ``forcefields`` cannot parameterize.

    Used like :func:`boonza.gaff2_patch`, and merged the same way::

        patch = boonza.openff_patch(system, ["aa.amber.ff19SB", "water.tip3p"])
        ff = boonza.merge_forcefields("aa.amber.ff19SB", patch)
        out = boonza.parameterize(system, [ff, "water.tip3p"])

    Each group from :func:`boonza.find_unmatched` must be one free molecule in
    one residue; anything bonded to the protein, and any amino acid, raises
    and points at GAFF2.  ``ligandff`` is an OpenFF force field
    openmmforcefields has (``openff-2.2.1``, Sage).  ``charges`` maps residue
    names to formal charges where the input has none; ``charge_method`` and
    ``charge_geometry`` are :func:`boonza.gaff.run_gaff2`'s, which computes
    the AM1-BCC charges.  ``tag`` makes the types (``C1~<tag>``) and template
    names of this patch unique, so patches made separately can join one force
    field.  With no force fields, the patch is a complete force field.
    """
    from . import gaff

    ffs = gaff._load(forcefields, path)
    b = gaff._Builder(system, ffs, charges or {}, {}, None, None)
    groups = b.groups()
    for residues in groups:  # what this cannot do, before anything is installed or run
        label = "+".join(f"{b.resnames[r] or 'LIG'}{b.resid[r]}" for r in residues)
        atoms = [a for r in residues for a in b.res_atoms[r]]
        inside = set(atoms)
        if any(b._amino(r) for r in residues) or any(
                x not in inside and b.anum[x] > 0 for a in atoms for x in b.nbrs[a]):  # fmt: skip
            raise ViparrError(
                f"{label} is an amino acid or is bonded to one: OpenFF templates are for "
                "free ligands, and a covalent adduct or a non-standard residue takes GAFF2 "
                "(ligandff = \"gaff-2.11\")")  # fmt: skip
        if len(residues) != 1:
            raise ViparrError(f"{label} is one molecule in {len(residues)} residues: an OpenFF "
                              "template is one residue (GAFF2 takes it as it is)")  # fmt: skip
    generator = find_openff(ligandff) if groups else None
    templates, params = [], {}
    made = 0
    for residues in groups:
        r = residues[0]
        atoms = b.res_atoms[r]
        so_far = ViparrForcefield("made", Rules(), templates, {})
        if templates and b.P._find_uncached(so_far, atoms)[0] is not None:
            continue  # another copy of a molecule already made
        made += 1
        frag, f = b._capped(atoms, residues)
        q = b.charges.get(b.resnames[r])
        net = int(q) if q is not None else int(f.atoms["formal_charge"].sum())
        am1 = gaff.run_gaff2(f, net, charge_method=charge_method,
                             charge_geometry=charge_geometry, amberhome=amberhome)  # fmt: skip
        charge = np.asarray(am1.atoms["charge"], float)
        charge = charge + (net - charge.sum()) / len(charge)  # antechamber's rounding
        name = f"{b.resnames[r] or 'LIG'}_openff"
        if tag is not None:
            name = f"{tag}_{name}"
        taken = {t.name for t in templates} | (
            {t.name for t in b.host.templates} if b.host else set())  # fmt: skip
        base, n = name, 1
        while name in taken:
            n += 1
            name = f"{base}{n}"
        types = str(made) if tag is None else f"{tag}" + (f".{made}" if made > 1 else "")
        one = patch_from_system(_typed(f, charge, generator, ligandff), name, types,
                                rules=b.host.rules if b.host else None)  # fmt: skip
        templates += one.templates
        for table, rows in one.params.items():
            params.setdefault(table, []).extend(rows)
    if b.host is None:
        rules = Rules(info=[f"OpenFF {ligandff} (openmmforcefields)"],
                      vdw_func="lj12_6_sig_epsilon", vdw_comb_rule="arithmetic/geometric",
                      es_scale=[0.0, 0.0, 1 / 1.2], lj_scale=[0.0, 0.0, 0.5],
                      plugins=list(_PLUGINS))  # fmt: skip
    else:
        gaff._check_host(b.host)
        rules = Rules(es_scale=[], lj_scale=[])
    out = ViparrForcefield("openff-patch", rules, templates, params)
    out.parents, out.units, out.drawings = [], [], []  # as a GAFF2 patch has them
    out.ligandff = ligandff
    return out


def _typed(f: System, charge, generator, ligandff: str) -> System:
    """Molecule ``f`` (all hydrogens, bond orders) as ``ligandff`` parameterizes it
    with the partial charges ``charge``: a system with its force-field tables."""
    import openmm.app as app
    import openmm.unit as u
    from openff.toolkit import Molecule
    from openff.units import unit as offunit

    from .chem import to_rdkit
    from .omm import from_openmm

    mol = Molecule.from_rdkit(to_rdkit(f, implicit_hydrogens=False),
                              allow_undefined_stereo=True, hydrogens_are_explicit=True)  # fmt: skip
    if [a.atomic_number for a in mol.atoms] != f.atoms["anum"].tolist():
        raise OpenFFError("the OpenFF toolkit changed the atoms of the molecule")
    mol.partial_charges = np.asarray(charge, float) * offunit.elementary_charge
    ff = app.ForceField()
    ff.registerTemplateGenerator(generator(molecules=[mol], forcefield=ligandff).generator)
    topology = mol.to_topology().to_openmm()
    try:
        omm = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None,
                              rigidWater=False, removeCMMotion=False)  # fmt: skip
    except Exception as e:
        raise OpenFFError(f"{ligandff} could not parameterize the molecule: {e}") from e
    out = from_openmm(topology, omm, np.asarray(f.positions, float) / 10.0 * u.nanometer)
    if not np.allclose(out.atoms["charge"], charge, atol=1e-5):
        raise OpenFFError("openmmforcefields did not keep the AM1-BCC charges it was given")
    return out


def patch_directory(workdir) -> Path:
    """Where a run keeps its OpenFF templates."""
    return Path(workdir) / "openff_patch"
