"""OpenFF (SMIRNOFF) templates for ligands, through openmmforcefields.

The other ligand force field, next to GAFF2 (:mod:`boonza.gaff`): each
molecule the force fields cannot parameterize is typed by an OpenFF force
field such as Sage (``openff-2.2.1``) with openmmforcefields'
``SMIRNOFFTemplateGenerator``, and :func:`openff_patch` turns what it makes
into a viparr patch, merged onto an Amber protein force field like a GAFF2
patch is.

Charges are the force field's own kind (:data:`CHARGE_MODELS`).  A release
that specifies NAGL charges (``openff-2.3.0``) gets them from openff-nagl, a
graph network trained on AM1-BCC: no quantum calculation, half a second for
a molecule of five hundred atoms, and the same in every pose.  The earlier
releases specify AM1-BCC, which AmberTools computes as boonza's GAFF2 path
does (so ``charge_geometry`` means the same).  Either can be asked for by
name.  The valence terms and Lennard-Jones are
OpenFF's to the last digit; the 1-4 electrostatics take the host force
field's scale (viparr's Amber files say 0.8333 where Sage says 0.833333),
which is a few thousandths of a kcal/mol in a drug-sized molecule.

Only free molecules of one residue: a ligand bonded to the protein or a
non-standard residue mixes protein types with the ligand's, which is built on
GAFF2, and :func:`ligand_patch` gives those GAFF2 templates beside the OpenFF
ones.  A pseudopeptide is refused: it is a free ligand, and GAFF2's to mix.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .system import System
from .viparr import Rules, ViparrError, ViparrForcefield, patch_from_system

__all__ = ["CHARGE_MODELS", "OpenFFError", "find_openff", "ligand_patch", "openff_patch"]

#: the OpenFF force field taken when none is named: Sage
DEFAULT = "openff-2.2.1"
#: where a ligand's partial charges come from: "auto", what its force field
#: specifies (NAGL for openff-2.3.0, AM1-BCC before it, and always for GAFF2);
#: "am1bcc", AmberTools' sqm; "nagl", openff-nagl's graph network
CHARGE_MODELS = ("auto", "am1bcc", "nagl")
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


def nagl_model(ligandff: str = DEFAULT) -> str | None:
    """The NAGL model ``ligandff`` specifies for its charges, or None where it
    specifies AM1-BCC."""
    find_openff(ligandff)
    from openff.toolkit import ForceField

    handlers = ForceField(f"{ligandff}.offxml")._parameter_handlers
    return str(handlers["NAGLCharges"].model_file) if "NAGLCharges" in handlers else None


def whole_molecule(ligandff) -> bool:
    """Whether ``ligandff`` types a molecule whole, as OpenFF and MMFF94 do: its
    templates are for free ligands, and :func:`ligand_patch` makes them."""
    from . import mmff

    return str(ligandff).startswith("openff") or mmff.is_mmff(ligandff)


def describe_ff(ligandff) -> str:
    """The ligand force field and where it comes from, for a log."""
    from . import mmff

    if mmff.is_mmff(ligandff):
        return f"{mmff._title(ligandff)} (RDKit)"
    if str(ligandff).startswith("openff"):
        return f"OpenFF {ligandff} (openmmforcefields)"
    return f"GAFF2 {ligandff}"


def check_charge_model(ligandff: str, model: str = "auto") -> None:
    """Refuse a charge model the ligand force field does not have: GAFF2 has
    AM1-BCC alone, MMFF94 its own.  Nothing is imported, so it can be asked anywhere."""
    from . import mmff

    if model not in CHARGE_MODELS:
        raise ValueError(f"charge_model must be one of {CHARGE_MODELS}")
    if mmff.is_mmff(ligandff):
        mmff.check(ligandff, model)
        return
    if model == "nagl" and not str(ligandff).startswith("openff"):
        raise ValueError(f"charge_model = 'nagl' goes with an OpenFF ligandff, not {ligandff}")


def charge_model(ligandff: str, model: str = "auto") -> str:
    """ "am1bcc" or "nagl": ``model``, or for "auto" what ``ligandff`` specifies."""
    from . import mmff

    check_charge_model(ligandff, model)
    if mmff.is_mmff(ligandff):
        return "mmff94"
    if not str(ligandff).startswith("openff"):
        return "am1bcc"
    if model != "auto":
        return model
    return "nagl" if nagl_model(ligandff) else "am1bcc"


def describe_charges(ligandff: str, model: str = "auto", geometry: str = "optimized") -> str:
    """The charges a run's ligands get, for its log."""
    from . import mmff

    check_charge_model(ligandff, model)
    if mmff.is_mmff(ligandff):
        return "MMFF94 charges"
    try:
        model = charge_model(ligandff, model)
    except OpenFFError:  # not installed: the build itself will say so
        return "the charges its force field specifies"
    if model == "nagl":
        return "NAGL charges"
    return f"AM1-BCC charges, {geometry} geometry"


def _nagl_charges(mol, ligandff: str) -> np.ndarray:
    """Partial charges of an OpenFF molecule from openff-nagl: the model the force
    field names, or the newest production AM1-BCC model where it names none."""
    try:
        from openff.nagl_models import get_models_by_type
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

        wrapper = NAGLToolkitWrapper()
    except Exception as e:
        raise OpenFFError(
            "openff-nagl was not found, and NAGL charges need it: install it with 'conda "
            "install -c conda-forge openff-nagl openff-nagl-models', or ask for charge_model "
            f"= \"am1bcc\" ({e})") from e  # fmt: skip
    model = nagl_model(ligandff)
    if model is None:
        model = Path(get_models_by_type("am1bcc", production_only=True)[-1]).name
    mol.assign_partial_charges(model, toolkit_registry=wrapper)
    return np.asarray(mol.partial_charges.m_as("elementary_charge"), float)


def openff_patch(system: System, forcefields=(), *, ligandff: str = DEFAULT, charges=None,
                 charge_model: str = "auto", charge_method: str = "bcc",
                 charge_geometry: str = "optimized", amberhome=None, path=None, tag=None,
                 bound: str = "raise") -> ViparrForcefield:  # fmt: skip
    """A viparr patch with OpenFF templates for what ``forcefields`` cannot parameterize.

    Used like :func:`boonza.gaff2_patch`, and merged the same way::

        patch = boonza.openff_patch(system, ["aa.amber.ff19SB", "water.tip3p"])
        ff = boonza.merge_forcefields("aa.amber.ff19SB", patch)
        out = boonza.parameterize(system, [ff, "water.tip3p"])

    Each group from :func:`boonza.find_unmatched` must be one free molecule in
    one residue; anything bonded to the protein, and any amino acid, raises
    and points at GAFF2 -- or, with ``bound="skip"``, is left without a
    template for GAFF2 to make (:func:`ligand_patch` does both).  ``ligandff``
    is an OpenFF force field openmmforcefields has (``openff-2.2.1``, Sage).
    ``charges`` maps residue names to formal charges where the input has none.
    ``charge_model`` (:data:`CHARGE_MODELS`) says where partial charges come
    from: by default what the force field specifies, NAGL for openff-2.3.0
    and AM1-BCC before it.  ``charge_method`` and ``charge_geometry`` are
    :func:`boonza.gaff.run_gaff2`'s, which computes AM1-BCC charges; NAGL
    charges need neither, and no AmberTools.  ``tag`` makes the types (``C1~<tag>``) and template
    names of this patch unique, so patches made separately can join one force
    field.  With no force fields, the patch is a complete force field.
    """
    from . import gaff

    if bound not in ("raise", "skip"):
        raise ValueError('bound is "raise" or "skip"')

    ffs = gaff._load(forcefields, path)
    b = gaff._Builder(system, ffs, charges or {}, {}, None, None)
    groups, skipped = free_groups(b, bound, "OpenFF")
    used = used_tags(b)
    generator = find_openff(ligandff) if groups else None
    model = globals()["charge_model"](ligandff, charge_model) if groups else "am1bcc"
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
        if model == "nagl":
            charge = _nagl_charges(_molecule(f, net), ligandff)
        else:
            am1 = gaff.run_gaff2(f, net, charge_method=charge_method,
                                 charge_geometry=charge_geometry, amberhome=amberhome)  # fmt: skip
            charge = np.asarray(am1.atoms["charge"], float)
        charge = charge + (net - charge.sum()) / len(charge)  # rounding, to the charge
        name, types = new_names(b, templates, used, r, "openff", tag, made)
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
    out.ligandff, out.skipped, out.charge_model = ligandff, skipped, model
    return out


def free_groups(b, bound: str, what: str) -> tuple[list, list[str]]:
    """The groups of ``b`` (a GAFF2 builder) that are one free molecule in one residue,
    and the labels of the others, which raise unless ``bound`` is "skip": templates
    made from a whole molecule (``what``: OpenFF, MMFF94) are not for those."""
    groups, skipped = [], []
    for residues in b.groups():  # what this cannot do, before anything is installed or run
        label = "+".join(f"{b.resnames[r] or 'LIG'}{b.resid[r]}" for r in residues)
        atoms = [a for r in residues for a in b.res_atoms[r]]
        inside = set(atoms)
        if any(b._amino(r) for r in residues) or any(
                x not in inside and b.anum[x] > 0 for a in atoms for x in b.nbrs[a]):  # fmt: skip
            if bound == "skip":
                skipped.append(label)
                continue
            raise ViparrError(
                f"{label} is an amino acid or is bonded to one: {what} templates are for "
                "free ligands, and a covalent adduct or a non-standard residue takes GAFF2 "
                "(ligandff = \"gaff-2.11\")")  # fmt: skip
        if len(residues) != 1:
            if bound == "skip":
                skipped.append(label)
                continue
            raise ViparrError(f"{label} is one molecule in {len(residues)} residues: "
                              f"{'an' if what[0] in 'AEIOU' else 'a'} {what} "
                              "template is one residue (GAFF2 takes it as it is)")  # fmt: skip
        groups.append(residues)
    return groups, skipped


def used_tags(b) -> set[str]:
    """Type tags the templates of the force field a patch joins already have: a patch
    made later (a second boonza fep prep) must not name its atoms as an earlier one
    named its own."""
    return {t.partition("~")[2] for tp in (b.host.templates if b.host else ())
            for t in tp.btype if "~" in t}  # fmt: skip


def new_names(b, templates, used: set, r: int, suffix: str, tag, made: int) -> tuple[str, str]:
    """A template name (``<residue>_<suffix>``) and a type tag for residue ``r`` that
    nothing made so far or in the host force field has; the tag joins ``used``."""
    name = f"{b.resnames[r] or 'LIG'}_{suffix}"
    if tag is not None:
        name = f"{tag}_{name}"
    taken = {t.name for t in templates} | (
        {t.name for t in b.host.templates} if b.host else set())  # fmt: skip
    base, n = name, 1
    while name in taken:
        n += 1
        name = f"{base}{n}"
    k, types = made, None
    while types is None or types in used:
        types = str(k) if tag is None else f"{tag}" + (f".{k}" if k > 1 else "")
        k += 1
    used.add(types)
    return name, types


def ligand_patch(system: System, forcefields, *, ligandff: str = DEFAULT, charges=None,
                 charge_model: str = "auto", charge_geometry: str = "optimized", tag=None,
                 path=None, log=None, **gaff2) -> ViparrForcefield:  # fmt: skip
    """One patch for everything ``forcefields`` cannot parameterize, as boonza md,
    swim and fep make it: OpenFF templates (MMFF94 ones for ``ligandff`` "mmff94" or
    "mmff94s", :func:`boonza.mmff_patch`) for the free ligands, and GAFF2 templates
    (``gaff2``: :func:`boonza.gaff2_patch`'s other arguments) for what those are not
    for -- a ligand bonded to the protein, a non-standard residue -- which ``log`` is
    told of.  ``forcefields`` is a list; the patch joins its Amber protein force field."""
    from . import gaff, mmff
    from .viparr import merge_forcefields

    ffs = gaff._load(forcefields, path)
    if mmff.is_mmff(ligandff):
        mmff.check(ligandff, charge_model)
        patch = mmff.mmff_patch(system, ffs, ligandff=ligandff, charges=charges, tag=tag,
                                bound="skip")  # fmt: skip
    else:
        patch = openff_patch(system, ffs, ligandff=ligandff, charges=charges,
                             charge_model=charge_model, charge_geometry=charge_geometry,
                             tag=tag, bound="skip")  # fmt: skip
    if not patch.skipped:
        return patch
    if log is not None:
        log(f"GAFF2 templates for {', '.join(patch.skipped)}: bonded to the protein, or an "
            f"amino acid, which {mmff._title(ligandff) if mmff.is_mmff(ligandff) else ligandff} "
            "templates are not for")  # fmt: skip
    host = gaff.host_index(ffs)
    joined = list(ffs)
    if patch.templates:
        joined[host] = merge_forcefields(ffs[host], patch)
    rest = gaff.gaff2_patch(system, joined, charges=charges, charge_geometry=charge_geometry,
                            tag=tag, **gaff2)  # fmt: skip
    params = {t: list(rows) for t, rows in patch.params.items()}
    for table, rows in rest.params.items():
        params.setdefault(table, []).extend(rows)
    out = ViparrForcefield(patch.name, rest.rules if not ffs else patch.rules,
                           [*patch.templates, *rest.templates], params)  # fmt: skip
    out.parents, out.units, out.drawings = rest.parents, rest.units, rest.drawings
    out.ligandff, out.skipped, out.charge_model = ligandff, patch.skipped, patch.charge_model
    return out


def refuse_pseudopeptide(args) -> None:
    """A pseudopeptide is a free ligand, so OpenFF would take it whole: the protein
    types its amino acids are asked to keep are GAFF2's to mix."""
    pseudo = getattr(args, "pseudopeptide", None) or "off"
    if pseudo != "off":
        raise ValueError(f"pseudopeptide = '{pseudo}' mixes protein types with GAFF2's: "
                         f"it goes with ligandff = 'gaff-2.11', not {args.ligandff}")  # fmt: skip


def _typed(f: System, charge, generator, ligandff: str) -> System:
    """Molecule ``f`` (all hydrogens, bond orders) as ``ligandff`` parameterizes it
    with the partial charges ``charge``: a system with its force-field tables."""
    import openmm.app as app
    import openmm.unit as u
    from openff.units import unit as offunit

    from .omm import from_openmm

    mol = _molecule(f)
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


def _molecule(f: System, net: int | None = None):
    """``f`` (all hydrogens, bond orders) as an OpenFF molecule, atoms in order."""
    from openff.toolkit import Molecule

    from .chem import to_rdkit

    mol = Molecule.from_rdkit(to_rdkit(f, implicit_hydrogens=False),
                              allow_undefined_stereo=True, hydrogens_are_explicit=True)  # fmt: skip
    if [a.atomic_number for a in mol.atoms] != f.atoms["anum"].tolist():
        raise OpenFFError("the OpenFF toolkit changed the atoms of the molecule")
    if net is not None and int(mol.total_charge.m) != int(net):
        raise OpenFFError(f"the molecule's formal charges sum to {int(mol.total_charge.m):+d}, "
                          f"not the {int(net):+d} asked for: NAGL charges follow the formal "
                          "charges, so give them on the atoms")  # fmt: skip
    return mol


def patch_directory(workdir, ligandff=DEFAULT) -> Path:
    """Where a run keeps its OpenFF templates, or its MMFF94 ones."""
    from . import mmff

    if mmff.is_mmff(ligandff):
        return mmff.patch_directory(workdir)
    return Path(workdir) / "openff_patch"
