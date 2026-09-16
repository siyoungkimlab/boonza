"""Settings for ``boonza md``: defaults, TOML files, the command line, and
what a restart restores.

Precedence: built-in defaults < the TOML file (``--config``) < options on
the command line. Keys and options are ommflow's, except that force fields
are ``forcefields`` (``-f``/``-m``: viparr force fields, or OpenMM XML files),
plus ``ligand_charges`` (``--charge``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tomllib
from pathlib import Path

PLATFORMS = ("CUDA", "OpenCL", "Metal", "CPU", "Reference")
PRECISIONS = ("mixed", "single", "double")
DIHEDRAL_RESTRAINTS = ("none", "bb", "ss")
#: Pressure coupling: one scale factor for the box, x and y together with z
#: free (a planar membrane), or constant volume.
BAROSTATS = ("isotropic", "membrane", "none")
LIGAND_MODES = ("disabled", "auto")
LIGAND_FORCE_FIELDS = ("gaff-2.11",)
PROTEIN_EXTENTS = ("matched", "cb")
HYDROGEN_MASS_AMU = 4.0
HMR_INTEGRATION_FS = 4.0

#: viparr force fields used when none are given. Each entry is a force field
#: and the patches merged onto it: phosphorylated residues join ff19SB, since a
#: phosphorylated chain must be matched by one force field. The first force
#: field matching a molecule wins, so Joung-Cheatham monovalent ions come
#: before the Li/Merz set that supplies the multivalent ones.
DEFAULT_FORCEFIELDS = (
    ("aa.amber.ff19SB", "aa.amber.phosaa19SB"),
    ("water.tip3p",),
    ("ions.amber1jc.tip3p",),
    ("ions.amber1234lm_anton.tip3p",),
)

DEFAULTS: dict = {
    "input_structure": None,
    "workdir": "openmm_md",
    "forcefields": None,
    "ligand_mode": "auto",
    "ligandff": "gaff-2.11",
    "ligand_charges": None,
    "parents": None,
    "protein_extent": "matched",
    "solvate": True,
    "padding_nm": 1.0,
    "box_nm": None,
    "cutoff_nm": None,
    "saltM": 0.15,
    "temperature": 298.0,
    "pressure": 1.0,
    "barostat": "isotropic",
    "surface_tension": 0.0,
    "equilibration_ns": 0.1,
    "equilibration_report_interval_ns": 0.01,
    "production_ns": 100.0,
    "production_report_interval_ns": 1.0,
    "checkpoint_interval_ns": 0.01,
    "performance_interval_ns": 1.0,
    "integration_fs": 2.0,
    "hmr": False,
    "dihedral_restraint": "none",
    "dihedral_restraint_kJ": 20.0,
    "dihedral_restraint_selection": None,
    "repulsion_selection": None,
    "repulsion_distance_nm": 0.5,
    "repulsion_kJ": 500.0,
    "seed": 0,
    "precision": "mixed",
    "platform": None,
    "early_stop": False,
    "monitor_ligand": None,
    "monitor_chain": None,
    "monitor_component": None,
    "monitor_selection": None,
    "monitor_interval_ns": 0.1,
    "pocket_cutoff_nm": 0.5,
    "contact_cutoff_nm": 0.5,
    "detach_cutoff_nm": 0.8,
    "confirmation_checks": 2,
}
_NUMBERS = {
    "padding_nm",
    "box_nm",
    "cutoff_nm",
    "saltM",
    "temperature",
    "pressure",
    "surface_tension",
    "equilibration_ns",
    "equilibration_report_interval_ns",
    "production_ns",
    "production_report_interval_ns",
    "checkpoint_interval_ns",
    "performance_interval_ns",
    "integration_fs",
    "dihedral_restraint_kJ",
    "monitor_interval_ns",
    "pocket_cutoff_nm",
    "contact_cutoff_nm",
    "detach_cutoff_nm",
    "repulsion_distance_nm",
    "repulsion_kJ",
}
_INTEGERS = {"seed", "confirmation_checks"}
_BOOLEANS = {"solvate", "hmr", "early_stop"}
_CHOICES = {
    "ligand_mode": LIGAND_MODES,
    "ligandff": LIGAND_FORCE_FIELDS,
    "protein_extent": PROTEIN_EXTENTS,
    "dihedral_restraint": DIHEDRAL_RESTRAINTS,
    "precision": PRECISIONS,
    "platform": PLATFORMS,
    "barostat": BAROSTATS,
}
MONITOR_SELECTORS = ("monitor_ligand", "monitor_chain", "monitor_component", "monitor_selection")
#: Settings a restart takes from ``final.toml`` unless they are given again.
RESTARTABLE = (
    "production_ns",
    "integration_fs",
    "hmr",
    "dihedral_restraint",
    "dihedral_restraint_kJ",
    "equilibration_ns",
    "equilibration_report_interval_ns",
    "production_report_interval_ns",
    "checkpoint_interval_ns",
    "performance_interval_ns",
    "platform",
    "precision",
    "early_stop",
    *MONITOR_SELECTORS,
    "monitor_interval_ns",
    "pocket_cutoff_nm",
    "contact_cutoff_nm",
    "detach_cutoff_nm",
    "confirmation_checks",
)


def forcefield_spec(value) -> tuple[tuple[str, ...], ...]:
    """Force fields as ((name, patch, ...), ...): each entry is a name, or a
    list of a name and the patches merged onto it (viparr's ``-m``)."""
    if not isinstance(value, list | tuple) or not value:
        raise ValueError("'forcefields' must be a non-empty list")
    out = []
    for entry in value:
        names = [entry] if isinstance(entry, str) else entry
        if (
            not isinstance(names, list | tuple)
            or not names
            or not all(isinstance(n, str) and n for n in names)
        ):
            raise ValueError(f"'forcefields' entry {entry!r} must be a name or a list of names")
        out.append(tuple(names))
    return tuple(out)


def is_xml(name: str) -> bool:
    """An OpenMM XML force field file, rather than a viparr force field."""
    return name.lower().endswith(".xml")


def forcefield_kind(spec) -> str:
    """The route: "xml" when every force field is an OpenMM XML file, "viparr" when none is."""
    xml = [is_xml(n) for entry in spec for n in entry]
    if not any(xml):
        return "viparr"
    if not all(xml):
        raise ValueError("give viparr force fields or OpenMM XML files, not both")
    if any(len(entry) > 1 for entry in spec):
        raise ValueError("OpenMM XML force fields take no patches (-m): list each file with -f")
    return "xml"


def describe_forcefields(spec) -> str:
    """``a + patch, b, c``: the force fields in priority order."""
    return ", ".join(" + ".join(entry) for entry in spec)


def ion_water_mismatch(spec) -> str | None:
    """A note when an ion set was fitted for another water model than the
    water force field's, by viparr's names: ``ions.<set>.<water>`` and
    ``water.<model>`` (``tip3p_charmm`` and ``tip3p-fb`` count as TIP3P)."""
    names = [Path(n).name for entry in spec for n in entry]
    waters = [n.split(".", 1)[1] for n in names if n.startswith("water.")]
    if len(waters) != 1:
        return None
    model = re.split(r"[_-]", waters[0])[0]
    odd = [n for n in names if n.startswith("ions.") and n.count(".") == 2
           and n.rsplit(".", 1)[1] not in ("all", model)]  # fmt: skip
    if not odd:
        return None
    return f"{', '.join(odd)}: fitted for another water model than water.{waters[0]}"


def water_model_mismatch(spec) -> str | None:
    """Why the force fields do not pair the protein force field with one of
    its own water models, or None. OpenMM XML files: by
    :func:`boonza.ffxml.water_mismatch`. viparr: TIP3P, which CHARMM
    (``water.tip3p_charmm``, with Lennard-Jones hydrogens) and Amber
    (``water.tip3p``) define differently; other water models are not checked."""
    if forcefield_kind(spec) == "xml":
        from ..ffxml import water_mismatch

        return water_mismatch(entry[0] for entry in spec)
    names = [Path(n).name for entry in spec for n in entry]
    charmm = [n for n in names if n.startswith("aa.charmm.")]
    amber = [n for n in names if n.startswith(("aa.amber.", "aa.DES-Amber"))]
    if charmm and "water.tip3p" in names:
        return f"{charmm[0]} takes CHARMM's TIP3P, water.tip3p_charmm, not water.tip3p"
    if amber and "water.tip3p_charmm" in names:
        return f"{amber[0]} takes water.tip3p, not CHARMM's water.tip3p_charmm"
    return None


def check_settings(values: dict, where: str) -> dict:
    """Validate the types and choices of settings read from a file."""
    legacy = sorted({"proteinff", "waterff"} & set(values))
    if legacy:
        raise ValueError(
            f"{where}: {' and '.join(legacy)} are gone; list OpenMM XML files in "
            'forcefields instead, e.g. forcefields = ["amber19/protein.ff19SB.xml", '
            '"amber19/opc.xml"]'
        )
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"{where}: unknown setting(s): {', '.join(sorted(unknown))}")
    out = dict(values)
    for key, v in values.items():
        if key in _BOOLEANS:
            ok = isinstance(v, bool)
        elif key in _INTEGERS:
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif key in _NUMBERS:
            ok = isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
        elif key == "forcefields":
            out[key] = forcefield_spec(v)
            ok = True
        elif key == "ligand_charges":
            ok = isinstance(v, dict) and all(
                isinstance(q, int) and not isinstance(q, bool) for q in v.values()
            )
        elif key == "parents":
            ok = isinstance(v, dict) and all(isinstance(p, str) and p for p in v.values())
        else:
            ok = isinstance(v, str) and v != ""
        if not ok:
            raise ValueError(f"{where}: '{key}' has the wrong type: {v!r}")
        if key in _CHOICES and v not in _CHOICES[key]:
            raise ValueError(f"{where}: '{key}' must be one of: {', '.join(_CHOICES[key])}")
    return out


def load_configuration(path) -> dict:
    """Settings from a TOML file, validated."""
    with Path(path).open("rb") as fh:
        return check_settings(tomllib.load(fh), str(path))


def build_parser(prog: str = "boonza md") -> argparse.ArgumentParser:
    """The ``boonza md`` parser; an option left out is left out of the result."""
    d = DEFAULTS
    p = argparse.ArgumentParser(
        prog=prog,
        argument_default=argparse.SUPPRESS,
        description="Prepare and run explicit-solvent MD with OpenMM. Running the same "
        "command on a used work directory resumes it.",
    )
    p.add_argument(
        "input_structure",
        nargs="?",
        metavar="INPUT_STRUCTURE",
        help="structure with hydrogens (PDB, DMS, MAE, mmCIF, SDF, GRO, ...)",
    )
    p.add_argument("--config", help="TOML settings; options given here override it")
    p.add_argument(
        "--write-default-config",
        metavar="FILE",
        help="write a commented settings template and exit",
    )
    p.add_argument(
        "--list-components",
        action="store_true",
        help="list the molecules of INPUT_STRUCTURE with their selectors and exit",
    )
    p.add_argument("--workdir", help=f"output directory (default: {d['workdir']})")
    p.add_argument(
        "-f",
        "--ff",
        dest="ff_options",
        action="append",
        type=lambda v: ("f", v),
        metavar="FF",
        help="force field, in priority order: a viparr force field, or OpenMM XML "
        "files (e.g. -f amber19/protein.ff19SB.xml -f amber19/opc.xml), not both "
        "(default: ff19SB + phosaa19SB, TIP3P, Joung-Cheatham and Li/Merz ions)",
    )
    p.add_argument(
        "-m",
        "--merge",
        dest="ff_options",
        action="append",
        type=lambda v: ("m", v),
        metavar="PATCH",
        help="patch the previous -f viparr force field",
    )
    for old in ("--proteinff", "--waterff"):  # ommflow's, replaced by -f XML files
        p.add_argument(old, help=argparse.SUPPRESS)
    p.add_argument(
        "--ligand-mode",
        dest="ligand_mode",
        choices=LIGAND_MODES,
        help="auto (default): GAFF2 templates for what the force fields cannot "
        "match; disabled: that is an error",
    )
    p.add_argument("--ligandff", choices=LIGAND_FORCE_FIELDS, help="default: gaff-2.11")
    p.add_argument(
        "--protein-extent",
        dest="protein_extent",
        choices=PROTEIN_EXTENTS,
        help="amino acids with GAFF2 atoms keep protein types as far as they match "
        "(matched, the default) or on the backbone and CB only (cb)",
    )
    p.add_argument(
        "--charge",
        dest="charge_options",
        action="append",
        metavar="RES=Q",
        help="formal charge of a ligand residue whose file has none",
    )
    p.add_argument(
        "--parent",
        dest="parent_options",
        action="append",
        metavar="RES=PARENT",
        help="the standard residue a modified residue comes from, e.g. MSE=MET",
    )
    floats = [
        ("--padding-nm", "padding_nm", "solute to box edge (nm)"),
        ("--cutoff-nm", "cutoff_nm", "nonbonded cutoff (nm; 0.9 Amber, 1.2 CHARMM)"),
        ("--saltM", "saltM", "NaCl added beyond neutralizing (mol/L, counted against waters)"),
        ("--temperature", "temperature", "K"),
        ("--pressure", "pressure", "bar"),
        ("--surface-tension", "surface_tension", "membrane barostat only (bar nm)"),
        ("--equilibration-ns", "equilibration_ns", "NVT and NPT equilibration, each (ns)"),
        ("--equilibration-report-interval-ns", "equilibration_report_interval_ns", "ns"),
        ("--production-ns", "production_ns", "absolute production target (ns)"),
        ("--production-report-interval-ns", "production_report_interval_ns", "ns"),
        ("--checkpoint-interval-ns", "checkpoint_interval_ns", "ns"),
        ("--performance-interval-ns", "performance_interval_ns", "rows of performance.csv"),
        ("--integration-fs", "integration_fs", "time step (fs; 4 with --hmr)"),
        ("--dihedral-restraint-kJ", "dihedral_restraint_kJ", "restraint strength (kJ/mol)"),
        ("--monitor-interval-ns", "monitor_interval_ns", "ns between detachment checks"),
        ("--pocket-cutoff-nm", "pocket_cutoff_nm", "target-to-pocket cutoff (nm)"),
        ("--contact-cutoff-nm", "contact_cutoff_nm", "contact distance (nm)"),
        ("--detach-cutoff-nm", "detach_cutoff_nm", "detachment distance (nm)"),
        ("--repulsion-distance-nm", "repulsion_distance_nm", "repulsion wall distance (nm)"),
        ("--repulsion-kJ", "repulsion_kJ", "repulsion strength (kJ/mol/nm^2)"),
    ]
    for opt, dest, text in floats:
        default = f"; default: {d[dest]}" if d[dest] is not None else ""
        p.add_argument(opt, dest=dest, type=float, help=text + default)
    p.add_argument("--seed", type=int, help="random seed (default: 0)")
    p.add_argument(
        "--dihedral-restraint-selection",
        dest="dihedral_restraint_selection",
        help="restrain only torsions whose atoms this selects, e.g. 'protein' (default: all)",
    )
    p.add_argument(
        "--repulsion-selection",
        dest="repulsion_selection",
        help="molecules kept from sticking together (heavy atoms of different ones repel), "
        "e.g. 'chain L'",
    )
    p.add_argument(
        "--confirmation-checks",
        dest="confirmation_checks",
        type=int,
        help="detached checks in a row that stop production (default: 2)",
    )
    p.add_argument(
        "--barostat",
        choices=BAROSTATS,
        help="pressure coupling: isotropic (default), membrane (x and y together, z "
        "free: planar bilayers), or none (constant volume)",
    )
    p.add_argument(
        "--solvate",
        action=argparse.BooleanOptionalAction,
        help="add water and ions (default: on); --no-solvate runs INPUT_STRUCTURE as it "
        "is, with its own water, ions and periodic cell",
    )
    p.add_argument(
        "--hmr",
        action=argparse.BooleanOptionalAction,
        help="hydrogen mass repartitioning to 4 amu, water untouched (default: off)",
    )
    p.add_argument(
        "--early-stop",
        dest="early_stop",
        action=argparse.BooleanOptionalAction,
        help="stop production once the target has detached (default: off)",
    )
    p.add_argument(
        "--dihedral-restraint",
        dest="dihedral_restraint",
        choices=DIHEDRAL_RESTRAINTS,
        help="restrain phi/psi to the input: bb all, ss helices and sheets",
    )
    p.add_argument("--precision", choices=PRECISIONS, help="GPU precision (default: mixed)")
    p.add_argument("--platform", choices=PLATFORMS, help="default: the fastest that works")
    for sel, text in (
        ("--monitor-ligand", "a ligand-N ID"),
        ("--monitor-chain", "an input chain ID"),
        ("--monitor-component", "a component-N ID"),
        ("--monitor-selection", "atoms of the input, e.g. 'resname LIG and chain L'"),
    ):
        p.add_argument(sel, dest=sel[2:].replace("-", "_"), help=f"early-stop target: {text}")
    return p


def parse_arguments(argv=None, parser=None) -> argparse.Namespace:
    """The resolved settings; ``args.specified`` names those given in the
    TOML file or on the command line, and ``args.extra`` the options of a
    ``parser`` built on :func:`build_parser` that are not settings."""
    parser = build_parser() if parser is None else parser
    given = vars(parser.parse_args(argv))
    legacy = [f"--{k}" for k in ("proteinff", "waterff") if k in given]
    if legacy:
        parser.error(f"{' and '.join(legacy)} are gone: give OpenMM XML files with -f, e.g. "
                     "-f amber19/protein.ff19SB.xml -f amber19/opc.xml")  # fmt: skip
    from_file: dict = {}
    if "config" in given:
        try:
            from_file = load_configuration(given["config"])
        except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
            parser.error(f"cannot use {given['config']}: {e}")
    cli = {k: v for k, v in given.items() if k in DEFAULTS}
    if "ff_options" in given:
        cli["forcefields"] = _cli_forcefields(given["ff_options"], parser)
    if "charge_options" in given:
        cli["ligand_charges"] = {
            **(from_file.get("ligand_charges") or {}),
            **_cli_charges(given["charge_options"], parser),
        }
    if "parent_options" in given:
        cli["parents"] = {**(from_file.get("parents") or {}),
                          **_cli_parents(given["parent_options"], parser)}  # fmt: skip
    args = argparse.Namespace(**{**DEFAULTS, **from_file, **cli})
    args.config = given.get("config")
    args.write_default_config = given.get("write_default_config")
    args.list_components = given.get("list_components", False)
    args.specified = set(from_file) | set(cli)
    own = {"config", "write_default_config", "list_components", "ff_options", "charge_options",
           "parent_options"}  # fmt: skip
    args.extra = {k: v for k, v in given.items() if k not in DEFAULTS and k not in own}
    try:
        finish(args)
    except ValueError as e:
        parser.error(str(e))
    return args


def _cli_forcefields(options, parser) -> tuple[tuple[str, ...], ...]:
    out: list[list[str]] = []
    for kind, name in options:
        if kind == "f":
            out.append([name])
        elif not out:
            parser.error(f"-m {name}: give a force field with -f before patching it")
        else:
            out[-1].append(name)
    return tuple(tuple(x) for x in out)


def _cli_parents(options, parser) -> dict[str, str]:
    out = {}
    for item in options:
        res, _, parent = item.partition("=")
        if not res or not parent:
            parser.error(f"--parent {item}: give RESNAME=PARENT, e.g. MSE=MET")
        out[res] = parent
    return out


def _cli_charges(options, parser) -> dict[str, int]:
    out = {}
    for item in options:
        res, _, q = item.partition("=")
        if not res or not q.lstrip("+-").isdigit():
            parser.error(f"--charge {item}: give RESNAME=CHARGE, e.g. LIG=-1")
        out[res] = int(q)
    return out


def finish(args) -> None:
    """Fill settings that depend on others and reject inconsistent ones."""
    positive = [
        "padding_nm",
        "temperature",
        "pressure",
        "equilibration_ns",
        "equilibration_report_interval_ns",
        "production_ns",
        "production_report_interval_ns",
        "checkpoint_interval_ns",
        "performance_interval_ns",
        "integration_fs",
        "monitor_interval_ns",
        "pocket_cutoff_nm",
        "contact_cutoff_nm",
        "detach_cutoff_nm",
    ]
    for key in positive + ["cutoff_nm", "box_nm", "repulsion_distance_nm", "repulsion_kJ"]:
        v = getattr(args, key)
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError(f"'{key}' must be positive")
    if args.saltM < 0:
        raise ValueError("'saltM' cannot be negative")
    if not math.isfinite(args.dihedral_restraint_kJ) or args.dihedral_restraint_kJ == 0:
        raise ValueError("'dihedral_restraint_kJ' must be a nonzero number")
    if args.confirmation_checks < 1:
        raise ValueError("'confirmation_checks' must be at least 1")
    if args.surface_tension and args.barostat != "membrane":
        raise ValueError("'surface_tension' needs barostat = 'membrane' (x and y coupled, z free)")
    if args.forcefields is None:
        args.forcefields = DEFAULT_FORCEFIELDS
    args.forcefields = forcefield_spec(args.forcefields)
    forcefield_kind(args.forcefields)  # viparr or XML, not a mixture
    problem = water_model_mismatch(args.forcefields)
    if problem:
        raise ValueError(problem)
    if args.cutoff_nm is None:
        args.cutoff_nm = default_cutoff_nm(args)
    if args.hmr and "integration_fs" not in args.specified:
        args.integration_fs = HMR_INTEGRATION_FS
    args.ligand_charges = dict(args.ligand_charges or {})
    args.parents = dict(args.parents or {})
    chosen = [k for k in MONITOR_SELECTORS if getattr(args, k) is not None]
    if len(chosen) > 1:
        raise ValueError("choose at most one early-stop target: " + ", ".join(chosen))


def default_cutoff_nm(args) -> float:
    """0.9 nm for Amber force fields and 1.2 nm for CHARMM, as ommflow."""
    names = [n.lower() for entry in args.forcefields for n in entry]
    return 1.2 if any("charmm" in n for n in names) else 0.9


def toml_value(v) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, dict):
        return (
            "{" + ", ".join(f"{json.dumps(str(k))} = {toml_value(x)}" for k, x in v.items()) + "}"
        )
    if isinstance(v, list | tuple):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def settings_of(args) -> dict:
    """The resolved settings of ``args`` (those that are set)."""
    return {k: getattr(args, k) for k in DEFAULTS if getattr(args, k, None) is not None}


def write_settings(path, settings: dict) -> None:
    """Write resolved settings as ``final.toml``, in a stable order."""
    lines = [
        "# Resolved boonza md settings of this run.",
        "# A restart reads them back; options given again override them.",
    ]
    lines += [f"{k} = {toml_value(settings[k])}" for k in DEFAULTS if settings.get(k) is not None]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def restore_settings(args, path) -> None:
    """Take restartable settings not given again from ``final.toml``."""
    saved = load_configuration(path)
    if "production_ns" not in args.specified and "production_ns" not in saved:
        raise ValueError(f"cannot resume: {path} has no production_ns")
    for key in RESTARTABLE:
        if key not in args.specified and key in saved:
            setattr(args, key, saved[key])
    for key in MONITOR_SELECTORS:  # a selector given now replaces a saved one
        if key not in args.specified and any(k in args.specified for k in MONITOR_SELECTORS):
            setattr(args, key, None)


def commit_settings(args, path) -> None:
    """Record the restartable settings of a resumed run in ``final.toml``."""
    saved = load_configuration(path)
    for key in RESTARTABLE:
        v = getattr(args, key, None)
        if v is None:
            saved.pop(key, None)
        else:
            saved[key] = v
    write_settings(path, saved)


_TEMPLATE = """\
# boonza md settings. Options given on the command line override this file.
input_structure = "protein.pdb"
workdir = "openmm_md"

# viparr force fields in priority order (the first matching a molecule wins); a
# list is a force field with patches merged onto it. The default:
# forcefields = [["aa.amber.ff19SB", "aa.amber.phosaa19SB"], "water.tip3p",
#                "ions.amber1jc.tip3p", "ions.amber1234lm_anton.tip3p"]
# Or OpenMM XML files, in OpenMM's data directories or by path (no GAFF2 ligands):
# forcefields = ["amber19/protein.ff19SB.xml", "amber19/opc.xml"]

# GAFF2 (AM1-BCC, AmberTools) templates for ligands and covalent adducts.
ligand_mode = "auto"
ligandff = "gaff-2.11"
# Formal charges of ligand residues read from files without them (PDB):
# ligand_charges = { LIG = -1 }
# The standard residue each modified residue comes from, where the file has no
# MODRES record and the PDB's dictionary does not know it (an unclear guess stops):
# parents = { XYZ = "LYS" }
# Atoms of an amino acid that has GAFF2 atoms (a covalent adduct, a non-standard
# residue) keep protein types "matched": as far as they and their neighbours
# match the parent residue; or "cb": on the backbone, CB and CB's hydrogens only.
protein_extent = "matched"

# solvate = false runs INPUT_STRUCTURE as it is, water, ions and box included
# (it must have a periodic cell); padding_nm, box_nm and saltM are then unused.
solvate = true
padding_nm = 1.0
# cutoff_nm defaults to 0.9 for Amber and 1.2 for CHARMM.
# cutoff_nm = 0.9
# NaCl beyond neutralizing, counted against the number of waters (msys).
saltM = 0.15
temperature = 298.0
pressure = 1.0
# Pressure coupling: isotropic, membrane (x and y together and z free, for a
# planar bilayer), or none (constant volume). surface_tension (bar nm, 0 for
# a tensionless membrane) needs the membrane barostat.
barostat = "isotropic"
surface_tension = 0.0
equilibration_ns = 0.1
production_ns = 100.0
equilibration_report_interval_ns = 0.01
production_report_interval_ns = 1.0
checkpoint_interval_ns = 0.01
performance_interval_ns = 1.0
integration_fs = 2.0
# Hydrogen mass repartitioning to 4 amu; integration_fs then defaults to 4.
hmr = false

# Restrain backbone phi/psi to the input: none, bb, or ss (helices and sheets).
dihedral_restraint = "none"
dihedral_restraint_kJ = 20.0
# Only torsions whose atoms this selects (boonza swim: not the ligands):
# dihedral_restraint_selection = "not chain LIG"
seed = 0

# Keep the molecules a selection picks (ligand copies) from sticking together:
# E = k (d0 - r)^2 between heavy atoms of different ones closer than d0.
# repulsion_selection = "chain L"
repulsion_distance_nm = 0.5
repulsion_kJ = 500.0

# GPU precision: mixed, single or double. Left out, a platform that cannot
# honour mixed falls back to its own; set, it is an error.
# precision = "mixed"

# Stop production once a target has left its pocket (off by default).
early_stop = false
# monitor_ligand = "ligand-0"
# monitor_chain = "B"
# monitor_component = "component-2"
# Or atoms of the input structure, in boonza's (msys) selection language:
# monitor_selection = "resname LIG and chain L"
monitor_interval_ns = 0.1
pocket_cutoff_nm = 0.5
contact_cutoff_nm = 0.5
detach_cutoff_nm = 0.8
confirmation_checks = 2

# Left out, the fastest platform that works is used.
# platform = "CUDA"
"""


def write_default_configuration(path) -> None:
    """Write the commented settings template."""
    Path(path).write_text(_TEMPLATE, encoding="utf-8")
