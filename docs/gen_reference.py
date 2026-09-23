"""Generate docs/reference.md from boonza's signatures and docstrings.

    python docs/gen_reference.py

The reference is produced from the code itself so it cannot drift from it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import boonza
from boonza import io as bio
from boonza import pbc

SECTIONS = [
    ("Files", ["load", "save", "open_trajectory", "open_writer", "Trajectory", "Frames", "Frame",
               "DMSError"]),
    ("System and handles", ["System", "AtomSel", "Atom", "Bond", "Residue", "Chain", "Ct",
                            "StaleHandleError", "NonbondedInfo"]),
    ("Force-field tables", ["TermTable", "Term", "ParamTable", "Param", "OverrideTable",
                            "TERM_SCHEMAS", "NONBONDED_SCHEMAS", "update_exclusions"]),
    ("Reports and checks", ["describe", "ForceFieldReport", "validate", "Problem", "find_knots",
                            "diff", "Difference"]),
    ("Molecules and rings", ["distinct_fragments", "sssr", "ring_systems"]),
    ("Geometry", ["Glue", "make_whole"]),
    ("Superposition", ["matchmaker", "MatchResult", "needleman_wunsch", "chimerax_ss", "cealign",
                       "CEResult", "ce_align", "superpose", "kabsch", "rmsd"]),
    ("Sequences", ["align_sequences", "SequenceAlignment", "sequence", "identity_matrix"]),
    ("Analysis", ["rmsd_trajectory", "rmsf", "radius_of_gyration", "rdf", "residue_contacts",
                  "sasa", "hbonds", "HBonds", "baker_hubbard", "wernet_nilsson", "dssp",
                  "backbone_dihedrals", "backbone_hbonds"]),
    ("RDKit", ["to_rdkit", "from_rdkit", "fragments_to_rdkit", "assign_bond_orders"]),
    ("OpenMM", ["to_openmm", "from_openmm", "openmm_energies"]),
    ("Martini", ["martinize", "Martinized", "martini.parameters", "martini.solvate",
                 "martini.bilayer", "martini.lipid_templates", "martini.equilibrate"]),
]  # fmt: skip
CLASSES_WITH_MEMBERS = {"System", "AtomSel", "Atom", "Bond", "Residue", "Chain", "Ct",
                        "TermTable", "ParamTable", "Term", "OverrideTable", "Trajectory",
                        "SequenceAlignment", "HBonds", "MatchResult", "Martinized"}  # fmt: skip


def _sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _doc(obj) -> str:
    return inspect.getdoc(obj) or ""


def _entry(name, obj) -> list[str]:
    out = []
    if inspect.isclass(obj):
        out.append(f"### `class {name}{_sig(obj)}`")
    elif callable(obj):
        out.append(f"### `{name}{_sig(obj)}`")
    else:
        out.append(f"### `{name}`")
        if isinstance(obj, dict):
            out.append(f"\n{len(obj)} entries: {', '.join(f'`{k}`' for k in sorted(obj))}")
        return out + [""]
    d = _doc(obj)
    if d:
        out += ["", d]
    if inspect.isclass(obj) and name in CLASSES_WITH_MEMBERS:
        rows = []
        for mname, member in inspect.getmembers(obj):
            if mname.startswith("_"):
                continue
            static = inspect.getattr_static(obj, mname)
            if isinstance(static, property):
                first = (_doc(member) or "").split("\n")[0]
                rows.append(f"- `{mname}` (property){': ' + first if first else ''}")
            elif callable(member):
                first = (_doc(member) or "").split("\n")[0]
                rows.append(f"- `{mname}{_sig(member)}`{': ' + first if first else ''}")
        if rows:
            out += ["", "Members:", ""] + rows
    return out + [""]


def main():
    lines = ["# API reference", "",
             "Generated from the code by `docs/gen_reference.py`; see the [guide](index.md) "
             "for explanations and examples.", ""]  # fmt: skip
    documented = set()
    for title, names in SECTIONS:
        lines += [f"## {title}", ""]
        for name in names:
            obj = boonza
            for part in name.split("."):  # submodule members too, e.g. martini.solvate
                obj = getattr(obj, part)
            lines += _entry(f"boonza.{name}", obj)
            documented.add(name)
    lines += ["## Per-format readers and writers (`boonza.io`)", ""]
    for name in sorted(dir(bio)):
        if name.startswith(("load_", "save_")):
            lines += _entry(f"boonza.io.{name}", getattr(bio, name))
    lines += ["## Periodic geometry (`boonza.pbc`)", "", _doc(pbc), ""]
    for name in sorted(dir(pbc)):
        f = getattr(pbc, name)
        if not name.startswith("_") and inspect.isfunction(f) and f.__module__ == pbc.__name__:
            lines += _entry(f"boonza.pbc.{name}", f)
    missing = sorted(set(boonza.__all__) - documented - {"pbc", "analysis"})
    if missing:
        lines += ["## Other", ""]
        for name in missing:
            lines += _entry(f"boonza.{name}", getattr(boonza, name))
    out = Path(__file__).with_name("reference.md")
    out.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
