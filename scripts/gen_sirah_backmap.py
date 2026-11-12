"""SIRAH Tools' backmapping table, as the JSON boonza reads.

    python scripts/gen_sirah_backmap.py /path/to/sirah.amber/tools/sirah_vmdtk.tcl \
        src/boonza/data/sirah/backmap.json

``sirah_vmdtk.tcl`` (SIRAH Tools, GPL v2) builds the table in its ``sirah_mapdb``
procedure, in plain Tcl: for every coarse-grained residue, the all-atom residue it
stands for, which atoms sit on which beads, and how the others are placed from three
atoms already there (a distance, an angle, a dihedral).  That procedure is run with
``tclsh`` and the table it leaves written out; nothing of it is retyped here.  The
JSON is a derived work of that file and carries its licence (see NOTICE).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DUMP = """
source [lindex $argv 0]
sirah_mapdb
dict for {res entry} $backmap { puts "@@ $res"; puts $entry }
"""


def words(text: str) -> list:
    """A Tcl list as nested Python lists: braces group, whitespace separates."""
    out, stack, word = [], [], ""
    stack.append(out)
    for ch in text:
        if ch == "{":
            if word:
                stack[-1].append(word)
                word = ""
            new: list = []
            stack[-1].append(new)
            stack.append(new)
        elif ch == "}":
            if word:
                stack[-1].append(word)
                word = ""
            stack.pop()
        elif ch.isspace():
            if word:
                stack[-1].append(word)
                word = ""
        else:
            word += ch
    if word:
        stack[-1].append(word)
    if len(stack) != 1:
        raise ValueError("unbalanced braces")
    return out


def pairs(items: list) -> dict:
    if len(items) % 2:
        raise ValueError(f"an odd number of words where a dictionary was expected: {items[:4]}")
    return {items[k]: items[k + 1] for k in range(0, len(items), 2)}


def rules(build: list) -> dict:
    """``{atom: [":", bead, ...]}`` or ``{atom: ["+", [a, b, c], [r, angle, dihedral]]}``."""
    out = {}
    for atom, how in pairs(build).items():
        if how[0] == ":":
            out[atom] = [":", *how[1:]]
        elif how[0] == "+":
            out[atom] = ["+", list(how[1]), [float(x) for x in how[2]]]
        else:
            raise ValueError(f"{atom}: building function {how[0]!r}")
    return out


def entry(items) -> dict:
    if not isinstance(items, list):  # a bare word: no instructions (print, for one)
        return {}
    out = {}
    for key, value in pairs(items).items():
        if key == "build":
            out[key] = rules(value)
        elif key in ("first", "last"):
            inner = pairs(value)
            out[key] = {k: rules(v) if k == "build" else v for k, v in inner.items()}
        else:
            out[key] = value
    return out


def main(tcl: str, target: str) -> None:
    source = Path(tcl).read_text(errors="replace")
    begin = source.index("proc sirah_mapdb")
    end = source.index("proc sirah_build")
    version = re.search(r"VERSION:\s+([\d.]+)", source[:4000])
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "mapdb.tcl").write_text(source[begin:end])
        (Path(tmp) / "dump.tcl").write_text(DUMP)
        text = subprocess.run(["tclsh", "dump.tcl", "mapdb.tcl"], cwd=tmp, check=True,
                              capture_output=True, text=True).stdout  # fmt: skip
    table = {}
    for block in text.split("@@ ")[1:]:  # an entry is printed as it was written: many lines
        name, _, body = block.partition("\n")
        table[name.strip()] = entry(words(body))
    out = {
        "source": "SIRAH Tools, sirah_vmdtk.tcl, procedure sirah_mapdb"
                  + (f" (version {version.group(1)})" if version else ""),
        "licence": "GPL-2.0 (SIRAH Tools); see NOTICE and LICENSES/GPL-2.0.txt",
        "residues": table,
    }  # fmt: skip
    Path(target).write_text(json.dumps(out, indent=1) + "\n")
    built = sum("build" in e for e in table.values())
    print(f"{target}: {len(table)} residues, {built} with building instructions")


if __name__ == "__main__":
    main(*sys.argv[1:3])
