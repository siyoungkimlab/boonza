"""Regenerate martinize2's reference topologies for tests/test_martinize.py.

    python tests/data/martini/regenerate.py

Needs martinize2 (vermouth) on PATH; the references in this directory were
made with vermouth 0.15.1.dev86 (commit a9b62ee).  Each case is martinized
with the secondary structure boonza's DSSP assigns, so both tools see the
same one.  Command-line headers are dropped from the itp files.

The Go case writes the contact map as well, with ``-go-write-file``.  That
flag raises IndexError in this vermouth on a structure holding a residue whose
alpha carbon has no coordinates -- it indexes the alpha-carbon positions by
residue and collects only the ones it has -- which 1TEN.pdb does, its first
residue being an arginine present as a C and an O.  It is not hit here because
the input is ``select("protein")``, which leaves that residue out before
martinize2 sees it, so both tools see the same 89 residues.
"""

import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path

import boonza

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CASES = {  # name: (input, elastic)
    "1TEN": (DATA / "1TEN.pdb", True),
    "2TRX": (DATA / "2TRX.pdb", False),
    "1HHO": (DATA / "1HHO.pdb", True),
    "5PTI": (HERE / "5PTI_protein.pdb.gz", True),
}
#: The same, martinized as Martini 2.2, under <name>.martini22.  Its residues are
#: CHARMM's: give martinize2 a residue called HIS and it builds the HSD block but
#: leaves the name, which its own protein_resnames macro does not list, so every
#: link skips the residue and the backbone is left severed there.  The structures
#: are renamed before it sees them, as martini22 expects.
#: 1TEN is left out: it carries a protonated aspartate, and martinize2 drops a
#: whole set of modifications when one of them has no mapping -- there the
#: N-terminus with it, which boonza applies, the two having nothing to do with
#: each other.
ALSO_MARTINI22 = ("2TRX",)
#: The same, with its fold held by native contacts instead of rubber bands,
#: under <name>.go: martinize2's own Go model, and beside it the contact map it
#: was built from, so boonza can build the same wells from the same contacts.
#: 1TEN is enough to pin it -- one chain, 89 mapped residues, 287 beads, 89
#: virtual sites and 185 wells, which boonza reaches bead for bead.
ALSO_GO = ("1TEN",)
CHARMM_NAMES = {"HIS": "HSD", "HID": "HSD", "HIE": "HSE", "HIP": "HSP"}


def load(path: Path):
    if path.suffix == ".gz":
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / path.stem
            plain.write_bytes(gzip.decompress(path.read_bytes()))
            return boonza.load(plain)
    return boonza.load(path)


def main():
    for name, (src, elastic) in CASES.items():
        for forcefield in ("martini3001", *(("martini22",) if name in ALSO_MARTINI22 else ())):
            one(name, src, elastic, forcefield)
        if name in ALSO_GO:
            one(name, src, elastic=False, forcefield="martini3001", go=True)


def one(name, src, elastic, forcefield, go=False):
    s = load(src)
    s = s.select("protein").clone()
    if forcefield == "martini22":
        for r in range(s.nresidues):
            want = CHARMM_NAMES.get(str(s.residues["name"][r]).strip())
            if want:
                s.residue(r).name = want
    ss = "".join("C" if c in ("NA", " ") else c for c in boonza.dssp(s)[0])
    with tempfile.TemporaryDirectory() as tmp:
        boonza.save(s, Path(tmp) / "in.pdb")
        cmd = ["martinize2", "-f", "in.pdb", "-o", "topol.top", "-x", "cg.pdb",
                   "-ff", forcefield, "-ss", ss, "-maxwarn", "100"]  # fmt: skip
        if elastic:
            cmd += ["-elastic", "-ef", "700", "-el", "0", "-eu", "0.9"]
        if go:  # the defaults, named so the reference says what it was built with
            cmd += ["-go", "-go-write-file", "go_contacts.out", "-go-eps", "9.414",
                    "-go-low", "0.3", "-go-up", "1.1", "-go-res-dist", "3"]  # fmt: skip
        subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)
        out = HERE / (f"{name}.go" if go else
                      name if forcefield == "martini3001" else f"{name}.{forcefield}")  # fmt: skip
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        (out / "ss.txt").write_text(ss + "\n")
        tmp = Path(tmp)
        # martinize2 writes molecule_0.itp, or molecule.itp for a Go model,
        # whose moltype names the virtual-site atom types as well
        files = [tmp / "topol.top", tmp / "cg.pdb", *tmp.glob("molecule*.itp")]
        if go:
            files += [tmp / "go_contacts.out", tmp / "go_atomtypes.itp",
                      tmp / "go_nbparams.itp"]  # fmt: skip
        for f in files:
            lines = f.read_text().splitlines(keepends=True)
            if f.name.startswith("molecule"):  # drop the command-line header
                lines = lines[lines.index("[ moleculetype ]\n") :]
            text = "".join(lines).encode()
            (out / (f.name + ".gz")).write_bytes(gzip.compress(text, mtime=0))
    print(name, forcefield, "go" if go else "", "done")


if __name__ == "__main__":
    main()
