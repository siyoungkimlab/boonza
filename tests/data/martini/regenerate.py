"""Regenerate martinize2's reference topologies for tests/test_martinize.py.

    python tests/data/martini/regenerate.py

Needs martinize2 (vermouth) on PATH; the references in this directory were
made with vermouth 0.15.1.dev86 (commit a9b62ee).  Each case is martinized
with the secondary structure boonza's DSSP assigns, so both tools see the
same one.  Command-line headers are dropped from the itp files.
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


def load(path: Path):
    if path.suffix == ".gz":
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / path.stem
            plain.write_bytes(gzip.decompress(path.read_bytes()))
            return boonza.load(plain)
    return boonza.load(path)


def main():
    for name, (src, elastic) in CASES.items():
        s = load(src)
        s = s.select("protein").clone()
        ss = "".join("C" if c in ("NA", " ") else c for c in boonza.dssp(s)[0])
        with tempfile.TemporaryDirectory() as tmp:
            boonza.save(s, Path(tmp) / "in.pdb")
            cmd = ["martinize2", "-f", "in.pdb", "-o", "topol.top", "-x", "cg.pdb",
                   "-ff", "martini3001", "-ss", ss, "-maxwarn", "100"]  # fmt: skip
            if elastic:
                cmd += ["-elastic", "-ef", "700", "-el", "0", "-eu", "0.9"]
            subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)
            out = HERE / name
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir()
            (out / "ss.txt").write_text(ss + "\n")
            tmp = Path(tmp)
            for f in [tmp / "topol.top", tmp / "cg.pdb", *tmp.glob("molecule_*.itp")]:
                lines = f.read_text().splitlines(keepends=True)
                if f.suffix == ".itp":
                    lines = lines[lines.index("[ moleculetype ]\n") :]
                text = "".join(lines).encode()
                (out / (f.name + ".gz")).write_bytes(gzip.compress(text, mtime=0))
        print(name, "done")


if __name__ == "__main__":
    main()
