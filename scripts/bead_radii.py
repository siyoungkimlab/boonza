"""Self-pair Lennard-Jones size of every bead type, per coarse-grained force field.

These force fields write their nonbonded terms per PAIR of types (NBFIX), so a
bead's own size is the pair it makes with itself -- which is the number a radius
means, and the one a per-type table does not always carry.  The three write it
three different ways:

  martini22    [ nonbond_params ] as C6 and C12   -> sigma = (C12/C6)^(1/6)
  martini3001  [ nonbond_params ] as sigma, eps   -> as written
  sirah        [ atomtypes ] as sigma, eps, with [ nonbond_params ] overriding

Everything here is written in angstrom, as boonza is throughout; the .itp files
are in nm.
"""

import csv
import io
import pathlib
import sys
import zipfile

DATA = pathlib.Path(__file__).resolve().parent.parent / "src/boonza/data"
RMIN = 2.0 ** (1.0 / 6.0)


def sections(text, name):
    keep, out = False, []
    for line in text.splitlines():
        bare = line.split(";")[0].strip()
        if bare.startswith("["):
            keep = bare.strip("[] \t") == name
            continue
        if keep and bare:
            out.append(bare.split())
    return out


def martini22():
    text = (DATA / "martini/params/martini_v2.2.itp").read_text()
    out = {}
    for row in sections(text, "nonbond_params"):
        if len(row) < 5 or row[0] != row[1]:
            continue
        c6, c12 = float(row[3]), float(row[4])
        if c6 <= 0 or c12 <= 0:
            continue
        out[row[0]] = ((c12 / c6) ** (1 / 6), c6 * c6 / (4 * c12))
    return out


def martini3001():
    text = (DATA / "martini/params/martini_v3.0.0.itp").read_text()
    out = {}
    for row in sections(text, "nonbond_params"):
        if len(row) >= 5 and row[0] == row[1]:
            out[row[0]] = (float(row[3]), float(row[4]))
    return out


def sirah():
    with zipfile.ZipFile(DATA / "sirah/sirah_x2.2.zip") as z:
        text = io.TextIOWrapper(z.open("ffnonbonded.itp"), "utf-8").read()
    out = {}
    for row in sections(text, "atomtypes"):
        if len(row) >= 6:
            out[row[0]] = (float(row[4]), float(row[5]))
    for row in sections(text, "nonbond_params"):  # a self-pair here wins
        if len(row) >= 5 and row[0] == row[1]:
            out[row[0]] = (float(row[3]), float(row[4]))
    return out


OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "cg_radii"
OUT.mkdir(parents=True, exist_ok=True)
for name, table in (("martini2", martini22()), ("martini3", martini3001()),
                    ("sirah", sirah())):  # fmt: skip
    path = OUT / f"{name}_bead_radii.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "sigma_A", "rmin_A", "radius_sigma_A", "radius_rmin_A",
                    "epsilon_kJ_per_mol"])  # fmt: skip
        for t, (s_nm, eps) in sorted(table.items()):
            s = 10.0 * s_nm
            w.writerow([t, f"{s:.4f}", f"{RMIN*s:.4f}", f"{s/2:.4f}",
                        f"{RMIN*s/2:.4f}", f"{eps:.4f}"])  # fmt: skip
    sizes = sorted(10.0 * v[0] for v in table.values())
    print(f"{name:<9} {len(table):4} types, sigma {sizes[0]:.2f}-{sizes[-1]:.2f} A -> {path}")
