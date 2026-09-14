"""Bond records (PDB SSBOND/LINK, mmCIF _struct_conn) vs gemmi, and PDB vs mmCIF bonds."""

import json
import subprocess

import pytest
from test_cif import GEMMI_PYTHON, HERE, _gemmi_available

import boonza

DATA = HERE / "data"
IDS = ["1LYZ", "1HHO", "1MBN", "2TRX"]


def _gemmi_connections(path):
    if not _gemmi_available():
        pytest.skip("gemmi not available; set BOONZA_GEMMI_PYTHON")
    r = subprocess.run([GEMMI_PYTHON, str(HERE / "gemmi_oracle.py"), "connections", str(path)],
                       capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    return {frozenset((tuple(a[:4]), tuple(b[:4]))) for a, b, _ in json.loads(r.stdout)}


def _bond_labels(s):
    """Bonds as pairs of (chain, resid, insertion, atom name); altloc copies collapse."""
    res = s.atoms["residue"]
    chain = s.chains["name"][s.residues["chain"][res]]
    lab = list(
        zip(
            chain.tolist(),
            s.residues["resid"][res].tolist(),
            s.residues["insertion"][res].tolist(),
            s.atoms["name"].tolist(),
            strict=True,
        )
    )
    return {frozenset((lab[i], lab[j]))
            for i, j in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True)}  # fmt: skip


@pytest.mark.parametrize("pid", IDS)
@pytest.mark.parametrize("fmt", ["pdb", "cif"])
def test_record_bonds_match_gemmi(pid, fmt):
    path = DATA / f"{pid}.{fmt}"
    kw = {"conect": False} if fmt == "pdb" else {}
    s = boonza.load(path, guess_bonds=False, **kw)  # only SSBOND/LINK or _struct_conn
    want = _gemmi_connections(path)
    assert want and _bond_labels(s) == want


@pytest.mark.parametrize("pid", IDS)
def test_pdb_and_mmcif_give_the_same_bonds(pid):
    p = boonza.load(DATA / f"{pid}.pdb", conect=False)
    c = boonza.load(DATA / f"{pid}.cif")
    assert p.natoms == c.natoms
    assert _bond_labels(p) == _bond_labels(c)
    plain = boonza.load(DATA / f"{pid}.cif", struct_conn=False)
    assert _bond_labels(plain) < _bond_labels(c)


def _atom(serial, name, alt, resname, resid, x, element):
    return (
        f"{'HETATM' if resname == 'ZN' else 'ATOM':<6}{serial:5d} {name:<4}{alt}{resname:>3} A"
        f"{resid:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2}"
    )


def _link(name1, alt1, res1, seq1, name2, res2, seq2):
    rec = [" "] * 72
    for col, text in ((0, "LINK"), (12, f"{name1:<4}"), (16, alt1), (17, res1), (21, "A"),
                      (22, f"{seq1:4d}"), (42, f"{name2:<4}"), (47, res2), (51, "A"),
                      (52, f"{seq2:4d}"), (59, "  1555"), (66, "  1555")):  # fmt: skip
        rec[col : col + len(text)] = list(text)
    return "".join(rec)


def test_link_and_alternate_locations(tmp_path):
    atoms = [_atom(1, "NE2", "A", "HIS", 10, 0.0, "N"), _atom(2, "NE2", "B", "HIS", 10, 0.4, "N"),
             _atom(3, "ZN", " ", "ZN", 100, 6.0, "ZN")]  # fmt: skip
    path = tmp_path / "link.pdb"

    def zn_partners(alt):
        path.write_text("\n".join([_link("NE2", alt, "HIS", 10, "ZN", " ZN", 100), *atoms,
                                   "END"]) + "\n")  # fmt: skip
        s = boonza.load(path)
        return {int(b) for b in s.bonded_atoms(2)}

    assert zn_partners(" ") == {0, 1}  # no altloc given: both copies are linked
    assert zn_partners("A") == {0}
    path.write_text("\n".join([_link("NE2", " ", "HIS", 10, "ZN", " ZN", 100), *atoms]) + "\n")
    assert not boonza.load(path, link=False).bonded_atoms(2).size
