import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.secondary import backbone_hbonds, dssp

HERE = Path(__file__).parent
MDTRAJ_PYTHON = os.path.expanduser(
    os.environ.get("BOONZA_MDTRAJ_PYTHON", "~/miniforge3/envs/ommflow/bin/python")
)
DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
FILES = [
    DATA / "1hvr.pdb",
    DATA / "4E43.pdb",
    DATA / "adk_closed.pdb",
    DATA / "adk_open.pdb",
    DATA / "cobrotoxin.pdb",
    DATA / "19hc.pdb.gz",
    DATA / "1a28.pdb.gz",
    DATA / "1osm.pdb.gz",
    DATA / "nmr_neopetrosiamide.pdb",
    Path("~/msys/tests/files/3RYZ.pdb").expanduser(),
]


@functools.cache
def _mdtraj_available() -> bool:
    try:
        r = subprocess.run([MDTRAJ_PYTHON, "-c", "import mdtraj"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def run_mdtraj(path):
    if not _mdtraj_available():
        pytest.skip("mdtraj not available; set BOONZA_MDTRAJ_PYTHON")
    r = subprocess.run([MDTRAJ_PYTHON, str(HERE / "mdtraj_oracle.py"), str(path)],
                       capture_output=True, text=True)  # fmt: skip
    if r.returncode:
        raise RuntimeError(f"mdtraj oracle failed:\n{r.stderr}")
    return json.loads(r.stdout)


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


def _assigned(keys, codes):
    return [(k[0], k[1], c) for k, c in zip(keys, codes, strict=True) if c != "NA"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_matches_mdtraj(path):
    ref = run_mdtraj(_need(path))
    s = boonza.load(path)
    assert s.ncts == len(ref["codes"])  # one ct per model
    for model, ref_codes in enumerate(ref["codes"]):
        sub = s.clone(s.ct_atoms(model)) if s.ncts > 1 else s
        res = sub.residues
        keys = list(zip(res["chain"].tolist(), res["resid"].tolist(), strict=True))
        keys = [(sub.chain(c).name, r) for c, r in keys]
        ours = _assigned(keys, dssp(sub, breaks=False)[0].tolist())  # mdtraj: no breaks
        # mdtraj reports a blank chain id as " "
        theirs = _assigned([(k[0].strip(), k[1]) for k in ref["residues"]], ref_codes)
        assert ours == theirs


def test_frames_and_simplified():
    s = boonza.load(_need(DATA / "adk_open.pdb"))
    rng = np.random.default_rng(0)
    frames = np.stack([s.positions, s.positions + rng.normal(0, 0.3, s.positions.shape)])
    both = dssp(s, frames)
    assert both.shape == (2, s.nresidues)
    np.testing.assert_array_equal(both[0], dssp(s)[0])
    np.testing.assert_array_equal(both[1], dssp(s, frames[1])[0])
    simple = dssp(s, simplified=True)[0]
    assert set(simple.tolist()) <= {"H", "E", "C", "NA"}
    full = dssp(s)[0]
    assert ((full == "G") <= (simple == "H")).all() and ((full == "B") <= (simple == "E")).all()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_backbone_dihedrals_match_mdtraj(path, capsys):
    from boonza.secondary import backbone_dihedrals

    _need(path)
    if not _mdtraj_available():
        pytest.skip("mdtraj not available; set BOONZA_MDTRAJ_PYTHON")
    r = subprocess.run([MDTRAJ_PYTHON, str(HERE / "mdtraj_oracle.py"), "dihedrals", str(path)],
                       capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    ref = json.loads(r.stdout)
    s = boonza.load(path)
    if s.ncts > 1:
        s = s.clone(s.ct_atoms(0))
    # mdtraj renames residues (HSD -> HIS, SOL -> HOH), so align by chain and resid only
    chains, resids = s.residues["chain"].tolist(), s.residues["resid"].tolist()
    keys = [[s.chain(c).name, int(rid)] for c, rid in zip(chains, resids, strict=True)]
    assert keys == [[k[0].strip(), k[1]] for k in ref["residues"]]
    ours = backbone_dihedrals(s)
    nres = s.nresidues
    atom_res, atom_names = s.atoms["residue"], s.atoms["name"]
    for name in ("phi", "psi", "omega"):
        mine = getattr(ours, name)[0]
        theirs = np.array([np.nan if v is None else v for v in ref[name]])
        both = ~np.isnan(mine) & ~np.isnan(theirs)
        assert both.sum() > 0.8 * (~np.isnan(theirs)).sum(), name
        diff = (mine[both] - theirs[both] + 180) % 360 - 180
        assert np.abs(diff).max() < 0.02, name
        assert not (~np.isnan(mine) & np.isnan(theirs)).any(), name
        # mdtraj links consecutive residues regardless of geometry; we stop at chain breaks
        for i in np.flatnonzero(np.isnan(mine) & ~np.isnan(theirs)).tolist():
            j = i + 1 if name == "psi" else i  # the residue after the peptide bond
            assert 0 < j < nres
            n_atom = np.flatnonzero((atom_res == j) & (atom_names == "N"))[0]
            c_atom = np.flatnonzero((atom_res == j - 1) & (atom_names == "C"))[0]
            assert np.linalg.norm(s.positions[n_atom] - s.positions[c_atom]) >= 2.5


def test_backbone_dihedral_frames_and_cli(capsys):
    from boonza.cli import main
    from boonza.secondary import backbone_dihedrals

    path = _need(DATA / "adk_open.pdb")
    s = boonza.load(path)
    frames = np.stack([s.positions, s.positions + 5.0])
    phi, psi, omega = backbone_dihedrals(s, frames)
    assert phi.shape == (2, s.nresidues)
    np.testing.assert_allclose(phi[0], phi[1], atol=1e-9, equal_nan=True)
    assert np.isnan(phi[0, 0]) and np.isnan(psi[0, -1]) and np.nanmean(np.abs(omega)) > 170
    assert main(["phipsi", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["chain", "resid", "res", "phi", "psi", "omega"]
    assert len(lines) == s.nresidues + 1


def test_backbone_hbonds():
    s = boonza.load(_need(DATA / "adk_open.pdb"))
    donor, acceptor, energy = backbone_hbonds(s)
    assert len(donor) > 100
    assert (energy < -0.5).all() and (energy >= -9.9).all()
    assert not np.any(s.residues["name"][donor] == "PRO")


def _wrapped(xyz, box, shift):
    """Atoms wrapped one by one into the cell: the protein is split across its faces."""
    frac = (xyz + shift) @ np.linalg.inv(box)
    return (frac - np.floor(frac)) @ box


def _widths(box):
    a, b, c = box
    vol = abs(np.linalg.det(box))
    return vol / np.array([np.linalg.norm(np.cross(b, c)), np.linalg.norm(np.cross(c, a)),
                           np.linalg.norm(np.cross(a, b))])  # fmt: skip


def test_periodic_boxes_orthorhombic_and_triclinic():
    s = boonza.load(_need(DATA / "adk_open.pdb"))
    ref = dssp(s)[0]
    donors = backbone_hbonds(s)
    ext = np.full(3, np.ptp(s.positions, axis=0).max() + 45.0)  # room for the tilted box
    boxes = [np.diag(ext), np.array([[ext[0], 0, 0], [0.3 * ext[1], ext[1], 0],
                                     [-0.25 * ext[2], 0.2 * ext[2], ext[2]]])]  # fmt: skip
    for box in boxes:
        assert (_widths(box) > np.ptp(s.positions, axis=0).max() + 10).all()  # no self-contact
        frames = np.stack([_wrapped(s.positions, box, sh) for sh in ([0, 0, 0], [31, -17, 44])])
        np.testing.assert_array_equal(dssp(s, frames, box=box), np.stack([ref, ref]))
        assert (dssp(s, frames[1]) != ref).any()  # without the box the split protein differs
        block = boonza.trajectory.Frames(np.arange(2), frames, np.stack([box, box]),
                                         np.zeros(2), np.zeros(2))  # fmt: skip
        np.testing.assert_array_equal(dssp(s, block), np.stack([ref, ref]))  # boxes from Frames
        donor, acceptor, energy = backbone_hbonds(s, frames[1], box=box)
        np.testing.assert_array_equal(donor, donors[0])
        np.testing.assert_array_equal(acceptor, donors[1])
        np.testing.assert_allclose(energy, donors[2], atol=1e-4)  # float32 of wrapped coordinates


def test_trajectory_input(tmp_path):
    s = boonza.load(_need(DATA / "adk_open.pdb"))
    box = np.diag(np.ptp(s.positions, axis=0) + 30.0)
    frames = np.stack([_wrapped(s.positions, box, [7.0 * k, 0, -5.0 * k]) for k in range(3)])
    path = tmp_path / "wrapped.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for f in frames:
            w.write(f, box=box)
    got = dssp(s, boonza.open_trajectory(path, s))
    np.testing.assert_array_equal(got, np.stack([dssp(s)[0]] * 3))


def test_geometric_chain_breaks():
    s = boonza.load(_need(DATA / "adk_open.pdb"))
    ref = dssp(s)[0]
    np.testing.assert_array_equal(ref, dssp(s, breaks=False)[0])  # an intact chain: no breaks
    # delete a residue in the middle of the longest helix
    runs, start = [], None
    for i, c in enumerate([*ref.tolist(), "-"]):
        if c == "H" and start is None:
            start = i
        elif c != "H" and start is not None:
            runs.append((i - start, start))
            start = None
    length, first = max(runs)
    assert length >= 12
    gap = first + length // 2
    cut = s.copy()
    cut.delete_residues([gap])  # the residue after the gap now has index `gap`
    broken = dssp(cut)[0]
    # a helix cannot continue across the break: the residue after it is never alpha
    assert broken[gap] != "H"
    # far from the gap nothing changes
    far = np.abs(np.arange(cut.nresidues) - gap) > 8
    theirs = np.delete(ref, gap)
    assert (broken[far] == theirs[far]).mean() > 0.97
    # mdtraj-style (no geometric breaks) lets the helix run through the gap
    assert (dssp(cut, breaks=False)[0][gap - 2 : gap + 2] != broken[gap - 2 : gap + 2]).any()
