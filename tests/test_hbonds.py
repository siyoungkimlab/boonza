"""boonza.hbonds against MDAnalysis HydrogenBondAnalysis, and baker_hubbard /
wernet_nilsson against mdtraj.  Both references compute in float32, so bonds
sitting on a cutoff may differ; MDAnalysis's grid search also misses a few
pairs in the triclinic adk box, which its own calc_bonds/calc_angles confirm."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file

import boonza
from boonza.hbonds import baker_hubbard, donor_hydrogen_pairs, hbonds, wernet_nilsson

HERE = Path(__file__).parent
DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
GRO, XTC, PDB = DATA / "adk_oplsaa.gro", DATA / "adk_oplsaa.xtc", DATA / "adk_oplsaa.pdb"


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


def _index_sel(ids):
    """Compact MDAnalysis 'index' selection with inclusive ranges."""
    ids = np.unique(ids)
    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    runs = np.split(ids, breaks)
    return "index " + " ".join(f"{r[0]}:{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in runs)


def _geometry(system, frame_xyz, box, d, h, a):
    from boonza.hbonds import _angles

    dist = np.linalg.norm(boonza.pbc.minimum_image(frame_xyz[a] - frame_xyz[d], box))
    ang = np.degrees(_angles(frame_xyz[[d]], frame_xyz[[h]], frame_xyz[[a]], box))[0]
    return dist, ang


@pytest.fixture(scope="module")
def adk():
    s = boonza.load(_need(GRO))
    frames = boonza.open_trajectory(_need(XTC), s).read()
    return s, frames


def test_matches_mdanalysis_over_trajectory(adk):
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

    s, frames = adk
    ours = hbonds(s, frames)
    dh = donor_hydrogen_pairs(s)
    anum = s.atoms["anum"]
    u = mda.Universe(str(GRO), str(XTC))
    hba = HydrogenBondAnalysis(
        u, donors_sel=_index_sel(dh[:, 0]), hydrogens_sel=_index_sel(dh[:, 1]),
        acceptors_sel=_index_sel(np.flatnonzero((anum == 7) | (anum == 8))),
        d_h_cutoff=1.2, d_a_cutoff=3.0, d_h_a_angle_cutoff=150, update_selections=False,
    )  # fmt: skip
    hba.run()
    ref = hba.results.hbonds
    assert len(frames) == 10 and len(ref) > 1000
    from MDAnalysis.lib.distances import calc_angles, calc_bonds

    for f in range(len(frames)):
        theirs = {tuple(map(int, r[1:4])) for r in ref[ref[:, 0] == f]}
        sel = ours.frame == f
        mine = set(zip(ours.donor[sel].tolist(), ours.hydrogen[sel].tolist(),
                       ours.acceptor[sel].tolist(), strict=True))  # fmt: skip
        u.trajectory[f]
        pos, dims = u.atoms.positions, u.dimensions
        for d, h, a in mine - theirs:
            # MDAnalysis's own geometry confirms the bond: its pair search missed it
            dist = calc_bonds(pos[[d]], pos[[a]], box=dims)[0]
            ang = np.degrees(calc_angles(pos[[d]], pos[[h]], pos[[a]], box=dims)[0])
            assert 1.0 < dist <= 3.0 + 1e-4 and ang > 150.0 - 1e-3, (f, d, h, a, dist, ang)
        for d, h, a in theirs - mine:  # only bonds sitting on a cutoff
            dist, ang = _geometry(s, frames.positions[f], frames.boxes[f], d, h, a)
            assert abs(dist - 3.0) < 1e-4 or abs(ang - 150.0) < 1e-3, (f, d, h, a, dist, ang)
        assert len(mine ^ theirs) <= max(5, len(theirs) // 1000)  # each one checked above
    per = ours.per_frame()
    assert per.sum() == len(ours) and len(per) == 10
    ref_d = {tuple(map(int, r[:4])): r[4] for r in ref}
    for k in range(0, len(ours), 97):  # distances agree with MDAnalysis
        key = (int(ours.frame[k]), int(ours.donor[k]), int(ours.hydrogen[k]), int(ours.acceptor[k]))
        if key in ref_d:
            assert ours.distance[k] == pytest.approx(ref_d[key], abs=1e-4)


def _mdtraj_hbonds(system, pdb, tmp_path):
    from test_dssp import MDTRAJ_PYTHON, _mdtraj_available

    if not _mdtraj_available():
        pytest.skip("mdtraj not available; set BOONZA_MDTRAJ_PYTHON")
    bonds = tmp_path / "bonds.json"
    bonds.write_text(json.dumps(np.column_stack([system.bonds["i"], system.bonds["j"]]).tolist()))
    r = subprocess.run([MDTRAJ_PYTHON, str(HERE / "mdtraj_oracle.py"), "hbonds", str(bonds),
                        str(pdb)], capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.parametrize("name", ["adk", "ww"])
def test_matches_mdtraj(name, tmp_path):
    if name == "adk":
        pdb = _need(PDB)
    else:  # the solvated WW domain from msys, through PDB so both read the same coordinates
        pdb = tmp_path / "ww.pdb"
        boonza.save(boonza.load(msys_file("ww.dms")), pdb)
    s = boonza.load(pdb)
    ref = _mdtraj_hbonds(s, pdb, tmp_path)
    assert sorted(ref["water"]) == s.select("water").ids.tolist()
    box = s.cell
    for label, ours, theirs, check in (
        ("baker_hubbard", baker_hubbard(s), ref["baker_hubbard"], (2.5, 120.0)),
        ("wernet_nilsson", wernet_nilsson(s)[0], ref["wernet_nilsson"][0], None),
    ):
        mine = set(map(tuple, np.asarray(ours).tolist()))
        want = set(map(tuple, theirs))
        assert len(want) > 10, label
        for d, h, a in mine ^ want:
            dist_da, _ = _geometry(s, s.positions, box, d, h, a)
            dist_ha = np.linalg.norm(boonza.pbc.minimum_image(s.positions[a] - s.positions[h], box))
            from boonza.hbonds import _angles

            if check:  # H...A distance or D-H...A angle on the cutoff
                ang = np.degrees(_angles(s.positions[[d]], s.positions[[h]], s.positions[[a]],
                                         box))[0]  # fmt: skip
                assert abs(dist_ha - 2.5) < 1e-4 or abs(ang - 120.0) < 1e-3, (label, d, h, a)
            else:  # the cone boundary
                theta = np.degrees(_angles(s.positions[[h]], s.positions[[d]],
                                           s.positions[[a]], box))[0]  # fmt: skip
                assert abs(dist_da - (3.3 - 0.00044 * theta**2)) < 1e-4 or abs(theta - 45) < 1e-3
        assert len(mine ^ want) <= 2, label


def test_frames_between_and_counts(adk):
    s, frames = adk
    all_hb = hbonds(s, frames.positions[:3], box=frames.boxes[:3])
    same = hbonds(s, boonza.trajectory.Frames(frames.indices[:3], frames.positions[:3],
                                              frames.boxes[:3], frames.times[:3],
                                              frames.steps[:3]))  # fmt: skip
    assert len(all_hb) == len(same) and (all_hb.acceptor == same.acceptor).all()
    pw = hbonds(s, frames.positions[:3], box=frames.boxes[:3], between=["protein", "water"])
    prot = set(s.select("protein").ids.tolist())
    wat = set(s.select("water").ids.tolist())
    assert 0 < len(pw) < len(all_hb)
    for d, a in zip(pw.donor.tolist(), pw.acceptor.tolist(), strict=True):
        assert (d in prot and a in wat) or (d in wat and a in prot)
    freq = all_hb.frequency()
    assert max(freq.values()) == 1.0 and min(freq.values()) >= 1 / 3
    assert sum(all_hb.counts().values()) == len(all_hb)
    single = hbonds(s)  # system positions and cell
    assert len(single) > 0 and single.nframes == 1
