import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza import viewer
from boonza.cli import _parser

DATA = Path(__file__).parent / "data"


@pytest.fixture
def served():
    """A function serving a Session on a free port; returns (session, get, post)."""
    servers = []

    def serve(session):
        server = viewer.make_server(session, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        base = f"http://127.0.0.1:{server.server_address[1]}/"

        def get(path):
            try:
                with urllib.request.urlopen(base + path) as r:
                    return r.status, r.headers, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.headers, e.read()

        def post(path, body, token=server.RequestHandlerClass.token):
            req = urllib.request.Request(base + path, json.dumps(body).encode(), method="POST",
                                         headers={"X-Boonza-Token": token})  # fmt: skip
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        return get, post

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


def test_topology_matches_the_system():
    s = boonza.load(DATA / "1LYZ.cif")
    topo = viewer.Session(s).topology()
    assert not topo["cg"] and topo["natoms"] == s.natoms and topo["nframes"] == 1
    assert topo["bonds"]["i"] == s.bonds["i"].tolist()
    assert topo["bonds"]["j"] == s.bonds["j"].tolist()
    assert topo["bonds"]["order"] == s.bonds["order"].tolist()
    assert len(topo["ss"]) == s.nresidues and "H" in topo["ss"] and "E" in topo["ss"]
    ca = s.select("protein and name CA").ids
    assert sorted(t for t in topo["residues"]["trace"] if t >= 0) == ca.tolist()
    assert [r["style"] for r in topo["reps"]][0] == "cartoon"
    json.dumps(topo)  # plain Python types only


@pytest.mark.parametrize("path, cg", [
    ("1HHO.pdb", False),
    ("1LYZ.cif", False),
    ("martini/2TRX/cg.pdb.gz", True),
    ("martini/DPPC_bilayer/min.gro.gz", True),  # no bonds at all
    ("sirah/crambin_cg.dms.gz", True),  # N-CA-O spacing on the backbone
    ("sirah/dna_cg.dms.gz", True),
])  # fmt: skip
def test_coarse_grained_by_spacing(path, cg):
    assert viewer.is_coarse_grained(boonza.load(DATA / path)) is cg


def test_coarse_grained_topology():
    s = boonza.load(DATA / "sirah/crambin_cg.dms.gz")
    topo = viewer.Session(s).topology()
    assert topo["cg"] and {"H", "E"} <= set(topo["ss"])  # from the trace (P-SEA)
    # bead sizes from the force field: sigma / 2, not element radii
    from boonza.sites import particle_radii

    expected = particle_radii(s, np.arange(s.natoms))
    assert np.allclose(topo["atoms"]["radius"], expected, atol=1e-3)
    gc = [a for a, n in enumerate(topo["atoms"]["name"]) if n == "GC"]
    assert sorted(t for t in topo["residues"]["trace"] if t >= 0) == gc
    assert topo["reps"][0]["style"] == "cartoon"

    martini = viewer.Session(boonza.load(DATA / "martini/2TRX/cg.pdb.gz")).topology()
    assert set(martini["atoms"]["radius"]) == {viewer.CG_RADIUS}  # no types to size by
    bb = [a for a, n in enumerate(martini["atoms"]["name"]) if n == "BB"]
    assert sorted(t for t in martini["residues"]["trace"] if t >= 0) == bb


def test_server_frames_selections_and_files(served):
    s = boonza.load(DATA / "1LYZ.cif")
    get, post = served(viewer.Session(s, whole=False))

    status, _, body = get("api/topology")
    assert status == 200 and json.loads(body)["natoms"] == s.natoms
    status, headers, body = get("api/frame?i=0")
    assert status == 200
    assert np.allclose(np.frombuffer(body, "<f4").reshape(-1, 3), s.positions, atol=1e-4)
    assert np.allclose(json.loads(headers["X-Box"]), s.cell)
    assert get("api/frame?i=1")[0] == 400

    sel = "same residue as within 5 of resname NAG"
    status, out = post("api/select", {"sel": sel})
    assert status == 200 and out["ids"] == s.select(sel).ids.tolist()
    status, out = post("api/select", {"sel": "resname ("})
    assert status == 400 and "error" in out

    assert get("")[0] == 200 and get("main.js")[0] == 200
    for path in ("../viewer.py", "..%2Fviewer.py", "nope.js", "data/viewer/main.js"):
        assert get(path)[0] == 404


def test_trajectory_glue(served, tmp_path):
    s = boonza.load(DATA / "1LYZ.cif")
    path = tmp_path / "moved.dcd"
    shifts = np.array([[0, 0, 0], [5, 0, 0], [0, -3, 2]], float)
    with boonza.open_writer(path, s.natoms) as w:
        for shift in shifts:
            w.write(s.positions + shift)
    traj = boonza.open_trajectory(path, s)

    session = viewer.Session(s, traj, whole=False)
    get, post = served(session)
    topo = json.loads(get("api/topology")[2])
    assert topo["nframes"] == 3
    frame = np.frombuffer(get("api/frame?i=1")[2], "<f4").reshape(-1, 3)
    assert np.allclose(frame, s.positions + shifts[1], atol=1e-3)
    # within: evaluated on the frame asked for
    out = post("api/select", {"sel": "x > 20", "frame": 1})[1]
    assert out["ids"] == np.flatnonzero(s.positions[:, 0] + 5 > 20).tolist()

    # fit every frame onto the first: rigid shifts disappear
    assert post("api/glue", {"whole": False, "fit": "name CA"}) == (200, {"ok": True})
    frame = np.frombuffer(get("api/frame?i=2")[2], "<f4").reshape(-1, 3)
    assert np.allclose(frame, s.positions, atol=1e-3)
    assert post("api/glue", {"fit": "resname XXX"})[0] == 400


def test_viewed_subset():
    s = boonza.load(DATA / "1HHO.pdb")
    session = viewer.Session(s, atoms="not water")
    assert session.system.natoms == len(s.select("not water"))
    pos, _ = session.frame(0)
    assert np.allclose(pos, s.positions[s.select("not water").ids], atol=1e-4)
    with pytest.raises(ValueError, match="no atoms"):
        viewer.Session(s, atoms="resname XXX")


def test_cli_arguments():
    args = _parser().parse_args(["view", "top.top", "run.xtc", "-c", "conf.gro", "-s",
                                 "not water", "--fit", "name CA", "--port", "0",
                                 "--no-browser"])  # fmt: skip
    assert (args.paths, args.coordinates) == (["top.top", "run.xtc"], "conf.gro")
    assert args.selection == "not water" and args.fit == "name CA" and args.no_browser


# --- folders boonza's commands write ---------------------------------------------


def _rotated(s, angle: float, shift):
    c, d = np.cos(angle), np.sin(angle)
    out = s.copy()
    out.positions = s.positions @ np.array([[c, -d, 0], [d, c, 0], [0, 0, 1]]).T + shift
    return out


def make_modes(d: Path) -> Path:
    """A boonza modes folder: three medoid poses, their rows, the ligand drawn."""
    s = boonza.load(DATA / "1HHO.pdb")
    (d / "structures").mkdir(parents=True)
    rows = ["run,mode,mode_runs,mode_spread,medoid,is_medoid,drifted,population"]
    for n, (count, run) in enumerate([(5, "r07"), (3, "r02"), (1, "r11")]):
        boonza.save(s, d / "structures" / f"mode{n:02d}_n{count}_{run}_pose.pdb")
        for k in range(count):
            name = run if k == 0 else f"{run}x{k}"
            rows.append(f"{name},{n},{count},0.1,{run},{int(k == 0)},{0.5 * n},0.9")
    (d / "modes.csv").write_text("\n".join(rows) + "\n")
    (d / "modes.json").write_text(json.dumps({"ligandsel": "resname HEM and chain A"}))
    return d


def make_triage(d: Path) -> Path:
    """A boonza triage folder: each run's pose where its own MD box left it."""
    s = boonza.load(DATA / "1HHO.pdb")
    (d / "structures").mkdir(parents=True)
    summary = ["run,outcome,frames,drifted,population,settled"]
    for k, run in enumerate(["run_a", "run_b", "run_c"]):
        boonza.save(
            _rotated(s, 0.4 * k, [20.0 * k, -5.0 * k, 3.0]), d / "structures" / f"{run}_pose.pdb"
        )
        summary.append(f"{run},finished,100,{k}.5,0.{9 - k},{k % 2}")
    (d / "summary.csv").write_text("\n".join(summary) + "\n")
    (d / "triage.json").write_text(json.dumps({"ligandsel": "resname HEM"}))
    return d


def make_cavity(d: Path, holo: bool = False, old: bool = False) -> Path:
    """A boonza cavity folder: the structure and its ligand (or apo, and holo with
    the ligand), two pockets, hotspots; ``old``: the ligand in a file of its own, as
    written before ligand.json."""
    s = boonza.load(DATA / "1HHO.pdb")
    d.mkdir(parents=True)
    lig = "chain A and resid 143 and resname HEM"
    if old:
        boonza.save(s.select("protein").clone(), d / "structure.pdb")
        boonza.save(s.select(lig).clone(), d / "ligand.pdb")
    elif holo:
        boonza.save(s.select("protein").clone(), d / "apo.pdb")
        boonza.save(s, d / "holo.pdb")
        (d / "ligand.json").write_text(json.dumps({"file": "holo.pdb", "selection": lig}))
    else:
        boonza.save(s, d / "structure.pdb")
        (d / "ligand.json").write_text(json.dumps({"file": "structure.pdb", "selection": lig}))
    heme = s.select("resname HEM and chain A").ids
    lines = []
    for k, centre in ((1, s.positions[heme].mean(0)), (2, s.positions[:50].mean(0))):
        for j in range(4):
            x, y, z = centre + [j, 0, 0]
            lines.append(f"ATOM  {len(lines) + 1:5d}    C STP  {k:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                         f"  0.00   0.50")  # fmt: skip
    (d / "sites.pqr").write_text("\n".join(lines + ["END"]) + "\n")
    (d / "pockets.csv").write_text("rank,p,volume,lining\n1,0.91,320,A:HIS87\n2,0.40,150,\n")
    x, y, z = s.positions[heme].mean(0)
    (d / "pharm_1.pdb").write_text(f"HETATM    1  X   ACC H   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                                   "  3.00  4.20           X\nEND\n")  # fmt: skip
    (d / "view.pml").write_text("alter pharm_1 and id 1, vdw=1.75\n")
    return d


def test_modes_folder(tmp_path, served):
    v = viewer.from_paths([make_modes(tmp_path / "modes")])
    assert [m.name for m in v.models] == ["mode 0", "mode 1", "mode 2"]
    assert [m.shown for m in v.models] == [True, False, False]
    assert v.models[0].row["runs"] == 5 and v.models[0].row["share"] == round(5 / 9, 3)
    assert v.columns[:4] == ["mode", "runs", "share", "run"] and "drifted" in v.columns
    get, post = served(v)
    listing = json.loads(get("api/models")[2])
    assert len(listing["models"]) == 3 and listing["columns"] == v.columns
    topo = json.loads(get("api/topology?m=1")[2])
    assert topo["ligand"] == "resname HEM and chain A"
    styles = [(r["style"], r["color"]) for r in topo["reps"]]
    # cartoon, pocket, ligand (in the mode's colour)
    assert styles[:3] == [
        ("cartoon", "white"),
        ("sticks", "element"),
        ("sticks", "element/magenta"),
    ]
    s = v.session(1).system
    pocket = s.select(topo["reps"][1]["sel"])
    assert len(pocket) and not set(pocket.ids) & set(s.select("resname HEM and chain A").ids)
    assert get("api/topology?m=3")[0] == 400


def test_triage_folder_is_fitted(tmp_path):
    v = viewer.from_paths([make_triage(tmp_path / "triage")])
    assert v.align and [m.row.get("drifted") for m in v.models] == [0.5, 1.5, 2.5]
    first, last = v.session(0).system, v.session(2).system
    ca = first.select("name CA").ids
    assert np.sqrt(((first.positions[ca] - last.positions[ca]) ** 2).sum(1).mean()) < 1e-3


@pytest.mark.parametrize("old", [False, True])
def test_cavity_folder(tmp_path, old):
    v = viewer.from_paths([make_cavity(tmp_path / "cavity", old=old)])
    names = [m.name for m in v.models]
    assert names == ["structure", "pocket 1", "pocket 2", "pharmacophore 1"]
    # the ligand is the structure's, so its pocket is drawn
    reps = v.session(0).topology()["reps"]
    assert [r["style"] for r in reps[:3]] == ["cartoon", "sticks", "sticks"]
    if old:
        return
    pocket = v.models[1]
    assert pocket.label == "1" and pocket.row["p"] == 0.91 and pocket.group == "pockets"
    s = v.session(1)
    topo = s.topology()
    assert topo["natoms"] == 4 and topo["atoms"]["radius"] == [0.5] * 4
    assert topo["reps"] == [{"sel": "all", "style": "spheres", "color": "cyan"}]
    hot = v.session(3).topology()
    assert hot["atoms"]["radius"] == [1.75] and hot["reps"][0]["sel"] == "resname ACC"
    assert not v.models[3].shown


def test_cavity_folder_with_holo(tmp_path):
    """apo is the protein alone; the ligand is holo's, a model of its own; holo is off."""
    v = viewer.from_paths([make_cavity(tmp_path / "cavity", holo=True)])
    assert [m.name for m in v.models][:3] == ["apo", "ligand", "holo"]
    assert [m.shown for m in v.models][:3] == [True, True, False]
    apo, lig, holo = (v.session(k).topology() for k in range(3))
    assert "HEM" not in apo["residues"]["name"]
    assert lig["natoms"] == 43 and set(lig["residues"]["name"]) == {"HEM"}
    assert holo["natoms"] > apo["natoms"] and "HEM" in holo["residues"]["name"]


def test_default_reps_and_contacts():
    s = boonza.load(DATA / "1HHO.pdb")
    session = viewer.Session(s, ligand="resname HEM and chain A")
    reps = session.topology()["reps"]
    assert [r["style"] for r in reps[:3]] == ["cartoon", "sticks", "sticks"]
    contacts = session.contacts()
    heme = set(s.select("resname HEM and chain A").ids)
    assert contacts and all(i in heme and j not in heme and d <= 3.5 for i, j, d in contacts)
    # coarse-grained: no ligand in a lone protein, so cartoon alone
    cg = viewer.Session(boonza.load(DATA / "martini/2TRX/cg.pdb.gz")).topology()
    assert [r["style"] for r in cg["reps"]] == ["cartoon"]


def test_several_files_and_a_manifest(tmp_path):
    a = tmp_path / "a.pdb"
    boonza.save(boonza.load(DATA / "1LYZ.cif"), a)
    v = viewer.from_paths([a, DATA / "1HHO.pdb"])
    assert [m.name for m in v.models] == ["a.pdb", "1HHO.pdb"] and len(v.listing()["models"]) == 2

    (tmp_path / "view.json").write_text(json.dumps({
        "title": "mine", "columns": ["score"],
        "models": [{"name": "lyso", "file": "a.pdb", "row": {"score": "1.5"},
                    "reps": [{"sel": "all", "style": "lines", "color": "white"}]}]}))  # fmt: skip
    v = viewer.from_paths([tmp_path])
    assert v.title == "mine" and v.models[0].row == {"score": 1.5}
    assert v.session(0).topology()["reps"][0]["style"] == "lines"
    with pytest.raises(ValueError, match="not a folder"):
        viewer.from_folder(DATA / "martini")


# --- editing, dihedrals and minimization ---------------------------------------------


def _hydrogens(s, a):
    from boonza.viewer.edit import neighbors

    return sum(1 for b, _ in neighbors(s, a) if s.atoms["anum"][b] == 1)


def test_edits_follow_valence():
    from boonza.viewer import edit

    s = boonza.from_smiles("CCO")
    o = int(np.flatnonzero(s.atoms["anum"] == 8)[0])
    edit.set_element(s, [o], "N")
    edit.adjust_hydrogens(s, [o])
    assert str(s.atoms["name"][o]).startswith("N") and _hydrogens(s, o) == 2
    edit.set_charge(s, [o], 1)
    edit.adjust_hydrogens(s, [o])
    assert _hydrogens(s, o) == 3
    c = [b for b, _ in edit.neighbors(s, o) if s.atoms["anum"][b] == 6][0]
    for h in (b for b, _ in edit.neighbors(s, o) if s.atoms["anum"][b] == 1):
        u, v = s.positions[h] - s.positions[o], s.positions[c] - s.positions[o]
        angle = np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
        assert 0.9 < np.linalg.norm(u) < 1.1 and 100 < angle < 120
    edit.set_charge(s, [o], -1)  # an amide anion: one hydrogen
    edit.adjust_hydrogens(s, [o])
    assert _hydrogens(s, o) == 1

    c1, c2 = (int(x) for x in np.flatnonzero(s.atoms["anum"] == 6))
    edit.set_bond(s, c1, c2, 2)
    edit.adjust_hydrogens(s, [c1, c2])
    assert s.find_bond(c1, c2).order == 2 and _hydrogens(s, c1) == 2
    edit.set_bond(s, c1, c2, 0)
    assert s.find_bond(c1, c2) is None
    n = s.natoms
    edit.delete(s, [c1])  # with its hydrogens
    assert s.natoms == n - 3


def test_dihedrals():
    from boonza.viewer.edit import dihedral_angle, rotating_side, set_dihedral

    s = boonza.from_smiles("CCCCO")
    c = [int(x) for x in np.flatnonzero(s.atoms["anum"] == 6)]
    before = s.positions.copy()
    assert abs(set_dihedral(s, *c, 75.0) - 75.0) < 1e-6
    assert abs(dihedral_angle(s.positions, *c) - 75.0) < 1e-6
    # bond lengths are kept, and the side that did not turn stayed put
    moving, _ = rotating_side(s, c[1], c[2])
    still = np.setdiff1d(np.arange(s.natoms), moving)
    assert np.allclose(s.positions[still], before[still])
    for i, j in zip(s.bonds["i"], s.bonds["j"], strict=True):
        d0, d1 = (
            np.linalg.norm(before[i] - before[j]),
            np.linalg.norm(s.positions[i] - s.positions[j]),
        )
        assert abs(d0 - d1) < 1e-6
    ring = boonza.from_smiles("c1ccccc1C")
    with pytest.raises(ValueError, match="ring"):
        rotating_side(ring, 0, 1)
    with pytest.raises(ValueError, match="bonded"):
        rotating_side(ring, 0, 3)


def test_session_edits_undo_and_save(tmp_path, served):
    s = boonza.from_smiles("CCO")
    session = viewer.Session(s)
    get, post = served(session)
    o = int(np.flatnonzero(s.atoms["anum"] == 8)[0])
    status, out = post("api/edit", {"op": "element", "atoms": [o], "value": "N"})
    assert status == 200 and out["topology"] and out["picked"] == [o]
    topo = json.loads(get("api/topology")[2])
    assert topo["natoms"] == 10 and topo["version"] == 1 and topo["editable"]
    status, out = post("api/edit", {"op": "charge", "atoms": [o], "value": 1})
    assert out["natoms"] == 11 and out["picked"] == [o]
    # deleting renumbers: the picks say where atoms went
    from boonza.viewer.edit import neighbors

    hs = [b for b, _ in neighbors(session.system, o) if session.system.atoms["anum"][b] == 1]
    h = hs[0]
    status, out = post("api/edit", {"op": "delete", "atoms": [h], "hydrogens": False})
    assert out["natoms"] == 10 and out["picked"] == [-1]
    assert post("api/edit", {"op": "charge", "atoms": []})[0] == 400
    assert post("api/undo", {})[1]["natoms"] == 11
    assert post("api/save", {"path": str(tmp_path / "out.sdf")})[0] == 200
    assert boonza.load(tmp_path / "out.sdf").natoms == 11

    traj = viewer.Session(s, boonza.open_trajectory(*_one_frame(tmp_path, s)))
    with pytest.raises(ValueError, match="trajectory"):
        traj.edit("charge", [0], 1)


def test_dihedral_endpoints(served):
    s = boonza.from_smiles("CCCCO")
    get, post = served(viewer.Session(s))
    c = [int(x) for x in np.flatnonzero(s.atoms["anum"] == 6)]
    status, out = post("api/dihedral", {"atoms": c})
    assert status == 200 and len(out["moving"]) and out["sign"] in (1, -1)
    status, out = post("api/set_dihedral", {"atoms": c, "angle": -120})
    assert status == 200 and out["angle"] == pytest.approx(-120, abs=1e-3) and not out["topology"]
    pos = np.frombuffer(get("api/frame?i=0")[2], "<f4").reshape(-1, 3).astype(float)
    from boonza.viewer.edit import dihedral_angle

    assert dihedral_angle(pos, *c) == pytest.approx(-120, abs=1e-2)
    ring = viewer.Session(boonza.from_smiles("c1ccccc1"))
    with pytest.raises(ValueError, match="ring"):
        ring.dihedral([0, 1, 2, 3])


def _one_frame(tmp_path, s):
    path = tmp_path / "one.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        w.write(s.positions)
    return path, s


def _amber_tools() -> bool:
    from boonza.gaff import AmberToolsError, find_amber_tools

    try:
        find_amber_tools()
    except AmberToolsError:
        return False
    return True


def test_minimize_holds_the_rest():
    # a protonated peptide the force fields match whole (no AmberTools needed)
    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    session = viewer.Session(s, atoms="protein", cg=False)
    mobile = "resid 2"
    before = session.system.positions.copy()
    out = session.minimize(mobile, restraint=10.0)
    assert out["energy_after"] < out["energy_before"]
    assert out["mobile_atoms"] == len(session.system.select(mobile)) > 0
    after = session.system.positions
    free = session.system.select(mobile).ids
    held = session.system.select(f"not ({mobile}) and not hydrogen").ids
    assert out["held_atoms"] == len(held)
    shift = np.linalg.norm(after[held] - before[held], axis=1)
    assert shift.max() == pytest.approx(out["held_max"], abs=1e-3)
    # hydrogens are never held: those outside the mobile part still relax
    hs = session.system.select(f"not ({mobile}) and hydrogen").ids
    assert np.linalg.norm(after[hs] - before[hs], axis=1).max() > out["held_max"]
    assert out["mobile_max"] > 0 and out["version"] == 1 and not out["topology"]
    assert np.linalg.norm(after[free] - before[free], axis=1).max() == pytest.approx(
        out["mobile_max"], abs=1e-3
    )
    assert session.undo()["version"] == 2 and np.allclose(session.system.positions, before)


@pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")
def test_dihedral_scan_with_a_ligand():
    s = boonza.from_smiles("CCCC")
    session = viewer.Session(s, ligand="all")
    c = [int(x) for x in np.flatnonzero(s.atoms["anum"] == 6)]
    scan = session.dihedral_scan(c, step=30)
    angles, torsion = np.array(scan["angles"]), np.array(scan["torsion"])
    assert len(angles) == 13 and min(torsion) == 0
    # butane: lowest anti (180), highest eclipsed (0)
    assert abs(abs(angles[np.argmin(scan["total"])]) - 180) <= 30
    assert abs(angles[np.argmax(scan["total"])]) <= 30


def test_dihedral_from_a_bond():
    from boonza.viewer.edit import dihedral_from_bond, dihedral_name

    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro").select("protein").clone()

    def at(resid, name):
        return int(s.select(f"resid {resid} and name {name}").ids[0])

    def named(q):
        return [f"{s.atoms['name'][a]}{s.residues['resid'][s.atoms['residue'][a]]}" for a in q]

    cases = {  # bond -> the IUPAC dihedral
        (1, "CA", 1, "CB"): (["N1", "CA1", "CB1", "CG1"], "chi1"),
        (1, "CB", 1, "CA"): (["CG1", "CB1", "CA1", "N1"], "chi1"),  # either way round
        (1, "CD", 1, "NE"): (["CG1", "CD1", "NE1", "CZ1"], "chi4"),
        (1, "NE", 1, "CZ"): (["CD1", "NE1", "CZ1", "NH11"], "chi5"),
        (3, "CB", 3, "CG"): (["CA3", "CB3", "CG3", "OD13"], "chi2"),
        (3, "N", 3, "CA"): (["C2", "N3", "CA3", "C3"], "phi"),
        (1, "CA", 1, "C"): (["N1", "CA1", "C1", "N2"], "psi"),
        (2, "C", 3, "N"): (["CA2", "C2", "N3", "CA3"], "omega"),
    }
    for (r1, n1, r2, n2), (atoms, name) in cases.items():
        q = dihedral_from_bond(s, at(r1, n1), at(r2, n2))
        assert named(q) == atoms and dihedral_name(s, q) == name

    # an alkane: the only heavy neighbours; a branch: the chain that goes on
    butane = boonza.from_smiles("CCCC")
    c = [int(x) for x in np.flatnonzero(butane.atoms["anum"] == 6)]
    assert dihedral_from_bond(butane, c[1], c[2]) == c
    assert butane.atoms["anum"][dihedral_from_bond(butane, c[0], c[1])[0]] == 1  # methyl: an H
    branched = boonza.from_smiles("CC(C)CCC")
    q = dihedral_from_bond(branched, 1, 3)
    assert q[1:3] == [1, 3] and branched.atoms["anum"][q[0]] == 6 and q[3] == 4
    with pytest.raises(ValueError, match="not bonded"):
        dihedral_from_bond(butane, c[0], c[2])


def test_dihedral_endpoint_takes_a_bond(served):
    s = boonza.from_smiles("CCCCO")
    get, post = served(viewer.Session(s))
    c = [int(x) for x in np.flatnonzero(s.atoms["anum"] == 6)]
    status, out = post("api/dihedral", {"atoms": [c[1], c[2]]})
    assert status == 200 and out["atoms"] == c and out["name"] == ""
    assert post("api/dihedral", {"atoms": [c[0], c[2]]})[0] == 400


# --- building, geometry edits, superposition, 2D ---------------------------------------


def _peptide():
    g = DATA / "gmx_amber"
    return boonza.load(g / "topol.top", coordinates=g / "conf.gro").select("protein").clone()


def test_functional_groups_and_molecules(served):
    from boonza.viewer.edit import GROUPS, neighbors

    s = boonza.from_smiles("CCO")
    get, post = served(viewer.Session(s))
    h = [b for b, _ in neighbors(s, 0) if s.atoms["anum"][b] == 1][0]
    status, out = post("api/edit", {"op": "attach", "atoms": [h], "value": "Phenyl"})
    assert status == 200 and out["natoms"] == 9 - 1 + 11  # an H out, C6H5 in
    topo = json.loads(get("api/topology")[2])
    assert (
        topo["atoms"]["element"].count("C") == 8
        and len(set(topo["atoms"]["name"])) == topo["natoms"]
    )
    pos = np.frombuffer(get("api/frame?i=0")[2], "<f4").reshape(-1, 3)
    heavy = [k for k, e in enumerate(topo["atoms"]["element"]) if e != "H"]
    d = np.linalg.norm(pos[heavy][:, None] - pos[heavy][None], axis=2) + 9 * np.eye(len(heavy))
    assert d.min() > 1.2  # bonded, not overlapping
    assert {"Phenyl", "Morpholino", "Trifluoromethyl"} <= set(GROUPS)
    out = post(
        "api/edit", {"op": "molecule", "value": {"smiles": "c1ccccc1", "centre": [10, 0, 0]}}
    )[1]
    assert out["natoms"] == 19 + 12
    assert post("api/edit", {"op": "attach", "atoms": [0], "value": "C"})[0] in (200, 400)


def test_mutations_and_caps_stay_ff19sb():
    from boonza import gaff, viparr
    from boonza.md.config import DEFAULT_FORCEFIELDS
    from boonza.viewer.edit import _residue_atom, cap, mutate, neighbors

    p = _peptide()
    for new in ("GLY", "PHE", "PRO", "LYS", "ASP", "TRP"):
        mutate(p, 1, new)
        assert str(p.residues["name"][1]) == new
        n = _residue_atom(p, 1, "N")
        hs = [b for b, _ in neighbors(p, n) if p.atoms["anum"][b] == 1]
        assert len(hs) == (0 if new == "PRO" else 1)
    assert int(p.atoms["formal_charge"][p.residue_atoms(1)].sum()) == 0  # TRP
    cap(p, 0, "N")
    cap(p, 0, "C")
    # files list the residues in order: ACE first, NME last
    listed = [str(p.residues["name"][r]) for r in dict.fromkeys(p.atoms["residue"].tolist())]
    assert listed == ["ACE", "ARG", "TRP", "ASP", "NME"]
    ffs = [viparr.load_forcefield(b) for b, *_ in DEFAULT_FORCEFIELDS]
    assert not gaff.find_unmatched(p, ffs)  # ff19SB has a template for every residue
    with pytest.raises(ValueError, match="capped"):
        cap(p, 0, "C")


def test_geometry_edits_redo_and_constraints(served):
    from boonza.viewer.edit import bond_angle

    s = boonza.from_smiles("CCCO")
    session = viewer.Session(s)
    get, post = served(session)
    c = [int(x) for x in np.flatnonzero(s.atoms["anum"] > 1)]
    out = post("api/set_distance", {"atoms": c[1:3], "value": 1.7})[1]
    assert out["value"] == pytest.approx(1.7, abs=1e-3) and not out["topology"]
    out = post("api/set_angle", {"atoms": c[:3], "value": 125})[1]
    assert out["value"] == pytest.approx(125, abs=1e-2)
    assert bond_angle(session.system.positions, *c[:3]) == pytest.approx(125, abs=1e-2)
    xyz = session.system.positions[[0]] + 1.0
    post("api/set_positions", {"atoms": [0], "xyz": xyz.ravel().tolist()})
    assert np.allclose(session.system.positions[0], xyz[0])
    post("api/undo", {})
    assert not np.allclose(session.system.positions[0], xyz[0])
    post("api/redo", {})
    assert np.allclose(session.system.positions[0], xyz[0])
    assert post("api/redo", {})[0] == 400  # nothing left to redo

    # constraints: fixed atoms stay exactly, tethered heavy atoms barely move
    p = viewer.Session(_peptide(), cg=False)
    before = p.system.positions.copy()
    fixed = p.system.select("resid 1").ids.tolist()
    tethered = p.system.select("resid 3 and not hydrogen").ids.tolist()
    out = p.minimize(fixed=fixed, tethered=tethered)
    assert out["mobile"] == "constraints" and out["fixed_atoms"] == len(fixed)
    assert np.allclose(p.system.positions[fixed], before[fixed])
    assert out["energy_after"] < out["energy_before"]


def test_align_models(served, tmp_path):
    s = boonza.load(DATA / "1LYZ.cif")
    moved = _rotated(s, 1.1, [30.0, -12.0, 4.0])
    boonza.save(moved, tmp_path / "moved.pdb")
    v = viewer.from_paths([DATA / "1LYZ.cif", tmp_path / "moved.pdb"])
    get, post = served(v)
    for method in ("matchmaker", "superpose", "cealign"):
        status, out = post("api/align", {"m": 1, "ref": 0, "method": method})
        assert status == 200 and out["rmsd"] < 0.05 and out["pairs"] > 100, (method, out)
    ca = s.select("name CA").ids
    assert np.allclose(v.session(1).system.positions[ca], s.positions[ca], atol=0.05)
    assert post("api/align", {"m": 1, "ref": 0, "method": "nope"})[0] == 400


def test_ligand_2d_and_isolated_profile(served):
    s = boonza.load(DATA / "1HHO.pdb")
    session = viewer.Session(s, ligand="resname HEM and chain A")
    get, post = served(session)
    status, _, body = get("api/ligand2d?dark=1")
    out = json.loads(body)
    assert (
        status == 200 and out["svg"].startswith("<?xml") and len(out["atoms"]) == len(out["coords"])
    )
    assert out["name"].startswith("HEM")
    # a molecule alone: the "isolated" curve is its own energy (here, the whole molecule)
    from boonza.viewer.edit import dihedral_scan

    b = boonza.from_smiles("CCCC")
    c = [int(x) for x in np.flatnonzero(b.atoms["anum"] == 6)]
    alone = viewer.Session(b)
    if not _amber_tools():
        pytest.skip("AmberTools (GAFF2) not installed")
    scan = dihedral_scan(b, *c, alone._parameterized(), step=30, alone=list(range(b.natoms)))
    assert np.allclose(scan["isolated"], scan["total"], atol=1e-3)


# --- the Python console and sessions ------------------------------------------------------


def test_python_console(served, tmp_path):
    a = tmp_path / "a.pdb"
    boonza.save(boonza.load(DATA / "1LYZ.cif"), a)
    v = viewer.from_paths([a, DATA / "1HHO.pdb"])
    get, post = served(v)
    status, out = post("api/python", {"code": "model0.natoms"})
    assert status == 200 and out["output"].strip() == str(boonza.load(a).natoms)
    # a line that moves a model: the page hears of it, and Undo puts it back
    before = v.session(0).system.positions.copy()
    out = post("api/python", {"code": "m.positions = m.positions + 1.0", "m": 0})[1]
    assert out["changed"] == [{"m": 0, "topology": False}]
    assert np.allclose(v.session(0).system.positions, before + 1)
    post("api/undo", {"m": 0})
    assert np.allclose(v.session(0).system.positions, before)
    # names stay from line to line; view asks the page for things; errors come back as text
    post("api/python", {"code": "x = m.select('name CA')"})
    out = post(
        "api/python", {"code": "view.select(x); view.color('resname HEM', 'cyan', model=1)"}
    )[1]
    assert [a["do"] for a in out["actions"]] == ["select", "color"]
    assert len(out["actions"][0]["ids"]) == len(v.session(0).system.select("name CA"))
    assert out["actions"][1]["m"] == 1
    out = post("api/python", {"code": "sel", "m": 0, "sel": [0, 1, 2]})[1]
    assert "AtomSel" in out["output"]
    assert "ZeroDivisionError" in post("api/python", {"code": "1/0"})[1]["output"]
    assert post("api/python", {"code": "models['1HHO.pdb'].natoms"})[1]["output"].strip() == "2396"
    # only the page the server wrote may change things
    assert post("api/python", {"code": "1"}, token="nope")[0] == 403
    assert post("api/undo", {"m": 0}, token="")[0] == 403


def test_python_needs_localhost():
    server = viewer.make_server(
        viewer.Session(boonza.load(DATA / "1LYZ.cif")), port=0, python=False
    )
    try:
        assert server.RequestHandlerClass.console is None
    finally:
        server.server_close()
    server = viewer.make_server(viewer.Session(boonza.load(DATA / "1LYZ.cif")), port=0)
    try:
        assert server.RequestHandlerClass.console is not None  # 127.0.0.1 by default
    finally:
        server.server_close()


def test_session_round_trip(served, tmp_path):
    a = tmp_path / "a.pdb"
    boonza.save(boonza.load(DATA / "1HHO.pdb"), a)
    v = viewer.from_paths([a, make_cavity(tmp_path / "cavity")])
    get, post = served(v)
    post("api/set_positions", {"m": 0, "atoms": [0], "xyz": [1.0, 2.0, 3.0]})  # an edit, unsaved
    page = {"camera": {"distance": 42}, "models": {"0": {"shown": True}}}
    status, out = post("api/save_session", {"path": str(tmp_path / "work"), "page": page})
    assert status == 200 and out["path"].endswith("work.bview")
    w = viewer.from_paths([tmp_path / "work.bview"])
    assert [m.name for m in w.models] == [m.name for m in v.models]
    assert np.allclose(w.session(0).system.positions[0], [1, 2, 3])  # the edit came along
    assert w.page["camera"]["distance"] == 42 and w.page["indices"] == list(range(len(v.models)))
    pocket = next(k for k, m in enumerate(w.models) if m.name == "pocket 1")
    assert w.session(pocket).radii is not None  # spheres stay spheres
    assert get("api/models")[0] == 200


# --- bond orders, atom information, the coarse-grained cartoon --------------------------


@pytest.mark.parametrize("smiles", [
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "CC(=O)Oc1ccccc1C(=O)O", "N#Cc1ccccc1", "C#CCO", "CS(=O)(=O)Nc1ccccc1",
    "O=[N+]([O-])c1ccccc1", "C=CC(=O)N", "c1ccncc1C(=O)[O-]",
])  # fmt: skip
def test_guess_bond_orders(smiles):
    def per_atom(s):
        t = np.zeros(s.natoms)
        for i, j, o in zip(s.bonds["i"], s.bonds["j"], s.bonds["order"], strict=True):
            t[i] += o
            t[j] += o
        return t

    # with hydrogens, and without (as a crystal structure has them): every heavy atom
    # gets back its total bond order, whichever Kekule structure the rings take
    s = boonza.from_smiles(smiles)
    want, heavy = per_atom(s), np.asarray(s.atoms["anum"]) > 1
    s.bonds["order"] = np.ones(s.nbonds, np.int64)
    assert boonza.guess_bond_orders(s) > 0
    assert np.allclose(per_atom(s)[heavy], want[heavy])
    t = boonza.from_smiles(smiles)
    keep = np.flatnonzero(np.asarray(t.atoms["anum"]) > 1)
    nh = np.zeros(t.natoms)
    for i, j in zip(t.bonds["i"], t.bonds["j"], strict=True):
        nh[j] += t.atoms["anum"][i] == 1
        nh[i] += t.atoms["anum"][j] == 1
    u = t.clone(keep)
    u.bonds["order"] = np.ones(u.nbonds, np.int64)
    boonza.guess_bond_orders(u)
    assert np.allclose(per_atom(u), (per_atom(t) - nh)[keep])
    assert boonza.guess_bond_orders(u) == 0  # orders there already: left alone


def test_bond_orders_of_a_crystal_structure():
    s = boonza.load(DATA / "1HHO.pdb")
    session = viewer.Session(s)
    assert session.guessed_orders > 400  # backbone carbonyls, rings, heme

    def order(sel_i, sel_j):
        a, b = s.select(sel_i).ids[0], s.select(sel_j).ids[0]
        return s.find_bond(int(a), int(b)).order

    assert order("resid 1 and chain A and name C", "resid 1 and chain A and name O") == 2
    phe = int(s.select("resname PHE and name CG").ids[0])
    r = int(s.atoms["residue"][phe])
    ring = [f"residue {r} and name {n}" for n in ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")]
    orders = [order(ring[k], ring[(k + 1) % 6]) for k in range(6)]
    assert sorted(orders) == [1, 1, 1, 2, 2, 2]  # a Kekule ring


def test_atom_information(served):
    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    get, post = served(viewer.Session(s))
    info = json.loads(get("api/atom?a=4")[2])
    assert info["index"] == 4 and info["force_field"] == "file"
    assert info["charge"] == pytest.approx(float(s.atoms["charge"][4]), abs=1e-4)
    assert info["sigma"] > 1 and info["epsilon"] > 0 and info["type"]
    bare = viewer.Session(boonza.load(DATA / "1LYZ.cif"))
    assert "charge" not in bare.atom_info(0)  # no force field until one is assigned


def test_coarse_grained_secondary_structure():
    sirah = viewer.Session(boonza.load(DATA / "sirah/crambin_cg.dms.gz"))
    ss = "".join(x or "-" for x in sirah.ss(0))
    assert "HHHHHHHHH" in ss and "EEE" in ss  # crambin: two helices, a small sheet
    cg = viewer.Session(boonza.load(DATA / "martini/2TRX/cg.pdb.gz"))
    aa = viewer.Session(boonza.load(DATA / "2TRX.pdb"))
    a, c = [x for x in aa.ss(0) if x], [x for x in cg.ss(0) if x]
    n = min(len(a), len(c))
    assert sum(x == y for x, y in zip(a[:n], c[:n], strict=True)) / n > 0.7  # DSSP on the atoms


def test_export_writes_what_the_server_answers(tmp_path):
    from boonza.viewer.export import export

    s = boonza.load(DATA / "1FNA.pdb")
    v = viewer.Viewer([viewer.Model("1FNA", lambda: viewer.Session(s, title="1FNA"))], title="1FNA")
    got = export(v, tmp_path)
    assert got["models"] == 1 and got["frames"] == 1
    assert 'name="boonza-static"' in (tmp_path / "index.html").read_text()
    assert (tmp_path / "main.js").is_file()
    listing = json.loads((tmp_path / "data/models.json").read_text())
    assert listing["static"] and listing["models"][0]["name"] == "1FNA"
    topo = json.loads((tmp_path / "data/topology-m0.json").read_text())
    assert topo["natoms"] == s.natoms
    assert all("ids" in rep for rep in topo["reps"])
    blob = np.fromfile(tmp_path / "data/frame-m0-i0.bin", "<f4")
    assert blob.size == 3 * s.natoms + 9  # positions, then the box
    np.testing.assert_allclose(blob[:-9].reshape(-1, 3), s.positions, atol=1e-3)


def test_sizes_set_radii_by_selection():
    s = boonza.load(DATA / "1FNA.pdb")
    topo = viewer.Session(s, sizes={"name CA": 0.25}).topology()
    ca = s.select("name CA").ids
    assert np.allclose(np.asarray(topo["atoms"]["radius"])[ca], 0.25)


def test_surface_charges_and_hydrophobicity():
    from boonza.viewer.properties import charges, hydrophobicity

    s = boonza.load(DATA / "1FNA.pdb")  # no partial or formal charges: the pH-7 template
    q = charges(s)
    names = [str(x) for x in s.atoms["name"]]
    rnames = [str(s.residues["name"][r]) for r in s.atoms["residue"]]
    asp = [a for a in range(s.natoms) if rnames[a] == "ASP" and names[a] in ("OD1", "OD2")]
    assert asp and np.allclose(q[asp], -0.5)
    lys = [a for a in range(s.natoms) if rnames[a] == "LYS" and names[a] == "NZ"]
    assert lys and np.allclose(q[lys], 1)
    h = hydrophobicity(s, q)
    leu = [a for a in range(s.natoms) if rnames[a] == "LEU" and names[a] in ("CD1", "CD2")]
    assert leu and np.all(h[leu] == 1)
    assert np.all(h[asp] == -1)
    # beads without a topology: the charged side-chain bead (Martini 2 and 3 alike)
    s = boonza.load(DATA / "martini" / "2TRX" / "cg.pdb.gz")
    q = charges(s, cg=True)
    names = [str(x) for x in s.atoms["name"]]
    rnames = [str(s.residues["name"][r]) for r in s.atoms["residue"]]
    for res, bead, charge in (("ASP", "SC1", -1), ("GLU", "SC1", -1), ("LYS", "SC2", 1)):
        here = [a for a in range(s.natoms) if rnames[a] == res and names[a] == bead]
        assert here and np.all(q[here] == charge)
    assert np.count_nonzero(q) == sum(r in ("ASP", "GLU", "LYS", "ARG") for r in s.residues["name"])


def test_memory_trajectory_and_frame_links():
    s = boonza.load(DATA / "1FNA.pdb")
    pos = np.asarray(s.positions, float)
    traj = viewer.MemoryTrajectory(np.array([pos, pos + 1.0]))
    session = viewer.Session(s, traj, whole=False)
    assert session.nframes == 2
    np.testing.assert_allclose(session.frame(1)[0], pos + 1.0, atol=1e-4)
    m = viewer.Model("site", lambda: session, frame={"model": "frames", "i": 1})
    assert m.info()["frame"] == {"model": "frames", "i": 1}


def test_a_model_swatch_apart_from_its_protein():
    s = boonza.load(DATA / "1FNA.pdb")
    m = viewer.Model("pocket 1", lambda: viewer.Session(s), color="gray", swatch="cyan")
    assert (m.info()["color"], m.info()["swatch"]) == ("gray", "cyan")
    assert "swatch" not in viewer.Model("apo", lambda: viewer.Session(s)).info()


def test_an_empty_view_builds_molecules(served):
    get, post = served(viewer.Viewer([]))
    assert json.loads(get("api/models")[2])["models"] == []
    assert _parser().parse_args(["view"]).paths == []
    status, out = post("api/build", {"smiles": "c1ccccc1O", "centre": [1, 2, 3]})
    assert status == 200 and out["added"] == [0] and out["models"][0]["group"] == "built"
    pos = np.frombuffer(get("api/frame?m=0&i=0")[2], "<f4").reshape(-1, 3)
    assert np.allclose(pos.mean(0), [1, 2, 3], atol=1e-3)
    mols = json.loads(get("api/smiles?m=0")[2])["molecules"]
    assert [x["smiles"] for x in mols] == ["Oc1ccccc1"]
    # a second molecule into the same model: a residue number of its own, its own SMILES
    acid = {"smiles": "CC(=O)O", "centre": [9, 0, 0]}
    out = post("api/edit", {"m": 0, "op": "molecule", "value": acid})[1]
    topo = json.loads(get("api/topology?m=0")[2])
    assert len(set(topo["residues"]["resid"])) == 2
    mols = json.loads(get(f"api/smiles?m=0&sel={out['natoms'] - 1}")[2])["molecules"]
    assert [x["smiles"] for x in mols] == ["CC(=O)O"]
    assert len(json.loads(get("api/smiles?m=0")[2])["molecules"]) == 2
    # built: every atom drawn, hydrogens too
    assert topo["reps"] == [{"sel": "all", "style": "ballstick", "color": "element"}]


def test_delete_from_a_trajectory_and_from_points(tmp_path):
    s = boonza.load(DATA / "1LYZ.cif")
    path = tmp_path / "moved.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for shift in ([0, 0, 0], [3, 0, 0]):
            w.write(s.positions + shift)
    session = viewer.Session(s, boonza.open_trajectory(path, s), whole=False)
    gone = s.select("resid 20").ids
    out = session.edit("delete", gone.tolist())
    keep = np.setdiff1d(np.arange(s.natoms), gone)
    assert out["natoms"] == len(keep) and set(out["picked"]) == {-1}
    pos, _ = session.frame(1)
    assert np.allclose(pos, s.positions[keep] + [3, 0, 0], atol=1e-3)  # the file is untouched
    session.undo()
    assert session.system.natoms == s.natoms and session.frame(1)[0].shape == (s.natoms, 3)
    with pytest.raises(ValueError, match="every atom"):
        session.edit("delete", list(range(s.natoms)))

    points = viewer.outputs.sphere_session(np.eye(3) * 4, np.array([1.0, 2.0, 3.0]), [], "red")
    points.edit("delete", [1])
    assert points.system.natoms == 2 and points.radii.tolist() == [1.0, 3.0]


def test_hydrogens_get_names_of_their_own():
    from boonza.viewer import edit

    assert edit._free_h_name({"H1"}, "") == "H2"
    assert edit._free_h_name({"HD11", "HD12"}, "D11") not in {"HD11", "HD12"}
    # a backbone N left without its peptide bond wants two hydrogens: both named apart
    s = boonza.load(DATA / "1LYZ.cif")
    n = int(s.select("resid 21 and name N").ids[0])
    index = edit.delete(s, s.select("resid 20").ids)
    n = int(index[n])
    edit.adjust_hydrogens(s, [n])
    hs = [b for b, _ in edit.neighbors(s, n) if s.atoms["anum"][b] == 1]
    names = [str(x) for x in s.atoms["name"][s.residue_atoms(int(s.atoms["residue"][n]))]]
    assert len(hs) == 2 and len(names) == len(set(names))
    # a structure without hydrogens gets none from an edit
    s = boonza.load(DATA / "1LYZ.cif")
    session = viewer.Session(s)
    session.edit("delete", s.select("resid 20").ids.tolist(), hydrogens=True)
    assert not (np.asarray(session.system.atoms["anum"]) == 1).any()


def test_clean_up_relaxes_a_molecule():
    s = boonza.from_smiles("CCO")
    pos = np.array(s.positions)
    pos[1] += [0.8, 0, 0]  # a C-C bond stretched
    s.positions = pos
    session = viewer.Session(s)
    out = session.edit("cleanup", [])
    assert out["molecules"] == 1 and out["forcefield"] == "MMFF94"
    p = session.system.positions
    assert abs(np.linalg.norm(p[0] - p[1]) - 1.51) < 0.05


@pytest.mark.parametrize("path", ["1LYZ.pdb", "2TRX.pdb", "1FNA.pdb"])
def test_hydrogens_from_the_templates_parameterize(path):
    """A crystal structure (no hydrogens, alternate locations, distorted geometry,
    unresolved side chains) takes ff19SB after Add Hydrogens."""
    from boonza.viewer import edit

    s = boonza.load(DATA / path)
    out = edit.protonate(s)
    assert out["hydrogens"] > 600 and not out["incomplete"]
    names = [str(x) for x in s.atoms["name"]]
    first = s.residue_atoms(0)
    assert {"H1", "H2", "H3"} <= {names[a] for a in first}  # NH3+
    assert int(np.asarray(s.atoms["formal_charge"]).sum()) == round(
        float(
            np.sum(
                edit.parameterize(
                    s.clone(s.select("protein or water").ids), log=lambda *a: None
                ).atoms["charge"]
            )
        )  # fmt: skip
    )
    res = np.asarray(s.atoms["residue"])
    assert (np.diff(res) >= 0).all()  # each residue in one piece
    if path == "1FNA.pdb":
        assert out["rebuilt"] == ["ARG6", "ASP7"]
    if path == "2TRX.pdb":
        assert out["alternates"] == 44


def test_add_hydrogens_then_minimize_a_crystal_structure():
    session = viewer.Session(boonza.load(DATA / "1LYZ.pdb"))
    out = session.edit("protonate")
    assert out["hydrogens"] == 1161 and out["topology"]
    heavy = session.system.select("not hydrogen").ids.tolist()
    report = session.minimize(fixed=[], tethered=heavy, restraint=10)
    assert report["energy_after"] < 0 < report["energy_before"]
    assert session.atom_info(0, assign=True)["type"] == "N3"
    session.undo()
    session.undo()
    assert session.system.natoms == 1102


def test_a_placeholder_cell_is_no_cell(tmp_path):
    """Cryo-EM and NMR entries carry a 1 x 1 x 1 A cell: no box, so secondary
    structure and distances are those of the structure itself."""
    s = boonza.load(DATA / "1LYZ.pdb")
    ss = viewer.Session(s).topology()["ss"]
    text = (DATA / "1LYZ.pdb").read_text().splitlines()
    text = [ln for ln in text if not ln.startswith("CRYST1")]
    text.insert(0, "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1")
    path = tmp_path / "em.pdb"
    path.write_text("\n".join(text) + "\n")
    em = boonza.load(path)  # the cell is read as msys reads it: 1 A
    assert viewer.Session(em).topology()["ss"] == ss and "H" in ss
    # a crystal's cell is kept, but distances do not wrap in it for one structure
    assert s.cell.any()
    near = viewer.Session(s).select("within 3 of resid 1")
    assert len(near) == len(s.select("within 3 of resid 1"))


def test_download_upload_copy_into_and_hbonds(served):
    v = viewer.Viewer([])
    get, post = served(v)
    post("api/build", {"smiles": "CC(=O)Nc1ccc(O)cc1", "centre": [0, 0, 0]})
    post("api/edit", {"m": 0, "op": "molecule", "value": {"smiles": "CCO", "centre": [8, 0, 0]}})
    # SDF: one entry per molecule, and a file name to keep
    status, _ = post("api/download", {"m": 0, "fmt": "xyz"})
    assert status == 400
    session = v.session(0)
    data, name = session.export("sdf")
    assert data.decode().count("$$$$") == 2 and name.endswith("_molecules.sdf")
    data, name = session.export("pdb", [0, 1, 2])
    lines = data.decode().splitlines()
    assert (
        name.endswith("_selection.pdb") and sum(ln[:6] in ("ATOM  ", "HETATM") for ln in lines) == 3
    )
    with pytest.raises(ValueError, match="download as"):
        session.export("xyz")
    # a file from the browser: opened, saved by default where boonza view runs
    added = v.upload("lyz.pdb", (DATA / "1LYZ.pdb").read_bytes())
    lyz = v.session(added[0])
    assert lyz.system.natoms == 1102 and lyz.path.name == "lyz.pdb"
    # the ligand into the protein, and back out with Undo
    out = v.copy_into(0, added[0])
    assert out["added"] == session.system.natoms and lyz.system.natoms == 1102 + out["added"]
    lyz.undo()
    assert lyz.system.natoms == 1102
    # hydrogen bonds: without hydrogens by N and O distance; with them by geometry
    plain = lyz.hbonds("resid 1 to 20", "resid 21 to 129")
    assert plain["method"].startswith("no hydrogens") and len(plain["pairs"]) > 10
    lyz.edit("protonate")
    placed = lyz.hbonds("resid 1 to 20", "resid 21 to 129")
    assert placed["method"].startswith("D-A") and placed["pairs"]
    hho = viewer.Session(boonza.load(DATA / "1HHO.pdb"))
    chains = np.asarray(hho.system.residues["chain"])[np.asarray(hho.system.atoms["residue"])]
    pairs = hho.hbonds("@chains")["pairs"]
    assert pairs and all(chains[i] != chains[j] for i, j, _ in pairs)


def test_view_takes_a_pdb_id(monkeypatch, tmp_path):
    """boonza view 1HHO: an ID that is not a file here is fetched; a file wins."""
    assert viewer.is_pdb_id("1hho")
    assert not viewer.is_pdb_id("hho1") and not viewer.is_pdb_id("1HHO.pdb")
    fetched, served = [], []

    def fetch(code, folder=None):
        fetched.append(code)
        return DATA / "1HHO.cif"

    monkeypatch.setattr(viewer, "fetch_pdb", fetch)
    monkeypatch.setattr(viewer, "serve", lambda what, **kw: served.append(what))
    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args(["view", "1hho", "--no-browser"])
    assert args.run(args) == 0
    assert fetched == ["1hho"] and served[0].system.natoms > 0
    (tmp_path / "2TRX").write_text("")  # a file named like an ID is read as a file
    args = _parser().parse_args(["view", "2TRX", "--no-browser"])
    with pytest.raises(ValueError):
        args.run(args)
    assert fetched == ["1hho"]


def test_amber_charges_for_a_structure_without_them():
    """A PDB entry: hydrogens on a copy, ff19SB charges folded onto the heavy atoms."""
    from boonza.viewer.edit import amber_charges

    s = boonza.load(DATA / "1FNA.pdb")
    q, has = amber_charges(s)
    assert has.all()  # its own hydrogens too, and the side chains rebuilt
    assert abs(q[has].sum() - round(q[has].sum())) < 1e-3  # whole charges
    names = [str(x) for x in s.atoms["name"]]
    nh3 = [a for a in s.residue_atoms(0) if names[a] in ("N", "H1", "H2", "H3")]
    assert abs(q[nh3].sum() - 0.8) < 0.15  # NH3+ (with CA and HA: +1)
    session = viewer.Session(s)
    assert session.topology()["q_source"] == "template"
    out = session.surface_charges()
    assert out["assigned"] == int(has.sum())
    assert np.allclose(np.array(out["q"])[has], q[has], atol=1e-4)


def test_open_takes_a_pdb_id(monkeypatch, tmp_path):
    """File | Open, the command line's open: a PDB ID that is no file here is fetched."""
    from boonza.viewer import outputs

    fetched = []

    def fetch(code, folder=None):
        fetched.append(code)
        return DATA / "1HHO.cif"

    monkeypatch.setattr(outputs, "fetch_pdb", fetch)
    monkeypatch.chdir(tmp_path)
    v = viewer.Viewer([])
    assert v.open("1hho") == [0] and fetched == ["1hho"]
    with pytest.raises(FileNotFoundError):
        v.open("no_such_file.pdb")


def test_residue_names_numbers_and_chains(served):
    p = _peptide()
    session = viewer.Session(p)
    get, post = served(session)
    one = [int(a) for a in p.residue_atoms(1)]
    value = {"name": "LIG", "resid": "52a", "chain": ""}
    status, out = post("api/edit", {"op": "residue", "atoms": one, "value": value})
    s = session.system
    assert status == 200 and out["topology"] and out["picked"] == one
    assert (str(s.residues["name"][1]), int(s.residues["resid"][1])) == ("LIG", 52)
    assert str(s.residues["insertion"][1]) == "A"
    every = list(range(s.natoms))
    post("api/edit", {"op": "residue", "atoms": every, "value": {"resid": "10", "chain": "B"}})
    s = session.system
    assert s.residues["resid"].tolist() == [10, 11, 12] and s.chains["name"].tolist() == ["B"]
    # part of a chain, to a new one; then back, and the empty chain goes
    post("api/edit", {"op": "residue", "atoms": [int(a) for a in s.residue_atoms(2)],
                      "value": {"chain": "C"}})  # fmt: skip
    s = session.system
    assert s.chains["name"].tolist() == ["B", "C"] and s.residues["chain"].tolist() == [0, 0, 1]
    post("api/edit", {"op": "residue", "atoms": [int(a) for a in s.residue_atoms(2)],
                      "value": {"chain": "B"}})  # fmt: skip
    assert session.system.chains["name"].tolist() == ["B"]
    status, out = post("api/edit", {"op": "residue", "atoms": one, "value": {"name": "TOO LONG"}})
    assert status == 400
    assert post("api/undo", {})[0] == 200


def test_a_port_in_use_moves_on():
    import socket

    from boonza.viewer.server import PORTS, _free_server

    v = viewer.Viewer([])
    first = _free_server(v, "127.0.0.1", None)
    try:
        second = _free_server(v, "127.0.0.1", None)  # the first one's port is taken
        port = first.server_address[1]
        assert second.server_address[1] != port
        assert port in PORTS or port not in range(1024)
        with pytest.raises(OSError, match="in use"):
            _free_server(v, "127.0.0.1", port)  # asked for: no moving on
        second.server_close()
    finally:
        first.server_close()
    with socket.socket() as s:  # and it was let go
        s.bind(("127.0.0.1", port))


def test_the_ligand_alone_is_a_representation():
    """Besides the ribbon with its ligand, a model can show its ligand alone, in sticks or
    space filling, with no ribbon: a pocket's points, or a holo ligand over another model."""
    import re

    ui = (Path(boonza.__file__).parent / "data/viewer/ui.js").read_text()
    presets = dict(re.findall(r'\["(\w+)", "([^"]+)"\]', ui[ui.index("const PRESETS") :]))
    assert presets["lig"] == "Ligand" and presets["ligspace"] == "Ligand, Space Filling"
    body = ui[ui.index("async function preset") : ui.index("function colorModel")]
    assert "lig: 1, ligspace: 4" in body  # sticks, spheres
    assert '"lig", "ligspace"' in body  # the ligand's atoms only
    assert not "lig".startswith("ribbon")  # no ribbon
