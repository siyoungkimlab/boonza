"""Many models in one viewer: files, and the folders boonza's commands write.

    boonza view a.mae b.mae              # one model each
    boonza view MODES_OUT                # boonza modes: a pose per mode, biggest first
    boonza view TRIAGE_OUT               # boonza triage: a pose per run, fitted on the first
    boonza view SITEMAP_OUT              # boonza sitemap: the protein and its pockets

A folder with a ``view.json`` is read from it (see :func:`from_manifest`), and
any pipeline can write one::

    {"title": "...", "align": "name CA",           # fit every model on the first
     "columns": ["mode", "runs"],                  # the table's columns
     "models": [{"name": "mode 0", "file": "structures/a.mae",
                 "trajectory": "run.dcd",          # optional
                 "group": "modes", "shown": true, "ligand": "resname LIG",
                 "carbon": "yellow", "row": {"mode": 0, "runs": 12},
                 "reps": [{"sel": "all", "style": "sticks", "color": "element"}]},
                {"name": "pocket 1", "file": "sites.pqr", "spheres": true,
                 "residue": 1, "radius": 0.5, "color": "red", "label": "1"}]}

Models load when first shown, so a triage of a thousand runs opens at once.
The table lists every model; the page shows one model of a group at a time
when a row is clicked, and steps through them with the arrow keys.
"""

from __future__ import annotations

import csv
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .session import Session

#: what poses are fitted on: alpha carbons, or a coarse-grained protein's
#: backbone beads (Martini BB, SIRAH GC), as boonza modes fits its medoids
ALIGN = "(protein and name CA) or name BB GC"
#: ligand carbons of each mode in turn, as boonza modes' sessions colour them
MODE_COLOURS = ("yellow", "magenta", "green", "orange", "cyan", "salmon", "purple", "lime",
                "slate", "wheat", "pink", "teal")  # fmt: skip
#: the first ten pockets each a colour of their own, the rest gray: not red or
#: yellow (a cartoon's helices and strands), white (its coil) or pink (the selection)
POCKET_COLOURS = ("cyan", "green", "blue", "orange", "purple", "magenta", "lime", "teal",
                  "slate", "wheat")  # fmt: skip
#: sitemap's pharmacophore codes: donor sky blue, acceptor salmon, cation blue,
#: anion red, aromatic orange, hydrophobe green
PHARM_COLOURS = {"DON": "#87ceeb", "ACC": "#fa8072", "CAT": "#3f6fff", "ANI": "#ff3030",
                 "ARO": "#ffa500", "HYD": "#4caf50"}  # fmt: skip


@dataclass
class Model:
    """One entry of the viewer: loaded by ``load`` the first time it is needed."""

    name: str
    load: Callable[[], Session]
    group: str = ""
    shown: bool = True
    row: dict = field(default_factory=dict)
    label: str = ""  # drawn at the model's centre (pocket numbers)
    color: str = ""  # its protein's colour (cartoon and carbons): a name or #rrggbb
    # its own numbers for the view's corner, as Viewer.summary's, shown while it is current
    summary: list = field(default_factory=list)
    # tied to frame i of another model, by name: {"model": name, "i": i} (a site and the
    # MD frame it was found in); showing this one shows that one at that frame
    frame: dict | None = None
    closed: bool = False
    _session: Session | None = None

    def info(self) -> dict:
        out = {"name": self.name, "group": self.group, "shown": self.shown,
               "row": self.row, "label": self.label}  # fmt: skip
        if self.color:
            out["color"] = self.color
        if self.summary:
            out["summary"] = self.summary
        if self.frame:
            out["frame"] = self.frame
        return out


class Viewer:
    """The models a page shows; ``align`` fits each onto the first as it loads."""

    def __init__(self, models: list[Model], title: str = "", columns=(), align: str | None = None):
        self.models = list(models)  # none: an empty view, to build or open into
        self.title = title
        self.columns = list(columns)
        self.align = align
        self.page = None  # a saved session's page state, handed to the page as it opens
        # a few numbers about the whole view, drawn in its top-right corner:
        # [{"label": ..., "value": ..., "title": what it is}]
        self.summary: list[dict] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.models)

    def session(self, m: int) -> Session:
        if not 0 <= m < len(self.models):
            raise IndexError(f"model {m} of {len(self.models)}")
        model = self.models[m]
        with self._lock:
            if model._session is None:
                s = model.load()
                if self.align and m:
                    self._fit(s, self.session(0))
                if getattr(model, "upload", None) is not None:
                    s.path = model.upload  # where Save suggests: the folder it runs in
                model._session = s
            return model._session

    def _fit(self, s: Session, ref: Session) -> None:
        from ..align import superpose

        a, b = s.system.select(self.align).ids, ref.system.select(self.align).ids
        if len(a) < 3 or len(b) < 3:
            return
        # the same system pairs in order, another one by sequence
        superpose(s.system, ref.system, sel=self.align,
                  match="order" if len(a) == len(b) else "sequence")  # fmt: skip
        s.set_glue(s.whole, s.center, s.fit)  # drop frames cached before the fit

    def listing(self) -> dict:
        from .edit import GROUPS

        out = {"title": self.title, "columns": self.columns, "groups": GROUPS,
               "models": [{**m.info(), "closed": m.closed} for m in self.models]}  # fmt: skip
        if self.page is not None:
            out["page"] = self.page
        if self.summary:
            out["summary"] = self.summary
        return out

    def save_session(self, path, page: dict) -> str:
        """The whole session in one file (a zip): every open model as it is now (DMS,
        edits included; a trajectory by its path), and the page's state (``page``:
        what each atom shows, colours, selection, meters, the camera)."""
        import tempfile
        import zipfile

        from ..io import save

        path = Path(path).expanduser()
        if path.suffix != ".bview":
            path = path.with_suffix(".bview")
        entries = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z,
        ):
            for k, m in enumerate(self.models):
                if m.closed:
                    continue
                s = self.session(k)
                f = Path(tmp) / f"model{k}.dms"
                save(s.system, f)
                z.write(f, f"models/model{k}.dms")
                traj = s.trajectory
                entries.append({
                    "index": k, "name": m.name, "group": m.group, "shown": m.shown, "row": m.row,
                    "label": m.label, "file": f"models/model{k}.dms", "title": s.title,
                    "trajectory": str(Path(traj.path).resolve()) if traj is not None else None,
                    "atoms": s.ids.tolist() if s.ids is not None else None,
                    "ligand": s.ligand, "carbon": s.carbon, "reps": s.reps, "cg": s.cg,
                    "radii": s.radii.tolist() if s.radii is not None else None,
                    "glue": {"whole": s.whole, "center": s.center, "fit": s.fit},
                })  # fmt: skip
            z.writestr("session.json", json.dumps({
                "format": "boonza view session", "version": 1, "title": self.title,
                "columns": self.columns, "models": entries, "page": page}))  # fmt: skip
        return str(path)

    def open(self, path) -> list[int]:
        """Add the models of a file or folder (as from_paths reads it); their indices.
        A PDB ID that is not a file here is fetched (:meth:`fetch`)."""
        path = Path(path).expanduser()
        if not path.exists() and is_pdb_id(str(path)):
            return self.fetch(str(path))
        if not path.exists():
            raise FileNotFoundError(path)
        opened = from_paths([path])
        new = opened.models
        if opened.page is not None:  # a session: its page state, by the new indices
            self.page = {**opened.page, "offset": len(self.models)}
        start = len(self.models)
        self.models += new
        for m in new:
            if self.title and m.group == "":
                m.group = "files"
        return list(range(start, start + len(new)))

    def fetch(self, code: str, folder=None) -> list[int]:
        """Download a PDB entry (mmCIF, from RCSB) into ``folder`` (default the one
        boonza view was started in) and open it."""
        return self.open(fetch_pdb(code, folder))

    def build(self, smiles: str, centre=(0.0, 0.0, 0.0), name: str = "") -> list[int]:
        """A new model of one molecule from SMILES (3D, hydrogens added), centred at
        ``centre``: what the builder starts from when no model can take it."""
        from ..system import System
        from .edit import add_molecule

        s = System()
        add_molecule(s, smiles, np.asarray(centre, float))
        k = sum(1 for m in self.models if m.group == "built") + 1
        name = name or f"molecule {k}"
        # every atom, hydrogens too: they are where the builder grows the molecule
        reps = [{"sel": "all", "style": "ballstick", "color": "element"}]
        session = Session(s, title=name, reps=reps)
        self.models.append(Model(name, lambda: session, group="built", _session=session))
        return [len(self.models) - 1]

    def upload(self, name: str, data: bytes) -> list[int]:
        """A file the page sent (dropped or picked in the browser): written to a folder
        of its own, opened, and saved by default where boonza view was started."""
        import atexit
        import shutil
        import tempfile

        name = Path(name).name
        if not name or name.startswith("."):
            raise ValueError("a file needs a name")
        if getattr(self, "_uploads", None) is None:
            self._uploads = Path(tempfile.mkdtemp(prefix="boonza-view-"))
            atexit.register(shutil.rmtree, self._uploads, True)
        path = self._uploads / name
        path.write_bytes(data)
        added = self.open(path)
        for k in added:
            self.models[k].upload = Path.cwd() / name
        return added

    def copy_into(self, src: int, dst: int, atoms=None, frame: int = 0) -> dict:
        """Copy model ``src``'s ``atoms`` (default all) into model ``dst`` where they
        are (a built ligand into its protein); undone as an edit of ``dst``."""
        if src == dst:
            raise ValueError("copy into another model")
        a, b = self.session(src), self.session(dst)
        b._editable()
        piece = a.system.clone(atoms if atoms else None)
        pos = np.asarray(a.frame(frame)[0], float)
        piece.positions = pos[np.asarray(atoms, np.int64)] if atoms else pos
        with b._lock:
            b._remember()
            new = b.system.append(piece)
            return {**b._changed(topology=True), "added": len(new)}

    def close(self, m: int) -> None:
        """Close a model; indices stay as they are (the page keys models by them)."""
        if not 0 <= m < len(self.models):
            raise IndexError(f"model {m} of {len(self.models)}")
        self.models[m].closed = True
        self.models[m]._session = None


# --- reading -------------------------------------------------------------------


def is_pdb_id(text: str) -> bool:
    """Four characters, a digit first (1IRK): what RCSB names an entry."""
    return re.fullmatch(r"[0-9][A-Za-z0-9]{3}", text.strip()) is not None


def fetch_pdb(code: str, folder=None) -> Path:
    """A PDB entry's mmCIF in ``folder`` (default the current one), downloaded from
    RCSB unless it is there already."""
    import urllib.request

    code = code.strip().upper()
    if not is_pdb_id(code):
        raise ValueError(f"{code!r} is not a PDB ID (four characters, e.g. 1IRK)")
    path = Path(folder or Path.cwd()) / f"{code}.cif"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{code}.cif"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                path.write_bytes(r.read())
        except Exception as e:
            raise ValueError(f"could not download {code}: {e}") from e
    return path


def _session(path, trajectory=None, coordinates=None, **kwargs) -> Session:
    import boonza

    path = Path(path)
    s = boonza.load(path, coordinates=coordinates) if coordinates else boonza.load(path)
    traj = boonza.open_trajectory(trajectory, s) if trajectory else None
    session = Session(s, traj, title=path.name, **kwargs)
    session.path = path
    return session


def points(path, residue: int | None = None):
    """Centres, radii and residue names of the atoms of a PQR or PDB file of spheres
    (fpocket's and boonza's sites), all of them or one residue number's."""
    xyz, radii, names = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if residue is not None and int(line[22:26]) != residue:
            continue
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        rest = line[54:].split()
        # PQR: charge and radius after the coordinates; PDB: occupancy, B-factor
        radii.append(float(rest[1]) if path.suffix == ".pqr" and len(rest) >= 2 else np.nan)
        names.append(line[17:20].strip())
    return np.array(xyz, float).reshape(-1, 3), np.array(radii, float), names


def sphere_session(xyz, radii, names, color: str, title: str = "") -> Session:
    """Spheres as a model: one atom each, no bonds, drawn at their own radii."""
    from ..system import System

    n = len(xyz)
    s = System.from_arrays(np.asarray(xyz, float), names=["X"] * n, anum=np.zeros(n, int),
                           resnames=names or ["PNT"] * n, resids=np.arange(1, n + 1))  # fmt: skip
    reps = [{"sel": "all", "style": "spheres", "color": color}]
    return Session(s, title=title, reps=reps, radii=radii, cg=False)


def from_session(path) -> Viewer:
    """A session saved by :meth:`Viewer.save_session`: its models as they were, and
    the page's state (``Viewer.page``) for the page to put back."""
    import atexit
    import shutil
    import tempfile
    import zipfile

    import boonza

    path = Path(path).expanduser()
    tmp = Path(tempfile.mkdtemp(prefix="boonza-view-"))
    atexit.register(shutil.rmtree, tmp, True)
    with zipfile.ZipFile(path) as z:
        spec = json.loads(z.read("session.json"))
        if spec.get("format") != "boonza view session":
            raise ValueError(f"{path}: not a boonza view session")
        z.extractall(tmp)
    models = []
    for e in spec["models"]:

        def load(e=e):
            s = boonza.load(tmp / e["file"])
            traj = None
            if e.get("trajectory"):
                if not Path(e["trajectory"]).exists():
                    raise FileNotFoundError(f"the session's trajectory {e['trajectory']} is gone")
                full = s if e.get("atoms") is None else None
                if full is None:
                    raise ValueError(
                        "a session of part of a trajectory's atoms is not supported yet"
                    )
                traj = boonza.open_trajectory(e["trajectory"], s)
            g = e.get("glue") or {}
            session = Session(s, traj, title=e.get("title", ""), ligand=e.get("ligand"),
                              carbon=e.get("carbon", "green"), reps=e.get("reps"),
                              radii=e.get("radii"), cg=e.get("cg"), whole=g.get("whole"),
                              center=g.get("center"), fit=g.get("fit"))  # fmt: skip
            session.path = path.with_name(f"{path.stem}_{e['name']}")
            return session

        models.append(Model(e["name"], load, group=e.get("group", ""), shown=e.get("shown", True),
                            row=e.get("row", {}), label=e.get("label", "")))  # fmt: skip
    v = Viewer(models, title=spec.get("title") or path.name, columns=spec.get("columns", ()))
    # the page keys its state by the models' old indices: now 0, 1, 2 ... in order
    page = spec.get("page") or {}
    page["indices"] = [e["index"] for e in spec["models"]]
    v.page = page
    return v


def from_paths(paths, **kwargs) -> Viewer:
    """Files (a model each), folders (see the module docstring) and saved sessions
    (.bview), in order."""
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and paths[0].suffix == ".bview":
        return from_session(paths[0])
    if len(paths) == 1 and paths[0].is_dir():
        return from_folder(paths[0])
    models, columns = [], []
    for p in paths:
        if p.is_dir():
            v = from_folder(p)
            for m in v.models:
                m.group = m.group or p.name
            models += v.models
            columns += [c for c in v.columns if c not in columns]
        elif not p.exists():
            raise FileNotFoundError(p)
        else:
            models.append(Model(p.name, lambda p=p: _session(p, **kwargs), group="files"))
    title = " + ".join(p.name for p in paths)
    return Viewer(models, title=title, columns=columns)


def from_folder(d) -> Viewer:
    """What a folder holds: a view.json, or the output of modes, triage or sitemap."""
    d = Path(d)
    if (d / "view.json").is_file():
        return from_manifest(d / "view.json")
    if (d / "modes.csv").is_file() and (d / "structures").is_dir():
        return from_modes(d)
    if (d / "summary.csv").is_file() and (d / "structures").is_dir():
        return from_triage(d)
    if (d / "pockets.csv").is_file() and (d / "sites.pqr").is_file():
        return from_sitemap(d)
    raise ValueError(f"{d}: not a folder boonza view knows (a view.json, or the output of "
                     "boonza modes, triage or sitemap)")  # fmt: skip


def _rows(path, key: str) -> dict:
    if not Path(path).is_file():
        return {}
    with open(path, newline="") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def _number(v):
    """CSV text as a number where it is one, for sorting the table."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in str(v) else round(f, 3)


def from_manifest(path) -> Viewer:
    path = Path(path)
    spec = json.loads(path.read_text())
    here = path.parent
    models = []
    for k, m in enumerate(spec.get("models", [])):
        if "file" not in m:
            raise ValueError(f"{path}: model {k} has no file")
        f = here / m["file"]
        name = m.get("name") or Path(m["file"]).name
        if m.get("spheres"):

            def load(f=f, m=m, name=name):
                xyz, r, names = points(f, m.get("residue"))
                r = np.where(np.isfinite(r), r, m.get("radius", 1.0))
                if "radii" in m:
                    r = np.asarray(m["radii"], float)
                return sphere_session(xyz, r, names, m.get("color", "gray"), name)
        else:

            def load(f=f, m=m):
                kw = {k: m[k] for k in ("ligand", "carbon", "reps") if m.get(k)}
                traj = here / m["trajectory"] if m.get("trajectory") else None
                coords = here / m["coordinates"] if m.get("coordinates") else None
                return _session(f, traj, coords, **kw)

        models.append(Model(name, load, group=m.get("group", ""), shown=m.get("shown", True),
                            row={c: _number(v) for c, v in m.get("row", {}).items()},
                            label=m.get("label", "")))  # fmt: skip
    return Viewer(models, title=spec.get("title", here.name), columns=spec.get("columns", ()),
                  align=spec.get("align"))  # fmt: skip


MODE_FILE = re.compile(r"mode(\d+)_n(\d+)_(.+)_pose\.[^.]+(\.gz|\.bz2)?$")
#: columns of boonza triage's summary worth a glance in the table
RUN_COLUMNS = ("drifted", "population", "late_population", "n_poses", "settled", "outcome")


def from_modes(d) -> Viewer:
    """boonza modes: the medoid pose of each mode, biggest first, the first shown;
    the ligand is the one boonza modes drew (modes.json)."""
    d = Path(d)
    rows = _rows(d / "modes.csv", "run")
    f = d / "modes.json"
    ligand = json.loads(f.read_text()).get("ligandsel") if f.is_file() else _triage_ligand(d)
    total = len(rows) or 1
    found = []
    for f in (d / "structures").iterdir():
        m = MODE_FILE.match(f.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), m.group(3), f))
    if not found:
        raise ValueError(f"{d / 'structures'}: no mode*_pose files")
    models = []
    for k, (n, count, run, f) in enumerate(sorted(found)):
        row = {"mode": n, "runs": count, "share": round(count / total, 3), "run": run}
        r = rows.get(run, {})
        row.update({c: _number(r[c]) for c in ("mode_spread", *RUN_COLUMNS) if r.get(c, "") != ""})
        colour = MODE_COLOURS[k % len(MODE_COLOURS)]
        models.append(Model(f"mode {n}", lambda f=f, c=colour: _session(f, carbon=c, ligand=ligand),
                            group="modes", shown=k == 0, row=row))  # fmt: skip
    columns = ["mode", "runs", "share", "run"]
    columns += [c for c in ("mode_spread", *RUN_COLUMNS) if any(c in m.row for m in models)]
    return Viewer(models, title=f"{d.name}: {len(models)} modes", columns=columns)


def _triage_ligand(d) -> str | None:
    f = Path(d) / "triage.json"
    return json.loads(f.read_text()).get("ligandsel") if f.is_file() else None


def from_triage(d) -> Viewer:
    """boonza triage: each run's pose, in summary order, fitted on the first."""
    d = Path(d)
    rows = _rows(d / "summary.csv", "run")
    ligand = _triage_ligand(d)
    poses = {}
    for f in (d / "structures").iterdir():
        m = re.match(r"(.+)_pose\.[^.]+(\.gz|\.bz2)?$", f.name)
        if m:
            poses[m.group(1)] = f
    order = [r for r in rows if r in poses] + sorted(r for r in poses if r not in rows)
    if not order:
        raise ValueError(f"{d / 'structures'}: no *_pose files")
    models = []
    for k, run in enumerate(order):
        r = rows.get(run, {})
        row = {"run": run, **{c: _number(r[c]) for c in RUN_COLUMNS if r.get(c, "") != ""}}
        models.append(Model(run, lambda f=poses[run]: _session(f, ligand=ligand),
                            group="runs", shown=k == 0, row=row))  # fmt: skip
    columns = ["run", *(c for c in RUN_COLUMNS if any(c in m.row for m in models))]
    return Viewer(models, title=f"{d.name}: {len(models)} runs", columns=columns, align=ALIGN)


#: pockets.csv columns shown in the table, those the file has, in this order
POCKET_COLUMNS = ("rank", "p", "p_max", "p_q90_open", "occupancy", "SiteScore", "score",
                  "volume", "PPc", "MOc", "LVC")  # fmt: skip


def _pharm_radii(d, k: int) -> dict[int, float]:
    """Hotspot radii of pocket k, from view.pml's ``alter pharm_k and id n, vdw=r``."""
    pml = Path(d) / "view.pml"
    if not pml.is_file():
        return {}
    pat = re.compile(rf"alter pharm_{k} and id (\d+), vdw=([\d.]+)")
    return {int(n): float(r) for n, r in pat.findall(pml.read_text())}


def _with_ligand(structure, ligand) -> Session:
    """A structure with its ligand from another file appended, as one model."""
    import boonza

    s = boonza.load(structure)
    ids = s.append(boonza.load(ligand))
    return Session(s, title=Path(structure).name, ligand=f"index {ids[0]} to {ids[-1]}")


def from_sitemap(d) -> Viewer:
    """boonza sitemap: the protein (or apo structure), its ligand, and each pocket's
    medoid site; pharmacophore hotspots and the holo structure are listed, off."""
    d = Path(d)
    rows = _rows(d / "pockets.csv", "rank")
    models = []

    def structure(name, f, shown, **kw):
        models.append(Model(name, lambda: _session(f, **kw), group="structure", shown=shown))

    main = next((d / n for n in ("apo.pdb", "structure.pdb") if (d / n).is_file()), None)
    beads = sorted(d.glob("beads.*"))
    ligand = d / "ligand.pdb"
    if main is not None and ligand.is_file():
        # one model, so the pocket around the ligand is drawn as in any other
        models.append(Model(main.stem, lambda: _with_ligand(main, ligand), group="structure"))
    elif main is not None:
        structure(main.stem, main, True)
    if beads:
        structure("beads", beads[0], main is None)
    if (d / "holo.pdb").is_file():
        structure("holo", d / "holo.pdb", False)
    if ligand.is_file() and main is None:
        structure("ligand", ligand, True, ligand="all")

    pharmacophores = []
    ranks = sorted({int(line[22:26]) for line in (d / "sites.pqr").read_text().splitlines()
                    if line.startswith(("ATOM", "HETATM"))})  # fmt: skip
    for k in ranks:
        colour = POCKET_COLOURS[k - 1] if k <= len(POCKET_COLOURS) else "gray"

        def load(k=k, colour=colour):
            xyz, r, names = points(d / "sites.pqr", k)
            return sphere_session(xyz, np.where(np.isfinite(r), r, 0.5), names, colour,
                                  f"pocket {k}")  # fmt: skip

        r = rows.get(str(k), {})
        row = {c: _number(r[c]) for c in POCKET_COLUMNS if r.get(c, "") != ""}
        row.setdefault("rank", k)
        models.append(Model(f"pocket {k}", load, group="pockets", row=row, label=str(k)))
        pharm = d / f"pharm_{k}.pdb"
        if pharm.is_file():

            def hotspots(f=pharm, k=k):
                xyz, _, names = points(f)
                radius = _pharm_radii(d, k)
                r = np.array([radius.get(n, 1.0) for n in range(1, len(xyz) + 1)])
                s = sphere_session(xyz, r, names, "gray", f"pharmacophore {k}")
                s.reps = [
                    {"sel": f"resname {code}", "style": "spheres", "color": colour}
                    for code, colour in PHARM_COLOURS.items()
                    if code in names
                ]
                return s

            pharmacophores.append(Model(f"pharmacophore {k}", hotspots, group="pharmacophore",
                                        shown=False, row={"rank": k}))  # fmt: skip
    models += pharmacophores
    columns = [c for c in POCKET_COLUMNS if any(c in m.row for m in models)]
    return Viewer(models, title=f"{d.name}: {len(ranks)} pockets", columns=columns)
