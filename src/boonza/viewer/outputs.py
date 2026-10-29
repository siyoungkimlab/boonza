"""Many models in one viewer: files, and the folders boonza's commands write.

    boonza view a.mae b.mae              # one model each
    boonza view MODES_OUT                # boonza modes: a pose per mode, biggest first
    boonza view TRIAGE_OUT               # boonza triage: a pose per run, fitted on the first
    boonza view CAVITY_OUT              # boonza cavity: the protein and its pockets

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
import math
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
#: boonza cavity's pharmacophore codes: donor sky blue, acceptor salmon, cation blue,
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
    swatch: str = ""  # its swatch in the list, when not its protein's colour (a site's points)
    # its own numbers for the view's corner, as Viewer.summary's, shown while it is current
    summary: list = field(default_factory=list)
    # tied to frame i of another model, by name: {"model": name, "i": i} (a site and the
    # MD frame it was found in); showing this one shows that one at that frame
    frame: dict | None = None
    # a design's receptor (a series member's): the model it is made against, by index
    receptor: int | None = None
    closed: bool = False
    _session: Session | None = None

    def info(self) -> dict:
        out = {"name": self.name, "group": self.group, "shown": self.shown,
               "row": self.row, "label": self.label}  # fmt: skip
        if self.color:
            out["color"] = self.color
        if self.swatch:
            out["swatch"] = self.swatch
        if self.summary:
            out["summary"] = self.summary
        if self.frame:
            out["frame"] = self.frame
        if self.receptor is not None:
            out["receptor"] = self.receptor
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
                    "receptor": self.receptor(k),
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
        for m in new:  # receptors by index in what was opened: now after what is here
            if m.receptor is not None:
                m.receptor += start
        taken = {m.group for m in self.models[:start]}
        renamed: dict[str, str] = {}
        for m in new:
            if self.title and m.group == "":
                m.group = "files"
            if m.group.startswith("series") and (m.group in taken or m.group in renamed):
                if m.group not in renamed:  # a series of that name is open: the next one
                    k = next(k for k in range(1, len(self.models) + 2)
                             if f"series {k}" not in taken | set(renamed.values()))  # fmt: skip
                    renamed[m.group] = f"series {k}"
                m.group = renamed[m.group]
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

    # --- series: ligand designs, a model each, saved together as one SDF ---------------

    def series(self, group: str) -> list[int]:
        """The open members of series ``group``, in order."""
        out = [k for k, m in enumerate(self.models) if m.group == group and not m.closed]
        if not group.startswith("series") or not out:
            raise ValueError(f"no series {group!r}")
        return out

    def _member(self, s, group: str, name: str, receptor: int | None = None) -> int:
        reps = [{"sel": "all", "style": "ballstick", "color": "element"}]
        session = Session(s, title=name, reps=reps)
        self.models.append(Model(name, lambda: session, group=group, receptor=receptor,
                                 _session=session))  # fmt: skip
        return len(self.models) - 1

    def create_series(self, src: int, atoms, frame: int = 0) -> list[int]:
        """A series of designs from model ``src``'s ``atoms`` (a ligand, where it sits):
        a group of its own whose first member is a copy of them, with hydrogens when it
        had none, and ``src`` its receptor.  Members are edited on their own;
        :meth:`save_series` writes them."""
        from .edit import adjust_hydrogens

        a = self.session(src)
        atoms = [int(x) for x in atoms]
        if not atoms:
            raise ValueError("select the ligand first (double-click it)")
        if a.cg or a.radii is not None:
            raise ValueError("a series is of molecules with atoms, not beads or points")
        piece = a.system.clone(atoms)
        piece.positions = np.asarray(a.frame(frame)[0], float)[atoms]
        anum = np.asarray(piece.atoms["anum"])
        if not (anum == 1).any():  # a PDB entry's ligand: designs need its hydrogens
            adjust_hydrogens(piece, np.flatnonzero(anum > 1).tolist())
        taken = {m.group for m in self.models}
        k = next(k for k in range(1, len(self.models) + 2) if f"series {k}" not in taken)
        base = str(piece.residues["name"][0]).strip() or "design"
        return [self._member(piece, f"series {k}", f"{base}1", receptor=src)]

    def _base(self, m: int) -> str:
        """What a member's name numbers: its residue name (LIG of LIG3), else the name
        without its last digits."""
        name = self.models[m].name
        s = self.session(m).system
        res = str(s.residues["name"][0]).strip() if s.nresidues else ""
        if res and re.fullmatch(rf"{re.escape(res)}\d+", name):
            return res
        return re.sub(r"\d+$", "", name).strip() or "design"

    def add_to_series(self, m: int) -> list[int]:
        """A new member of model ``m``'s series: a copy of ``m``, to change into the next
        design, against the same receptor."""
        group = self.models[m].group
        members = self.series(group)
        base = self._base(members[0])
        numbered = re.compile(rf"{re.escape(base)}(\d+)")
        used = [int(x.group(1)) for k in members if (x := numbered.fullmatch(self.models[k].name))]
        name = f"{base}{max(used, default=0) + 1}"
        return [self._member(self.session(m).system.clone(), group, name, self.receptor(m))]

    def set_note(self, m: int, note: str) -> None:
        """A design's note: shown in the model list, saved in the series' SDF."""
        note = " ".join(str(note).split())
        if note:
            self.models[m].row["note"] = note
        else:
            self.models[m].row.pop("note", None)

    def receptor(self, m: int) -> int | None:
        """The open model design ``m`` is made against, if any."""
        r = self.models[m].receptor
        if r is None or not 0 <= r < len(self.models) or self.models[r].closed:
            return None
        return r

    def set_receptor(self, group: str, r: int | None) -> None:
        """What every member of series ``group`` is made against (None: nothing)."""
        if r is not None:
            if not 0 <= r < len(self.models) or self.models[r].closed:
                raise ValueError(f"no model {r + 1}")
            if self.models[r].group.startswith("series"):
                raise ValueError("a receptor is a structure, not a design")
            s = self.session(r)
            if s.cg or s.radii is not None:
                raise ValueError("a receptor needs atoms, not beads or points")
        for k in self.series(group):
            self.models[k].receptor = r

    def _receptor_atoms(self, m: int, frame: int = 0):
        """Design ``m``'s receptor: its index, the atoms of it that count and their
        positions.  The molecules the design sits on are left out: the ligand it was
        made from, waters it displaces (any heavy atom within 1.5 A of one of its own)."""
        r = self.receptor(m)
        if r is None:
            raise ValueError("this design has no receptor: right-click it, Receptor")
        rec, design = self.session(r), self.session(m)
        pos = np.asarray(rec.frame(min(frame, rec.nframes - 1))[0], float)
        s = rec.system
        keep = np.ones(s.natoms, bool)
        small = s.select("not (protein or nucleic)").ids
        small = small[np.asarray(s.atoms["anum"])[small] > 1]
        mine = np.asarray(design.system.positions, float)
        mine = mine[np.asarray(design.system.atoms["anum"]) > 1]
        if len(small) and len(mine):
            from scipy.spatial import cKDTree

            near = small[[bool(x) for x in cKDTree(mine).query_ball_point(pos[small], 1.5)]]
            frag = np.asarray(s.fragids)
            keep[np.isin(frag, frag[near])] = False
        return r, np.flatnonzero(keep), pos

    def contacts(self, m: int, frame: int = 0) -> list:
        """Model ``m``'s polar contacts; a design's with its receptor, as
        ``[i, j, distance, receptor]`` (``j`` an atom of the receptor)."""
        from .session import POLAR_CONTACT

        if self.receptor(m) is None:
            return self.session(m).contacts(frame)
        design = self.session(m)
        r, keep, pos = self._receptor_atoms(m, frame)
        rec = self.session(r).system
        on = np.isin(np.asarray(design.system.atoms["anum"]), (7, 8))
        a = np.flatnonzero(on)
        polar = np.isin(np.asarray(rec.atoms["anum"]), (7, 8))
        polar[rec.select("water").ids] = False
        b = keep[polar[keep]]
        if not len(a) or not len(b):
            return []
        mine = np.asarray(design.system.positions, float)
        d = np.linalg.norm(mine[a][:, None] - pos[b][None], axis=2)
        i, j = np.nonzero(d <= POLAR_CONTACT)
        return [[int(a[x]), int(b[y]), round(float(d[x, y]), 2), r]
                for x, y in zip(i, j, strict=True)]  # fmt: skip

    def minimize(self, m: int, mode: str | None = None, path: bool = False, rem: bool = False,
                 **kw) -> dict:  # fmt: skip
        """Minimize model ``m`` (see :meth:`Session.minimize`); a design with a receptor
        in it (:meth:`minimize_design`) unless it is given its own restraints.  The
        path's atoms are split by the model they are in."""
        constrained = kw.get("fixed") is not None or kw.get("tethered") is not None
        if self.receptor(m) is not None and not constrained:
            return self.minimize_design(m, mode or "both", path=path,
                                        restraint=kw.get("restraint", 10.0), rem=rem)  # fmt: skip
        protonated = _protonate_if_bare(self.session(m), m)
        out = self.session(m).minimize(mode=mode, path=path, rem=rem, **kw)
        if protonated:
            out["protonated"] = protonated
        if "path" in out:
            out["path"] = _split_path(out["path"], [(m, None)])
        return out

    def minimize_design(self, m: int, mode: str = "both", path: bool = False,
                        restraint: float = 10.0, rem: bool = False) -> dict:  # fmt: skip
        """Minimize design ``m`` in its receptor: ``mode`` both (it and its pocket free),
        ligand (it alone) or pocket (the pocket alone); the heavy atoms of the rest
        held by ``restraint`` (kcal/mol/A^2) springs; ``rem``: soft first (boonza.rem).
        Only the design keeps its new positions: the receptor is every design's.  A
        receptor without hydrogens (a PDB entry) gets them first, as Add / Adjust H adds
        them: an edit of it, undone with Undo."""
        from .edit import minimize, parameterize

        design = self.session(m)
        design._editable()
        r = self.receptor(m)
        protonated = _protonate_if_bare(self.session(r), r) if r is not None else None
        r, keep, pos = self._receptor_atoms(m)
        rec = self.session(r)
        s = rec.system.clone(keep)
        s.positions = pos[keep]
        n0 = s.natoms
        s.append(design.system.clone())
        ligand = np.arange(n0, s.natoms)
        lig = "index " + " ".join(map(str, ligand.tolist()))
        pocket = s.select(f"(same residue as ((protein or nucleic) and within 5 of ({lig})))").ids
        free = {"both": np.union1d(ligand, pocket), "ligand": ligand, "pocket": pocket}.get(mode)
        if free is None:
            raise ValueError(f"mode is both, ligand or pocket, not {mode!r}")
        if not len(free):
            raise ValueError("nothing to move: no pocket residues within 5 A of the design")
        # the rest's heavy atoms held by springs; hydrogens are never held
        held = np.setdiff1d(np.flatnonzero(np.asarray(s.atoms["anum"]) > 1), free)
        key = (r, getattr(rec, "topo_version", 0), keep.tobytes(),
               getattr(design, "topo_version", 0))  # fmt: skip
        cached = getattr(design, "_in_receptor", None)
        if cached is None or cached[0] != key:
            design._in_receptor = (key, parameterize(s))
        p = design._in_receptor[1]
        p.positions = np.asarray(s.positions)[np.asarray(p.atoms["view_index"])]
        with design._lock:
            report = minimize(s, restraint=restraint, fixed=[], tethered=held, parameterized=p,
                              path=path, rem=rem)  # fmt: skip
            design._remember()
            design.system.positions = np.asarray(s.positions)[n0:]
            out = {**design._changed(topology=False), **report, "mobile": mode,
                   "receptor": self.models[r].name}  # fmt: skip
        if protonated:
            out["protonated"] = protonated
        if "path" in out:
            out["path"] = _split_path(out["path"], [(m, n0), (r, keep)], n0)
        return out

    def series_sdf(self, group: str) -> str:
        """Series ``group`` as SDF text: a record per member, titled with its name, with
        the series, its SMILES and its note as fields."""
        import tempfile

        from ..io.sdf import save_sdf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "series.sdf"
            for n, k in enumerate(self.series(group)):
                s = self.session(k).system.clone()
                if not s.ncts:
                    s.add_ct()
                smiles = [x["smiles"] for x in self.session(k).smiles(range(s.natoms))]
                fields = {"series": group, "SMILES": ".".join(x for x in smiles if x)}
                if self.models[k].row.get("note"):
                    fields["note"] = self.models[k].row["note"]
                for c in range(s.ncts):
                    s._ct_names[c] = self.models[k].name
                    s._ct_props[c] = dict(fields)
                save_sdf(s, path, append=n > 0)
            return path.read_text()

    def save_series(self, group: str, path) -> str:
        path = Path(path).expanduser()
        if path.suffix.lower() != ".sdf":
            raise ValueError("a series is saved as SDF (a .sdf path)")
        path.write_text(self.series_sdf(group))
        return str(path)

    def close(self, m: int) -> None:
        """Close a model; indices stay as they are (the page keys models by them)."""
        if not 0 <= m < len(self.models):
            raise IndexError(f"model {m} of {len(self.models)}")
        self.models[m].closed = True
        self.models[m]._session = None


def _protonate_if_bare(session: Session, m: int | None) -> dict | None:
    """Hydrogens for a model without them (a PDB entry: its protein has none, though a
    ligand placed in it may), before a minimization: as Add / Adjust H adds them
    (Session.edit protonate), undone with Undo; what was done, for the page to reload
    model ``m`` and say so."""
    if m is None:
        return None
    s = session.system
    polymer = s.select("protein or nucleic").ids
    anum = np.asarray(s.atoms["anum"])
    if (anum[polymer] == 1).any() if len(polymer) else (anum == 1).any():
        return None
    report = session.edit("protonate")
    return {"m": m, "hydrogens": int(report.get("hydrogens", 0))
            + int(report.get("ligand_hydrogens", 0)),
            "incomplete": report.get("incomplete", [])}  # fmt: skip


def _split_path(path: dict, parts, n0: int = 0) -> dict:
    """A minimization's path (edit.MinimizationPath) by model: ``parts`` is
    [(model, where)], where None (every atom is the model's), an int (the atoms from
    there on, as the model's 0, 1 ...) or an array (atoms before ``n0``: the model's
    atoms of that array)."""
    atoms = np.asarray(path["atoms"], np.int64)
    frames = [np.asarray(f, np.float32).reshape(-1, 3) for f in path["frames"]]
    models = []
    for m, where in parts:
        if where is None:
            take, ids = np.ones(len(atoms), bool), atoms
        elif isinstance(where, int):
            take = atoms >= where
            ids = atoms[take] - where
        else:
            take = atoms < n0
            ids = np.asarray(where)[atoms[take]]
        if take.any():
            models.append({"m": m, "atoms": ids.tolist(),
                           "frames": [f[take].ravel().tolist() for f in frames]})  # fmt: skip
    return {"iterations": path["iterations"], "energy": path["energy"],
            "stages": path.get("stages", []), "models": models}  # fmt: skip


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
    # a design's receptor, by its old index: now where that model is in the list
    where = {e["index"]: k for k, e in enumerate(spec["models"])}
    for m, e in zip(models, spec["models"], strict=True):
        m.receptor = where.get(e.get("receptor"))
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
        elif members := _series_members(p):
            models += members
        else:
            models.append(Model(p.name, lambda p=p: _session(p, **kwargs), group="files"))
    title = " + ".join(p.name for p in paths)
    return Viewer(models, title=title, columns=columns)


def _series_members(path) -> list[Model]:
    """An SDF a series was saved as (every record has its ``series`` field): a model
    per record, in that series again; else none."""
    if path.suffix.lower() != ".sdf":
        return []
    from ..io.sdf import load_sdf

    s = load_sdf(path)
    if not s.ncts or any("series" not in s.ct(c).keys() for c in range(s.ncts)):
        return []
    reps = [{"sel": "all", "style": "ballstick", "color": "element"}]
    out = []
    for c in range(s.ncts):
        name = s._ct_names[c] or f"design {c + 1}"
        session = Session(s.clone(s.ct_atoms(c)), title=name, reps=reps)
        props = s.ct(c)
        row = {"note": str(props["note"])} if "note" in props.keys() else {}
        out.append(Model(name, lambda x=session: x, group=str(props["series"]), row=row,
                         _session=session))  # fmt: skip
    return out


def from_folder(d) -> Viewer:
    """What a folder holds: a view.json, or the output of modes, triage or cavity."""
    d = Path(d)
    if (d / "view.json").is_file():
        return from_manifest(d / "view.json")
    if (d / "modes.csv").is_file() and (d / "structures").is_dir():
        return from_modes(d)
    if (d / "summary.csv").is_file() and (d / "structures").is_dir():
        return from_triage(d)
    if (d / "pockets.csv").is_file() and (d / "sites.pqr").is_file():
        return from_cavity(d)
    raise ValueError(f"{d}: not a folder boonza view knows (a view.json, or the output of "
                     "boonza modes, triage or cavity)")  # fmt: skip


def _rows(path, key: str) -> dict:
    if not Path(path).is_file():
        return {}
    with open(path, newline="") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def _number(v):
    """CSV text as a number where it is one, for sorting the table.

    A value that is not finite becomes None.  boonza triage leaves a metric it
    could not measure as the text ``nan`` -- a run that detached has no dRMSD
    against its own first frame -- and JSON has no NaN: ``json.dumps`` writes a
    bare ``NaN`` token, which a browser's ``JSON.parse`` refuses.  It refuses
    the whole body, so one such cell and not one model in the response loads.
    None is ``null``, which the page shows as an empty cell, the same as the
    blank cell this already skips.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
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
                kw = {k: m[k] for k in ("ligand", "carbon", "reps", "numbers") if m.get(k)}
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
    """A structure with its ligand from another file appended, as one model (a folder
    written before ligand.json)."""
    import boonza

    s = boonza.load(structure)
    ids = s.append(boonza.load(ligand))
    return Session(s, title=Path(structure).name, ligand=f"index {ids[0]} to {ids[-1]}")


def from_cavity(d) -> Viewer:
    """boonza cavity: the protein (or apo structure), its ligand, and each pocket's
    medoid site; pharmacophore hotspots and the holo structure are listed, off.  The
    ligand is in the file ligand.json names: in the structure, it is part of that
    model; in holo, it is a model of its own over the apo protein, and holo is off."""
    import json

    d = Path(d)
    rows = _rows(d / "pockets.csv", "rank")
    models = []

    def structure(name, f, shown, **kw):
        models.append(Model(name, lambda: _session(f, **kw), group="structure", shown=shown))

    def first(stem):
        return next((f for f in sorted(d.glob(f"{stem}.*")) if f.suffix != ".json"), None)

    main, holo = first("apo") or first("structure"), first("holo")
    beads = sorted(d.glob("beads.*"))
    lig_file = lig_sel = None
    if (d / "ligand.json").is_file():
        info = json.loads((d / "ligand.json").read_text())
        lig_file, lig_sel = d / info["file"], info["selection"]
    old = first("ligand") if lig_file is None else None  # written before ligand.json
    if old is not None and holo is None and main is not None:  # the structure's own ligand
        models.append(Model(main.stem, lambda f=main, g=old: _with_ligand(f, g), group="structure"))
        main = None
    elif old is not None:
        lig_file, lig_sel = old, "all"
    with_main = lig_file is not None and lig_file == main
    if main is not None:
        structure(main.stem, main, True, **({"ligand": lig_sel} if with_main else {}))
    if beads:
        structure("beads", beads[0], not models)
    if lig_file is not None and not with_main:
        structure("ligand", lig_file, True, atoms=lig_sel, ligand="all")
    if holo is not None:
        structure("holo", holo, False, **({"ligand": lig_sel} if lig_file == holo else {}))

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
