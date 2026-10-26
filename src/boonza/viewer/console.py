"""A Python line in the viewer, with boonza at hand.

    >>> model0                      # the first model: a boonza System (model1, model2, ...)
    >>> m.select("resname LIG")     # m: the current model; sel: its selected atoms
    >>> boonza.matchmaker(model1, model0)        # moves model1; the page follows
    >>> view.select("within 5 of resname LIG")   # and drives the page
    >>> view.show(sel); view.color(sel, "cyan"); view.center(sel)

``models`` lists them all (``models[0]``, or by name: ``models["1IR3.cif"]``).
Whatever a line changes in a model (positions, atoms, bonds, charges) is
picked up afterwards: the page reloads that model, and Edit | Undo undoes it.

The console runs code as the user running boonza view, so the server offers
it only when it listens on this machine alone (127.0.0.1), and every request
must carry the token the server wrote into its own page.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import traceback

import numpy as np


class _Models:
    """The models' Systems, by position or by name; loaded on first use."""

    def __init__(self, viewer):
        self._viewer = viewer

    def _live(self):
        return [k for k, m in enumerate(self._viewer.models) if not m.closed]

    def __len__(self) -> int:
        return len(self._live())

    def __getitem__(self, key):
        if isinstance(key, str):
            for k in self._live():
                if self._viewer.models[k].name == key:
                    return self._viewer.session(k).system
            raise KeyError(key)
        return self._viewer.session(self._live()[key]).system

    def __iter__(self):
        return (self[k] for k in range(len(self)))

    def __repr__(self) -> str:
        return "models: " + ", ".join(
            f"model{k} = {self._viewer.models[k].name}" for k in self._live()
        )


class _View:
    """What a line asks of the page: each call adds an action the page carries out."""

    def __init__(self, console):
        self._console = console

    def _ids(self, atoms, model):
        k = self._console.current if model is None else self._console.index(model)
        s = self._console.viewer.session(k).system
        if atoms is None:
            return k, list(range(s.natoms))
        if isinstance(atoms, str):
            return k, s.select(atoms).ids.tolist()
        if hasattr(atoms, "ids"):
            return k, np.asarray(atoms.ids).tolist()
        return k, [int(a) for a in np.atleast_1d(atoms)]

    def _do(self, what, atoms=None, model=None, **more):
        k, ids = self._ids(atoms, model)
        self._console.actions.append({"do": what, "m": k, "ids": ids, **more})
        return len(ids)

    def select(self, atoms, model=None, add: bool = False):
        """Select atoms (a selection string, an AtomSel, or indices)."""
        return self._do("select", atoms, model, add=add)

    def show(self, atoms=None, model=None, style: str | None = None):
        """Show atoms, as a new representation; ``style``: lines, sticks, ballstick, spheres."""
        return self._do("show", atoms, model, style=style)

    def hide(self, atoms=None, model=None):
        return self._do("hide", atoms, model)

    def color(self, atoms, scheme: str, model=None):
        """Colour atoms: element, element/green, chain, restype, a name, #rrggbb."""
        return self._do("color", atoms, model, scheme=scheme)

    def center(self, atoms=None, model=None):
        return self._do("center", atoms, model)

    def label(self, atoms, mode: str = "name", model=None):
        """Label atoms: name, element, residue, charge, or off."""
        return self._do("label", atoms, model, mode=mode)

    def __repr__(self) -> str:
        return "view: select, show, hide, color, center, label (atoms, model=None)"


def _fingerprint(s):
    """(topology, positions) digests of a System, to see what a line changed."""
    topo = hashlib.sha1()
    for column in ("anum", "formal_charge", "residue"):
        topo.update(np.ascontiguousarray(s.atoms[column]).tobytes())
    for column in ("i", "j", "order"):
        topo.update(np.ascontiguousarray(s.bonds[column]).tobytes())
    topo.update(" ".join(map(str, s.atoms["name"])).encode())
    pos = hashlib.sha1(np.ascontiguousarray(s.positions).tobytes()).digest()
    return topo.digest(), pos


class Console:
    """One namespace for the whole session: names set by one line stay for the next."""

    def __init__(self, viewer):
        import boonza

        self.viewer = viewer
        self.current = 0
        self.actions: list = []
        self.ns = {"boonza": boonza, "np": np, "models": _Models(viewer), "view": _View(self)}

    def index(self, model) -> int:
        """A model's index from its index, its name or its System."""
        if isinstance(model, int):
            return model
        for k, m in enumerate(self.viewer.models):
            if m.name == model or (m._session is not None and m._session.system is model):
                return k
        raise ValueError(f"no model {model!r}")

    def _bind(self, selected) -> None:
        for k, m in enumerate(self.viewer.models):
            if m._session is not None and not m.closed:
                self.ns[f"model{k}"] = m._session.system
        cur = self.viewer.session(self.current)
        self.ns["m"] = cur.system
        self.ns["sel"] = cur.system.select(list(selected)) if selected else cur.system.select([])
        for k, m in enumerate(self.viewer.models):  # unloaded models: load on first mention
            if m._session is None and not m.closed:
                self.ns.pop(f"model{k}", None)

    def run(self, code: str, current: int = 0, selected=()) -> dict:
        self.current = int(current)
        self.actions = []
        # load the models a line names, so model3 works before it was ever shown
        for k, m in enumerate(self.viewer.models):
            if not m.closed and f"model{k}" in code:
                self.viewer.session(k)
        sessions = {k: m._session for k, m in enumerate(self.viewer.models)
                    if m._session is not None and not m.closed}  # fmt: skip
        self._bind(selected)
        before = {k: (_fingerprint(s.system), s.system.copy()) for k, s in sessions.items()
                  if s.trajectory is None and s.radii is None}  # fmt: skip
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                try:
                    compiled = compile(code, "<boonza>", "eval")
                except SyntaxError:
                    exec(compile(code, "<boonza>", "exec"), self.ns)
                else:
                    value = eval(compiled, self.ns)
                    if value is not None:
                        print(repr(value))
                        self.ns["_"] = value
            except Exception:
                lines = traceback.format_exc().splitlines()
                print("\n".join(lines[-4:]))
        changed = []
        for k, ((topo, pos), copy) in before.items():
            s = sessions[k]
            now_topo, now_pos = _fingerprint(s.system)
            if (now_topo, now_pos) == (topo, pos):
                continue
            s._undo = getattr(s, "_undo", [])
            s._undo.append((copy, getattr(s, "topo_version", 0), s.ids, s.radii))
            s._redo = []
            changed.append({"m": k, "topology": now_topo != topo})
            s._changed(topology=now_topo != topo)
        return {"output": out.getvalue(), "actions": self.actions, "changed": changed}
