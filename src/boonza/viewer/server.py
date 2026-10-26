"""The HTTP server between boonza and the page.

The page (``data/viewer``, plain WebGL2, no libraries) only draws.  It asks:

    GET  /api/models          the models, their table rows, groups, what is shown
    GET  /api/topology?m=M    model M's atoms, residues, chains, bonds with orders,
                              DSSP, default representations (JSON, once)
    GET  /api/frame?m=M&i=N   positions of frame N as little-endian float32
                              (natoms * 3), after Glue; the cell in X-Box
    GET  /api/ss?m=M&i=N      DSSP (simplified: H, E, C) of frame N
    GET  /api/contacts?m=M&i=N   the ligand's polar contacts in frame N
    GET  /api/smiles?m=M&sel=a,b  SMILES of the molecules of those atoms (default: ligands)
    POST /api/select          {"m": M, "sel": text, "frame": N} -> {"ids": [...]}
    POST /api/glue            {"m": M, "whole": bool, "center": sel, "fit": sel}
    POST /api/edit            {"m": M, "op": element|charge|bond|delete|hydrogens,
                               "atoms": [...], "value": ..., "hydrogens": bool}
    POST /api/undo, /redo, /save   {"m": M[, "path": file]}
    POST /api/set_positions   {"m": M, "atoms": [...], "xyz": [...]} (atoms moved by mouse)
    POST /api/set_distance, /api/set_angle   {"m": M, "atoms": [...], "value": x}
    POST /api/open            {"path": file or folder} -> the new models; /api/close {"m": M}
    POST /api/build           {"smiles": ..., "centre": [x, y, z]} -> a new model of it
    POST /api/upload          a file's bytes (its name in X-Filename) -> opened
    POST /api/copy_into       {"m": from, "into": to, "atoms": [...]} -> appended there
    POST /api/hbonds          {"m": M, "a": sel, "b": sel or none, "frame": N} -> pairs
    POST /api/download        {"m": M, "fmt": pdb|cif|sdf|mae|dms|gro, "atoms": [...]} -> a file
    POST /api/python          {"code": ..., "m": current, "sel": [...]} -> output, actions
                              (only on 127.0.0.1; every POST carries the page's token)
    POST /api/surface_charges {"m": M} -> the force field's charges for the surface
    POST /api/fetch           {"id": PDB ID} -> downloads it (mmCIF) and opens it
    POST /api/align           {"m": mobile, "ref": reference, "method", "sel", "ref_sel",
                               "chain", "ref_chain"}; method: matchmaker|cealign|superpose|pair
    POST /api/dihedral        {"m": M, "atoms": [a, b, c, d]} -> angle, atoms that turn
    POST /api/set_dihedral    {"m": M, "atoms": [...], "angle": degrees}
    POST /api/dihedral_scan   {"m": M, "atoms": [...], "step": 10} -> energy profiles
    POST /api/minimize        {"m": M, "mobile": sel, "restraint": kcal/mol/A^2}

Trajectories stay on the server and are read one frame at a time, so a
viewer started on a cluster node shows long runs through an SSH tunnel
(``ssh -L 8765:localhost:8765 host``) without copying them.
"""

from __future__ import annotations

import errno
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .outputs import Model, Viewer
from .session import Session

STATIC = Path(__file__).parent.parent / "data" / "viewer"
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


#: one request works on the models at a time: numba's default threading layer aborts
#: the process when two threads run its parallel code at once (two models loading)
WORK = threading.RLock()


class _Handler(BaseHTTPRequestHandler):
    viewer: Viewer  # set on the subclass made by make_server
    token: str = ""  # every POST carries it: only the page this server wrote can change things
    console = None  # the Python line (None: not offered)

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200, headers=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj, separators=(",", ":")).encode(), "application/json", status)

    def _guard(self, handle) -> None:
        try:
            handle()
        except (ValueError, IndexError, KeyError) as e:  # includes selection errors
            self._json({"error": str(e)}, 400)
        except FileNotFoundError as e:
            self._json({"error": str(e)}, 404)
        except ConnectionError:  # the page went away (reload, closed tab)
            pass
        except Exception as e:  # shown on the page rather than a dropped connection
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            raise

    def do_GET(self) -> None:
        if not self.path.startswith("/api/"):
            self._guard(self._get)
            return
        with WORK:
            self._guard(self._get)

    def do_POST(self) -> None:
        with WORK:
            self._guard(self._post)

    def _get(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)

        def arg(name):
            return int(query.get(name, ["0"])[0])

        if url.path == "/api/models":
            self._json(self.viewer.listing())
        elif url.path == "/api/topology":
            self._json(self.viewer.session(arg("m")).topology())
        elif url.path == "/api/frame":
            pos, box = self.viewer.session(arg("m")).frame(arg("i"))
            self._send(pos.tobytes(), "application/octet-stream",
                       headers={"X-Box": json.dumps(box.tolist())})  # fmt: skip
        elif url.path == "/api/ss":
            self._json({"ss": self.viewer.session(arg("m")).ss(arg("i"))})
        elif url.path == "/api/ligand2d":
            sel = [int(x) for x in query.get("sel", [""])[0].split(",") if x]
            dark = query.get("dark", ["1"])[0] == "1"
            self._json(self.viewer.session(arg("m")).ligand2d(sel, dark=dark))
        elif url.path == "/api/atom":
            self._json(self.viewer.session(arg("m")).atom_info(arg("a")))
        elif url.path == "/api/smiles":
            sel = [int(x) for x in query.get("sel", [""])[0].split(",") if x]
            self._json({"molecules": self.viewer.session(arg("m")).smiles(sel)})
        elif url.path == "/api/contacts":
            self._json({"contacts": self.viewer.session(arg("m")).contacts(arg("i"))})
        else:
            self._static(url.path)

    def _post(self) -> None:
        url = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if not secrets.compare_digest(self.headers.get("X-Boonza-Token", ""), self.token):
            self._json({"error": "this request did not come from the viewer's page"}, 403)
            return
        if url.path == "/api/upload":  # a file's bytes, its name in a header
            from urllib.parse import unquote

            name = unquote(self.headers.get("X-Filename", ""))
            self._json({"added": self.viewer.upload(name, raw), **self.viewer.listing()})
            return
        body = json.loads(raw or b"{}")
        if url.path == "/api/python":
            if self.console is None:
                self._json({"error": "the Python line is offered only when boonza view listens "
                                     "on this machine alone (127.0.0.1)"}, 403)  # fmt: skip
                return
            self._json(self.console.run(str(body.get("code", "")), int(body.get("m", 0)),
                                        body.get("sel", [])))  # fmt: skip
            return
        if url.path == "/api/open":
            self._json({"added": self.viewer.open(body["path"]), **self.viewer.listing()})
            return
        if url.path == "/api/build":
            added = self.viewer.build(str(body["smiles"]), body.get("centre") or (0, 0, 0))
            self._json({"added": added, **self.viewer.listing()})
            return
        if url.path == "/api/fetch":
            self._json({"added": self.viewer.fetch(body["id"]), **self.viewer.listing()})
            return
        if url.path == "/api/align":
            mobile = self.viewer.session(int(body["m"]))
            ref = self.viewer.session(int(body["ref"]))
            how = {k: body.get(k) for k in ("sel", "ref_sel", "chain", "ref_chain")}
            self._json(mobile.align(ref, body.get("method", "matchmaker"), **how))
            return
        if url.path == "/api/save_session":
            self._json({"path": self.viewer.save_session(body["path"], body.get("page") or {})})
            return
        if url.path == "/api/copy_into":
            self._json(self.viewer.copy_into(int(body["m"]), int(body["into"]),
                                             body.get("atoms") or None,
                                             int(body.get("frame", 0))))  # fmt: skip
            return
        if url.path == "/api/close":
            self.viewer.close(int(body["m"]))
            self._json(self.viewer.listing())
            return
        session = self.viewer.session(int(body.get("m", 0)))
        if url.path == "/api/select":
            ids = session.select(str(body.get("sel", "")), int(body.get("frame", 0)))
            self._json({"ids": ids.tolist()})
        elif url.path == "/api/download":
            data, name = session.export(str(body.get("fmt", "pdb")), body.get("atoms") or (),
                                        int(body.get("frame", 0)))  # fmt: skip
            saved = {"Content-Disposition": f'attachment; filename="{name}"'}
            self._send(data, "application/octet-stream", headers=saved)
        elif url.path == "/api/hbonds":
            a = body["a"] if isinstance(body["a"], list) else str(body["a"])
            self._json(session.hbonds(a, body.get("b") or None,
                                      int(body.get("frame", 0))))  # fmt: skip
        elif url.path == "/api/glue":
            session.set_glue(bool(body.get("whole", True)), body.get("center"), body.get("fit"))
            self._json({"ok": True})
        elif url.path == "/api/edit":
            self._json(session.edit(body["op"], body.get("atoms", ()), body.get("value"),
                                    hydrogens=bool(body.get("hydrogens", True))))  # fmt: skip
        elif url.path == "/api/undo":
            self._json(session.undo())
        elif url.path == "/api/redo":
            self._json(session.redo())
        elif url.path == "/api/surface_charges":
            self._json(session.surface_charges())
        elif url.path == "/api/assign_ff":
            s = session
            s._parameterized()
            self._json({"ok": True, "atom": s.atom_info(int(body["a"])) if "a" in body else None})
        elif url.path == "/api/set_ligand":
            self._json(session.set_ligand(body["atoms"]))
        elif url.path == "/api/set_positions":
            self._json(session.set_positions(body["atoms"], body["xyz"]))
        elif url.path == "/api/set_distance":
            self._json(session.set_distance(body["atoms"], float(body["value"])))
        elif url.path == "/api/set_angle":
            self._json(session.set_angle(body["atoms"], float(body["value"])))
        elif url.path == "/api/save":
            self._json({"path": session.save(body["path"])})
        elif url.path == "/api/dihedral":
            self._json(session.dihedral(body["atoms"]))
        elif url.path == "/api/set_dihedral":
            self._json(session.set_dihedral(body["atoms"], float(body["angle"])))
        elif url.path == "/api/dihedral_scan":
            self._json(session.dihedral_scan(body["atoms"], float(body.get("step", 10))))
        elif url.path == "/api/minimize":
            self._json(
                session.minimize(
                    body.get("mobile") or None,
                    float(body.get("restraint", 10.0)),
                    fixed=body.get("fixed"),
                    tethered=body.get("tethered"),
                )
            )
        else:
            self._json({"error": "not found"}, 404)

    def _static(self, path: str) -> None:
        if path == "/favicon.ico":  # none: say so without an error in the page's console
            self._send(b"", "image/x-icon", 204)
            return
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        f = STATIC / name
        # only files directly in the viewer directory
        if "/" in name or f.suffix not in TYPES or not f.is_file():
            self._json({"error": "not found"}, 404)
            return
        body = f.read_bytes()
        if name == "index.html":  # the token, for the page's own requests
            meta = f'<meta name="boonza-token" content="{self.token}">'
            meta += f'<meta name="boonza-python" content="{int(self.console is not None)}">'
            body = body.replace(b"</head>", meta.encode() + b"</head>", 1)
        self._send(body, TYPES[f.suffix] + "; charset=utf-8")


def _viewer(what) -> Viewer:
    if isinstance(what, Viewer):
        return what
    if isinstance(what, Session):
        return Viewer([Model(what.title or "model", lambda: what)], title=what.title)
    return Viewer([Model("model", lambda: Session(what))])


LOCAL = ("127.0.0.1", "localhost", "::1")


def make_server(what, host: str = "127.0.0.1", port: int = 8765,
                python: bool | None = None) -> ThreadingHTTPServer:  # fmt: skip
    """An HTTP server (not started) for a Viewer, a Session or a System; ``port=0``
    picks a free port.  ``python``: offer the Python line (default: when ``host`` is
    this machine alone)."""
    from .console import Console

    viewer = _viewer(what)
    python = host in LOCAL if python is None else python
    handler = type("Handler", (_Handler,), {
        "viewer": viewer, "token": secrets.token_urlsafe(24),
        "console": Console(viewer) if python else None})  # fmt: skip
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


#: where ``serve`` looks for a port when none is given: 8765, then the next ones
PORTS = range(8765, 8785)


def _free_server(viewer, host: str, port: int | None) -> ThreadingHTTPServer:
    """A server on ``port``; with none, on the first of PORTS not in use (another
    viewer, a tunnel), else any free one."""
    if port is not None:
        try:
            return make_server(viewer, host, port)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            why = (f"port {port} is in use (another boonza view, or a tunnel?): give "
                   "another --port, or leave it out to take a free one")  # fmt: skip
            raise OSError(e.errno, why) from None
    for p in [*PORTS, 0]:
        try:
            return make_server(viewer, host, p)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
    raise AssertionError("port 0 is always free")


def serve(what, trajectory=None, atoms=None, host: str = "127.0.0.1",
          port: int | None = None, open_browser: bool = True, **glue) -> None:  # fmt: skip
    """View a System (and ``trajectory``), a Session or a Viewer in the browser until
    interrupted.  ``port``: default the first free one from 8765 on.  ``glue``:
    ``whole``, ``center`` and ``fit`` as for :class:`Session`."""
    if not isinstance(what, (Viewer, Session)):
        what = Session(what, trajectory, atoms, **glue)
    viewer = _viewer(what)
    server = _free_server(viewer, host, port)
    port = server.server_address[1]
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    what = (f"{len(viewer)} models" if len(viewer) > 1 else
            viewer.title or ("1 model" if len(viewer) else "an empty view"))  # fmt: skip
    print(f"boonza view: {what} at {url}")
    print(f"  (on a remote machine: ssh -L {port}:localhost:{port} <host>, then open {url}; "
          "Ctrl-C stops)")  # fmt: skip
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
