"""Molecules in Jupyter notebooks, drawn by 3Dmol.js (loaded from a CDN when shown).

    s.view()                                        # cartoon, ligand sticks, ion spheres
    s.view("protein or resname LIG", style="sticks")
    boonza.view(s, positions=frames[::10])          # an animation, one model per frame
    boonza.view(s).save("complex.html")             # a standalone page

No Python package is needed: the view is HTML with a script, as py3Dmol
produces.  Coordinates go into the page as PDB text, so select what you want
to see from large solvated systems (water is hidden by default anyway).
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"
STYLES = ("auto", "cartoon", "sticks", "lines", "spheres")
_counter = itertools.count()


class MoleculeView:
    """One 3Dmol.js viewer: shown by Jupyter, or written as a page with ``save``."""

    def __init__(self, pdb: str, nframes: int, styles: list, width: int, height: int,
                 background: str, interval: int):  # fmt: skip
        self.pdb = pdb
        self.nframes = nframes
        self.styles = styles
        self.width, self.height = width, height
        self.background = background
        self.interval = interval

    def to_html(self) -> str:
        uid = f"boonza-view-{os.getpid()}-{next(_counter)}"
        add = "addModelsAsFrames" if self.nframes > 1 else "addModel"
        styles = "\n".join(f"    v.setStyle({json.dumps(sel)}, {json.dumps(style)});"
                           for sel, style in self.styles)  # fmt: skip
        animate = (f'    v.animate({{"loop": "forward", "interval": {self.interval}}});\n'
                   if self.nframes > 1 else "")  # fmt: skip
        return f"""<div id="{uid}" style="width: {self.width}px; height: {self.height}px; \
position: relative;"></div>
<script>
(function () {{
  function draw() {{
    var v = $3Dmol.createViewer(document.getElementById("{uid}"),
                                {{"backgroundColor": {json.dumps(self.background)}}});
    v.{add}({json.dumps(self.pdb)}, "pdb");
    v.setStyle({{}}, {{}});
{styles}
    v.zoomTo();
{animate}    v.render();
  }}
  if (window.$3Dmol) {{ draw(); return; }}
  var s = document.createElement("script");
  s.src = "{THREEDMOL_URL}";
  s.onload = draw;
  document.head.appendChild(s);
}})();
</script>"""

    def _repr_html_(self) -> str:
        return self.to_html()

    def save(self, path) -> None:
        """Write a standalone HTML page."""
        with open(os.fspath(path), "w") as fh:
            fh.write("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
                     "<title>boonza view</title></head><body>\n"
                     f"{self.to_html()}\n</body></html>\n")  # fmt: skip

    def __repr__(self) -> str:
        return f"<MoleculeView {self.nframes} frame(s); display in a notebook or .save(path)>"


def _name(nm: str) -> str:
    return nm[:4] if len(nm) >= 4 else " " + nm.ljust(3)


def _pdb_text(system, ids, frames, polymer, bonded) -> str:
    from .elements import msys_symbol

    A, R, C = system._atoms, system._residues, system._chains
    res = A.column("residue")[ids]
    ch = R.column("chain")[res]
    head = []
    for k, (a, r, c) in enumerate(zip(ids.tolist(), res.tolist(), ch.tolist(), strict=True)):
        record = "ATOM  " if polymer[k] else "HETATM"
        resname = str(R.column("name")[r])[:3]
        chain = (str(C.column("name")[c]) or " ")[:1]
        resid = int(R.column("resid")[r]) % 10000
        head.append(f"{record}{k + 1:5d} {_name(str(A.column('name')[a]))} {resname:>3s} "
                    f"{chain}{resid:4d}    ")  # fmt: skip
    elem = [f"  1.00  0.00          {msys_symbol(int(z))[:2]:>2s}"
            for z in A.column("anum")[ids].tolist()]  # fmt: skip
    conect = [f"CONECT{i + 1:5d}{j + 1:5d}" for i, j in bonded]
    out = []
    for f, X in enumerate(frames):
        if len(frames) > 1:
            out.append(f"MODEL     {f + 1:4d}")
        out.extend(f"{h}{x:8.3f}{y:8.3f}{z:8.3f}{e}"
                   for h, (x, y, z), e in zip(head, X.tolist(), elem, strict=True))  # fmt: skip
        out.extend(conect)
        out.append("ENDMDL" if len(frames) > 1 else "END")
    return "\n".join(out) + "\n"


def view(system, atoms=None, positions=None, style: str = "auto", width: int = 640,
         height: int = 480, water: bool = False, background: str = "white",
         interval: int = 100) -> MoleculeView:  # fmt: skip
    """A notebook view of ``atoms`` (default all), displayed with 3Dmol.js.

    ``style="auto"`` draws polymers as cartoons, other molecules as sticks,
    ions as spheres and hides water (``water=True`` shows it as lines);
    "cartoon", "sticks", "lines" and "spheres" apply to everything.
    ``positions``: one frame or several ((nframes, natoms, 3) or Frames),
    animated ``interval`` ms apart.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    ids = np.arange(system.natoms) if atoms is None else (
        system.select(atoms).ids if isinstance(atoms, str) else
        np.asarray(system._ids("atom", atoms), dtype=np.int64))  # fmt: skip
    if len(ids) == 0:
        raise ValueError("no atoms to draw")
    if len(ids) > 99999:
        raise ValueError(f"{len(ids)} atoms: select fewer than 100000 atoms to draw")
    xyz = system.positions if positions is None else np.asarray(
        getattr(positions, "positions", positions), dtype=np.float64)  # fmt: skip
    if xyz.ndim == 2:
        xyz = xyz[None]
    if xyz.shape[1:] != (system.natoms, 3):
        raise ValueError(f"positions must be (nframes, {system.natoms}, 3)")

    def mask(sel):
        m = np.zeros(system.natoms, bool)
        m[system.select(sel).ids] = True
        return m[ids]

    polymer, is_water, ion = mask("polymer"), mask("water"), mask("ions")
    local = np.full(system.natoms, -1, np.int64)
    local[ids] = np.arange(len(ids))
    bi, bj = local[system.bonds["i"]], local[system.bonds["j"]]
    keep = (bi >= 0) & (bj >= 0)
    bi, bj = bi[keep], bj[keep]
    keep = ~(polymer[bi] & polymer[bj])  # 3Dmol bonds standard residues itself
    bonded = list(zip(bi[keep].tolist(), bj[keep].tolist(), strict=True))
    pdb = _pdb_text(system, ids, xyz[:, ids], polymer, bonded)

    def serials(m):
        return {"serial": (np.flatnonzero(m) + 1).tolist()}

    if style == "auto":
        other = ~(polymer | is_water | ion)
        styles = []
        if polymer.any():
            styles.append((serials(polymer), {"cartoon": {"color": "spectrum"}}))
        if other.any():
            styles.append((serials(other), {"stick": {"radius": 0.2}}))
        if ion.any():
            styles.append((serials(ion), {"sphere": {"scale": 0.8}}))
        if water and is_water.any():
            styles.append((serials(is_water), {"line": {}}))
    else:
        key = {"cartoon": "cartoon", "sticks": "stick", "lines": "line", "spheres": "sphere"}[style]
        styles = [({}, {key: {"color": "spectrum"} if key == "cartoon" else {}})]
    return MoleculeView(pdb, len(xyz), styles, width, height, background, interval)
