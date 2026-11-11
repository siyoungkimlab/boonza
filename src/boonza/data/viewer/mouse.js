// The mouse in the rendering area (Help | Mouse), and right-drag to zoom:
//
//   left    click: select an atom (or a bond, or a ribbon's residue); Shift toggles;
//           Ctrl (or Cmd) takes the residue; double-click: the residue;
//           triple-click (or double-click it again): the molecule (of its class);
//           drag: select a box (Shift adds, Ctrl takes residues); click away: clear
//   middle  drag: rotate; in the perimeter: about z; Shift: pan; Ctrl: zoom;
//           click an atom: the center of rotation (away from atoms: back to the view's)
//   Alt + middle   the selected atoms: rotate them (perimeter: about z); Shift+Alt: move
//   Alt + left     with two bonded atoms selected: turn about that bond
//   right   the popup menu;  wheel: zoom;  Shift over an atom: what it is
//
// A two-button mouse (Help | Mouse) uses the right button as the middle one and
// opens the popup when Alt is pressed and released alone.  A trackpad (Help | Mouse)
// rotates with a two-finger swipe (in the perimeter: about z), pans with Shift and
// zooms with a pinch.
//
// That is the default, boonza's own.  Help | Mouse has two more modes, for hands that
// know another program (binding() below is the whole table):
//
//   ChimeraX   left: rotate; Ctrl + left: select (click or box; Shift toggles);
//              middle or right: move; wheel: zoom; Ctrl + double-click: the popup
//   PyMOL      (3-Button Viewing) left: rotate; middle: move, click: the center of
//              rotation; right: zoom, click: the popup; Shift + left / middle: a box
//              added / taken away; Shift + right: clip; Ctrl + left: move; Ctrl +
//              middle or right click: one atom; Ctrl + Shift + left click: one
//              residue; wheel: the slab (Shift: move it; Ctrl: zoom)
//
// A click selects as it does in boonza's, whatever the mode: the atom under it (Shift
// toggles it; away from atoms: nothing selected), twice its residue, three times its
// molecule.  The other programs want a key or take a residue for that, and a click
// that did nothing, or more than was asked, read as a selection that does not work.
// In both, Alt edits as in boonza's too: Alt + left rotates the selected atoms,
// Alt + middle (ChimeraX: or right) moves them, Shift + Alt + left turns a bond.

const PERIMETER = 0.07;  // the band at the edges where dragging turns about z

export const MOUSE_MODES = ["boonza", "chimerax", "pymol"];

// What a button does, with the keys held when it went down:
//   act    a drag: box, rotate, pan, zoom, clip, dihedral, rotateSel, panSel
//   box    what a box does to the selection: set, add, toggle, remove
//   tap    a click without a drag: select, pivot, popup, tool (a builder tool's click)
//   pick   how a tap selects: mode (set, toggle), residues (whole ones), keep (a click
//          away from atoms leaves the selection)
//   zoom   +1: down zooms out; -1: down zooms in
export function binding(mode, role, shift, ctrl, alt) {
  if (mode === "chimerax") {
    if (role === "left") {
      if (alt) return { act: shift ? "dihedral" : "rotateSel", tap: "tool" };
      if (ctrl) return { act: "box", box: shift ? "toggle" : "set", tap: "select",
                         pick: { mode: shift ? "toggle" : "set", keep: shift } };
      return { act: "rotate", tap: "select", pick: { mode: shift ? "toggle" : "set", keep: shift } };
    }
    return { act: alt ? "panSel" : "pan", tap: role === "middle" ? "pivot" : "popup" };
  }
  if (mode === "pymol") {
    if (role === "left") {
      if (alt) return { act: shift ? "dihedral" : "rotateSel", tap: "tool" };
      if (ctrl && shift) return { tap: "select", pick: { mode: "set", residues: true } };
      if (shift) return { act: "box", box: "add", tap: "select", pick: { mode: "toggle", keep: true } };
      if (ctrl) return { act: "pan", tap: "tool" };
      return { act: "rotate", tap: "select", pick: { mode: "set" } };
    }
    if (role === "middle") {
      if (alt) return { act: "panSel" };
      if (ctrl && shift) return { tap: "pivot" };
      if (shift) return { act: "box", box: "remove" };
      if (ctrl) return { tap: "select", pick: { mode: "set" } };
      return { act: "pan", tap: "pivot" };
    }
    if (shift) return { act: "clip" };
    if (ctrl) return { tap: "select", pick: { mode: "set" } };
    return { act: "zoom", zoom: -1, tap: "popup" };
  }
  if (role === "left") {
    if (alt) return { act: "dihedral" };
    return { act: "box", box: shift && ctrl ? "toggle" : shift || ctrl ? "add" : "set", residues: ctrl,
             tap: "select", pick: { mode: shift ? "toggle" : "set", residues: ctrl, keep: shift || ctrl } };
  }
  if (role === "middle") {
    if (alt) return { act: shift ? "panSel" : "rotateSel", tap: "pivot" };
    return { act: shift ? "pan" : ctrl ? "zoom" : "rotate", zoom: 1, tap: "pivot" };
  }
  return { act: "zoom", zoom: -1, tap: "popup" };  // right-drag: up out, down in
}

export function setupMouse(app) {
  const { canvas, camera, state } = app;
  const stage = document.getElementById("stage");
  const rubber = document.getElementById("rubber");
  const bubble = document.getElementById("bubble");
  state.twoButton = localStorage.getItem("boonza.twoButton") === "1";
  state.trackpad = localStorage.getItem("boonza.trackpad") === "1";
  const saved = localStorage.getItem("boonza.mouseMode");
  state.mouseMode = MOUSE_MODES.includes(saved) ? saved : "boonza";
  app.onMouseMode?.();

  let drag = null;
  const local = e => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const inPerimeter = (x, y) => {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = PERIMETER * Math.min(w, h);
    return x < b || y < b || x > w - b || y > h - b;
  };
  const role = button => {
    if (button === 0) return "left";
    if (button === 1) return "middle";
    return twoButton() ? "middle" : "right";
  };
  // the two-button setting is boonza's own: the other modes use both buttons already
  const twoButton = () => state.twoButton && state.mouseMode === "boonza";
  const clamp = v => Math.min(Math.max(v, 0.02), 1);
  function clipBy(front, back) {  // the clipping planes, as Display | Clipping sets them
    const c = camera.zclip;
    c[0] = clamp(c[0] + front); c[1] = clamp(c[1] + back);
    app.clipChanged?.();
  }

  canvas.addEventListener("contextmenu", e => e.preventDefault());

  canvas.addEventListener("pointerdown", e => {
    // the view takes the keys back (Delete, Ctrl Z ...) from a field or list clicked before
    const focused = document.activeElement;
    if (focused && focused !== document.body && focused.blur) focused.blur();
    app.stopFlight();
    altAlone = false;
    const [x, y] = local(e);
    const r = role(e.button), ctrl = e.ctrlKey || e.metaKey;
    drag = { ...binding(state.mouseMode, r, e.shiftKey, ctrl, e.altKey),
             role: r, x0: x, y0: y, x, y, moved: false, perimeter: inPerimeter(x, y) };
    if (drag.role === "left") app.message("");  // a click in the view puts a message away
    canvas.setPointerCapture(e.pointerId);
    if (drag.act === "dihedral") startDihedral();
    e.preventDefault();
  });

  // Alt + left (the other modes: with Shift): two bonded atoms selected turn the bond
  async function startDihedral() {
    const sel = app.selection();
    const d = drag;
    if (sel.length !== 1 || sel[0][1].length !== 2) {
      app.message("turning a bond: select its two atoms first", true);
      d.dihedral = "none";
      return;
    }
    const [m, atoms] = sel[0];
    d.dihedral = "loading";
    try {
      const out = await app.post("api/dihedral", { m: m.index, atoms });
      d.dihedral = { ...out, model: m, base: m.pos.slice(), now: out.angle };
    } catch (err) {
      d.dihedral = "none";
      app.message(err.message, true);
    }
  }

  canvas.addEventListener("pointermove", e => {
    const [x, y] = local(e);
    if (drag && !e.buttons) { cancel(); return; }  // the button went up where we did not hear it
    if (!drag) { hover(e, x, y); return; }
    const dx = x - drag.x, dy = y - drag.y;
    if (!dx && !dy) return;
    if (!drag.moved && Math.abs(x - drag.x0) + Math.abs(y - drag.y0) < 3) return;
    drag.moved = true;
    drag.x = x; drag.y = y;
    bubble.hidden = true;

    const act = drag.act;
    if (act === "dihedral") {
      const d = drag.dihedral;
      if (d && typeof d === "object") {
        d.now += dx * 0.6;
        d.now = ((d.now + 540) % 360) - 180;
        app.turnDihedral(d, d.now);
        app.message(`dihedral ${d.now.toFixed(1)}°`);
      }
      return;
    }
    if (act === "box") {
      const x0 = Math.min(drag.x0, x), y0 = Math.min(drag.y0, y);
      Object.assign(rubber.style, { left: `${x0}px`, top: `${y0}px`,
                                    width: `${Math.abs(x - drag.x0)}px`, height: `${Math.abs(y - drag.y0)}px` });
      rubber.hidden = false;
      return;
    }
    if (act === "rotateSel" || act === "panSel") {  // the selected atoms move
      if (!drag.moving) drag.moving = app.beginMove() ? "yes" : "no";
      if (drag.moving !== "yes") return;
      if (act === "panSel") app.translateSelected(dx, dy);
      else if (drag.perimeter) app.rotateSelected(0, 0, 1, zTurn(x, y, dx, dy));
      else app.rotateSelected(dy, dx, 0, Math.hypot(dx, dy) * 0.01);
      return;
    }
    if (act === "pan") {
      const scale = 2 * camera.distance * Math.tan(camera.fov / 2) / canvas.clientHeight;
      camera.pan(dx * scale, -dy * scale);
    } else if (act === "zoom") {
      camera.zoom(Math.exp(drag.zoom * dy * 0.01));
    } else if (act === "clip") {  // up and down: the front plane; left and right: the back
      clipBy(dy * 0.004, dx * 0.004);
    } else if (act !== "rotate") {
      return;
    } else if (drag.perimeter) {
      camera.rotate(0, 0, 1, zTurn(x, y, dx, dy));
    } else {
      camera.rotate(dy, dx, 0, Math.hypot(dx, dy) * 0.008);
    }
    app.draw();
  });

  // the angle a drag sweeps about the center of the rendering area
  function zTurn(x, y, dx, dy) {
    const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
    return Math.atan2(y - dy - cy, x - dx - cx) - Math.atan2(y - cy, x - cx);
  }

  canvas.addEventListener("pointerup", async e => {
    if (!drag) return;
    const d = drag;
    drag = null;
    const [x, y] = local(e);
    rubber.hidden = true;

    if (d.act === "dihedral") {
      const t = d.dihedral;
      if (t && typeof t === "object" && d.moved) {
        try {
          await app.post("api/set_dihedral", { m: t.model.index, atoms: t.atoms, angle: t.now });
          await app.reloadModel(t.model, false);
        } catch (err) { app.message(err.message, true); }
      }
      if (d.moved || !d.tap) return;
    }
    if (d.moving === "yes") { await app.endMove(); return; }
    if (d.moved) {
      if (d.act !== "box") return;
      const found = app.inRect(Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y));
      if (d.box === "set") app.clearSelection();
      for (const [m, ids] of found) {
        app.select(m, d.residues ? residuesOf(m, ids) : ids, d.box === "set" ? "add" : d.box);
      }
      if (!found.length) { app.rebuild(); app.onSelection(); }
      return;
    }
    if (d.tap === "select") { click(x, y, d.pick); return; }
    if (d.tap === "tool") { if (app.toolActive?.()) app.applyTool(app.pick(x, y), x, y); return; }
    if (d.tap === "pivot") {  // the center of rotation
      const hit = app.pick(x, y);
      if (hit && hit.atom !== undefined) {
        camera.pivot = app.xyz(hit.model, hit.atom);
        app.message(`center of rotation: ${app.atomLabel(hit.model, hit.atom, false)}`);
      } else {
        camera.pivot = null;
        app.message("center of rotation: the view's");
      }
      return;
    }
    if (d.tap === "popup") app.popup(e.clientX, e.clientY, app.pick(x, y));
  });

  // a drag the browser takes away (a trackpad gesture, a window switch): it ends
  // there, with nothing selected and no box left on the screen
  function cancel() {
    if (drag?.moving === "yes") app.endMove();
    drag = null;
    rubber.hidden = true;
  }
  canvas.addEventListener("pointercancel", cancel);
  canvas.addEventListener("lostpointercapture", () => { if (drag) cancel(); });
  window.addEventListener("blur", () => { if (drag) cancel(); });
  window.addEventListener("keydown", e => { if (e.key === "Escape" && drag) cancel(); });

  function residuesOf(m, ids) {
    const out = new Set();
    for (const a of ids) for (const b of m.topo.residueAtoms[m.topo.atoms.residue[a]]) out.add(b);
    return [...out];
  }

  function click(x, y, p) {
    const hit = app.pick(x, y);
    if (app.applyTool?.(hit, x, y)) return;  // a builder tool in hand takes the click
    if (!hit) {
      if (!p.keep) { app.clearSelection(); app.rebuild(); app.onSelection(); }
      return;
    }
    const m = hit.model;
    let ids = hit.atom !== undefined ? [hit.atom] : hit.atoms || m.topo.residueAtoms[hit.residue];
    if (p.residues) ids = residuesOf(m, ids);
    app.select(m, ids, p.mode);
    if (hit.atom !== undefined) app.describeAtom(m, hit.atom);
    else if (hit.residue !== undefined) app.message(`residue ${app.residueLabel(m, hit.residue)}`);
    else if (hit.atoms) app.message(`bond ${app.atomLabel(m, hit.atoms[0], false)} – ${app.atomLabel(m, hit.atoms[1], false)}`
      + `  (order ${m.topo.bonds.order[m.topo.bonds.i.findIndex((i, k) => (i === hit.atoms[0] && m.topo.bonds.j[k] === hit.atoms[1]) || (i === hit.atoms[1] && m.topo.bonds.j[k] === hit.atoms[0]))] ?? 1})`);
  }

  // double-click (an atom, a bond or a ribbon): its residue; a triple click, or a
  // double-click again on that residue: its molecule, every atom bonded to it,
  // receptor, ligand or cap alike; again: the residue
  let lastDouble = null;
  function residueOrMolecule(e, molecule) {
    if (app.toolActive?.()) return;
    const [x, y] = local(e);
    const hit = app.pick(x, y);
    if (!hit) return;
    const m = hit.model, A = m.topo.atoms;
    const seed = hit.atom ?? hit.atoms?.[0] ?? m.topo.residueAtoms[hit.residue][0];
    const r = A.residue[seed];
    const whole = molecule || (lastDouble?.model === m && lastDouble.residue === r && lastDouble.level === "residue");
    let ids = m.topo.residueAtoms[r];
    if (whole) {
      const f = A.fragment[seed];
      ids = app.range(m.topo.natoms).filter(a => A.fragment[a] === f);
    }
    lastDouble = { model: m, residue: r, level: whole ? "molecule" : "residue" };
    app.select(m, ids, e.shiftKey ? "add" : "set");
  }
  // ChimeraX's double-click with Ctrl opens the popup
  canvas.addEventListener("dblclick", e => {
    if (state.mouseMode === "chimerax" && (e.ctrlKey || e.metaKey)) {
      if (!app.toolActive?.()) app.popup(e.clientX, e.clientY, app.pick(...local(e)));
    } else {
      residueOrMolecule(e, false);
    }
  });
  canvas.addEventListener("click", e => {
    if (e.detail !== 3 || e.button !== 0) return;
    if (state.mouseMode === "chimerax" && (e.ctrlKey || e.metaKey)) return;
    residueOrMolecule(e, true);
  });

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    app.stopFlight();
    if (state.trackpad && e.ctrlKey) {  // a pinch
      camera.zoom(Math.exp(e.deltaY * 0.01));
    } else if (state.trackpad) {  // a two-finger swipe
      const dx = -e.deltaX, dy = -e.deltaY;
      const [x, y] = local(e);
      if (e.shiftKey) {
        const scale = 2 * camera.distance * Math.tan(camera.fov / 2) / canvas.clientHeight;
        camera.pan(dx * scale, -dy * scale);
      } else if (inPerimeter(x, y)) {
        camera.rotate(0, 0, 1, zTurn(x + dx, y + dy, dx, dy));
      } else if (dx || dy) {
        camera.rotate(dy, dx, 0, Math.hypot(dx, dy) * 0.008);
      }
    } else if (state.mouseMode === "pymol" && !e.ctrlKey && !e.metaKey) {
      const turn = e.deltaY || e.deltaX;  // with Shift, some browsers turn the wheel sideways
      if (e.shiftKey) clipBy(turn * 0.0008, -turn * 0.0008);  // the slab moves
      else { const f = Math.exp(-turn * 0.0015), c = camera.zclip; clipBy(c[0] * f - c[0], c[1] * f - c[1]); }
    } else {
      camera.zoom(Math.exp(e.deltaY * 0.0015));
    }
    app.draw();
  }, { passive: false });

  // Shift over an atom: a bubble saying what it is
  function hover(e, x, y) {
    if (!e.shiftKey) { bubble.hidden = true; return; }
    const hit = app.pick(x, y);
    if (!hit || hit.atom === undefined) { bubble.hidden = true; return; }
    const m = hit.model, a = hit.atom, A = m.topo.atoms;
    const lines = [app.atomLabel(m, a, false), `element ${A.element[a]}   charge ${A.charge[a]}`,
                   ["receptor", "ligand", "solvent"][A.class[a]]];
    bubble.textContent = lines.join("\n");
    const r = stage.getBoundingClientRect();
    Object.assign(bubble.style, { left: `${e.clientX - r.left + 14}px`, top: `${e.clientY - r.top + 10}px` });
    bubble.hidden = false;
  }
  window.addEventListener("keyup", e => { if (e.key === "Shift") bubble.hidden = true; });

  // two-button mouse: Alt pressed and released alone opens the popup
  let altAlone = false, last = [0, 0];
  window.addEventListener("pointermove", e => { last = [e.clientX, e.clientY]; });
  window.addEventListener("keydown", e => { altAlone = e.key === "Alt" && !e.repeat; });
  window.addEventListener("keyup", e => {
    if (e.key === "Alt" && altAlone && twoButton() && !drag) app.popup(...last);
    altAlone = false;
  });
}
