// The mouse in the rendering area (Help | Mouse), and right-drag to zoom:
//
//   left    click: select an atom (or a bond, or a ribbon's residue); Shift toggles;
//           Ctrl (or Cmd) takes the residue; double-click: the molecule (of its class);
//           drag: select a box (Shift adds, Ctrl takes residues); click away: clear
//   middle  drag: rotate; in the perimeter: about z; Shift: pan; Ctrl: zoom;
//           click an atom: the centre of rotation (away from atoms: back to the view's)
//   Alt + middle   the selected atoms: rotate them (perimeter: about z); Shift+Alt: move
//   Alt + left     with two bonded atoms selected: turn about that bond
//   right   the popup menu;  wheel: zoom;  Shift over an atom: what it is
//
// A two-button mouse (Help | Mouse) uses the right button as the middle one and
// opens the popup when Alt is pressed and released alone.

const PERIMETER = 0.07;  // the band at the edges where dragging turns about z

export function setupMouse(app) {
  const { canvas, camera, state } = app;
  const stage = document.getElementById("stage");
  const rubber = document.getElementById("rubber");
  const bubble = document.getElementById("bubble");
  state.twoButton = localStorage.getItem("boonza.twoButton") === "1";

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
    return state.twoButton ? "middle" : "right";
  };

  canvas.addEventListener("contextmenu", e => e.preventDefault());

  canvas.addEventListener("pointerdown", e => {
    // the view takes the keys back (Delete, Ctrl Z ...) from a field or list clicked before
    const focused = document.activeElement;
    if (focused && focused !== document.body && focused.blur) focused.blur();
    app.stopFlight();
    altAlone = false;
    const [x, y] = local(e);
    drag = { role: role(e.button), x0: x, y0: y, x, y, moved: false, shift: e.shiftKey,
             ctrl: e.ctrlKey || e.metaKey, alt: e.altKey, perimeter: inPerimeter(x, y) };
    if (drag.role === "left") app.message("");  // a click in the view puts a message away
    canvas.setPointerCapture(e.pointerId);
    if (drag.role === "left" && drag.alt) startDihedral();
    e.preventDefault();
  });

  // Alt + left: two bonded atoms selected turn the bond
  async function startDihedral() {
    const sel = app.selection();
    const d = drag;
    if (sel.length !== 1 || sel[0][1].length !== 2) {
      app.message("Alt-drag turns a bond: select its two atoms first", true);
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
    if (!drag) { hover(e, x, y); return; }
    const dx = x - drag.x, dy = y - drag.y;
    if (!dx && !dy) return;
    if (!drag.moved && Math.abs(x - drag.x0) + Math.abs(y - drag.y0) < 3) return;
    drag.moved = true;
    drag.x = x; drag.y = y;
    bubble.hidden = true;

    if (drag.role === "left") {
      if (drag.alt) {
        const d = drag.dihedral;
        if (d && typeof d === "object") {
          d.now += dx * 0.6;
          d.now = ((d.now + 540) % 360) - 180;
          app.turnDihedral(d, d.now);
          app.message(`dihedral ${d.now.toFixed(1)}°`);
        }
        return;
      }
      const x0 = Math.min(drag.x0, x), y0 = Math.min(drag.y0, y);
      Object.assign(rubber.style, { left: `${x0}px`, top: `${y0}px`,
                                    width: `${Math.abs(x - drag.x0)}px`, height: `${Math.abs(y - drag.y0)}px` });
      rubber.hidden = false;
      return;
    }
    if (drag.role === "right") {  // right-drag zooms: up out, down in (a right click opens the popup)
      camera.zoom(Math.exp(-dy * 0.01));
      app.draw();
      return;
    }
    if (drag.role !== "middle") return;

    if (drag.alt) {  // the selected atoms move
      if (!drag.moving) drag.moving = app.beginMove() ? "yes" : "no";
      if (drag.moving !== "yes") return;
      if (drag.shift) app.translateSelected(dx, dy);
      else if (drag.perimeter) app.rotateSelected(0, 0, 1, zTurn(x, y, dx, dy));
      else app.rotateSelected(dy, dx, 0, Math.hypot(dx, dy) * 0.01);
      return;
    }
    if (drag.shift) {
      const scale = 2 * camera.distance * Math.tan(camera.fov / 2) / canvas.clientHeight;
      camera.pan(dx * scale, -dy * scale);
    } else if (drag.ctrl) {
      camera.zoom(Math.exp(dy * 0.01));
    } else if (drag.perimeter) {
      camera.rotate(0, 0, 1, zTurn(x, y, dx, dy));
    } else {
      camera.rotate(dy, dx, 0, Math.hypot(dx, dy) * 0.008);
    }
    app.draw();
  });

  // the angle a drag sweeps about the centre of the rendering area
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

    if (d.role === "left" && d.alt) {
      const t = d.dihedral;
      if (t && typeof t === "object" && d.moved) {
        try {
          await app.post("api/set_dihedral", { m: t.model.index, atoms: t.atoms, angle: t.now });
          await app.reloadModel(t.model, false);
        } catch (err) { app.message(err.message, true); }
      }
      return;
    }
    if (d.role === "left" && d.moved) {  // a box
      const found = app.inRect(Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y));
      if (!d.shift && !d.ctrl) app.clearSelection();
      for (const [m, ids] of found) {
        const atoms = d.ctrl ? residuesOf(m, ids) : ids;
        app.select(m, atoms, d.shift && d.ctrl ? "toggle" : "add");
      }
      if (!found.length) { app.rebuild(); app.onSelection(); }
      return;
    }
    if (d.role === "left") { click(x, y, d); return; }
    if (d.role === "middle") {
      if (d.moving === "yes") { await app.endMove(); return; }
      if (!d.moved) {  // the centre of rotation
        const hit = app.pick(x, y);
        if (hit && hit.atom !== undefined) {
          camera.pivot = app.xyz(hit.model, hit.atom);
          app.message(`centre of rotation: ${app.atomLabel(hit.model, hit.atom, false)}`);
        } else {
          camera.pivot = null;
          app.message("centre of rotation: the view's");
        }
      }
      return;
    }
    if (d.role === "right" && !d.moved) app.popup(e.clientX, e.clientY, app.pick(x, y));
  });

  function residuesOf(m, ids) {
    const out = new Set();
    for (const a of ids) for (const b of m.topo.residueAtoms[m.topo.atoms.residue[a]]) out.add(b);
    return [...out];
  }

  function click(x, y, d) {
    const hit = app.pick(x, y);
    if (app.applyTool?.(hit, x, y)) return;  // a builder tool in hand takes the click
    if (!hit) {
      if (!d.shift && !d.ctrl) { app.clearSelection(); app.rebuild(); app.onSelection(); }
      return;
    }
    const m = hit.model;
    let ids = hit.atom !== undefined ? [hit.atom] : hit.atoms || m.topo.residueAtoms[hit.residue];
    if (d.ctrl) ids = residuesOf(m, ids);
    app.select(m, ids, d.shift ? "toggle" : "set");
    if (hit.atom !== undefined) app.describeAtom(m, hit.atom);
    else if (hit.atoms) app.message(`bond ${app.atomLabel(m, hit.atoms[0], false)} – ${app.atomLabel(m, hit.atoms[1], false)}`
      + `  (order ${m.topo.bonds.order[m.topo.bonds.i.findIndex((i, k) => (i === hit.atoms[0] && m.topo.bonds.j[k] === hit.atoms[1]) || (i === hit.atoms[1] && m.topo.bonds.j[k] === hit.atoms[0]))] ?? 1})`);
  }

  // double-click: the atom's molecule, as far as it is of the atom's class
  canvas.addEventListener("dblclick", e => {
    if (app.toolActive?.()) return;
    const [x, y] = local(e);
    const hit = app.pick(x, y);
    if (!hit || hit.atom === undefined) return;
    const m = hit.model, A = m.topo.atoms;
    const f = A.fragment[hit.atom], c = A.class[hit.atom];
    const ids = app.range(m.topo.natoms).filter(a => A.fragment[a] === f && A.class[a] === c);
    app.select(m, ids, e.shiftKey ? "add" : "set");
  });

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    app.stopFlight();
    camera.zoom(Math.exp(e.deltaY * 0.0015));
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
    if (e.key === "Alt" && altAlone && state.twoButton && !drag) app.popup(...last);
    altAlone = false;
  });
}
