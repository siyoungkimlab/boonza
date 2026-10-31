// The window around the view: the top bar (menus and actions), the popup menu, the
// side bar (the models, and the representations of the current one), the status bar
// (the selection's geometry to edit), the view's own buttons and trackball, the command
// line and the panels.  Commands that change what is drawn act on the selected atoms, or
// on all of them if none are.

import { colorAtoms, colorResidues, isColor, named } from "./colors.js";
import { GRIDS } from "./surface.js";

const $ = id => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  for (const k of kids) e.append(k);
  return e;
};
const CARBONS = ["gray", "green", "cyan", "magenta", "yellow", "orange", "salmon", "purple", "white", "slate"];
const SINGLE = ["white", "gray", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "pink", "wheat"];
const ELEMENTS = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"];

export function setupUI(app) {
  const { state } = app;

  // --- menus ------------------------------------------------------------------------
  // an item: {label, run, sub: [...] | () => [...], checked: () => bool, off: () => bool,
  // key}, or "-" for a separator

  let openMenus = [];
  function closeMenus(from = 0) {
    for (const m of openMenus.splice(from)) m.remove();
    for (const t of document.querySelectorAll(".menutitle.open")) if (!from) t.classList.remove("open");
  }
  function menu(items, x, y, depth = 0) {
    closeMenus(depth);
    const div = el("div", { className: "menu" });
    for (const it of typeof items === "function" ? items() : items) {
      if (it === "-") { div.append(el("div", { className: "sep" })); continue; }
      const sub = it.sub;
      const off = it.off && it.off();
      const row = el("div", { className: `item${sub ? " sub" : ""}${off ? " disabled" : ""}${it.checked && it.checked() ? " checked" : ""}` },
                     it.label);
      if (it.key) row.append(el("span", { className: "key" }, it.key));
      row.onmouseenter = () => {
        closeMenus(depth + 1);
        if (sub && !off) {
          const r = row.getBoundingClientRect();
          const m = menu(sub, r.right - 2, r.top - 5, depth + 1);
          const w = m.getBoundingClientRect().width;
          if (r.right - 2 + w > innerWidth) m.style.left = `${Math.max(0, r.left - w + 2)}px`;  // no room: to the left
        }
      };
      row.onclick = e => {
        e.stopPropagation();
        if (off || sub) return;
        closeMenus();
        it.run();
      };
      div.append(row);
    }
    document.body.append(div);
    // keep it on the screen
    const r = div.getBoundingClientRect();
    div.style.left = `${Math.max(0, Math.min(x, innerWidth - r.width - 2))}px`;
    div.style.top = `${Math.max(0, Math.min(y, innerHeight - r.height - 2))}px`;
    openMenus[depth] = div;
    return div;
  }
  window.addEventListener("pointerdown", e => {
    if (!e.target.closest(".menu") && !e.target.closest(".menutitle") && !e.target.closest(".pick")
        && !e.target.closest("#rep-presets")) closeMenus();
  });

  // --- what is drawn: representations ------------------------------------------------

  const STYLE_LABEL = Object.fromEntries(app.REP_STYLES);
  const classIds = (m, k) => app.range(m.topo.natoms).filter(a => m.topo.atoms.class[a] === k);

  // a new representation of these atoms in each model; the last one made is the current one
  function addRep(pairs, { style = "lines", color = "element", sel = null, label = null } = {}) {
    let last = null;
    for (const [m, ids] of pairs) {
      if (!m?.state || !ids.length) continue;
      const rep = app.makeRep(m, { ids, sel, label: label || sel || describeAtoms(m, ids), style, color });
      m.reps.push(rep);
      m.currentRep = rep.id;
      app.aggregate(m);
      last = [m, rep];
    }
    app.rebuild();
    if (last) { state.current = last[0].index; app.onReps(last[0]); }
    return last?.[1] || null;
  }

  // a few words for a set of atoms: a residue, a molecule, so many atoms
  function describeAtoms(m, ids) {
    const A = m.topo.atoms, R = m.topo.residues;
    if (ids.length === m.topo.natoms) return "all";
    const res = [...new Set(ids.map(a => A.residue[a]))];
    if (res.length === 1) return `${R.name[res[0]]}${R.resid[res[0]]}${ids.length < m.topo.residueAtoms[res[0]].length ? ` (${ids.length} atoms)` : ""}`;
    if (res.length <= 4) return res.map(r => `${R.name[r]}${R.resid[r]}`).join(" ");
    return `${ids.length} atoms`;
  }

  // these atoms drawn by no representation (their ribbons, surfaces and atoms alike)
  function hideAtoms(pairs, what = null) {
    for (const [m, ids] of pairs) {
      if (!m?.state) continue;
      const out = new Set(ids);
      for (const rep of m.reps) {
        let any = false;
        for (const a of out) if (rep.mask[a]) { rep.mask[a] = 0; any = true; }
        if (!any) continue;
        rep.version++;
        if (rep.sel && what) rep.sel = `(${rep.sel}) and not (${what})`;
        else rep.sel = null;
        app.colorRep(m, rep);
      }
      m.reps = m.reps.filter(r => r.mask.some(x => x));
      app.aggregate(m);
    }
    app.rebuild();
    app.onReps(app.currentModel());
  }

  // these atoms drawn in this style only: out of the other atom representations, into a new one
  function showAs(pairs, style, what = null) {
    for (const [m, ids] of pairs) {
      if (!m?.state || app.isRibbon({ style }) || style === "surface") continue;
      for (const rep of m.reps) {
        if (app.isRibbon(rep) || rep.style === "surface") continue;
        let any = false;
        for (const a of ids) if (rep.mask[a]) { rep.mask[a] = 0; any = true; }
        if (any) rep.sel = rep.sel && what ? `(${rep.sel}) and not (${what})` : null;
      }
      m.reps = m.reps.filter(r => r.mask.some(x => x));
    }
    return addRep(pairs, { style, sel: what });
  }

  // these atoms in this color wherever they are drawn
  function colorIn(pairs, spec) {
    for (const [m, ids] of pairs) {
      if (!m?.state) continue;
      for (const rep of m.reps) {
        if (rep.style === "surface") continue;
        const mine = ids.filter(a => rep.mask[a]);
        if (!mine.length) continue;
        if (mine.length === app.repIds(rep).length) { rep.color = spec; rep.tinted = false; app.colorRep(m, rep); continue; }
        if (app.isRibbon(rep)) {
          const res = [...new Set(mine.map(a => m.topo.atoms.residue[a]))].filter(r => m.topo.residues.trace[r] >= 0);
          colorResidues(named(spec) ? spec : spec.startsWith("element/") ? spec.split("/")[1] : spec, res, m.topo, m.ss, rep.rgb);
          rep.custom = true;
        } else {
          colorAtoms(spec, mine, m.topo, m.ss, rep.rgb);
          rep.custom = true;
        }
        rep.tinted = false;
      }
    }
    app.rebuild();
    app.onReps(app.currentModel());
  }

  function forTargets(fn) {
    for (const [m, ids] of app.targets()) fn(m, ids);
    app.rebuild();
  }

  function labels(mode) {
    if (mode !== "off") state.labelMode = mode;
    forTargets((m, ids) => { for (const a of ids) m.state.labels[a] = mode === "off" ? 0 : 1; });
  }

  async function serverSelect(m, sel) {
    try { return Array.from(await app.selectIds(m, sel)); } catch (e) { app.message(e.message, true); return []; }
  }
  const pocketSelection = m => m.topo.cg
    ? `(same residue as ((same residue as name BB GC PX) and within 6 of (${m.topo.ligand}))) and not (${m.topo.ligand})`
    : `(same residue as ((protein or nucleic) and within 4.5 of (${m.topo.ligand}))) and not (${m.topo.ligand})`;

  // the ligand and its pocket in view; drawn as the binding site preset if neither is drawn
  async function focusLigand() {
    const m = app.currentModel();
    if (!m?.topo) return;
    const lig = await serverSelect(m, `same residue as (${m.topo.ligand})`);
    if (!lig.length) { app.message("no ligand: right-click it and Set as Ligand", true); return; }
    const pocket = await serverSelect(m, pocketSelection(m));
    if (![...lig, ...pocket].some(a => m.state.show[a])) await preset(m, "site");
    const [c, r] = app.bounds(m, [...lig, ...pocket]);
    app.flyTo(c, Math.max(r, 8));
  }

  // --- selection commands --------------------------------------------------------------

  function selectWhere(test, mode = "set") {
    if (mode === "set") app.clearSelection();
    for (const m of app.shownModels()) {
      const ids = app.range(m.topo.natoms).filter(a => test(m, a));
      if (ids.length) app.select(m, ids, "add");
    }
    app.rebuild(); app.onSelection();
  }
  async function selectServer(make, models = app.shownModels()) {
    app.clearSelection();
    for (const m of models) {
      const ids = await serverSelect(m, make(m));
      if (ids.length) app.select(m, ids, "add");
    }
    app.rebuild(); app.onSelection();
  }
  function extend(how) {
    for (const [m, ids] of app.selection()) {
      const A = m.topo.atoms, more = new Set(ids);
      if (how === "residues") for (const a of ids) for (const b of m.topo.residueAtoms[A.residue[a]]) more.add(b);
      if (how === "chains") {
        const chains = new Set(ids.map(a => m.topo.residues.chain[A.residue[a]]));
        for (let a = 0; a < m.topo.natoms; a++) if (chains.has(m.topo.residues.chain[A.residue[a]])) more.add(a);
      }
      if (how === "molecules") {
        const frags = new Set(ids.map(a => A.fragment[a]));
        for (let a = 0; a < m.topo.natoms; a++) if (frags.has(A.fragment[a])) more.add(a);
      }
      if (how === "hydrogens") for (const a of ids) for (const b of m.topo.bondsOf[a]) if (A.element[b] === "H") more.add(b);
      app.select(m, [...more], "add");
    }
  }
  async function extendNear(cutoff = 4.5) {
    for (const [m, ids] of app.selection()) {
      const more = await serverSelect(m, `same residue as (within ${cutoff} of index ${ids.join(" ")})`);
      app.select(m, more, "add");
    }
  }
  function invert() {
    for (const m of app.shownModels()) for (let a = 0; a < m.topo.natoms; a++) m.state.sel[a] ^= 1;
    app.rebuild(); app.onSelection();
  }

  // which atoms a click can pick: receptor, ligand, solvent, ribbons
  const PICKABLE = [["Protein and other receptor atoms", 0], ["Ligand", 1], ["Solvent", 2], ["Ribbons", 3]];
  const selectItems = () => [
    { label: "All", run: () => selectWhere(() => true), key: "Ctrl A" },
    { label: "None", run: () => { app.clearSelection(); app.rebuild(); app.onSelection(); }, key: "Esc" },
    { label: "Invert", run: invert },
    "-",
    { label: "Ligand", run: () => selectServer(m => `same residue as (${m.topo.ligand})`) },
    { label: "Binding Site", run: () => selectServer(pocketSelection) },
    { label: "Receptor", run: () => selectWhere((m, a) => m.topo.atoms.class[a] === 0) },
    { label: "Solvent", run: () => selectWhere((m, a) => m.topo.atoms.class[a] === 2) },
    { label: "Hydrogens", run: () => selectWhere((m, a) => m.topo.atoms.element[a] === "H") },
    { label: "Element", sub: [
      ...["C", "N", "O", "S", "P"].map(e => ({ label: e, run: () => selectWhere((m, a) => m.topo.atoms.element[a] === e) })),
      { label: "Halogens", run: () => selectWhere((m, a) => ["F", "Cl", "Br", "I"].includes(m.topo.atoms.element[a])) },
    ] },
    "-",
    { label: "Grow", sub: extendItems },
    { label: "Clickable", sub: () => PICKABLE.map(([label, k]) => ({ label, checked: () => state.masks[k],
      run: () => { state.masks[k] = !state.masks[k]; } })) },
    "-",
    { label: "By Expression…", run: () => openCLI("select ") },
  ];
  const extendItems = () => [
    { label: "To Residues", run: () => extend("residues") },
    { label: "To Chains", run: () => extend("chains") },
    { label: "To Molecules", run: () => extend("molecules") },
    { label: "Their Hydrogens", run: () => extend("hydrogens") },
    { label: "Within 4.5 Å", run: () => extendNear() },
  ];

  // --- view items ------------------------------------------------------------------------

  const ATOM_STYLES = app.REP_STYLES.filter(([s]) => !["cartoon", "tube", "surface"].includes(s));
  const showAsItems = () => ATOM_STYLES.map(([s, label]) => ({ label, run: () => showAs(app.targets(), s) }));
  const labelItems = () => [
    ...[["Name", "name"], ["Element", "element"], ["Residue", "residue"], ["Charge", "charge"]]
      .map(([l, mode]) => ({ label: l, run: () => labels(mode) })),
    "-",
    { label: "Off", run: () => labels("off") },
  ];
  function togglePocketNumbers() {
    state.pocketNumbers = !state.pocketNumbers;
    app.message(`pocket numbers ${state.pocketNumbers ? "shown" : "hidden"}`);
    app.rebuild();
  }
  function setHydrogens(mode) {
    state.hydrogens = mode;
    $("f-hydrogens").textContent = `H: ${mode}`;
    app.rebuild();
  }
  function cycleHydrogens() {
    setHydrogens({ polar: "all", all: "none", none: "polar" }[state.hydrogens]);
    app.message(`hydrogens: ${{ all: "all shown", polar: "polar ones shown", none: "hidden" }[state.hydrogens]}`);
  }
  function toggleContacts() {
    state.contacts = !state.contacts;
    app.refreshContacts().then(app.rebuild);
  }
  function toggleCell() {
    state.cell = !state.cell;
    $("f-cell").classList.toggle("on", state.cell);
    app.rebuild();
  }
  const viewItems = () => [
    { label: "Representations", run: () => toggleSide(), checked: () => !$("window").classList.contains("nomodels") },
    "-",
    { label: "Fit All", run: () => app.viewAll(), key: "W" },
    { label: "Center on Selection", run: () => app.centerView() },
    { label: "Focus on Ligand", run: focusLigand },
    { label: "Reset", run: () => app.resetView(), key: "R" },
    "-",
    { label: "Show Selection As", sub: showAsItems },
    { label: "Hide Selection", run: () => hideAtoms(app.targets()) },
    { label: "Add Surface", sub: surfaceItems },
    { label: "Labels", sub: labelItems },
    { label: "Hydrogens", sub: [
      { label: "All", run: () => setHydrogens("all"), checked: () => state.hydrogens === "all" },
      { label: "Polar", run: () => setHydrogens("polar"), checked: () => state.hydrogens === "polar" },
      { label: "None", run: () => setHydrogens("none"), checked: () => state.hydrogens === "none" },
    ] },
    "-",
    { label: "Contacts", run: toggleContacts, checked: () => state.contacts },
    { label: "Pocket Numbers", run: togglePocketNumbers, checked: () => state.pocketNumbers },
    { label: "Unit Cell", run: toggleCell, checked: () => !!state.cell },
    { label: "Secondary Structure of Each Frame", run: () => { state.ssEachFrame = !state.ssEachFrame; app.setFrame(state.frame); },
      checked: () => !!state.ssEachFrame },
    { label: "2D Ligand", run: toggle2D, checked: () => !$("lig2d").hidden },
    { label: "Trackball", run: toggleTrackball, checked: () => !$("trackball").hidden },
    "-",
    { label: "Fog…", run: fogPanel },
    { label: "Clipping…", run: zclipPanel },
    { label: "Projection", sub: [
      { label: "Orthographic", run: () => projection(true), checked: () => app.camera.ortho },
      { label: "Perspective", run: () => projection(false), checked: () => !app.camera.ortho },
    ] },
    { label: "Lighting", sub: [
      { label: "Standard", run: () => lighting(false), checked: () => !app.renderer.publication },
      { label: "Outlines and Occlusion", run: () => lighting(true), checked: () => app.renderer.publication },
    ] },
    { label: "Theme", sub: themeItems },
    { label: "Full Screen", run: () => document.documentElement.requestFullscreen?.(), key: "F12" },
  ];
  // View | Add Surface: a surface representation of the selected atoms, else of each
  // shown model's receptor (else of all but its solvent)
  const SURFACES = [["electrostatic", "Electrostatic"], ["hydrophobic", "Hydrophobic"], ["white", "One Color"]];
  const surfaceItems = () => [
    ...SURFACES.map(([kind, label]) => ({ label, run: () => addSurface(kind) })),
    "-",
    { label: "Resolution", sub: () => GRIDS.map(([h, label]) => ({ label, run: () => surfaceGrid(h),
      checked: () => state.surfaceGrid === h })) },
  ];
  // the surface's grid spacing (A; beads twice it): finer is smoother and slower
  function surfaceGrid(h) {
    state.surfaceGrid = h;
    localStorage.setItem("boonza.surfaceGrid", String(h));
    app.rebuild();
  }
  function surfaceAtoms(m) {
    const A = m.topo.atoms, all = app.range(m.topo.natoms);
    const receptor = all.filter(a => A.class[a] === 0);
    return receptor.length ? receptor : all.filter(a => A.class[a] !== 2);
  }
  function addSurface(kind) {
    const sel = app.selection();
    const pairs = sel.length ? sel : app.shownModels().map(m => [m, surfaceAtoms(m)]);
    const rep = addRep(pairs, { style: "surface", color: kind, label: sel.length ? null : "receptor" });
    if (!rep) return;
    for (const [m] of pairs) if (kind === "electrostatic") forceFieldCharges(m);
    app.message(surfaceNote(kind));
  }
  const surfaceNote = kind => kind === "electrostatic" ? "electrostatic surface: red −10 to blue +10 kcal/(mol·e) (Coulomb, ε = 4r, 1.4 Å off the surface)"
    : kind === "hydrophobic" ? "hydrophobic surface: dark cyan hydrophilic, goldenrod hydrophobic" : "molecular surface";
  // an electrostatic surface's charges: the force field's (Amber ff19SB, hydrogens
  // added on a copy), once per topology, for a structure that has none of its own
  async function forceFieldCharges(m) {
    const t = m.topo;
    if (app.STATIC || t.cg || t.q_source !== "template" || t.qAssigned) return;
    t.qAssigned = true;
    try {
      const out = await app.busy("assigning Amber ff19SB charges for the surface",
                                 () => app.post("api/surface_charges", { m: m.index }));
      if (m.topo !== t) return;  // the model changed meanwhile
      t.atoms.q = out.q;
      for (const rep of m.reps) if (rep.style === "surface") rep.version++;
      app.rebuild();
      app.message(`electrostatic surface: Amber ff19SB charges on ${out.assigned} of ${out.natoms} atoms`
        + `${out.assigned < out.natoms ? " (the rest by residue)" : ""}; red −10 to blue +10 kcal/(mol·e)`);
    } catch (e) {
      app.message(`${e.message}\nthe surface's charges are by residue`, true);
    }
  }

  const themeItems = () => [
    { label: "Dark", run: () => theme("dark"), checked: () => document.body.classList.contains("dark") },
    { label: "Light", run: () => theme("light"), checked: () => document.body.classList.contains("light") },
  ];
  function lighting(on) {
    app.renderer.publication = on;
    localStorage.setItem("boonza.lighting", on ? "publication" : "standard");
    app.draw();
  }
  app.renderer.publication = localStorage.getItem("boonza.lighting") === "publication";
  function projection(ortho) {
    app.camera.ortho = ortho;
    localStorage.setItem("boonza.projection", ortho ? "ortho" : "persp");
    app.draw();
  }
  app.camera.ortho = localStorage.getItem("boonza.projection") !== "persp";

  // a session or a setting from before the themes were renamed
  const THEMES = { blanc: "light", noir: "dark", light: "light", dark: "dark" };
  app.setTheme = name => theme(name);
  function theme(name) {
    name = THEMES[name] || "dark";
    document.body.className = name;
    app.renderer.background = name === "dark" ? [0, 0, 0] : [1, 1, 1];
    localStorage.setItem("boonza.theme", name);
    app.draw();
  }

  // Render: how the view looks on screen (lighting, projection, background), and a
  // picture of it, drawn larger in detail and finished (outlines, occlusion), with the
  // labels: a PNG that carries its print resolution
  const PICTURE_SIZES = [["view3", "3 × the view"], ["view2", "2 × the view"], ["view4", "4 × the view"],
    ["col1", "1 journal column (3.5 in)"], ["col2", "2 journal columns (7 in)"],
    ["slide", "slide (1920 × 1080)"], ["uhd", "3840 × 2160"], ["custom", "custom"]];
  function renderPanel() {
    panel("picture", "Render", body => {
      const w0 = app.canvas.clientWidth, h0 = app.canvas.clientHeight;
      // on screen
      const pub = el("input", { type: "checkbox", checked: app.renderer.publication });
      pub.onchange = () => lighting(pub.checked);
      const ortho = el("input", { type: "checkbox", checked: app.camera.ortho });
      ortho.onchange = () => projection(ortho.checked);
      const dark = el("input", { type: "checkbox", checked: document.body.classList.contains("dark") });
      dark.onchange = () => theme(dark.checked ? "dark" : "light");
      body.append(el("h4", {}, "On screen"),
                  row(el("label", {}, pub, " outlines and ambient occlusion")),
                  row(el("label", {}, ortho, " orthographic (no perspective)")),
                  row(el("label", {}, dark, " dark background")),
                  el("h4", {}, "Picture"));
      const saved = (k, d) => { try { return localStorage.getItem(`boonza.picture.${k}`) ?? d; } catch (e) { return d; } };
      const keep = (k, v) => { try { localStorage.setItem(`boonza.picture.${k}`, v); } catch (e) { /* not kept */ } };
      const select = (pairs, value) => {
        const s = el("select");
        for (const [v, l] of pairs) s.add(new Option(l, v, false, v === value));
        return s;
      };
      const size = select(PICTURE_SIZES, saved("size", "view3"));
      const width = el("input", { type: "number", min: 16, max: 16000, style: "width:70px" });
      const height = el("input", { type: "number", min: 16, max: 16000, style: "width:70px" });
      const dpi = el("input", { type: "number", min: 72, max: 2400, value: saved("dpi", 300), style: "width:56px" });
      const fill = () => {
        const d = +dpi.value || 300, v = size.value;
        const set = (w, h) => { width.value = Math.round(w); height.value = Math.round(h); };
        if (v.startsWith("view")) set(+v[4] * w0, +v[4] * h0);
        else if (v === "col1" || v === "col2") { const w = (v === "col1" ? 3.5 : 7) * d; set(w, w * h0 / w0); }
        else if (v === "slide") set(1920, 1080);
        else if (v === "uhd") set(3840, 2160);
        info();
      };
      const info = () => {
        const d = +dpi.value || 300;
        note.textContent = `${width.value} × ${height.value} px = ${(width.value / d).toFixed(2)} × ${(height.value / d).toFixed(2)} in at ${d} dpi`
          + (Math.abs(width.value / height.value - w0 / h0) > 0.01 ? " · a different shape from the view: more or less is seen at the sides" : "");
      };
      const note = el("p", { className: "help" });
      size.onchange = () => { keep("size", size.value); fill(); };
      dpi.oninput = () => { keep("dpi", dpi.value); if (size.value.startsWith("col")) fill(); else info(); };
      width.oninput = height.oninput = () => { size.value = "custom"; info(); };
      const quality = select([["2", "good (2 × 2 samples)"], ["3", "fine (3 × 3)"], ["4", "finest (4 × 4)"]], saved("quality", "3"));
      quality.onchange = () => keep("quality", quality.value);
      const back = select([["white", "white"], ["black", "black"], ["none", "transparent"], ["view", "as the view"]], saved("back", "none"));
      back.onchange = () => keep("back", back.value);
      const check = (key, label, on) => {
        const box = el("input", { type: "checkbox", checked: saved(key, on ? "1" : "0") === "1" });
        box.onchange = () => keep(key, box.checked ? "1" : "0");
        return [box, el("label", {}, box, ` ${label}`)];
      };
      const [outline, outlineLabel] = check("outline", "outlines", true);
      const [ao, aoLabel] = check("ao", "ambient occlusion (shade in crevices)", true);
      const [fog, fogLabel] = check("fog", "fog (depth cue)", false);
      const [labels, labelsLabel] = check("labels", "labels and measurements", true);
      const font = el("input", { type: "number", min: 4, max: 36, value: saved("font", 8), style: "width:46px" });
      font.onchange = () => keep("font", font.value);
      const render = async () => {
        const w = Math.round(+width.value), h = Math.round(+height.value);
        if (!(w >= 16 && h >= 16 && w <= 16000 && h <= 16000)) throw new Error("a picture is 16 to 16000 pixels on a side");
        const bg = { white: [1, 1, 1], black: [0, 0, 0] }[back.value] || null;
        app.message(`drawing a ${w} × ${h} picture…`);
        await new Promise(ok => setTimeout(ok, 30));  // let the message show
        const pic = app.withPictureScene(r => r.picture(app.camera, w, h, { outline: outline.checked, ao: ao.checked,
          background: bg, transparent: back.value === "none", fog: fog.checked, quality: +quality.value }));
        const dark = back.value === "black" || (back.value === "view" && document.body.classList.contains("dark"));
        if (labels.checked) drawLabels(pic, (+font.value || 8) / 72 * (+dpi.value || 300), dark);
        const blob = await new Promise(ok => pic.toBlob(ok, "image/png"));
        return [withDpi(new Uint8Array(await blob.arrayBuffer()), +dpi.value || 300), pic];
      };
      const go = button("Save PNG", async () => {
        try {
          const [png, pic] = await render();
          const name = (state.title || app.currentModel()?.name || "boonza").replace(/[\s:/·+]+/g, "_").replace(/_+$/, "");
          el("a", { href: URL.createObjectURL(new Blob([png], { type: "image/png" })), download: `${name}-${state.frame + 1}.png` }).click();
          app.message(`saved a ${pic.width} × ${pic.height} picture (${dpi.value} dpi) to your downloads`);
        } catch (e) { app.message(e.message, true); }
        app.draw();
      }, "blue");
      const copy = button("Copy", async () => {
        try {
          // the clipboard is asked for at once, the picture still to come: Safari allows it only
          // while the click is fresh, and drawing the picture takes longer than that
          const drawn = render();
          const png = drawn.then(([png]) => new Blob([png], { type: "image/png" }));
          await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
          const [, pic] = await drawn;
          app.message(`copied a ${pic.width} × ${pic.height} picture`);
        } catch (e) { app.message(`could not copy the picture: ${e.message}`, true); }
        app.draw();
      });
      body.append(row(el("span", { className: "what" }, "size"), size));
      body.append(row(el("span", { className: "what" }, "pixels"), width, el("span", {}, "×"), height,
                      el("span", { className: "what", style: "width:auto" }, "dpi"), dpi));
      body.append(note);
      body.append(row(el("span", { className: "what" }, "quality"), quality));
      body.append(row(el("span", { className: "what" }, "background"), back));
      body.append(row(outlineLabel), row(aoLabel), row(fogLabel),
                  row(labelsLabel, el("span", { className: "what", style: "width:auto;margin-left:6px" }, "size (pt)"), font));
      body.append(row(go, copy));
      body.append(el("p", { className: "help" }, "Drawn in tiles at several samples per pixel, so any size is sharp: "
        + "rounder sticks and ribbons than on screen, lines drawn as thin sticks, the selection's pink left out."));
      fill();
    });
  }

  // the labels (atom labels, meters, pocket numbers) onto a picture, `px` pixels high
  function drawLabels(pic, px, dark) {
    const g = pic.getContext("2d");
    g.textAlign = "center"; g.textBaseline = "middle"; g.lineJoin = "round";
    for (const div of $("labels").children) {
      const v = app.camera.toView(JSON.parse(div.dataset.at));
      if (v[2] >= -0.5) continue;
      const [x, y] = app.camera.screen(v, pic.width, pic.height);
      const pocket = div.classList.contains("pocket"), meter = div.classList.contains("meter");
      const size = px * (pocket ? 1.3 : 1);
      g.font = `${pocket || meter ? "bold " : ""}${size}px Arial, Helvetica, sans-serif`;
      g.lineWidth = Math.max(1, size * 0.22);
      g.strokeStyle = dark ? "#000" : "#fff";
      g.fillStyle = meter ? (dark ? "#ffd34d" : "#a67c00") : dark ? "#fff" : "#000";
      g.strokeText(div.textContent, x, y);
      g.fillText(div.textContent, x, y);
    }
  }

  // a PNG with its resolution (a pHYs chunk after IHDR), so it places at its size in print
  const CRC = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; table[n] = c >>> 0; }
    return bytes => { let c = 0xffffffff; for (const b of bytes) c = table[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  })();
  function withDpi(png, dpi) {
    const chunk = new Uint8Array(21), view = new DataView(chunk.buffer), ppm = Math.round(dpi / 0.0254);
    view.setUint32(0, 9);
    chunk.set([0x70, 0x48, 0x59, 0x73], 4);  // "pHYs"
    view.setUint32(8, ppm); view.setUint32(12, ppm); chunk[16] = 1;  // per meter
    view.setUint32(17, CRC(chunk.subarray(4, 17)));
    const out = new Uint8Array(png.length + 21);
    out.set(png.subarray(0, 33)); out.set(chunk, 33); out.set(png.subarray(33), 54);
    return out;
  }

  // --- editing ----------------------------------------------------------------------

  // the one model the selected atoms are in
  function editTarget(need = 1) {
    const sel = app.selection();
    if (sel.length !== 1 || sel[0][1].length < need) {
      app.message(need === 1 ? "select atoms (in one model) first" : `select ${need} atoms first`, true);
      return null;
    }
    return sel[0];
  }
  // one edit at a time: one sent while another is still on the server would name
  // atoms by indices that edit is about to change
  let editing = false;
  async function serverEdit(m, url, body, topology = true) {
    if (editing) { app.message("an edit is still running: wait for it to finish", true); return null; }
    editing = true;
    try {
      const out = await app.post(url, { m: m.index, ...body });
      await app.reloadModel(m, out.topology ?? topology, out.was);
      return out;
    } catch (e) { app.message(e.message, true); return null; }
    finally { editing = false; }
  }
  const autoH = () => localStorage.getItem("boonza.autoH") !== "0";
  // the model a command acts on: the selection's, else the current one
  function actingModel() {
    const m = app.selection()[0]?.[0] || app.currentModel();
    if (!m || m.closed || !m.topo) { app.message("no model: build one (Builder) or open one (File | Open)", true); return null; }
    return m;
  }
  async function edit(op, value) {
    const t = editTarget(op === "bond" ? 2 : 1);
    if (!t) return;
    const [m, atoms] = t;
    if (op === "bond" && atoms.length !== 2) { app.message("select the bond's two atoms", true); return; }
    await serverEdit(m, "api/edit", { op, atoms, value, hydrogens: op === "name" ? false : autoH() });
  }
  // Undo and Redo step back through one history of both kinds of change: how the
  // models look (their representations, labels and colors: kept here, as snapshots)
  // and edits to a model (atoms, bonds, positions: undone by the server)
  const REP_KEYS = ["id", "sel", "label", "style", "color", "on", "tinted", "material", "opacity", "custom", "version", "fresh"];
  const looks = new WeakMap();  // model -> { state, snap }: how it looked after the last step
  const steps = [], undone = [];
  let replaying = false;
  const snapshot = m => ({ reps: m.reps.map(r => ({ ...r, mask: r.mask.slice(), rgb: r.rgb?.slice() ?? null })),
                           labels: m.state.labels.slice(), color: m.color, tintAll: m.tintAll });
  const same = (a, b) => {
    if (a === b) return true;
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };
  const sameLook = (m, snap) => m.color === snap.color && m.tintAll === snap.tintAll
    && same(m.state.labels, snap.labels) && m.reps.length === snap.reps.length
    && m.reps.every((r, k) => { const o = snap.reps[k]; return REP_KEYS.every(key => r[key] === o[key]) && same(r.mask, o.mask) && same(r.rgb, o.rgb); });
  const remember = step => {
    steps.push(step);
    if (steps.length > 50) steps.shift();
    undone.length = 0;
  };
  app.onRebuild = models => {
    if (replaying) return;
    const changed = [];
    for (const m of models) {
      if (!m.state) continue;
      const was = looks.get(m);
      if (was?.state === m.state && sameLook(m, was.snap)) continue;
      // a model loaded again after an edit starts afresh: the edit is its step
      if (was?.state === m.state) changed.push([m, was.snap]);
      looks.set(m, { state: m.state, snap: snapshot(m) });
    }
    if (changed.length) remember({ look: changed });
  };
  const reloadModel = app.reloadModel;
  app.reloadModel = async (m, topology, was = null) => {
    if (!replaying) remember({ edit: m, before: m.state ? snapshot(m) : null });
    return reloadModel(m, topology, was);
  };
  // a model just loaded again takes a look its atoms fit (what an edit undone or redone had)
  function lookAgain(m, snap) {
    if (snap && m.state && snap.labels.length === m.state.labels.length) {
      m.reps = snap.reps.map(r => ({ ...r, mask: r.mask.slice(), rgb: r.rgb?.slice() ?? null }));
      m.state.labels.set(snap.labels);
      m.color = snap.color; m.tintAll = snap.tintAll;
      app.repsChanged(m);
    }
    if (m.state) looks.set(m, { state: m.state, snap: snapshot(m) });
  }
  function putLook(pairs) {  // the snapshots back; how the models look now, to go the other way
    const now = [];
    for (const [m, snap] of pairs) {
      const was = looks.get(m);
      if (m.closed || was?.state !== m.state || snap.labels.length !== m.state.labels.length) continue;
      now.push([m, was.snap]);
      lookAgain(m, snap);
    }
    return now;
  }
  async function undo(redo = false) {
    const from = redo ? undone : steps, to = redo ? steps : undone;
    const step = from.pop();
    if (!step) {
      if (redo) { app.message("nothing to redo"); return; }
      // nothing done here: what the server keeps (an edit from before, say)
      const m = actingModel();
      if (!m) return;
      replaying = true;
      try { if (await serverEdit(m, "api/undo", {})) { undone.push({ edit: m }); app.message("undone"); } } finally { replaying = false; }
      return;
    }
    replaying = true;
    try {
      if (step.look) {
        to.push({ look: putLook(step.look) });
        app.onModels?.();
      } else {
        const m = step.edit, now = m.state ? snapshot(m) : null;
        const out = await serverEdit(m, redo ? "api/redo" : "api/undo", {});
        if (!out) return;
        lookAgain(m, redo ? step.after : step.before);
        to.push({ ...step, [redo ? "before" : "after"]: now });
      }
      app.message(redo ? "redone" : "undone");
    } finally { replaying = false; }
  }
  async function deleteSelected() {
    const sel = app.selection();
    if (!sel.length) { app.message("nothing selected to delete: click atoms, drag a box, or use Select", true); return; }
    let gone = 0;
    for (const [m, ids] of sel) {
      if (ids.length === m.topo.natoms) { await closeModel(m); gone += ids.length; continue; }
      const out = await serverEdit(m, "api/edit", { op: "delete", atoms: ids, hydrogens: autoH() });
      if (!out) return;  // its error is showing
      gone += ids.length;
    }
    app.message(`deleted ${gone} atom${gone === 1 ? "" : "s"} (and their hydrogens) · Edit | Undo puts them back`);
  }
  async function hydrogens(what) {
    const m = actingModel();
    if (!m) return;
    const sel = app.selected(m);
    const A = m.topo.atoms;
    const pool = sel.length ? sel : app.range(m.topo.natoms);
    if (what === "delete") {
      const hs = pool.filter(a => A.element[a] === "H");
      if (hs.length) await serverEdit(m, "api/edit", { op: "delete", atoms: hs, hydrogens: false });
    } else if (!sel.length && m.topo.residues.kind.some(k => k === 1)) {
      // a protein: every hydrogen as the force field names it (templates), then ligands'
      const out = await busyEdit(m, "adding hydrogens from the force field's templates", { op: "protonate" });
      if (!out) return;
      const notes = [];
      if (out.rebuilt.length) notes.push(`side chains built for ${out.rebuilt.join(", ")}`);
      if (out.alternates) notes.push(`${out.alternates} alternate-location atoms dropped`);
      if (out.incomplete.length) notes.push(`incomplete, left as they are: ${out.incomplete.join(", ")}`);
      if (out.skipped.length) notes.push(`${out.skipped.join(", ")} by valence (${out.ligand_hydrogens} H)`);
      app.message(`added ${out.hydrogens} hydrogens with ff19SB's names to ${out.residues} residues `
        + `(charged termini, Lys/Arg/Asp/Glu charged, His as HIE)${notes.length ? "; " + notes.join("; ") : ""}`,
        out.incomplete.length > 0);
    } else {
      const heavy = pool.filter(a => A.element[a] !== "H");
      if (heavy.length) await serverEdit(m, "api/edit", { op: "hydrogens", atoms: heavy });
    }
  }

  // restraints (Edit | Restraints): 0 free, 1 restrained, 2 frozen
  function constrain(kind) {
    const sel = app.selection();
    if (!sel.length && kind !== "freeall") { app.message("select atoms first", true); return; }
    if (kind === "freeall") for (const m of app.shownModels()) m.state.constraint.fill(0);
    for (const [m, ids] of sel) for (const a of ids) {
      if (kind === "unfix") { if (m.state.constraint[a] === 2) m.state.constraint[a] = 0; }
      else m.state.constraint[a] = { tether: 1, fix: 2, free: 0 }[kind];
    }
    const counts = constraintCounts(app.currentModel());
    app.message(`frozen ${counts.fixed}, restrained ${counts.tethered}`);
  }
  function constraintCounts(m) {
    const c = m?.state?.constraint || [];
    let fixed = 0, tethered = 0;
    for (const x of c) { if (x === 2) fixed++; else if (x === 1) tethered++; }
    return { fixed, tethered };
  }

  // what moves in each way to minimize; the rest is fixed
  const MIN_MODES = [["both", "Pocket + ligand", "the ligand and its pocket free"],
                     ["ligand", "Ligand", "the ligand free"],
                     ["pocket", "Pocket", "the pocket free, the ligand fixed"]];
  async function minimize({ mode = null, restraint = null } = {}) {
    const m = actingModel();
    if (!m) return;
    if (!m.topo?.editable) { app.message("this model cannot be minimized", true); return; }
    mode = mode || localStorage.getItem("boonza.minMode") || "both";
    const c = m.state.constraint, body = { m: m.index, path: true, rem: localStorage.getItem("boonza.rem") === "1" };
    if (c.some(x => x)) {  // Edit | Restraints: those, not a mode
      body.restraint = restraint ?? +(localStorage.getItem("boonza.restraint") || 10);
      body.fixed = app.range(c.length).filter(a => c[a] === 2);
      body.tethered = app.range(c.length).filter(a => c[a] === 1);
    } else Object.assign(body, { mode, restraint: restraint ?? +(localStorage.getItem("boonza.restraint") || 10) });
    try {
      const out = await app.busy("minimizing (the first time sets up the force field)", () => app.post("api/minimize", body));
      // a model without hydrogens got them first (the receptor, or this one): its atoms changed
      const added = out.protonated && state.models[out.protonated.m];
      if (added && added !== m) await app.reloadModel(added, true);
      await app.reloadModel(m, added === m);
      const how = out.mobile === "constraints" ? `${out.fixed_atoms} frozen, ${out.held_atoms} restrained`
        : `${MIN_MODES.find(([k]) => k === out.mobile)?.[2] ?? out.mobile}, the rest's heavy atoms held `
          + `(${body.restraint} kcal/mol/Å², moved at most ${out.held_max} Å)`
          + (out.receptor ? ` in ${out.receptor} (whose positions stay: every design shares it)` : "");
      const first = added ? `${out.protonated.hydrogens} hydrogens added to ${added.name} first (Edit | Undo takes them back)`
        + (out.protonated.incomplete.length ? `, incomplete residues left as they are: ${out.protonated.incomplete.join(", ")}` : "") + ". " : "";
      const rem = out.rem_energy_before !== undefined
        ? ` (after REM: soft E ${out.rem_energy_before} → ${out.rem_energy_after})` : "";
      app.message(`${first}Minimized: E ${out.energy_before} → ${out.energy_after} kJ/mol${rem}; ${how}`
        + (out.mobile_rmsd ? `; moved atoms RMSD ${out.mobile_rmsd} Å (largest ${out.mobile_max} Å)` : ""));
      if (out.path) showPath(out.path, m);
    } catch (e) { app.message(e.message, true); }
  }

  // --- a minimization played back: the atoms as they moved, the energy by iteration ----------
  let playback = null;
  function stopPath() {
    if (!playback) return;
    clearInterval(playback.timer);
    playback.restore();
    playback = null;
  }
  function showPath(path, m) {
    stopPath();
    const parts = path.models.map(x => ({ ...x, model: state.models[x.m] })).filter(x => x.model?.pos);
    if (!parts.length || path.iterations.length < 2) return;
    // where each model is now (after the minimization), and what playback last drew
    const base = parts.map(x => x.model.pos), drawn = parts.map(x => x.model.pos);
    const last = path.iterations.length - 1;
    const E = path.energy, eMin = Math.min(...E.map(([, e]) => e)), eMax = Math.max(...E.map(([, e]) => e));
    const energyAt = it => { let e = E[0][1]; for (const [k, v] of E) { if (k > it) break; e = v; } return e; };
    const pb = { frame: last, timer: null };
    pb.restore = () => {  // back to where they ended, unless something else moved them since
      parts.forEach((x, k) => { if (x.model.pos === drawn[k]) x.model.pos = base[k]; });
      app.rebuild();
    };
    playback = pb;
    // the energy on a log scale above its lowest, so the long tail shows past the first drop
    const W = 340, H = 170, L = 50, R = 8, T = 8, B = 22;
    const itMax = path.iterations[last] || 1;
    const x = it => L + it / itMax * (W - L - R);
    const y = e => T + (1 - Math.log10(e - eMin + 1) / Math.log10(eMax - eMin + 1 || 10)) * (H - T - B);
    const curve = E.map(([k, e], i) => `${i ? "L" : "M"}${x(k).toFixed(1)},${y(e).toFixed(1)}`).join("");
    const div = el("div", { className: "plot" });
    div.innerHTML = `<svg viewBox="0 0 ${W} ${H}"><text x="4" y="${T + 8}">${Math.round(eMax)}</text>`
      + `<text x="4" y="${H - B}">${Math.round(eMin)}</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${itMax}</text>`
      + `<text x="${L}" y="${H - 8}">0</text><text x="${(L + W) / 2}" y="${H - 8}" text-anchor="middle">iteration</text>`
      + (path.stages || []).slice(1).map(([name, k]) => `<line x1="${x(k)}" x2="${x(k)}" y1="${T}" y2="${H - B}" `
        + `stroke="rgba(128,128,128,0.6)" stroke-dasharray="3,3"/><text x="${x(k) + 3}" y="${T + 18}">${name}</text>`).join("")
      + ((path.stages || []).length > 1 ? `<text x="${L + 3}" y="${T + 18}">${path.stages[0][0]}</text>` : "")
      + `<path d="${curve}" fill="none" stroke="#f08c00" stroke-width="1.8"/>`
      + `<line class="now" y1="${T}" y2="${H - B}" stroke="#2e7d32" stroke-width="1.5"/></svg>`;
    const svg = div.querySelector("svg"), now = svg.querySelector(".now");
    const slider = el("input", { type: "range", min: 0, max: last, value: last, style: "width:240px" });
    const label = el("span", {});
    const play = button("▶ Play", () => {
      if (pb.timer) { clearInterval(pb.timer); pb.timer = null; play.textContent = "▶ Play"; return; }
      if (pb.frame >= last) show(0);
      play.textContent = "❚❚ Pause";
      pb.timer = setInterval(() => {
        if (pb.frame >= last) { clearInterval(pb.timer); pb.timer = null; play.textContent = "▶ Play"; return; }
        show(pb.frame + 1);
      }, 80);
    });
    function show(f) {
      pb.frame = f;
      parts.forEach((x, k) => {
        const out = base[k].slice(), xyz = x.frames[f];
        x.atoms.forEach((a, i) => { out[3 * a] = xyz[3 * i]; out[3 * a + 1] = xyz[3 * i + 1]; out[3 * a + 2] = xyz[3 * i + 2]; });
        x.model.pos = drawn[k] = out;
      });
      app.rebuild();
      const it = path.iterations[f];
      now.setAttribute("x1", x(it)); now.setAttribute("x2", x(it));
      slider.value = f;
      label.textContent = `iteration ${it} · E ${energyAt(it).toFixed(1)} kJ/mol`;
    }
    slider.oninput = () => show(+slider.value);
    svg.onclick = e => {  // the frame nearest the iteration clicked
      const r = svg.getBoundingClientRect(), it = ((e.clientX - r.left) / r.width * W - L) / (W - L - R) * itMax;
      let f = 0;
      path.iterations.forEach((k, i) => { if (Math.abs(k - it) < Math.abs(path.iterations[f] - it)) f = i; });
      show(f);
    };
    const pn = panel("minpath", `Minimization of ${m.name}`, body => {
      body.append(div, row(play, slider), row(label));
      body.append(el("p", { className: "help" }, "Potential energy (kJ/mol, log scale above the lowest) by iteration "
        + "of the minimizer; the atoms as they were at that point. Click the plot or drag the slider; "
        + "closing this puts them where the minimization ended."
        + ((path.stages || []).length > 1 ? " With REM the first stage's energy is the soft one, not the force field's." : "")));
    });
    pn.querySelector(".title button").addEventListener("click", stopPath);
    show(last);
  }

  function measureMeter() {
    const t = editTarget(2);
    if (!t) return;
    const [m, atoms] = t;
    if (atoms.length > 4) { app.message("a measurement is 2, 3 or 4 atoms", true); return; }
    state.meters.push({ model: m, atoms: orderChain(m, atoms) });
    app.rebuild();
  }

  // selected atoms in the order of the bonds between them (selection order is lost)
  function orderChain(m, atoms) {
    if (atoms.length < 3) return atoms;
    const set = new Set(atoms), nb = a => m.topo.bondsOf[a].filter(b => set.has(b));
    const end = atoms.find(a => nb(a).length === 1);
    if (end === undefined) return atoms;
    const out = [end];
    while (out.length < atoms.length) {
      const next = nb(out[out.length - 1]).find(b => !out.includes(b));
      if (next === undefined) return atoms;
      out.push(next);
    }
    return out;
  }

  // --- menu bar ------------------------------------------------------------------------

  const menus = {
    File: () => [
      { label: "Open…", run: openPanel, key: "Ctrl O" },
      { label: "Open from This Computer…", run: browse },
      { label: "Fetch from PDB…", run: fetchPanel },
      "-",
      { label: "Save Model…", run: savePanel, key: "Ctrl S" },
      { label: "Download", sub: () => DOWNLOADS.map(([fmt, label]) => ({ label, run: () => download(fmt) })) },
      { label: "Save Session…", run: saveSessionPanel },
      "-",
      { label: "Render Picture…", run: renderPanel },
      "-",
      { label: "Close Model", run: () => closeModel() },
    ],
    Edit: () => [
      { label: "Undo", run: () => undo(), key: "Ctrl Z" },
      { label: "Redo", run: () => undo(true), key: "Ctrl Y" },
      "-",
      { label: "Delete", run: deleteSelected, key: "Del" },
      { label: "Copy SMILES", run: copySmiles, key: "Ctrl C" },
      { label: "Rename Residue or Chain…", run: relabelPanel },
      "-",
      { label: "Hydrogens", sub: [
        { label: "Add Hydrogens", run: () => hydrogens("add") },
        { label: "Delete Hydrogens", run: () => hydrogens("delete") },
      ] },
      { label: "Restraints", sub: constrainItems },
      { label: "Set Selection as Ligand", run: chooseLigand },
    ],
    Select: selectItems,
    View: viewItems,
    Build: () => [
      { label: "Molecule Builder", run: builderPanel, checked: () => !$("ldock").hidden && dockTab === "builder" },
      { label: "Protein Builder", run: () => openDock("protein"), checked: () => !$("ldock").hidden && dockTab === "protein" },
      { label: "Clean Up Geometry", run: cleanUp },
      "-",
      { label: "Superpose Models…", run: () => alignPanel(), off: () => app.live().length < 2 },
    ],
    Analyze: () => [
      { label: "Assign Force Field", run: assignForceField },
      { label: "Minimize…", run: minimizePanel },
      { label: "Minimize Ligand Only", run: () => minimize({ mode: "ligand" }) },
      { label: "Torsion Profile…", run: () => profilePanel() },
      { label: "Ramachandran Plot…", run: () => ramaPanel() },
      "-",
      { label: "Contacts", run: toggleContacts, checked: () => state.contacts },
      { label: "Hydrogen Bonds…", run: hbondPanel, checked: () => !!state.hbonds },
      { label: "Measure", sub: measureItems },
      "-",
      { label: "Trajectory Player", run: () => { $("player").hidden = app.nframes() < 2; },
        off: () => app.nframes() < 2 },
    ],
    Window: () => [
      { label: "Models and Representations", run: toggleSide, checked: () => !$("window").classList.contains("nomodels") },
      { label: "Sequence", run: () => toggleSequence(), checked: () => !$("seqbar").hidden },
      { label: "Builder", run: builderPanel, checked: () => !$("ldock").hidden },
      { label: "Command Line", run: () => openCLI(), key: "Space" },
      { label: "Python Console", run: togglePython, key: "`", off: () => !app.PYTHON },
    ],
    Help: () => [
      { label: "Mouse…", run: mousePanel },
      { label: "Keys…", run: () => app.message("Space: command line · Ctrl Z / Ctrl Y: undo, redo · Del: delete · ←/→: frames · ↑/↓: next model · R: reset the view · W: fit all · F12: full screen · Esc: close menus, put a message away, else select none") },
    ],
  };
  // restraints for Minimize: 0 free, 1 restrained (a spring to where it is), 2 frozen
  const constrainItems = () => [
    { label: "Restrain Selection", run: () => constrain("tether") },
    { label: "Freeze Selection", run: () => constrain("fix") },
    { label: "Unfreeze Selection", run: () => constrain("unfix") },
    { label: "Release Selection", run: () => constrain("free") },
    "-",
    { label: "Release All", run: () => constrain("freeall") },
  ];
  const measureItems = () => [
    { label: "Measure Selection (2-4 atoms)", run: measureMeter },
    { label: "Clear Measurements", run: () => { state.meters = []; app.rebuild(); } },
  ];

  for (const [name, items] of Object.entries(menus)) {
    const t = el("div", { className: "menutitle" }, name);
    t.onclick = () => {
      const open = t.classList.contains("open");
      closeMenus();
      if (open) return;
      t.classList.add("open");
      const r = t.getBoundingClientRect();
      menu(items, r.left, r.bottom + 4);
    };
    t.onmouseenter = () => { if (document.querySelector(".menutitle.open") && !t.classList.contains("open")) t.click(); };
    $("menus").append(t);
  }

  // the models open, for the popup
  function modelItems() {
    return [
      ...app.live().map(m => ({ label: m.name, checked: () => m.shown,
                                run: () => { app.state.current = m.index; app.show([m], !m.shown); } })),
      "-",
      { label: "Models and Representations", run: toggleSide },
    ];
  }

  async function chooseLigand() {
    const t = editTarget();
    if (!t) return;
    const [m, atoms] = t;
    try {
      const out = await app.post("api/set_ligand", { m: m.index, atoms });
      m.topo.ligand = out.ligand;
      app.message("ligand chosen: Focus on Ligand, Binding Site, contacts and Minimize use it");
      await app.refreshContacts();
      app.rebuild();
    } catch (e) { app.message(e.message, true); }
  }

  // --- popup menu (right button) -------------------------------------------------------------

  // what an atom is, with its force field: index, charge, sigma, epsilon, type
  app.describeAtom = async (m, a) => {
    app.message(app.atomLabel(m, a));
    try {
      const info = await app.getJSON(`api/atom?m=${m.index}&a=${a}`);
      const ff = info.charge === undefined
        ? "  · no force field yet: Analyze | Assign Force Field"
        : `  · q ${info.charge.toFixed(4)} e · σ ${info.sigma?.toFixed(3) ?? "–"} Å · ε ${info.epsilon?.toFixed(4) ?? "–"} kcal/mol`
          + `${info.type ? ` · type ${info.type}` : ""}${info.force_field === "assigned" ? " (Amber ff19SB/GAFF2)" : ""}`;
      app.message(`${app.atomLabel(m, a)}${ff}`);
    } catch (e) { /* the label alone */ }
  };
  async function assignForceField() {
    const m = actingModel();
    if (!m) return;
    if (!m.topo?.editable) { app.message("this model has no atoms to assign a force field to", true); return; }
    try {
      await app.busy("assigning the force field (Amber ff19SB, GAFF2 for ligands)", () => app.post("api/assign_ff", { m: m.index }));
      const sel = app.selected(m);
      if (sel.length === 1) await app.describeAtom(m, sel[0]);
      else app.message("force field assigned: click an atom to see its charge, σ and ε");
    } catch (e) { app.message(e.message, true); }
  }

  // delete these atoms of a model: all of them closes it
  async function removeAtoms(m, ids, what) {
    if (ids.length >= m.topo.natoms) { await closeModel(m); return; }
    const out = await serverEdit(m, "api/edit", { op: "delete", atoms: ids, hydrogens: autoH() });
    if (out) app.message(`deleted ${what} (${ids.length} atoms) · Edit | Undo puts it back`);
  }
  // the atoms of what a right click landed on: its atom, residue, molecule
  function clicked(hit) {
    if (!hit) return null;
    const m = hit.model, A = m.topo.atoms;
    const a = hit.atom ?? hit.atoms?.[0] ?? m.topo.residues.trace[hit.residue];
    if (a === undefined || a < 0) return null;
    const r = A.residue[a], R = m.topo.residues;
    // the molecule as double-click takes it: bonded atoms of its kind (a heme on its
    // histidine is the heme, not the chain)
    const molecule = app.range(m.topo.natoms).filter(b => A.fragment[b] === A.fragment[a] && A.class[b] === A.class[a]);
    return { m, a, residue: m.topo.residueAtoms[r], molecule, resName: `${R.name[r]}${R.resid[r]}`,
             molName: molecule.length === m.topo.residueAtoms[r].length ? `${R.name[r]}${R.resid[r]}` : `molecule of ${R.name[r]}${R.resid[r]}` };
  }

  app.popup = (x, y, hit = null) => menu(() => [
    ...(c => c ? [
      { label: `${c.molName}${hit.atom !== undefined ? ` · ${c.m.topo.atoms.name[c.a]}` : ""}`, off: () => true },
      { label: "Select Molecule", run: () => app.select(c.m, c.molecule) },
      { label: "Set as Ligand", run: () => { app.select(c.m, c.molecule); chooseLigand(); } },
      // The selected part, when one is selected, and the atom under the cursor
      // otherwise: in a structure of a thousand superposed parts the cursor atom is
      // one of dozens within a few A of it, so a selection is what was meant.  The
      // item names the part either way, before it is clicked.
      ...(c.m.topo.sources ? [(t => ({ label: `Open Source of ${t.name}`,
                                       run: () => openSource(c.m, t.a) }))(sourceTarget(c))] : []),  // fmt: skip
      { label: `Delete Molecule (${c.molecule.length} atoms)`, run: () => removeAtoms(c.m, c.molecule, c.molName) },
      { label: `Delete Residue ${c.resName}`, run: () => removeAtoms(c.m, c.residue, c.resName) },
      ...(hit.atom !== undefined ? [{ label: `Delete Atom ${c.m.topo.atoms.name[c.a]}`, run: () => removeAtoms(c.m, [c.a], `atom ${c.m.topo.atoms.name[c.a]}`) }] : []),
      { label: "Copy Its SMILES", run: () => { app.select(c.m, c.molecule); copySmiles(); } },
      { label: "Copy Molecule Into", sub: copyIntoItems(c.m, c.molecule) },
      { label: "Center on It", run: () => { const [cc, r] = app.bounds(c.m, c.molecule); app.flyTo(cc, Math.max(r, 6)); } },
      "-",
      ...(c.m.topo.atoms.class[c.a] === 1 || isSeries(c.m) ? [...seriesItems(c.m, c.molecule), "-"] : []),
    ] : [])(clicked(hit)),
    { label: "Fit All", run: () => app.viewAll(), key: "W" },
    { label: "Center on Selection", run: () => app.centerView() },
    { label: "Focus on Ligand", run: focusLigand },
    "-",
    { label: "Show Selection As", sub: showAsItems },
    { label: "Hide Selection", run: () => hideAtoms(app.targets()) },
    { label: "New Representation of Selection", run: () => newRepFromSelection() },
    { label: "Add Surface", sub: surfaceItems },
    { label: "Labels", sub: labelItems },
    "-",
    { label: "Select", sub: selectItems },
    { label: "Grow Selection", sub: extendItems },
    { label: "Models", sub: modelItems },
    "-",
    { label: "Copy SMILES", run: copySmiles },
    { label: "Create Series from Selection", run: () => { const t = editTarget(); if (t) createSeries(...t); } },
    { label: "Rename Residue or Chain…", run: relabelPanel },
    { label: "Delete", run: deleteSelected },
  ], x, y);

  // --- series: designs of a ligand, a model each, saved together as one SDF ----------------

  const isSeries = m => m?.group?.startsWith("series");
  // a molecule's popup items: start a series from it, or (a member) the series' own
  function seriesItems(m, atoms) {
    if (!isSeries(m)) return [{ label: "Create Series from It", run: () => createSeries(m, atoms) }];
    return seriesMemberItems(m);
  }
  function seriesMemberItems(m) {
    return [
      { label: "New Analogue (a copy of this)", run: () => newAnalogue(m) },
      { label: "Note…", run: () => notePanel(m) },
      { label: "Receptor", sub: () => [
        ...app.live().filter(o => !isSeries(o)).map(o => ({ label: o.name, checked: () => m.receptor === o.index,
                                                             run: () => setReceptor(m.group, o.index) })),
        "-",
        { label: "None", checked: () => m.receptor === undefined, run: () => setReceptor(m.group, null) },
      ] },
      { label: `${m.group} as a Table`, run: () => seriesTablePanel(m.group) },
      { label: `Save ${m.group} as SDF…`, run: () => saveSeriesPanel(m.group) },
      { label: `Download ${m.group} (SDF)`, run: () => downloadSeries(m.group) },
    ];
  }
  async function createSeries(m, atoms) {
    try {
      const out = await app.post("api/series", { m: m.index, atoms, frame: Math.min(state.frame, m.topo.nframes - 1) });
      await addModels(out, false);
      app.clearSelection();
      const first = state.models[out.added[0]];
      app.message(`${first.group}: ${first.name}, a copy of the ligand · change it with the Builder; `
        + "right-click it for New Analogue and Save as SDF");
    } catch (e) { app.message(e.message, true); }
  }
  async function newAnalogue(m) {
    try {
      const out = await app.post("api/series_add", { m: m.index });
      for (const info of out.models.slice(state.models.length)) state.models.push(app.newModel(info, state.models.length));
      const next = state.models[out.added[0]];
      await app.solo(next);  // the new design alone of its series, over what it sits in
      app.message(`${next.name}: a copy of ${m.name} · ← and → step through ${m.group}`);
    } catch (e) { app.message(e.message, true); }
  }
  // the page's models take the server's listing (receptors, notes) as it is now
  function relist(out) {
    out.models.forEach((info, k) => {
      const m = state.models[k];
      if (!m) return;
      m.row = info.row;
      if (info.receptor === undefined) delete m.receptor; else m.receptor = info.receptor;
    });
    app.onModels();
  }
  async function setReceptor(group, receptor) {
    try {
      relist(await app.post("api/series_receptor", { group, receptor }));
      await app.refreshContacts();
      app.rebuild();
      app.message(receptor === null ? `${group}: no receptor` : `${group}: designs made against ${state.models[receptor].name} `
        + "(contacts and Minimize use it)");
    } catch (e) { app.message(e.message, true); }
  }
  function notePanel(m) {
    panel("note", `Note on ${m.name}`, body => {
      const input = el("input", { value: m.row?.note || "", style: "width:320px", spellcheck: true, placeholder: "what this design tries" });
      const go = button("Keep", async () => {
        try {
          relist(await app.post("api/series_note", { m: m.index, note: input.value }));
          panels.note?.remove(); delete panels.note;
        } catch (e) { app.message(e.message, true); }
      }, "blue");
      input.onkeydown = e => { if (e.key === "Enter") go.click(); };
      body.append(row(input, go), el("p", { className: "help" }, "Shown in the model list and saved with the series (SDF field note)."));
      setTimeout(() => input.focus(), 0);
    }, [innerWidth / 2 - 200, 90]);
  }
  // a series as a table: a row per member, a column per SDF field.  Click a row to
  // look at that member alone, a column's header to sort by it (again: the other way);
  // tick rows to plot only those (none ticked: every member)
  const picked = {};  // series -> the indices of its ticked members
  // series -> the member last clicked in its table or plot, marked in both until it is
  // clicked again or the plot's background is
  const focus = {};
  const lookAt = m => { focus[m.group] = m.index; app.solo(m); };
  const unfocus = group => { delete focus[group]; app.onModels(); };
  // series -> each member's properties as RDKit computes them from its structure (MW,
  // cLogP, ...), named apart from the SDF's own fields; not saved with the series
  const computed = {}, computing = {};
  const value = (m, c) => m.row?.[c] ?? computed[m.group]?.[m.index]?.[c];
  async function compute(group) {
    if (computing[group]) { computing[group] = "again"; return; }  // an edit while asking
    computing[group] = true;
    try {
      const { properties } = await app.post("api/series_properties", { group });
      const own = new Set(app.live().filter(m => m.group === group).flatMap(m => Object.keys(m.row || {})));
      const named = {};
      for (const [k, props] of Object.entries(properties)) {
        named[k] = Object.fromEntries(Object.entries(props).map(([c, v]) => [own.has(c) ? `${c} (calc)` : c, v]));
      }
      if (JSON.stringify(named) !== JSON.stringify(computed[group])) {
        computed[group] = named;
        for (const kind of ["table", "plot", "histogram"]) panels[`${kind} ${group}`]?.redraw?.();
      }
    } catch (e) { app.message(e.message, true); }
    const again = computing[group] === "again";
    computing[group] = false;
    if (again) compute(group);
  }
  function seriesTablePanel(group) {
    const id = `table ${group}`;
    let column = null, descending = false, chosen = [];
    panel(id, group, (body, p) => {
      const draw = () => {
        const members = app.live().filter(m => m.group === group);
        if (!members.length) { p.remove(); delete panels[id]; return; }
        const ticked = picked[group] ??= new Set();
        for (const k of ticked) if (!members.some(m => m.index === k)) ticked.delete(k);  // closed
        compute(group);  // its computed columns, again if a member was edited
        const fields = [], calc = [];
        for (const m of members) for (const k of Object.keys(m.row || {})) if (!fields.includes(k)) fields.push(k);
        for (const m of members) for (const k of Object.keys(computed[group]?.[m.index] || {})) if (!calc.includes(k)) calc.push(k);
        const columns = [...fields, ...calc];
        const rows = column ? [...members].sort(byColumn(column, descending)) : members;
        // a column of numbers as a histogram, two against each other
        const numeric = columns.filter(c => members.some(m => typeof value(m, c) === "number"));
        const tools = [];
        const pick = k => {
          const s = el("select");
          for (const c of numeric) s.add(new Option(c, c));
          s.value = numeric.includes(chosen[k]) ? chosen[k] : numeric[Math.min(k, numeric.length - 1)];
          s.onchange = () => { chosen[k] = s.value; };
          return s;
        };
        const which = el("span", { className: "help" }, ticked.size ? `${ticked.size} ticked` : "every member");
        if (numeric.length >= 2) {
          const xs = pick(0), ys = pick(1);
          tools.push(row(el("span", { className: "what" }, "plot"), ys, "against", xs,
                         button("Plot", () => scatterPanel(group, xs.value, ys.value), "blue"), which));
        }
        if (numeric.length) {
          const hs = pick(2);
          tools.push(row(el("span", { className: "what" }, "histogram"), hs,
                         button("Histogram", () => histogramPanel(group, hs.value), "blue"),
                         numeric.length < 2 ? which : ""));
        }
        const tick = (k, on) => {
          if (on) ticked.add(k); else ticked.delete(k);
          draw(); panels[`plot ${group}`]?.redraw?.(); panels[`histogram ${group}`]?.redraw?.();
        };
        const t = el("table", { className: "series-table" });
        const head = t.createTHead().insertRow();
        const all = el("input", { type: "checkbox", title: "tick or untick every row", checked: ticked.size === members.length });
        all.indeterminate = ticked.size > 0 && ticked.size < members.length;
        all.onchange = () => {
          for (const m of members) if (all.checked) ticked.add(m.index); else ticked.delete(m.index);
          draw(); panels[`plot ${group}`]?.redraw?.(); panels[`histogram ${group}`]?.redraw?.();
        };
        head.append(el("th", { className: "tick" }, all));
        for (const c of ["name", ...columns]) {
          const made = calc.includes(c);
          const th = el("th", { className: made ? "calc" : "", title: `${made ? "computed by RDKit from the structure (not in the SDF) · " : ""}sort by ${c}` },
                        c + (c === column ? (descending ? " ↓" : " ↑") : ""));
          th.onclick = () => { descending = c === column && !descending; column = c; draw(); };
          head.append(th);
        }
        const tbody = t.createTBody();
        for (const m of rows) {
          const tr = tbody.insertRow();
          if (m.index === focus[group]) tr.className = "current";
          const box = el("input", { type: "checkbox", checked: ticked.has(m.index), title: "plot this one" });
          box.onclick = e => e.stopPropagation();  // a tick, not a look at it
          box.onchange = () => tick(m.index, box.checked);
          tr.insertCell().append(box);
          for (const v of [m.name, ...columns.map(c => value(m, c))]) {
            const td = tr.insertCell();
            td.textContent = cell(v);
            td.title = td.textContent;
          }
          tr.onclick = () => m.index === focus[group] ? unfocus(group) : lookAt(m);
        }
        body.replaceChildren(...tools, el("div", { className: "series-scroll" }, t));
      };
      p.redraw = draw;
      draw();
    }).classList.add("resizable");
  }
  // one column of a series against another: a point per member that has both (the
  // ticked ones, if any are; click a point to look at that member), with Pearson's r
  // and Spearman's rho
  function scatterPanel(group, xc, yc) {
    const id = `plot ${group}`;
    const W = 480, H = 420, L = 56, B = 40, T0 = 10, R0 = 12;
    const log = { x: false, y: false };
    let degree = 1;  // of the least-squares polynomial drawn through the points; 0: none
    let white = false;  // drawn for a white page (black text) or a black one (white text)
    panel(id, `${group}: ${yc} against ${xc}`, (body, p) => {
      const head = el("div", { className: "help", style: "margin:0 0 6px" });
      const box = el("div", { className: "plot scatter" });
      const scale = axis => {
        const b = el("input", { type: "checkbox" });
        b.onchange = () => { log[axis] = b.checked; draw(); };
        return el("label", {}, b, ` log ${axis}`);
      };
      const file = `${group}_${yc}_vs_${xc}`.replace(/[^\w.-]+/g, "_");
      const order = el("input", { type: "number", min: 0, max: 6, step: 1, value: degree, style: "width:42px",
                                  title: "the fitted polynomial's degree (1: a straight line; 0: no line)" });
      order.onchange = () => { degree = Math.max(0, Math.min(6, Math.round(+order.value) || 0)); order.value = degree; draw(); };
      const fitNote = el("div", { className: "help" });
      const paper = el("input", { type: "checkbox", checked: white });
      paper.onchange = () => { white = paper.checked; box.classList.toggle("white", white); };
      body.append(head, box, fitNote, row(scale("x"), scale("y"), el("label", {}, "fit degree ", order), el("label", { title: "for a white page: black text and lines (else white, for a black one); saved with a clear background either way" }, paper, " white"), el("span", { style: "flex:1" }),
        button("SVG", () => savePlot(box, file, "svg", white)), button("PNG", () => savePlot(box, file, "png", white))));
      const draw = () => {
        const ticked = picked[group]?.size ? picked[group] : null;
        const have = app.live().filter(m => m.group === group && (!ticked || ticked.has(m.index)))
          .filter(m => Number.isFinite(value(m, xc)) && Number.isFinite(value(m, yc)));
        // a log axis has no place for zero or below
        const pts = have.filter(m => (!log.x || value(m, xc) > 0) && (!log.y || value(m, yc) > 0));
        const left = have.length - pts.length;
        const some = `${ticked ? "ticked " : ""}members`;
        const dropped = left ? ` · ${left} left out (not positive, on a log axis)` : "";
        fitNote.textContent = "";
        if (pts.length < 2) { head.textContent = `fewer than two ${some} have both ${xc} and ${yc}${dropped}`; box.innerHTML = ""; return; }
        const f = axis => log[axis] ? Math.log10 : v => v;  // value -> where it is plotted
        const X = pts.map(m => f("x")(value(m, xc))), Y = pts.map(m => f("y")(value(m, yc)));
        const span = v => { const lo = Math.min(...v), hi = Math.max(...v), pad = (hi - lo || Math.abs(lo) || 1) * 0.06; return [lo - pad, hi + pad]; };
        const [x0, x1] = span(X), [y0, y1] = span(Y);
        const x = v => L + (v - x0) / (x1 - x0) * (W - L - R0), y = v => T0 + (y1 - v) / (y1 - y0) * (H - T0 - B);
        const ticks = (axis, lo, hi) => axisTicks(log[axis], lo, hi);
        const grid = ticks("x", x0, x1).map(([v, s]) => `<line x1="${x(v)}" x2="${x(v)}" y1="${T0}" y2="${H - B}" class="grid"/><text x="${x(v)}" y="${H - B + 16}" text-anchor="middle">${s}</text>`).join("")
          + ticks("y", y0, y1).map(([v, s]) => `<line x1="${L}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 5}" y="${y(v) + 4}" text-anchor="end">${s}</text>`).join("");
        const dots = pts.map((m, k) => `<circle cx="${x(X[k]).toFixed(1)}" cy="${y(Y[k]).toFixed(1)}" r="${m.index === focus[group] ? 6.5 : 4.5}" class="${m.index === focus[group] ? "on" : ""}" data-k="${k}"><title>${esc(m.name)}: ${esc(xc)} ${cell(value(m, xc))}, ${esc(yc)} ${cell(value(m, yc))}</title></circle>`).join("");
        // the fit, in what is drawn (the logs, on a log axis); it needs more points than terms
        let curve = "";
        const fit = degree && pts.length > degree ? polyfit(X, Y, degree) : null;
        if (fit) {
          const at = v => fit.reduce((s, c, k) => s + c * v ** k, 0);
          const path = Array.from({ length: 121 }, (_, k) => x0 + (x1 - x0) * k / 120)
            .map((v, k) => `${k ? "L" : "M"}${x(v).toFixed(1)} ${y(at(v)).toFixed(1)}`).join("");
          curve = `<clipPath id="fitclip${++plotClips}"><rect x="${L}" y="${T0}" width="${W - L - R0}" height="${H - T0 - B}"/></clipPath><path d="${path}" class="fit" clip-path="url(#fitclip${plotClips})"/>`;
          const fy = Y.map((_, k) => at(X[k])), my = Y.reduce((s, v) => s + v, 0) / Y.length;
          const r2 = 1 - Y.reduce((s, v, k) => s + (v - fy[k]) ** 2, 0) / Y.reduce((s, v) => s + (v - my) ** 2, 0);
          const vx = log.x ? `log ${xc}` : xc, vy = log.y ? `log ${yc}` : yc, num = c => +c.toPrecision(4);
          const terms = fit.map((c, k) => k === 0 ? `${num(c)}` : `${c < 0 ? "−" : "+"} ${Math.abs(num(c))}·x${k > 1 ? "^" + k : ""}`);
          fitNote.textContent = `fit: y = ${terms.join(" ")} (y: ${vy}, x: ${vx}) · R² ${r2.toFixed(3)}`;
        } else if (degree) fitNote.textContent = `a degree ${degree} fit needs more than ${degree} points`;
        box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${grid}${curve}<text x="${(L + W - R0) / 2}" y="${H - 6}" text-anchor="middle">${esc(xc)}</text><text transform="translate(14 ${(T0 + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(yc)}</text>${dots}</svg>`;
        for (const c of box.querySelectorAll("circle")) {
          c.onclick = e => { e.stopPropagation(); const m = pts[+c.dataset.k]; m.index === focus[group] ? unfocus(group) : lookAt(m); };
        }
        box.querySelector("svg").onclick = () => { if (group in focus) unfocus(group); };
        // r of what is drawn (the logs, on a log axis); rho is the same either way
        head.textContent = `${pts.length} ${some} · Pearson r ${pearson(X, Y).toFixed(2)}${log.x || log.y ? " (as plotted)" : ""}`
          + ` · Spearman ρ ${pearson(ranks(X), ranks(Y)).toFixed(2)}${dropped}`;
      };
      p.redraw = draw;
      draw();
    });
    // a window to drag larger by its corner: the plot fills its width
    Object.assign(panels[id].style, { width: "560px" });
    panels[id].classList.add("resizable");
  }
  // one column of a series as a histogram: how many members (the ticked ones, if any
  // are) fall in each bin, with their names on hovering a bar; the clicked member is a
  // dot under its bin
  function histogramPanel(group, c) {
    const id = `histogram ${group}`;
    const W = 480, H = 380, L = 48, B = 40, T0 = 10, R0 = 12;
    let logx = false, bins = 0, white = false;  // bins 0: Sturges' rule, log2 n + 1
    panel(id, `${group}: ${c}`, (body, p) => {
      const head = el("div", { className: "help", style: "margin:0 0 6px" });
      const box = el("div", { className: "plot scatter" });
      const lx = el("input", { type: "checkbox" });
      lx.onchange = () => { logx = lx.checked; draw(); };
      const nb = el("input", { type: "number", min: 0, max: 200, step: 1, value: bins, style: "width:48px",
                               title: "how many bins (0: Sturges' rule, log2 n + 1)" });
      nb.onchange = () => { bins = Math.max(0, Math.min(200, Math.round(+nb.value) || 0)); nb.value = bins; draw(); };
      const paper = el("input", { type: "checkbox", checked: white });
      paper.onchange = () => { white = paper.checked; box.classList.toggle("white", white); };
      const file = `${group}_${c}_histogram`.replace(/[^\w.-]+/g, "_");
      body.append(head, box, row(el("label", {}, lx, " log x"), el("label", {}, "bins ", nb),
        el("label", { title: "for a white page: black text and lines (else white, for a black one); saved with a clear background either way" }, paper, " white"),
        el("span", { style: "flex:1" }),
        button("SVG", () => savePlot(box, file, "svg", white)), button("PNG", () => savePlot(box, file, "png", white))));
      const draw = () => {
        const ticked = picked[group]?.size ? picked[group] : null;
        const have = app.live().filter(m => m.group === group && (!ticked || ticked.has(m.index)) && Number.isFinite(value(m, c)));
        const pts = have.filter(m => !logx || value(m, c) > 0);
        const some = `${ticked ? "ticked " : ""}members`;
        const dropped = have.length > pts.length ? ` · ${have.length - pts.length} left out (not positive, on a log axis)` : "";
        if (!pts.length) { head.textContent = `no ${some} have ${c}${dropped}`; box.innerHTML = ""; return; }
        const V = pts.map(m => value(m, c)), X = V.map(v => logx ? Math.log10(v) : v);
        const n = bins || Math.ceil(Math.log2(X.length)) + 1;
        let lo = Math.min(...X), hi = Math.max(...X);
        if (hi === lo) { lo -= 0.5; hi += 0.5; }
        const w = (hi - lo) / n, counts = new Array(n).fill(0), names = Array.from({ length: n }, () => []);
        const binOf = v => Math.min(n - 1, Math.floor((v - lo) / w));
        X.forEach((v, k) => { counts[binOf(v)]++; names[binOf(v)].push(pts[k].name); });
        const top = Math.max(...counts) * 1.08;
        const x = v => L + (v - lo) / (hi - lo) * (W - L - R0), y = v => T0 + (1 - v / top) * (H - T0 - B);
        const grid = axisTicks(logx, lo, hi).map(([v, s]) => `<text x="${x(v)}" y="${H - B + 16}" text-anchor="middle">${s}</text>`).join("")
          + axisTicks(false, 0, top).filter(([v]) => Number.isInteger(v)).map(([v, s]) => `<line x1="${L}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 5}" y="${y(v) + 4}" text-anchor="end">${s}</text>`).join("");
        const edge = k => { const v = lo + k * w; return cell(logx ? +(10 ** v).toPrecision(3) : +v.toPrecision(4)); };
        const bars = counts.map((k, i) => k ? `<rect class="bar" x="${x(lo + i * w).toFixed(1)}" y="${y(k).toFixed(1)}" width="${(x(lo + w) - x(lo)).toFixed(1)}" height="${(y(0) - y(k)).toFixed(1)}"><title>${esc(c)} ${edge(i)} to ${edge(i + 1)}: ${k}\n${esc(names[i].slice(0, 20).join(", "))}${names[i].length > 20 ? ", ..." : ""}</title></rect>` : "").join("");
        const f = pts.find(m => m.index === focus[group]);
        const mark = f ? `<circle class="on" r="6" cx="${x(logx ? Math.log10(value(f, c)) : value(f, c)).toFixed(1)}" cy="${(y(0) - 7).toFixed(1)}"><title>${esc(f.name)}: ${esc(c)} ${cell(value(f, c))}</title></circle>` : "";
        box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${grid}${bars}<line x1="${L}" x2="${W - R0}" y1="${y(0)}" y2="${y(0)}" class="grid"/><text x="${(L + W - R0) / 2}" y="${H - 6}" text-anchor="middle">${esc(c)}</text><text transform="translate(14 ${(T0 + H - B) / 2}) rotate(-90)" text-anchor="middle">members</text>${mark}</svg>`;
        const dot = box.querySelector("circle.on");
        if (dot) dot.onclick = () => unfocus(group);
        const sorted = [...V].sort((a, b) => a - b), mid = sorted.length / 2;
        const median = sorted.length % 2 ? sorted[Math.floor(mid)] : (sorted[mid - 1] + sorted[mid]) / 2;
        const mean = V.reduce((s, v) => s + v, 0) / V.length;
        head.textContent = `${pts.length} ${some} · ${n} bins${logx ? " (in log)" : ""} · median ${cell(+median.toPrecision(4))} · mean ${cell(+mean.toPrecision(4))} · range ${cell(sorted[0])} to ${cell(sorted[sorted.length - 1])}${dropped}`;
      };
      p.redraw = draw;
      draw();
    });
    Object.assign(panels[id].style, { width: "560px" });
    panels[id].classList.add("resizable");
  }
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  // an axis's ticks, [where, label]: plain steps; on a log axis (lo and hi in log10) the
  // decades, with 2 and 5 between them when it spans few, or plain steps within a decade
  function axisTicks(log, lo, hi) {
    const linear = (lo, hi) => {
      const step = 10 ** Math.floor(Math.log10((hi - lo) / 4)), s = [1, 2, 5, 10].map(k => k * step).find(k => (hi - lo) / k <= 5);
      const out = [];
      for (let v = Math.ceil(lo / s) * s; v <= hi; v += s) out.push(+v.toPrecision(12));
      return out.map(v => [v, cell(v)]);
    };
    if (!log) return linear(lo, hi);
    const at = m => { const out = []; for (let d = Math.floor(lo); d <= Math.ceil(hi); d++) for (const k of m) { const t = Math.log10(k) + d; if (t >= lo && t <= hi) out.push(t); } return out; };
    const ts = hi - lo < 2 ? at([1, 2, 5]) : at([1]);
    if (ts.length >= 2) return ts.map(t => [t, cell(+(10 ** t).toPrecision(3))]);
    return linear(10 ** lo, 10 ** hi).filter(([v]) => v > 0).map(([v, s]) => [Math.log10(v), s]);
  }
  // a plot as a file, on a clear background: its SVG with the page's styles written into
  // it (Arial; black text and lines for a white page, white for a black one), editable in
  // Illustrator; or that drawn 4x as a PNG
  function savePlot(box, name, fmt, white = false) {
    const ink = white ? "#000" : "#fff", rule = white ? "#c8c8c8" : "#555";
    const svg = box.querySelector("svg");
    if (!svg) return;
    const out = svg.cloneNode(true);
    out.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    for (const t of out.querySelectorAll("text")) Object.assign(t.style, { font: "12px Arial, sans-serif", fill: ink });
    for (const g of out.querySelectorAll(".grid")) Object.assign(g.style, { stroke: rule, strokeWidth: "0.6" });
    for (const f of out.querySelectorAll(".fit")) Object.assign(f.style, { fill: "none", stroke: ink, strokeWidth: "1.2" });
    for (const r of out.querySelectorAll(".bar")) Object.assign(r.style, { fill: "#5b8def", stroke: "rgba(0,0,0,0.45)", strokeWidth: "0.5" });
    for (const c of out.querySelectorAll("circle")) {
      Object.assign(c.style, { fill: c.classList.contains("on") ? "#ff73d9" : "#5b8def", stroke: "rgba(0,0,0,0.45)", strokeWidth: "0.5" });
    }
    const text = new XMLSerializer().serializeToString(out);
    const save = (blob, ext) => el("a", { href: URL.createObjectURL(blob), download: `${name}.${ext}` }).click();
    if (fmt === "svg") { save(new Blob([text], { type: "image/svg+xml" }), "svg"); return; }
    const img = new Image(), k = 4, w = +svg.getAttribute("width"), h = +svg.getAttribute("height");
    img.onload = () => {
      const c = el("canvas", { width: w * k, height: h * k });
      c.getContext("2d").drawImage(img, 0, 0, w * k, h * k);
      c.toBlob(b => save(b, "png"), "image/png");
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
  }
  // the least-squares polynomial of ``degree`` through (x, y): its coefficients, constant
  // first.  Solved on x centred and scaled, then expanded back, so high degrees stay sound
  let plotClips = 0;  // each drawn fit's clip, its own id
  function polyfit(x, y, degree) {
    const n = degree + 1, mx = x.reduce((s, v) => s + v, 0) / x.length;
    const sx = Math.max(...x.map(v => Math.abs(v - mx))) || 1, u = x.map(v => (v - mx) / sx);
    const A = Array.from({ length: n }, (_, i) => Array.from({ length: n + 1 }, (_, j) =>
      j < n ? u.reduce((s, v) => s + v ** (i + j), 0) : u.reduce((s, v, k) => s + y[k] * v ** i, 0)));
    for (let i = 0; i < n; i++) {  // Gauss-Jordan with partial pivoting
      const p = A.slice(i).reduce((b, r, k) => Math.abs(r[i]) > Math.abs(A[b][i]) ? i + k : b, i);
      [A[i], A[p]] = [A[p], A[i]];
      if (!A[i][i]) return null;
      for (let k = 0; k < n; k++) if (k !== i) { const f = A[k][i] / A[i][i]; for (let j = i; j <= n; j++) A[k][j] -= f * A[i][j]; }
    }
    const b = A.map((r, i) => r[n] / r[i]);  // in u = (x - mx) / sx; expand to x
    const out = new Array(n).fill(0);
    const binom = (m, k) => { let c = 1; for (let t = 1; t <= k; t++) c = c * (m - t + 1) / t; return c; };
    for (let m = 0; m < n; m++) for (let k = 0; k <= m; k++) out[k] += b[m] / sx ** m * binom(m, k) * (-mx) ** (m - k);
    return out;
  }
  function pearson(a, b) {
    const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
    let sab = 0, saa = 0, sbb = 0;
    for (let k = 0; k < n; k++) { sab += (a[k] - ma) * (b[k] - mb); saa += (a[k] - ma) ** 2; sbb += (b[k] - mb) ** 2; }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : NaN;
  }
  // ranks from 1, ties sharing their mean rank
  function ranks(v) {
    const order = v.map((x, k) => k).sort((i, j) => v[i] - v[j]), out = new Array(v.length);
    for (let i = 0; i < order.length;) {
      let j = i;
      while (j + 1 < order.length && v[order[j + 1]] === v[order[i]]) j++;
      for (let k = i; k <= j; k++) out[order[k]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return out;
  }
  function saveSeriesPanel(group) {
    pathPanel("series", `Save ${group}`, `${group.replace(/\s+/g, "_")}.sdf`, async path => {
      try {
        const out = await app.post("api/series_save", { group, path });
        app.message(`saved ${group}: ${out.path} (one record per design; File | Open brings it back as a series)`);
        panels.series?.remove(); delete panels.series;
      } catch (e) { app.message(e.message, true); }
    });
  }
  async function downloadSeries(group) {
    try {
      const r = await fetch("api/series_download", { method: "POST", headers: { "Content-Type": "application/json", "X-Boonza-Token": app.TOKEN },
        body: JSON.stringify({ group }) });
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      const name = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "")?.[1] || "series.sdf";
      el("a", { href: URL.createObjectURL(await r.blob()), download: name }).click();
      app.message(`downloaded ${name}`);
    } catch (e) { app.message(e.message, true); }
  }

  // --- the view's own buttons, the status bar ---------------------------------------------------

  trackball(app);
  function toggleTrackball() {
    $("trackball").hidden = !$("trackball").hidden;
    try { localStorage.setItem("boonza.trackball", $("trackball").hidden ? "0" : "1"); } catch (e) { /* not kept */ }
  }
  try { $("trackball").hidden = localStorage.getItem("boonza.trackball") === "0"; } catch (e) { /* shown */ }

  $("vt-whole").onclick = () => app.viewAll();
  $("vt-center").onclick = () => app.centerView();
  $("vt-site").onclick = () => focusLigand();
  $("vt-reset").onclick = () => app.resetView();
  $("vt-2d").onclick = () => toggle2D();
  $("vt-fog").onclick = fogPanel;
  $("vt-clip").onclick = zclipPanel;
  $("f-hydrogens").onclick = cycleHydrogens;
  $("f-cell").onclick = toggleCell;
  function toggle2D() {
    const on = $("lig2d").hidden;
    $("lig2d").hidden = !on;
    $("vt-2d").classList.toggle("on", on);
    if (on) refresh2D();
  }
  let pending2D = 0;
  async function refresh2D() {
    if ($("lig2d").hidden) return;
    const ticket = ++pending2D;
    const sel = app.selection();
    const m = sel[0]?.[0] || app.currentModel();
    if (!m?.topo) return;
    const ids = sel[0]?.[1] || [];
    const dark = document.body.classList.contains("dark") ? 1 : 0;
    try {
      const out = await app.getJSON(`api/ligand2d?m=${m.index}&dark=${dark}&sel=${ids.slice(0, 400).join(",")}`);
      if (ticket !== pending2D) return;
      const box = $("lig2d").querySelector(".lig2d-svg");
      box.innerHTML = out.svg;
      $("lig2d").querySelector(".lig2d-title").textContent = `${out.name}  ·  click an atom to select it (Shift: toggle)`;
      out.coords.forEach(([x, y], k) => {
        const hit = el("div", { className: "hit", title: m.topo.atoms.name[out.atoms[k]] });
        Object.assign(hit.style, { left: `${x}px`, top: `${y}px` });
        hit.onclick = e => app.select(m, [out.atoms[k]], e.shiftKey ? "toggle" : "set");
        box.append(hit);
      });
    } catch (e) {
      $("lig2d").querySelector(".lig2d-title").textContent = e.message;
      $("lig2d").querySelector(".lig2d-svg").innerHTML = "";
    }
  }

  // the footer editing area: what the selection is, and its geometry to edit
  app.onSelection = () => {
    ramaRefresh?.();
    footerEdit();
    footerSmiles();
    refreshSmilesBox();
    paintSequence();
  };
  function footerEdit() {
    setTimeout(refresh2D, 0);
    const box = $("footer-edit");
    box.innerHTML = "";
    const sel = app.selection();
    const m = sel.length === 1 ? sel[0][0] : app.currentModel();
    $("f-cell").hidden = !(m?.box && m.box.flat().some(x => x));  // a periodic box: it can be drawn
    if (!m || !m.topo) return;
    const n = sel.reduce((k, [, ids]) => k + ids.length, 0);
    if (!n) {
      const R = m.topo.residues;
      box.append(el("span", { className: "what" },
        `${m.name}: ${m.topo.natoms} atoms, ${R.name.length} residues, ${m.topo.chains.name.length} chains`));
      return;
    }
    // one residue, whole (a ribbon clicked, a double-click): its name, number and chain
    const res = sel.length === 1 ? [...new Set(sel[0][1].map(a => m.topo.atoms.residue[a]))] : [];
    if (n > 1 && res.length === 1 && n === m.topo.residueAtoms[res[0]].length) {
      residueFields(box, m, sel[0][1]);
      return;
    }
    if (sel.length > 1 || n > 4) { box.append(el("span", { className: "what" }, `${n} atoms selected`)); return; }
    const atoms = orderChain(m, sel[0][1]);
    const { value } = n > 1 ? app.measure(m, atoms) : { value: 0 };
    const editable = m.topo.editable;
    const field = (v, digits, apply) => {
      const input = el("input", { value: v.toFixed(digits) });
      input.onkeydown = e => { if (e.key === "Enter") apply(+input.value); };
      const go = el("button", {}, "Apply");
      go.onclick = () => apply(+input.value);
      if (!editable) { input.disabled = true; go.disabled = true; }
      return [input, go];
    };
    if (n === 1) {
      const a = atoms[0];
      const name = el("input", { className: "name", value: m.topo.atoms.name[a] });
      name.onkeydown = e => { if (e.key === "Enter") edit("name", name.value); };
      name.disabled = !editable;
      box.append(el("span", { className: "what" }, "Atom"), name,
                 el("span", { className: "what" }, `[${m.topo.atoms.element[a]}]`));
      residueFields(box, m, atoms);
    } else if (n === 2) {
      const bonded = m.topo.bondsOf[atoms[0]].includes(atoms[1]);
      box.append(el("span", { className: "what" }, "Distance"),
                 ...field(value, 3, v => bonded ? serverEdit(m, "api/set_distance", { atoms, value: v }, false)
                                                : app.message("only a bond's length can be set", true)));
      if (bonded && editable) rotateWheel(box, m, atoms);
    } else if (n === 3) {
      box.append(el("span", { className: "what" }, "Angle"),
                 ...field(value, 2, v => serverEdit(m, "api/set_angle", { atoms, value: v }, false)));
    } else {
      box.append(el("span", { className: "what" }, "Dihedral"),
                 ...field(value, 2, v => serverEdit(m, "api/set_dihedral", { atoms, angle: v }, false)));
      const prof = el("button", {}, "Profile");
      prof.onclick = () => profilePanel(m, atoms);
      if (editable) box.append(prof);
    }
  }

  // the footer's residue name, number and chain, for one atom or one residue: Enter applies
  function residueFields(box, m, atoms) {
    const R = m.topo.residues, r = m.topo.atoms.residue[atoms[0]];
    const was = { name: R.name[r], resid: `${R.resid[r]}${R.insertion[r].trim()}`, chain: m.topo.chains.name[R.chain[r]] };
    const fields = {
      name: el("input", { className: "name", value: was.name, title: "residue name" }),
      resid: el("input", { className: "short", value: was.resid, title: "residue number (52A: with an insertion code)" }),
      chain: el("input", { className: "short", value: was.chain, title: "chain" }),
    };
    const apply = async () => {
      const value = {};
      for (const [k, f] of Object.entries(fields)) if (f.value.trim() !== was[k]) value[k] = f.value;
      if (!Object.keys(value).length) return;
      if (!await serverEdit(m, "api/edit", { op: "residue", atoms, value, hydrogens: false })) return;
      app.select(m, atoms, "set");  // relabelling keeps the atoms where they were
      app.message(`${was.name}${was.resid}: relabelled`);
    };
    for (const f of Object.values(fields)) {
      f.onkeydown = e => { if (e.key === "Enter") apply(); };
      f.disabled = !m.topo.editable;
    }
    box.append(el("span", { className: "what" }, "Residue"), fields.name, fields.resid,
               el("span", { className: "what" }, "Chain"), fields.chain);
  }

  // two bonded atoms: the Rotate wheel turns the bond, and Profile plots its energy
  async function rotateWheel(box, m, atoms) {
    let d;
    try { d = await app.post("api/dihedral", { m: m.index, atoms }); } catch (e) { return; }
    const wheel = el("input", { type: "range", min: -180, max: 180, step: 1, value: Math.round(d.angle),
                                title: `Rotate: ${d.atoms.map(a => m.topo.atoms.name[a]).join("–")}` });
    const label = el("span", { className: "what" }, `${d.name ? d.name.replace("chi", "χ") + " " : ""}${d.angle.toFixed(1)}°`);
    const turn = { ...d, model: m, base: m.pos.slice() };
    wheel.oninput = () => { app.turnDihedral(turn, +wheel.value); label.textContent = `${wheel.value}°`; };
    wheel.onchange = () => serverEdit(m, "api/set_dihedral", { atoms: d.atoms, angle: +wheel.value }, false);
    const prof = el("button", {}, "Profile");
    prof.onclick = () => profilePanel(m, d.atoms);
    box.append(el("span", { className: "what" }, "Rotate"), wheel, label, prof);
  }

  // --- command line -------------------------------------------------------------------------------

  function openCLI(text = "") {
    $("cli").hidden = false;
    $("cli-input").value = text;
    $("cli-input").focus();
  }
  const closeCLI = () => { $("cli").hidden = true; $("cli-input").blur(); };
  $("btn-cli").onclick = () => $("cli").hidden ? openCLI() : closeCLI();
  $("cli-close").onclick = closeCLI;
  // Enter runs a command and the line stays, empty, for the next; up and down bring
  // back the ones before; Esc, ✕ or Command again closes it
  const cliHistory = [];
  let cliAt = 0;
  $("cli").onsubmit = async e => {
    e.preventDefault();
    const input = $("cli-input"), text = input.value.trim();
    if (!text) return;
    if (cliHistory.at(-1) !== text) cliHistory.push(text);
    cliAt = cliHistory.length;
    input.value = "";
    await command(text);
  };
  $("cli-input").onkeydown = e => {
    const input = $("cli-input");
    if (e.key === "Escape") closeCLI();
    else if (e.key === "ArrowUp" && cliAt > 0) { e.preventDefault(); input.value = cliHistory[--cliAt]; }
    else if (e.key === "ArrowDown" && cliAt < cliHistory.length) { e.preventDefault(); input.value = cliHistory[++cliAt] ?? ""; }
  };

  // models by number (as the side bar lists them), by name, or all: "2", "1,3", "1FNA"
  function modelsNamed(spec) {
    const open = state.models.filter(m => !m.closed);
    if (spec === "all") return open;
    const out = [];
    for (const w of spec.split(",").filter(Boolean)) {
      const m = /^\d+$/.test(w) ? state.models[+w - 1] : open.find(m => m.name === w)
        ?? open.find(m => m.name.toLowerCase().includes(w.toLowerCase()));
      if (!m || m.closed) throw new Error(`no model ${w}: the side bar numbers them`);
      if (!out.includes(m)) out.push(m);
    }
    return out;
  }

  async function command(text) {
    if (!text) return;
    // "... in 2": the command acts on those models (shown first if hidden), not on every shown one
    let models = null;
    const at = text.match(/^(.*?)\s+in\s+(\S+)$/);
    if (at && !text.startsWith("model")) {
      try { models = modelsNamed(at[2]); } catch (e) { app.message(e.message, true); return; }
      text = at[1];
      const hidden = models.filter(m => !m.shown || !m.topo);
      if (hidden.length) await app.show(hidden);
      app.clearSelection();
      state.current = models[0].index;
      app.onSelection();
    }
    const [word, ...rest] = text.split(/\s+/);
    const arg = rest.join(" ");
    // the atoms of a selection in each model (or the models named), else the targets
    const pairsOf = async sel => {
      if (!sel && !models) return app.targets();
      const out = [];
      for (const m of models || app.shownModels()) out.push([m, sel ? await serverSelect(m, sel) : app.range(m.topo.natoms)]);
      return out;
    };
    const STYLE_WORDS = app.REP_STYLES.map(([s]) => s).join("|");
    switch (word) {
      case "help":
        app.message("a selection selects atoms (boonza's language, e.g. resname LIG or within 5 of resname LIG)\n"
          + `drawing: rep SEL [${STYLE_WORDS}] [COLOR] adds a representation · show SEL [STYLE] · hide SEL · `
          + "style STYLE [SEL] (those atoms in that style only) · color SCHEME [SEL] · label name|element|residue|charge|off [SEL] · "
          + "material opaque|glossy|matte|transparent|ghost · opacity 0.5 (the current representation's) · "
          + "surface [electrostatic|hydrophobic|COLOR] [coarse|standard|fine] · center [SEL] · fit · focus · reset\n"
          + "editing: select SEL · minimize · undo · redo · open PATH|PDB-ID · save PATH\n"
          + "models: model 2|1,3|NAME|all shows those alone · end a command with in 2 (or in 1,3, in NAME) to act on those models only");
        return;
      case "model": {  // show these models alone
        let these;
        try { these = modelsNamed(arg || "all"); } catch (e) { app.message(e.message, true); return; }
        const others = state.models.filter(m => m.shown && !these.includes(m));
        if (others.length) await app.show(others, false);
        await app.show(these);
        state.current = these[0].index;
        app.onSelection();
        app.message(`showing ${these.map(m => `${m.index + 1} ${m.name}`).join(", ")}`);
        return;
      }
      case "select": await selectServer(() => arg, models || undefined); return;
      case "rep": case "show": {  // rep SEL [STYLE] [COLOR]: the last words, when they are a style and a color
        const words = [...rest];
        let color = "element", style = "lines";
        if (words.length && isColor(words[words.length - 1]) && words.length > 1 && !app.repStyle(words[words.length - 1])) color = words.pop();
        if (words.length && app.repStyle(words[words.length - 1])) style = app.repStyle(words.pop());
        const sel = words.join(" ") || (app.selection().length ? null : "all");
        addRep(await pairsOf(sel), { style, color, sel });
        return;
      }
      case "hide": hideAtoms(await pairsOf(arg), arg || null); return;
      case "material": case "opacity": {  // the current representation's: material NAME, opacity 0-1 (or %)
        const m = app.currentModel(), rep = currentRep(m);
        if (!rep) { app.message("no representation: make one first (rep SEL)", true); return; }
        if (word === "material") {
          if (!app.MATERIALS.some(([k]) => k === arg)) { app.message(`materials: ${app.MATERIALS.map(([k]) => k).join(", ")}`, true); return; }
          setRepMaterial(m, rep, arg);
        } else {
          const v = parseFloat(arg);
          if (!(v > 0)) { app.message("opacity 0.5 (or 50%)", true); return; }
          rep.opacity = Math.min(1, v > 1 ? v / 100 : v);
          app.repsChanged(m);
        }
        return;
      }
      case "center": if (arg) await selectServer(() => arg, models || undefined); app.centerView(); return;
      case "style": {
        const [s, ...sel] = rest;
        const style = app.repStyle(s);
        if (!style) { app.message(`styles: ${STYLE_WORDS.replaceAll("|", ", ")}`, true); return; }
        const what = sel.join(" ");
        if (app.isRibbon({ style }) || style === "surface") addRep(await pairsOf(what), { style, color: style === "surface" ? "white" : "ss", sel: what || null });
        else showAs(await pairsOf(what), style, what || null);
        return;
      }
      case "color": {
        const [c, ...sel] = rest;
        if (!isColor(c)) { app.message("colors: element, element/green, chain, restype, resname, ss, spectrum, bfactor, or a name", true); return; }
        colorIn(await pairsOf(sel.join(" ")), c);
        return;
      }
      case "label": {
        const [mode, ...sel] = rest;
        if (mode !== "off") state.labelMode = mode;
        for (const [m, ids] of await pairsOf(sel.join(" "))) for (const a of ids) m.state.labels[a] = mode === "off" ? 0 : 1;
        app.rebuild();
        return;
      }
      case "minimize": await minimize(); return;
      case "undo": await undo(); return;
      case "redo": await undo(true); return;
      case "open": await openPath(arg); return;
      case "save": await savePath(arg); return;
      case "focus": await focusLigand(); return;
      case "fit": app.viewAll(); return;
      case "surface": {  // surface [electrostatic|hydrophobic|COLOR] [coarse|standard|fine]
        const grids = { coarse: 1.0, standard: 0.5, fine: 0.3 };
        for (const w of rest) if (w in grids) surfaceGrid(grids[w]);
        const kind = rest.find(w => ["electrostatic", "hydrophobic"].includes(w) || (named(w) && !(w in grids)));
        if (kind || !rest.length) addSurface(kind || "electrostatic");
        return;
      }
      case "reset": app.resetView(); return;
      default: await selectServer(() => text, models || undefined);  // a selection
    }
  }

  // --- the Python console ---------------------------------------------------------------------------

  const pyHistory = [];
  let pyAt = 0;
  function togglePython() {
    if (!app.PYTHON) { app.message("the Python console is offered only when boonza view listens on 127.0.0.1", true); return; }
    const box = $("pyconsole");
    box.hidden = !box.hidden;
    if (!box.hidden) {
      if (!$("py-out").textContent) pyPrint("boonza view, Python. model0, model1 …: the models (boonza Systems); m: the current one; "
        + "sel: its selected atoms (an AtomSel); models: all of them (also by name).\n"
        + "view.select(atoms), view.show(atoms, style=...), view.hide, view.color(atoms, \"cyan\"), view.label, view.center: "
        + "atoms are a selection string, an AtomSel or indices; model=1 for another model.\n"
        + "Changes to a model (positions, atoms, bonds, charges) show up after the line, and Edit | Undo undoes them.\n", "");
      $("py-input").focus();
    }
  }
  function pyPrint(text, cls) {
    const span = el("span", { className: cls }, text);
    $("py-out").append(span);
    $("py-out").scrollTop = $("py-out").scrollHeight;
  }
  async function runPython(code) {
    if (!code.trim()) return;
    pyHistory.push(code); pyAt = pyHistory.length;
    pyPrint(code.split("\n").map((l, k) => (k ? "... " : ">>> ") + l).join("\n") + "\n", "in");
    const m = app.selection()[0]?.[0] || app.currentModel();
    const sel = app.selected(m);
    try {
      const out = await app.post("api/python", { code, m: m.index, sel });
      if (out.output) pyPrint(out.output.endsWith("\n") ? out.output : out.output + "\n", /Traceback|Error/.test(out.output) ? "err" : "");
      for (const c of out.changed) {
        const model = state.models[c.m];
        if (model?.topo) await app.reloadModel(model, c.topology);
      }
      await doActions(out.actions);
    } catch (e) { pyPrint(e.message + "\n", "err"); }
  }
  async function doActions(actions) {
    for (const a of actions) {
      const m = state.models[a.m];
      if (!m) continue;
      if (!m.shown) await app.show([m]);
      await app.load(m);
      const st = m.state;
      if (a.do === "select") app.select(m, a.ids, a.add ? "add" : "set");
      else if (a.do === "show") addRep([[m, a.ids]], { style: app.repStyle(a.style || "lines") || "lines" });
      else if (a.do === "hide") hideAtoms([[m, a.ids]]);
      else if (a.do === "color") colorIn([[m, a.ids]], a.scheme);
      else if (a.do === "label") { if (a.mode !== "off") state.labelMode = a.mode; for (const k of a.ids) st.labels[k] = a.mode === "off" ? 0 : 1; }
      else if (a.do === "center" && a.ids.length) { const [c, r] = app.bounds(m, a.ids); app.flyTo(c, Math.max(r, 6)); }
    }
    app.rebuild();
  }
  $("btn-py").onclick = togglePython;
  if (!app.PYTHON) $("btn-py").disabled = true;
  $("py-input").onkeydown = e => {
    const input = $("py-input");
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const code = input.value;
      input.value = ""; input.rows = 1;
      runPython(code);
    } else if (e.key === "Enter") {
      input.rows = Math.min(8, input.value.split("\n").length + 1);
    } else if (e.key === "ArrowUp" && !input.value.slice(0, input.selectionStart).includes("\n") && pyHistory.length) {
      e.preventDefault(); pyAt = Math.max(0, pyAt - 1); input.value = pyHistory[pyAt];
    } else if (e.key === "ArrowDown" && !input.value.slice(input.selectionEnd).includes("\n") && pyAt < pyHistory.length) {
      e.preventDefault(); pyAt += 1; input.value = pyHistory[pyAt] ?? "";
    } else if (e.key === "Escape") {
      togglePython();
    }
    e.stopPropagation();
  };

  // --- keys -----------------------------------------------------------------------------------------

  window.addEventListener("keydown", e => {
    const typing = e.target.tagName === "INPUT" && !["range", "checkbox"].includes(e.target.type) || ["SELECT", "TEXTAREA"].includes(e.target.tagName);
    if (e.key === "Escape") {  // what is open closes; with nothing open, the selection clears
      const open = openMenus.some(Boolean) || !$("cli").hidden || app.toolActive();
      closeMenus(); if (!$("cli").hidden) closeCLI(); if (app.toolActive()) app.dropTool(); app.message("");
      if (!open && !typing && app.selection().length) { app.clearSelection(); app.rebuild(); app.onSelection(); }
      return;
    }
    if (typing) return;
    const ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && e.key.toLowerCase() === "c" && !String(window.getSelection())) { e.preventDefault(); copySmiles(); return; }
    if (ctrl && e.key.toLowerCase() === "z") { e.preventDefault(); undo(e.shiftKey); }
    else if (ctrl && e.key.toLowerCase() === "y") { e.preventDefault(); undo(true); }
    else if (ctrl && e.key.toLowerCase() === "a") { e.preventDefault(); selectWhere(() => true); }
    else if (ctrl && e.key.toLowerCase() === "o") { e.preventDefault(); openPanel(); }
    else if (ctrl && e.key.toLowerCase() === "s") { e.preventDefault(); savePanel(); }
    else if (e.key === " ") { e.preventDefault(); openCLI(); }
    else if (e.key === "`") { e.preventDefault(); togglePython(); }
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSelected(); }
    else if (e.key === "ArrowRight") app.setFrame(state.frame + 1);
    else if (e.key === "ArrowLeft") app.setFrame(state.frame - 1);
    else if (e.key === "ArrowDown" && app.live().length > 1) { e.preventDefault(); stepModel(1); }
    else if (e.key === "ArrowUp" && app.live().length > 1) { e.preventDefault(); stepModel(-1); }
    else if (e.key === "r" || e.key === "R") app.resetView();
    else if (e.key === "w" || e.key === "W") app.viewAll();
    else if (e.key === "F12") document.documentElement.requestFullscreen?.();
  });

  // --- the player -----------------------------------------------------------------------------------

  $("slider").oninput = () => app.setFrame(parseInt($("slider").value));
  $("play").onclick = () => app.play();
  window.addEventListener("resize", app.draw);

  // --- files to and from this computer -----------------------------------------------------------

  const DOWNLOADS = [["pdb", "PDB"], ["cif", "mmCIF"], ["sdf", "SDF (each molecule)"], ["mae", "Maestro (.mae)"],
                     ["dms", "DMS"], ["gro", "GROMACS (.gro)"]];
  // the selection's model and atoms (else the current model, whole) as a file in Downloads
  async function download(fmt, m = null, atoms = null) {
    const sel = app.selection();
    m = m || sel[0]?.[0] || app.currentModel();
    if (!m?.topo) { app.message("no model to download", true); return; }
    atoms = atoms ?? (sel.find(([x]) => x === m)?.[1] || []);
    try {
      const r = await fetch("api/download", { method: "POST", headers: { "Content-Type": "application/json", "X-Boonza-Token": app.TOKEN },
        body: JSON.stringify({ m: m.index, fmt, atoms, frame: Math.min(state.frame, m.topo.nframes - 1) }) });
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      const name = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "")?.[1] || `model.${fmt}`;
      el("a", { href: URL.createObjectURL(await r.blob()), download: name }).click();
      app.message(`downloaded ${name}${atoms.length && fmt !== "sdf" ? ` (${atoms.length} selected atoms)` : ""}`);
    } catch (e) { app.message(e.message, true); }
  }
  // files from this computer (dropped on the window, or picked): sent to boonza view and opened
  async function uploadFiles(files) {
    for (const f of files) {
      try {
        const r = await app.busy(`opening ${f.name}`, () => fetch("api/upload", { method: "POST", body: f,
          headers: { "X-Boonza-Token": app.TOKEN, "X-Filename": encodeURIComponent(f.name) } }));
        const out = await r.json();
        if (!r.ok) throw new Error(out.error || r.statusText);
        await addModels(out);
        app.message(`opened ${f.name}: ${out.added.length} model${out.added.length === 1 ? "" : "s"}`);
      } catch (e) { app.message(`${f.name}: ${e.message}`, true); }
    }
  }
  function browse() {
    const input = el("input", { type: "file", multiple: true });
    input.onchange = () => { panels.open?.remove(); delete panels.open; uploadFiles([...input.files]); };
    input.click();
  }
  if (!app.STATIC) {
    const files = e => [...(e.dataTransfer?.types || [])].includes("Files");
    window.addEventListener("dragover", e => { if (!files(e)) return; e.preventDefault(); $("stage").classList.add("dropping"); });
    window.addEventListener("dragleave", e => { if (!e.relatedTarget) $("stage").classList.remove("dropping"); });
    window.addEventListener("drop", e => {
      if (!files(e)) return;
      e.preventDefault();
      $("stage").classList.remove("dropping");
      uploadFiles([...e.dataTransfer.files]);
    });
  }

  // copy a model's atoms (a built ligand) into another model, where they are
  const copyIntoItems = (m, atoms) => () => {
    const others = app.live().filter(o => o !== m);
    return others.length ? others.map(o => ({ label: o.name, run: () => copyInto(m, o, atoms) }))
      : [{ label: "(no other model)", off: () => true }];
  };
  async function copyInto(m, o, atoms) {
    try {
      await app.load(o);
      const out = await app.post("api/copy_into", { m: m.index, into: o.index, atoms,
                                                     frame: Math.min(state.frame, m.topo.nframes - 1) });
      state.current = o.index;
      await app.reloadModel(o, true);
      await app.show([o]);
      app.message(`copied ${out.added} atoms of ${m.name} into ${o.name} · Edit | Undo with ${o.name} current takes them out`);
    } catch (e) { app.message(e.message, true); }
  }

  // Analyze | Ramachandran Plot: phi and psi of each residue of the current model at this
  // frame, over the regions residues usually take; the selected residues are marked, and
  // a click on a point selects its residue (Shift: adds or takes it away)
  const RAMA_CLUSTERS = [  // phi, psi, spread of each, weight: where residues are found
    [-63, -43, 15, 15, 1], [-120, 130, 25, 25, 0.8], [-65, 145, 15, 20, 0.7], [-90, 0, 20, 15, 0.3], [57, 47, 12, 12, 0.25],
    // and the broad bands around them that are allowed, if less often taken
    [-100, 30, 35, 55, 0.12], [-115, 160, 45, 40, 0.15], [-80, -40, 35, 30, 0.12], [65, 30, 18, 25, 0.08]];
  const ramaDensity = (phi, psi) => {
    const wrap = d => ((d + 540) % 360) - 180;
    let s = 0;
    for (const [p, q, sp, sq, w] of RAMA_CLUSTERS) s += w * Math.exp(-(wrap(phi - p) ** 2) / (2 * sp * sp) - (wrap(psi - q) ** 2) / (2 * sq * sq));
    return s;
  };
  const FAVORED = 0.35, ALLOWED = 0.06;
  // phi, psi of each residue with a backbone and bonded neighbors on both sides
  function phiPsi(m) {
    const T = m.topo, A = T.atoms, R = T.residues;
    const atom = (r, name) => T.residueAtoms[r].find(a => A.name[a] === name);
    const close = (a, b) => a !== undefined && b !== undefined && Math.hypot(...app.xyz(m, a).map((x, d) => x - app.xyz(m, b)[d])) < 2.0;
    const out = [];
    for (let r = 1; r < R.name.length - 1; r++) {
      if (R.kind[r] !== 1) continue;
      const n = atom(r, "N"), ca = atom(r, "CA"), c = atom(r, "C"), cp = atom(r - 1, "C"), nn = atom(r + 1, "N");
      if ([n, ca, c].some(a => a === undefined) || !close(cp, n) || !close(c, nn)) continue;
      out.push({ r, phi: app.measure(m, [cp, n, ca, c]).value, psi: app.measure(m, [n, ca, c, nn]).value });
    }
    return out;
  }
  let ramaRefresh = null;
  app.onFrame = () => ramaRefresh?.();
  function ramaPanel() {
    const m = app.selection()[0]?.[0] || app.currentModel();
    if (!m?.topo) { app.message("no model", true); return; }
    if (m.topo.cg) { app.message("a Ramachandran plot needs an all-atom protein (N, CA and C of each residue)", true); return; }
    const W = 330, H = 330, L = 34, B = 26, T0 = 8, R0 = 8;
    const x = phi => L + (phi + 180) / 360 * (W - L - R0), y = psi => T0 + (180 - psi) / 360 * (H - T0 - B);
    // the regions, drawn once as a small picture the plot stretches: favored darker than allowed
    const N = 72, pic = el("canvas", { width: N, height: N }), g = pic.getContext("2d"), img = g.createImageData(N, N);
    const accent = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#f0a23b";
    const rgb = /^#([0-9a-f]{6})$/i.test(accent) ? [0, 2, 4].map(k => parseInt(accent.slice(1 + k, 3 + k), 16)) : [240, 162, 59];
    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const d = ramaDensity(-180 + (col + 0.5) * 360 / N, 180 - (row + 0.5) * 360 / N), k = 4 * (row * N + col);
        img.data.set([...rgb, d >= FAVORED ? 95 : d >= ALLOWED ? 38 : 0], k);
      }
    }
    g.putImageData(img, 0, 0);
    const cells = `<image href="${pic.toDataURL()}" x="${L}" y="${T0}" width="${W - L - R0}" height="${H - T0 - B}" preserveAspectRatio="none"/>`;
    const ticks = [-180, -90, 0, 90, 180].map(v => `<line x1="${x(v)}" x2="${x(v)}" y1="${T0}" y2="${H - B}" class="grid"/>`
      + `<line x1="${L}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}" class="grid"/>`
      + `<text x="${x(v)}" y="${H - 10}" text-anchor="middle">${v}</text><text x="${L - 4}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join("");
    panel("rama", "Ramachandran Plot", body => {
      const head = el("div", { className: "help", style: "margin:0 0 6px" });
      const box = el("div", { className: "plot rama" });
      const legend = el("div", { className: "help" });
      body.append(head, box, legend);
      const draw = () => {
        if (!m.topo || m.closed) return;
        const pts = phiPsi(m), A = m.topo.atoms, Rr = m.topo.residues;
        const picked = new Set(app.selected(m).map(a => A.residue[a]));
        const kind = r => Rr.name[r] === "GLY" ? "gly" : Rr.name[r] === "PRO" ? "pro" : "res";
        const dots = pts.map(p => `<circle cx="${x(p.phi).toFixed(1)}" cy="${y(p.psi).toFixed(1)}" r="${picked.has(p.r) ? 3.6 : 2.6}" class="${kind(p.r)}${picked.has(p.r) ? " on" : ""}" data-r="${p.r}"><title>${Rr.name[p.r]} ${Rr.resid[p.r]}${Rr.insertion[p.r].trim()} ${m.topo.chains.name[Rr.chain[p.r]] || ""}: φ ${p.phi.toFixed(0)}°, ψ ${p.psi.toFixed(0)}°</title></circle>`).join("");
        box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${cells}${ticks}<text x="${(L + W) / 2}" y="${H}" text-anchor="middle">φ</text><text x="9" y="${(T0 + H - B) / 2}" text-anchor="middle">ψ</text>${dots}</svg>`;
        for (const c of box.querySelectorAll("circle")) c.onclick = e => app.select(m, m.topo.residueAtoms[+c.dataset.r], e.shiftKey ? "toggle" : "set");
        const general = pts.filter(p => kind(p.r) === "res");
        const fav = general.filter(p => ramaDensity(p.phi, p.psi) >= FAVORED).length;
        const out = general.filter(p => ramaDensity(p.phi, p.psi) < ALLOWED).length;
        head.textContent = `${m.name}${app.nframes() > 1 ? `, frame ${state.frame + 1}` : ""}: ${pts.length} residues`
          + (picked.size ? `, ${pts.filter(p => picked.has(p.r)).length} selected` : "")
          + (general.length ? ` · ${Math.round(100 * fav / general.length)}% favored, ${out} outside (glycine and proline aside)` : "");
      };
      legend.innerHTML = '<span class="key res"></span> residue <span class="key gly"></span> glycine <span class="key pro"></span> proline · '
        + "shaded: where residues are usually found (darker: favored), a rough guide rather than a validation. "
        + "Click a point to select its residue, Shift to add it.";
      draw();
      ramaRefresh = () => { if (box.isConnected) draw(); else ramaRefresh = null; };
    });
  }

  // Analyze | Hydrogen Bonds: between two selections, drawn green and dashed
  const HBOND_SETS = [["@ligand", "the ligand and the rest"], ["@sel", "the selection and the rest"],
    ["@chains", "chains (every interface)"], ["protein|water", "protein and water"], ["custom", "two selections…"]];
  function hbondPanel() {
    panel("hbonds", "Hydrogen Bonds", body => {
      const kind = el("select");
      for (const [v, l] of HBOND_SETS) kind.add(new Option(l, v));
      if (app.selection().length) kind.value = "@sel";
      const a = el("input", { placeholder: "selection A, e.g. chain A", spellcheck: false, style: "width:220px" });
      const b = el("input", { placeholder: "selection B (empty: the rest)", spellcheck: false, style: "width:220px" });
      const custom = [row(el("span", { className: "what" }, "A"), a), row(el("span", { className: "what" }, "B"), b)];
      const sync = () => custom.forEach(r => { r.style.display = kind.value === "custom" ? "" : "none"; });
      kind.onchange = sync;
      const result = el("p", { className: "help" });
      const go = button("Show", async () => {
        const v = kind.value;
        state.hbonds = v === "@sel" ? { sel: true } : v === "custom" ? { a: a.value.trim(), b: b.value.trim() || null }
          : v === "protein|water" ? { a: "protein", b: "water" } : { a: v, b: null };
        if (v === "custom" && !state.hbonds.a) { app.message("give selection A", true); return; }
        if (v === "@sel" && !app.selection().length) { app.message("select atoms first", true); return; }
        const { n, how } = await app.busy("finding hydrogen bonds", () => app.refreshHbonds());
        // the residues at both ends, so no bond hangs in the air: in sticks, those not drawn yet
        const ends = app.shownModels().map(m => {
          const ids = new Set();
          for (const [i, j] of m.hbonds || []) {
            for (const r of [m.topo.atoms.residue[i], m.topo.atoms.residue[j]]) {
              for (const a of m.topo.residueAtoms[r]) if (!m.state.show[a]) ids.add(a);
            }
          }
          return [m, [...ids]];
        });
        if (ends.some(([, ids]) => ids.length)) addRep(ends, { style: "sticks", label: "hydrogen-bonded residues" });
        else app.rebuild();
        result.textContent = `${n} hydrogen bond${n === 1 ? "" : "s"} (${how})`;
        app.message(result.textContent);
      }, "blue");
      const clear = button("Clear", () => {
        state.hbonds = null;
        for (const m of app.live()) m.hbonds = [];
        app.rebuild(); result.textContent = "";
      });
      body.append(row(el("span", { className: "what" }, "between"), kind), ...custom, row(go, clear), result);
      body.append(el("p", { className: "help" }, "With hydrogens: donor-acceptor within 3.5 Å and D-H…A above 120°. "
        + "Without (most PDB entries): N and O of different residues within 3.5 Å; Edit | Hydrogens | "
        + "Add Hydrogens adds them. Followed frame by frame."));
      sync();
    }, [innerWidth / 2 - 180, 80]);
  }

  // --- panels -----------------------------------------------------------------------------------------

  const panels = {};
  function panel(id, title, build, where = null) {
    if (panels[id]) { panels[id].remove(); delete panels[id]; }
    const p = el("div", { className: "panel" });
    const close = el("button", { title: "close" }, "✕");
    close.onclick = () => { p.remove(); delete panels[id]; };
    const bar = el("div", { className: "title" }, el("span", {}, title), close);
    const body = el("div", { className: "body" });
    p.append(bar, body);
    build(body, p);
    $("panels").append(p);
    // by default beside the button bar, anchored on the right so a panel that grows
    // (a plot arriving) stays on the screen
    if (where) { p.style.left = `${where[0]}px`; p.style.top = `${where[1]}px`; }
    else { p.style.right = "96px"; p.style.top = `${40 + Object.keys(panels).length * 28}px`; }
    let start = null;
    bar.onpointerdown = e => {
      if (e.target.closest("button")) return;  // the close button: a click, not a drag
      start = [e.clientX - p.offsetLeft, e.clientY - p.offsetTop];
      p.style.left = `${p.offsetLeft}px`; p.style.right = "auto";
      bar.setPointerCapture(e.pointerId);
    };
    bar.onpointermove = e => { if (start) { p.style.left = `${e.clientX - start[0]}px`; p.style.top = `${e.clientY - start[1]}px`; } };
    bar.onpointerup = () => { start = null; };
    panels[id] = p;
    return p;
  }
  const row = (...kids) => el("div", { className: "row" }, ...kids);
  const button = (label, run, cls = "") => { const b = el("button", { className: cls }, label); b.onclick = run; return b; };

  // --- the builder, docked on the left: atoms and groups; amino acids and caps ---------------

  const AMINO = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"];
  let dockTab = "builder";

  function openDock(tab = dockTab) {
    dockTab = tab;
    $("ldock").hidden = false;
    $("window").classList.add("docked");
    for (const b of document.querySelectorAll("#ldock-tabs [data-tab]")) b.classList.toggle("on", b.dataset.tab === tab);
    const body = $("ldock-body");
    body.innerHTML = "";
    (tab === "protein" ? proteinBuilder : atomBuilder)(body);
    if (tab === "builder") {  // the hydrogens are where molecules grow: those of shown molecules too
      setHydrogens("all");
      for (const m of app.shownModels()) {
        const A = m.topo.atoms;
        for (const rep of m.reps) {
          if (app.isRibbon(rep) || rep.style === "surface") continue;
          let grew = false;
          for (let a = 0; a < m.topo.natoms; a++) {
            if (A.element[a] !== "H" || A.class[a] !== 1 || rep.mask[a]) continue;
            const heavy = m.topo.bondsOf[a][0];
            if (heavy !== undefined && rep.mask[heavy]) { rep.mask[a] = 1; grew = true; }
          }
          if (grew) app.colorRep(m, rep);
        }
        app.aggregate(m);
      }
      app.rebuild();
    }
    app.draw();
  }
  function closeDock() { setTool(null); $("ldock").hidden = true; $("window").classList.remove("docked"); app.draw(); }
  function toggleDock(tab) { if (!$("ldock").hidden && dockTab === tab) closeDock(); else openDock(tab); }
  for (const b of document.querySelectorAll("#ldock-tabs [data-tab]")) b.onclick = () => openDock(b.dataset.tab);
  $("ldock-close").onclick = closeDock;
  const grid = (cls = "grid") => el("div", { className: cls });

  // Draw tools: pick one, then click in the view -- a hydrogen grows the molecule from
  // it, an atom is changed, a bond takes the order, empty space starts a new molecule.
  // With atoms selected, a tool acts on them at once instead.
  const ELEMENT_SMILES = { C: "C", N: "N", O: "O", S: "S", P: "P", F: "F", Cl: "Cl", Br: "Br", I: "I" };
  const RINGS = [["Benzene", "c1ccccc1"], ["Pyridine", "c1ccncc1"], ["Pyrimidine", "c1cncnc1"], ["Pyrrole", "c1cc[nH]c1"],
    ["Imidazole", "c1c[nH]cn1"], ["Pyrazole", "c1cn[nH]c1"], ["Thiophene", "c1ccsc1"], ["Furan", "c1ccoc1"],
    ["Cyclopropane", "C1CC1"], ["Cyclopentane", "C1CCCC1"], ["Cyclohexane", "C1CCCCC1"], ["Piperidine", "C1CCNCC1"],
    ["Piperazine", "C1CNCCN1"], ["Morpholine", "C1COCCN1"], ["Naphthalene", "c1ccc2ccccc2c1"], ["Indole", "c1ccc2[nH]ccc2c1"]];
  let tool = null;  // {kind: element|smiles|bond|charge|erase, value, label}

  function setTool(next) {
    const same = tool && next && tool.kind === next.kind && tool.value === next.value;
    tool = same ? null : next;
    for (const b of document.querySelectorAll("#ldock [data-tool]")) b.classList.toggle("armed", !!tool && b.dataset.tool === `${tool.kind}:${tool.value}`);
    $("stage").classList.toggle("drawing", !!tool);
    if (tool) app.message(toolHelp(tool));
    else app.message("");
  }
  function toolHelp(t) {
    if (t.kind === "bond") return `${t.label}: click a bond (Esc: done)`;
    if (t.kind === "charge") return `charge ${t.label}: click an atom (Esc: done)`;
    if (t.kind === "erase") return "Erase: click an atom or a bond to remove it (Esc: done)";
    return `${t.label}: click a hydrogen to grow from it, ${t.kind === "element" ? "an atom to make it " + t.value + ", " : "an atom to attach it, "}`
      + "or empty space for a new molecule (Esc: done)";
  }
  app.toolActive = () => !!tool;
  app.dropTool = () => setTool(null);

  // a tool's button: with atoms selected it acts on them, else it is picked up
  function toolButton(label, next, cls = "") {
    const b = button(label, () => {
      const sel = app.selection();
      if (sel.length && next.kind !== "erase") { applyToAtoms(next, sel); return; }
      setTool(next);
    }, cls);
    b.dataset.tool = `${next.kind}:${next.value}`;
    if (tool && tool.kind === next.kind && tool.value === next.value) b.classList.add("armed");
    return b;
  }

  async function applyToAtoms(t, sel) {
    const [m, atoms] = sel[0];
    if (t.kind === "element") await serverEdit(m, "api/edit", { op: "element", atoms, value: t.value, hydrogens: autoH() });
    else if (t.kind === "charge") await serverEdit(m, "api/edit", { op: "charge", atoms, value: t.value, hydrogens: autoH() });
    else if (t.kind === "bond") {
      if (atoms.length !== 2) { app.message("select the bond's two atoms (or click a bond with the tool)", true); return; }
      await serverEdit(m, "api/edit", { op: "bond", atoms, value: t.value, hydrogens: autoH() });
    } else if (t.kind === "smiles") await serverEdit(m, "api/edit", { op: "attach", atoms: [atoms[0]], value: t.value, hydrogens: false });
  }

  // the model new molecules go into: the current one if it can take them, else a new one
  const canTake = m => m && !m.closed && m.shown && m.topo?.editable;
  async function newMolecule(smiles, center, intoNew = false) {
    const m = app.currentModel();
    try {
      if (!intoNew && canTake(m)) {
        await serverEdit(m, "api/edit", { op: "molecule", value: { smiles, centre: center } });
      } else {
        const out = await app.busy("building", () => app.post("api/build", { smiles, centre: center }));
        await addModels(out, false);
        setHydrogens("all");
      }
      app.message(`added ${smiles}: grow it by clicking its hydrogens with a tool`);
    } catch (e) { app.message(e.message, true); }
  }

  // a click in the view while a tool is picked up: true when the tool took it
  app.applyTool = (hit, x, y) => {
    if (!tool) return false;
    const t = tool;
    (async () => {
      if (!hit) {
        if (t.kind === "element" || t.kind === "smiles") await newMolecule(t.kind === "element" ? ELEMENT_SMILES[t.value] || t.value : t.value, app.unproject(x, y));
        return;
      }
      const m = hit.model;
      if (!m.topo.editable) { app.message("this model cannot be edited (a trajectory or a set of points)", true); return; }
      const A = m.topo.atoms;
      let atom = hit.atom;
      if (atom === undefined && hit.atoms) {
        if (t.kind === "bond" || t.kind === "erase") {
          await serverEdit(m, "api/edit", { op: "bond", atoms: hit.atoms, value: t.kind === "erase" ? 0 : t.value, hydrogens: autoH() });
          return;
        }
        atom = app.nearerAtom(hit.model, hit.atoms, x, y);
      }
      if (atom === undefined) { app.message("click an atom or a bond", true); return; }
      if (t.kind === "bond") { app.message("click a bond (between two atoms), not an atom", true); return; }
      if (t.kind === "erase") { await serverEdit(m, "api/edit", { op: "delete", atoms: [atom], hydrogens: autoH() }); return; }
      if (t.kind === "charge") { await serverEdit(m, "api/edit", { op: "charge", atoms: [atom], value: t.value, hydrogens: autoH() }); return; }
      if (t.kind === "element" && A.element[atom] !== "H") {
        await serverEdit(m, "api/edit", { op: "element", atoms: [atom], value: t.value, hydrogens: autoH() });
        return;
      }
      const smiles = t.kind === "element" ? ELEMENT_SMILES[t.value] : t.value;
      if (!smiles) { app.message("hydrogens grow from atoms: pick C, N, O or a group", true); return; }
      await serverEdit(m, "api/edit", { op: "attach", atoms: [atom], value: smiles, hydrogens: false });
    })();
    return true;
  };

  // SMILES of the selected molecules (else of the current model's ligands)
  async function smilesOf(m, ids = []) {
    return (await app.getJSON(`api/smiles?m=${m.index}&sel=${ids.slice(0, 5000).join(",")}`)).molecules;
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) {
      const ta = el("textarea", { value: text });
      document.body.append(ta); ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    }
  }
  async function copySmiles() {
    const sel = app.selection();
    const m = sel[0]?.[0] || app.currentModel();
    if (!m?.topo) { app.message("no molecule: build or open one", true); return; }
    try {
      const mols = (await smilesOf(m, sel[0]?.[1] || [])).filter(x => x.smiles);
      if (!mols.length) { app.message("no molecule with a SMILES here: select one (double-click picks a molecule)", true); return; }
      const text = mols.map(x => x.smiles).join("\n");
      app.message((await copyText(text)) ? `copied: ${mols.map(x => x.smiles).join("  ")}` : `could not reach the clipboard: ${text}`);
    } catch (e) { app.message(e.message, true); }
  }
  app.copySmiles = copySmiles;

  // the builder's SMILES box: what is selected, or the model's molecules
  let smilesTicket = 0;
  async function refreshSmilesBox() {
    const box = document.querySelector("#ldock .smiles-box");
    if (!box) return;
    const ticket = ++smilesTicket;
    const sel = app.selection();
    const m = sel[0]?.[0] || app.currentModel();
    box.innerHTML = "";
    if (!m?.topo || m.closed) { box.append(el("p", { className: "help" }, "Nothing built yet.")); return; }
    let mols;
    try { mols = await smilesOf(m, sel[0]?.[1] || []); } catch (e) { mols = []; }
    if (ticket !== smilesTicket) return;
    if (!mols.length) { box.append(el("p", { className: "help" }, "Select a molecule (double-click it) to see its SMILES.")); return; }
    for (const x of mols.slice(0, 12)) {
      const text = el("code", { title: x.smiles || x.note }, x.smiles || x.note);
      const copy = button("Copy", async () => app.message((await copyText(x.smiles)) ? `copied ${x.smiles}` : x.smiles));
      copy.disabled = !x.smiles;
      box.append(el("div", { className: "smiles-row" }, el("span", { className: "what" }, `${x.name} · ${x.heavy}`), text, copy));
      if (x.smiles && x.note) box.append(el("p", { className: "help" }, x.note));
    }
    box.append(row(
      ...(mols.length > 1 ? [button("Copy all", async () => {
        const all = mols.filter(x => x.smiles).map(x => x.smiles).join("\n");
        app.message((await copyText(all)) ? `copied ${mols.length} SMILES` : all);
      })] : []),
      button("Download SDF", () => download("sdf", m, sel[0]?.[1] || []))));
  }

  function atomBuilder(body) {
    body.append(el("h3", {}, "SMILES"));
    body.append(el("div", { className: "smiles-box" }));
    refreshSmilesBox();
    body.append(el("p", { className: "help" }, "Pick a tool, then click in the view: a hydrogen grows the molecule "
      + "from it, an atom changes, empty space starts a new molecule. With atoms selected, a tool acts on them. Esc puts the tool down."));
    body.append(el("h3", {}, "Atoms"));
    const g = grid();
    for (const e of ELEMENTS) if (e !== "H") g.append(toolButton(e, { kind: "element", value: e, label: e }));
    g.append(toolButton("H", { kind: "element", value: "H", label: "H" }));
    g.append(toolButton("Erase", { kind: "erase", value: "", label: "Erase" }));
    body.append(g);
    const other = el("input", { placeholder: "other element (e.g. Se)", spellcheck: false });
    body.append(row(other, button("Pick", () => other.value.trim() && setTool({ kind: "element", value: other.value.trim(), label: other.value.trim() }))));
    body.append(el("h3", {}, "Hydrogens, geometry"));
    body.append(row(button("Add / Adjust H", () => hydrogens("add")), button("Delete H", () => hydrogens("delete")),
                    button("Clean Up", cleanUp)));
    body.append(row(button("Delete selected", deleteSelected), button("Undo", () => undo()), button("Redo", () => undo(true))));
    const auto = el("input", { type: "checkbox", checked: autoH() });
    auto.onchange = () => localStorage.setItem("boonza.autoH", auto.checked ? "1" : "0");
    body.append(row(el("label", {}, auto, " adjust hydrogens after each edit")));
    body.append(el("h3", {}, "Bonds and charges"));
    body.append(row(...[["Single", 1], ["Double", 2], ["Triple", 3]].map(([l, v]) => toolButton(l, { kind: "bond", value: v, label: `${l} bond` }))));
    body.append(row(...[["−1", -1], ["0", 0], ["+1", 1]].map(([l, v]) => toolButton(l, { kind: "charge", value: v, label: l }))));
    body.append(el("h3", {}, "Rings"));
    const rings = grid("grid wide");
    for (const [n, smi] of RINGS) {
      const b = toolButton("", { kind: "smiles", value: smi, label: n }, "group");
      b.title = `${n}: ${smi}`;
      b.append(el("span", {}, n), el("small", {}, smi));
      rings.append(b);
    }
    body.append(rings);
    body.append(el("h3", {}, "Functional groups"));
    const fg = grid("grid wide");
    for (const [n, smi] of Object.entries(app.state.groups || {})) {  // the name, its SMILES below
      const b = toolButton("", { kind: "smiles", value: smi, label: n }, "group");
      b.title = `${n}: ${smi} (the first atom bonds)`;
      b.append(el("span", {}, n), el("small", {}, smi));
      fg.append(b);
    }
    body.append(fg);
    const smiles = el("input", { placeholder: "a group as SMILES, first atom bonds", spellcheck: false });
    body.append(row(smiles, button("Pick", () => smiles.value.trim() && setTool({ kind: "smiles", value: smiles.value.trim(), label: smiles.value.trim() }))));
    body.append(el("h3", {}, "New molecule from SMILES"));
    const mol = el("input", { placeholder: "e.g. CC(=O)Oc1ccccc1C(=O)O", spellcheck: false });
    const fresh = el("input", { type: "checkbox", checked: !canTake(app.currentModel()) || app.currentModel()?.group !== "built" });
    const add = button("Add", () => mol.value.trim() && newMolecule(mol.value.trim(), app.camera.center.slice(), fresh.checked));
    mol.onkeydown = e => { if (e.key === "Enter") add.click(); };
    body.append(row(mol, add), row(el("label", {}, fresh, " as a new model")));
    body.append(el("p", { className: "help" }, "Clean Up relaxes the selected molecules (else every small molecule) "
      + "with MMFF94, alone. Alt + middle drag moves the selected atoms, Alt + left drag turns a selected bond."));
  }

  async function cleanUp() {
    const m = actingModel();
    if (!m) return;
    const sel = app.selected(m);
    const out = await busyEdit(m, "cleaning up", { op: "cleanup", atoms: sel });
    if (out) app.message(`cleaned up ${out.molecules} molecule${out.molecules === 1 ? "" : "s"} (${out.forcefield})`);
  }

  function proteinBuilder(body) {
    const sel = app.selection();
    const residues = [];
    for (const [m, ids] of sel) {
      const seen = new Set(ids.map(a => m.topo.atoms.residue[a]));
      for (const r of seen) residues.push(`${m.topo.residues.name[r]}${m.topo.residues.resid[r]}`);
    }
    body.append(el("h3", {}, "Mutate"));
    body.append(el("p", { className: "help" }, residues.length
      ? `Selected: ${residues.slice(0, 8).join(", ")}${residues.length > 8 ? "…" : ""}`
      : "Select an atom of each residue to mutate (Ctrl-click takes the residue)."));
    const g = grid();
    for (const aa of AMINO) g.append(button(aa, () => mutate(aa)));
    body.append(g);
    body.append(el("p", { className: "help" }, "The backbone stays; the side chain is built and turned to the "
      + "rotamer that clashes least; Lys, Arg, Asp and Glu charged as at pH 7."));
    body.append(el("h3", {}, "Caps"));
    body.append(row(button("ACE (N terminus)", () => capChains("N")), button("NME (C terminus)", () => capChains("C"))));
    body.append(row(button("Both ends", () => capChains("both"))));
    body.append(el("p", { className: "help" }, "On the chains of the selected atoms, or every chain if none are "
      + "selected: a trans peptide bond, Amber's names (ACE: C O CH3; NME: N H CH3)."));
  }
  async function mutate(aa) {
    const t = editTarget();
    if (!t) return;
    const out = await busyEdit(t[0], `mutating to ${aa}`, { op: "mutate", atoms: t[1], value: aa });
    if (out) { app.message(`mutated to ${aa}`); openDock("protein"); }
  }
  async function capChains(end) {
    const sel = app.selection();
    const m = sel[0]?.[0] || app.currentModel();
    const out = await busyEdit(m, "capping", { op: "cap", atoms: sel[0]?.[1] || [], value: end });
    if (out) app.message(end === "both" ? "capped with ACE and NME" : `capped with ${end === "N" ? "ACE" : "NME"}`);
  }
  async function busyEdit(m, what, body) {
    try {
      const out = await app.busy(what, () => app.post("api/edit", { m: m.index, ...body }));
      await app.reloadModel(m, true);
      return out;
    } catch (e) { app.message(e.message, true); return null; }
  }
  function builderPanel() { toggleDock("builder"); }

  // the selected residues' name, number and chain (Edit | Rename Residue or Chain)
  function relabelPanel() {
    const t = editTarget();
    if (!t) return;
    const [m, atoms] = t, R = m.topo.residues;
    const rs = [...new Set(atoms.map(a => m.topo.atoms.residue[a]))].sort((a, b) => a - b);
    const common = f => { const v = new Set(rs.map(f)); return v.size === 1 ? [...v][0] : ""; };
    const number = `${R.resid[rs[0]]}${R.insertion[rs[0]].trim()}`;
    const what = rs.length === 1 ? `${R.name[rs[0]]}${number}` : `${rs.length} residues`;
    panel("relabel", "Residue and Chain", body => {
      const field = v => el("input", { value: v, placeholder: "keep", style: "width:70px" });
      const name = field(common(r => R.name[r])), resid = field(number);
      const chain = field(common(r => m.topo.chains.name[R.chain[r]]));
      body.append(el("p", { className: "help" }, `${what} of ${m.name}. A blank field stays as it is; `
        + "more than one residue are numbered on from the number. A chain name no other chain has "
        + "renames the chains the residues fill; otherwise the residues move to that chain."));
      body.append(row(el("span", { className: "what" }, "Residue name"), name));
      body.append(row(el("span", { className: "what" }, "Residue number"), resid));
      body.append(row(el("span", { className: "what" }, "Chain"), chain));
      const apply = async () => {
        // the number as it was is not sent: it would renumber the residues after the first
        const value = { name: name.value, resid: resid.value === number ? "" : resid.value, chain: chain.value };
        const out = await serverEdit(m, "api/edit", { op: "residue", atoms, value, hydrogens: false });
        if (out) app.message(`${what}: relabelled`);
      };
      for (const f of [name, resid, chain]) {
        f.onkeydown = e => { if (e.key === "Enter") apply(); };
        f.disabled = !m.topo.editable;
      }
      body.append(row(button("Apply", apply, "blue")));
    });
  }

  function minimizePanel() {
    panel("minimize", "Energy Minimize", body => {
      const m = app.currentModel();
      const c = constraintCounts(m);
      const k = el("input", { type: "number", value: localStorage.getItem("boonza.restraint") || 10, min: 0, step: 1, style: "width:60px" });
      k.onchange = () => localStorage.setItem("boonza.restraint", k.value);
      const spring = row(el("span", { className: "what" }, "Restraint"), k, el("span", {}, "kcal/mol/Å²"));
      if (c.fixed || c.tethered) {
        body.append(el("p", { className: "help" }, `Restraints: ${c.fixed} frozen, ${c.tethered} restrained `
          + "(Edit | Restraints); the rest is free. Edit | Restraints | Free All goes back to the choices below."));
        body.append(spring, row(button("Minimize", () => minimize({ restraint: +k.value }), "blue")));
      } else {
        let mode = localStorage.getItem("boonza.minMode") || "both";
        const choices = MIN_MODES.map(([key, label]) => {
          const b = button(label, () => {
            mode = key;
            localStorage.setItem("boonza.minMode", key);
            choices.forEach(x => x.classList.toggle("armed", x === b));
          });
          b.classList.toggle("armed", key === mode);
          return b;
        });
        body.append(el("p", { className: "help" }, "What moves. The heavy atoms of the rest are held by springs: they barely "
          + "move, and a crystal structure's strain can relax (frozen, it stays, and so does an energy of 10⁵ kJ/mol). "
          + "The pocket is the whole residues within 5 Å of the ligand."));
        body.append(row(...choices), spring);
        body.append(row(button("Minimize", () => minimize({ mode, restraint: +k.value }), "blue")));
        if (m?.receptor !== undefined) body.append(el("p", { className: "help" }, "A design is minimized in its receptor "
          + `(${state.models[m.receptor]?.name ?? "none"}); it keeps its new positions, the receptor its own.`));
      }
      const rem = el("input", { type: "checkbox", checked: localStorage.getItem("boonza.rem") === "1" });
      rem.onchange = () => localStorage.setItem("boonza.rem", rem.checked ? "1" : "0");
      body.append(row(el("label", { title: "for a structure that clashes: Lennard-Jones and Coulomb made soft and bounded "
        + "(A 100, C 50 kJ/mol, as mstool's REM), then the force field; nothing holds chirality meanwhile" },
        rem, " REM first (soft potentials, for clashes)")));
      body.append(el("p", { className: "help" }, "Hydrogens are never held; a structure without any (or a design's receptor) "
        + "gets them first, as Add / Adjust H adds them. Force field: Amber ff19SB, "
        + "GAFF2 for ligands, AM1-BCC charges at a conformer built from the bonds (the same in every pose); "
        + "vacuum with a 12 Å reaction field. The minimization plays back after."));
    });
  }

  async function profilePanel(m = null, atoms = null) {
    if (!m) {
      const t = editTarget(2);
      if (!t) return;
      [m, atoms] = t;
      if (atoms.length !== 2 && atoms.length !== 4) { app.message("select a bond (2 atoms) or 4 atoms", true); return; }
    }
    const p = panel("profile", "Torsion Profile", body => { body.append(el("p", { className: "help" }, "computing…")); });
    const body = p.querySelector(".body");
    try {
      const d = await app.post("api/dihedral", { m: m.index, atoms: orderChain(m, atoms) });
      const scan = await app.busy("torsion profile", () => app.post("api/dihedral_scan", { m: m.index, atoms: d.atoms, step: 10 }));
      body.innerHTML = "";
      const name = d.name ? `${d.name.replace("chi", "χ")}: ` : "";
      body.append(el("div", {}, `${name}${d.atoms.map(a => m.topo.atoms.name[a]).join("–")}`));
      body.append(plot(scan, async angle => {
        await serverEdit(m, "api/set_dihedral", { atoms: d.atoms, angle }, false);
        marker(angle);
      }));
      const alone = d.atoms && scan.alone_atoms ? ` (${scan.alone_atoms} atoms)` : "";
      body.append(el("p", { className: "help" }, "Energy (kcal/mol, from each curve's lowest) as the bond turns, "
        + `the rest rigid. Orange: in its surroundings; red bands: clashes, not allowed. Blue dotted: the residue `
        + `or molecule alone${alone}. Yellow dashes: the angle now; green: set. Click to set the angle.`));
      app.message("");
    } catch (e) { body.innerHTML = ""; body.append(el("p", { className: "help" }, e.message)); }
  }

  let marker = () => {};
  // two curves: in context (the rest rigid; angles that clash are forbidden,
  // shaded) and the residue or molecule alone (dotted)
  function plot(scan, onPick) {
    const W = 340, H = 190, L = 34, R = 6, T = 8, B = 22;
    const iso = scan.isolated || scan.torsion, ctx = scan.total;
    const finite = [...iso, ...ctx].filter(v => v < 40);
    const top = Math.min(40, Math.max(5, Math.ceil(Math.max(...finite, 1) * 1.25)));
    const x = a => L + (a + 180) / 360 * (W - L - R);
    const y = e => T + (1 - Math.min(e, top) / top) * (H - T - B);
    const step = scan.angles.length > 1 ? scan.angles[1] - scan.angles[0] : 10;
    // a curve, broken where it leaves the scale
    const path = vs => {
      let d = "", pen = false;
      vs.forEach((e, k) => {
        if (e > top) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(scan.angles[k]).toFixed(1)},${y(e).toFixed(1)}`;
        pen = true;
      });
      return d;
    };
    const forbidden = ctx.map((e, k) => e > top
      ? `<rect x="${x(scan.angles[k] - step / 2).toFixed(1)}" y="${T}" width="${(x(step) - x(0)).toFixed(1)}" height="${H - T - B}" fill="rgba(229,57,53,0.18)"/>` : "").join("");
    const ticks = [-180, -90, 0, 90, 180].map(a => `<line x1="${x(a)}" x2="${x(a)}" y1="${T}" y2="${H - B}" stroke="rgba(128,128,128,0.3)"/>`
      + `<text x="${x(a)}" y="${H - 8}" text-anchor="middle">${a}°</text>`).join("");
    const div = el("div", { className: "plot" });
    div.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${forbidden}${ticks}<text x="4" y="${T + 8}">${top}</text><text x="4" y="${H - B}">0</text>
      <path d="${path(ctx)}" fill="none" stroke="#f08c00" stroke-width="1.8"/>
      <path d="${path(iso)}" fill="none" stroke="#3fa0ff" stroke-width="1.6" stroke-dasharray="4,3"/>
      <line class="now" y1="${T}" y2="${H - B}" stroke="#2e7d32" stroke-width="1.5"/>
      <line x1="${x(scan.current)}" x2="${x(scan.current)}" y1="${T}" y2="${H - B}" stroke="#d4a000" stroke-dasharray="3,3"/></svg>`;
    const svg = div.querySelector("svg");
    marker = angle => { const l = svg.querySelector(".now"); l.setAttribute("x1", x(angle)); l.setAttribute("x2", x(angle)); };
    marker(scan.current);
    svg.onclick = e => {
      const r = svg.getBoundingClientRect();
      const angle = Math.round(((e.clientX - r.left) / r.width * W - L) / (W - L - R) * 360 - 180);
      if (angle >= -180 && angle <= 180) onPick(angle);
    };
    return div;
  }

  // --- the side bar: the models (on/off, color, the table's data), and below them the
  // representations of the current one: each a selection, a style and a color, switched
  // on and off, edited, added and deleted one by one -------------------------------------

  // a preset: the model's representations replaced by these [atoms, style, color];
  // "keep" keeps the color the atoms have now (a site's points), "carbon" the ligand's
  // carbons in the model's own ligand color
  const PRESETS = [
    ["default", "Default", null],
    ["site", "Binding site", [["polymer", "cartoon", "white"], ["pocket", "sticks", "element"], ["ligand", "sticks", "carbon"]]],
    ["ribbon", "Cartoon", [["polymer", "cartoon", "ss"]]],
    ["ribbonlig", "Cartoon + ligand", [["polymer", "cartoon", "ss"], ["ligand", "sticks", "carbon"]]],
    ["ribbonpocket", "Cartoon + pocket lines", [["polymer", "cartoon", "white"], ["pocket", "lines", "element"], ["ligand", "sticks", "carbon"]]],
    ["ribbonlines", "Cartoon + lines", [["polymer", "cartoon", "ss"], ["nosolvent", "lines", "element"]]],
    ["ribbonspace", "Cartoon + ligand spheres", [["polymer", "cartoon", "ss"], ["ligand", "spheres", "carbon"]]],
    ["lig", "Ligand", [["ligand", "sticks", "keep"]]],
    ["ligspace", "Ligand spheres", [["ligand", "spheres", "keep"]]],
    ["line", "Lines", [["all", "lines", "element"]]],
    ["stick", "Sticks", [["all", "sticks", "element"]]],
    ["ballstick", "Ball & stick", [["all", "ballstick", "element"]]],
    ["space", "Spheres", [["all", "spheres", "element"]]],
    ["surface", "Surface + ligand", [["receptor", "surface", "electrostatic"], ["ligand", "sticks", "carbon"]]],
  ];
  const MODEL_COLORS = [["default", "Default"], ["chain", "By chain"], ...["green", "cyan", "magenta", "yellow",
    "orange", "salmon", "purple", "slate", "white", "gray"].map(c => [c, c])];

  async function presetAtoms(m, what) {
    const A = m.topo.atoms, R = m.topo.residues, all = app.range(m.topo.natoms);
    if (what === "all") return all;
    if (what === "nosolvent") return all.filter(a => A.class[a] !== 2);
    if (what === "receptor") return surfaceAtoms(m);
    if (what === "polymer") return all.filter(a => R.trace[A.residue[a]] >= 0);
    if (what === "ligand") return serverSelect(m, `same residue as (${m.topo.ligand})`);
    if (what === "pocket") return serverSelect(m, pocketSelection(m));
    return [];
  }
  const PRESET_SEL = { all: "all", nosolvent: "not solvent", ligand: null, pocket: null, polymer: null, receptor: null };
  const PRESET_LABEL = { all: "all", nosolvent: "all but solvent", ligand: "ligand", pocket: "binding site",
                         polymer: "polymer", receptor: "receptor" };

  async function preset(m, kind) {
    await app.load(m);
    const spec = PRESETS.find(([k]) => k === kind)?.[2];
    if (spec === undefined) return;
    const before = m.reps;
    m.reps = [];
    if (spec === null) {  // the model's own: what it came with
      app.applyReps(m, await Promise.all(m.topo.reps.map(async r => ({ ...r, ids: r.ids || await serverSelect(m, r.sel) }))));
    } else {
      for (const [what, style, color] of spec) {
        const ids = await presetAtoms(m, what);
        if (!ids.length) continue;
        let c = color;
        if (color === "carbon") c = `element/${m.topo.carbon || "green"}`;
        if (color === "keep") c = before.find(r => !app.isRibbon(r) && r.style !== "surface" && r.mask[ids[0]])?.color || "element";
        const rep = app.makeRep(m, { ids, sel: PRESET_SEL[what] || null, label: PRESET_LABEL[what], style, color: c });
        m.reps.push(rep);
        if (style === "surface" && c === "electrostatic") forceFieldCharges(m);
      }
    }
    m.currentRep = m.reps[0]?.id;
    app.repsChanged(m);
  }

  // a model's own color: every representation it has not been given a color of its own
  function colorModel(m, spec) {
    if (!m.state) return;
    m.color = spec;
    m.tintAll = true;
    for (const rep of m.reps) {
      if (!rep.tinted) continue;
      if (spec === "chain" && rep.color !== "chain") {
        if (app.isRibbon(rep)) rep.color = "chain";
        else if (rep.color.startsWith("element")) rep.color = "element/chain";
      }
      app.colorRep(m, rep);
    }
    app.repsChanged(m);
  }

  // the representations of the current model, and the one being edited
  const SCHEMES_ATOMS = [["element", "Element"], ["chain", "Chain"], ["restype", "Residue type"], ["resname", "Residue name"],
                         ["ss", "Secondary structure"], ["spectrum", "Spectrum (N to C)"], ["bfactor", "B-factor"]];
  const SCHEMES_RIBBON = [["ss", "Secondary structure"], ["chain", "Chain"], ["spectrum", "Spectrum (N to C)"],
                          ["restype", "Residue type"], ["resname", "Residue name"], ["bfactor", "B-factor"]];
  const SCHEMES_SURFACE = [["electrostatic", "Electrostatic"], ["hydrophobic", "Hydrophobic"]];
  const rgbCss = c => `rgb(${c.map(x => Math.round(x * 255)).join(",")})`;

  function chipColor(rep) {
    if (rep.custom) return "conic-gradient(#e53935, #ffd54f, #4d4, #3bd, #66f, #e5c, #e53935)";
    const c = rep.color, [scheme, carbon] = c.split("/");
    if (scheme === "element") return carbon && named(carbon) ? `linear-gradient(135deg, ${rgbCss(named(carbon))} 55%, #3355ff 55% 75%, #ff2a2a 75%)`
      : "linear-gradient(135deg, #999 55%, #3355ff 55% 75%, #ff2a2a 75%)";
    if (named(c)) return rgbCss(named(c));
    return { chain: "linear-gradient(135deg,#5c5 33%,#3bd 33% 66%,#e5c 66%)", ss: "linear-gradient(135deg,#e53935 50%,#ffd54f 50%)",
             spectrum: "linear-gradient(90deg,#3355ff,#33ddff,#4d4,#fe3,#f43)", electrostatic: "linear-gradient(90deg,#d63027,#fff,#3061db)",
             hydrophobic: "linear-gradient(90deg,#008c8c,#fff,#d9a621)", bfactor: "linear-gradient(90deg,#33f,#fff,#f33)" }[c]
      || "linear-gradient(135deg,#ccc 50%,#888 50%)";
  }
  const colorName = c => {
    const [scheme, carbon] = c.split("/");
    if (scheme === "element") return carbon ? `element, ${carbon} C` : "element";
    return [...SCHEMES_ATOMS, ...SCHEMES_RIBBON, ...SCHEMES_SURFACE].find(([k]) => k === c)?.[1].toLowerCase() || c;
  };

  function currentRep(m) {
    if (!m?.reps?.length) return null;
    return m.reps.find(r => r.id === m.currentRep) || m.reps[0];
  }

  function repsPanel() {
    const m = app.currentModel(), list = $("rep-list"), edit = $("rep-edit");
    list.innerHTML = ""; edit.innerHTML = "";
    $("reps-model").textContent = m && !m.closed ? `${m.index + 1}  ${m.name}` : "";
    $("reps-model").title = m?.name || "";
    for (const b of ["rep-dup", "rep-del"]) $(b).disabled = !currentRep(m);
    $("rep-new").disabled = $("rep-presets").disabled = !m?.state || m.closed;
    if (!m?.state || m.closed) { list.append(el("p", { className: "mgroup" }, m && !m.closed ? "loading…" : "no model")); edit.hidden = true; return; }
    const cur = currentRep(m);
    for (const rep of m.reps) {
      const row = el("div", { className: `rep${rep.on ? " on" : " off"}${rep === cur ? " current" : ""}`,
                              title: `${STYLE_LABEL[rep.style]} · ${rep.custom ? "colors of its own" : colorName(rep.color)} · ${rep.sel || rep.label} (${app.repIds(rep).length} atoms)` });
      const eye = el("button", { className: "eye", title: rep.on ? "drawn: click to hide" : "hidden: click to draw" });
      eye.onclick = e => { e.stopPropagation(); rep.on = !rep.on; app.repsChanged(m); };
      const chip = el("span", { className: "chip" });
      chip.style.background = chipColor(rep);
      const x = el("button", { className: "x", title: "delete this representation" }, "✕");
      x.onclick = e => { e.stopPropagation(); deleteRep(m, rep); };
      const alpha = app.repAlpha(rep);
      row.append(eye, el("span", { className: "style" }, STYLE_LABEL[rep.style], ...(alpha < 1 ? [el("small", {}, ` ${Math.round(100 * alpha)}%`)] : [])), chip,
                 el("span", { className: "what" }, rep.label || rep.sel), x);
      row.onclick = () => { m.currentRep = rep.id; repsPanel(); };
      row.ondblclick = () => { rep.on = !rep.on; app.repsChanged(m); };
      list.append(row);
    }
    if (!m.reps.length) list.append(el("p", { className: "mgroup" }, "nothing drawn: + New, or a preset"));
    edit.hidden = !cur;
    if (cur) repEditor(m, cur, edit);
  }

  function repEditor(m, rep, box) {
    const sel = el("textarea", { value: rep.sel ?? "", rows: 2, spellcheck: false,
                                 placeholder: rep.sel ? "" : `${rep.label}: type a selection to replace them` });
    const apply = button("Apply", () => setRepSelection(m, rep, sel.value.trim()));
    sel.onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); apply.click(); } e.stopPropagation(); };
    const picked = button("Use selected atoms", () => {
      const ids = app.selected(m);
      if (!ids.length) { app.message("nothing selected in this model", true); return; }
      rep.mask = new Uint8Array(m.topo.natoms);
      for (const a of ids) rep.mask[a] = 1;
      rep.sel = null; rep.label = describeAtoms(m, ids); rep.version++;
      app.colorRep(m, rep); app.repsChanged(m);
    });
    box.append(el("div", { className: "lab" }, "Atoms"), sel, el("div", { className: "row", style: "margin-top:4px" }, apply, picked));
    // style
    const seg = el("div", { className: "seg" });
    for (const [s, label] of app.REP_STYLES) {
      const b = button(label, () => setRepStyle(m, rep, s), rep.style === s ? "on" : "");
      seg.append(b);
    }
    box.append(el("div", { className: "lab" }, "Style"), seg);
    // color: a scheme, carbons of a color, or one color
    const schemes = rep.style === "surface" ? SCHEMES_SURFACE : app.isRibbon(rep) ? SCHEMES_RIBBON : SCHEMES_ATOMS;
    const scheme = el("select");
    scheme.add(new Option(rep.custom ? "colors of its own" : schemes.some(([k]) => k === rep.color.split("/")[0]) ? "" : `${colorName(rep.color)}`, "", false, true));
    for (const [k, label] of schemes) scheme.add(new Option(label, k, false, !rep.custom && (rep.color === k || (k === "element" && rep.color.startsWith("element")))));
    scheme.onchange = () => scheme.value && setRepColor(m, rep, scheme.value);
    const dots = (label, names, spec, title) => {
      const d = el("div", { className: "dots" }, el("span", { className: "dots-label" }, label));
      for (const c of names) {
        const b = el("button", { className: `dot${rep.color === spec(c) ? " on" : ""}`, title: title(c) });
        b.style.background = rgbCss(named(c));
        b.onclick = () => setRepColor(m, rep, spec(c));
        d.append(b);
      }
      return d;
    };
    box.append(el("div", { className: "lab" }, "Color"), el("div", { className: "row" }, scheme));
    if (!app.isRibbon(rep) && rep.style !== "surface") box.append(dots("carbons", CARBONS, c => `element/${c}`, c => `element, ${c} carbons`));
    box.append(dots("one color", SINGLE, c => c, c => c));
    // material: how it shines and how opaque it is; the slider sets opacity apart from it
    const mat = el("div", { className: "seg" });
    for (const [k, label] of app.MATERIALS) mat.append(button(label, () => setRepMaterial(m, rep, k), rep.material === k ? "on" : ""));
    const alpha = app.repAlpha(rep);
    const slider = el("input", { type: "range", min: 5, max: 100, step: 5, value: Math.round(100 * alpha), className: "opacity" });
    const pct = el("span", { className: "pct" }, `${Math.round(100 * alpha)}%`);
    slider.oninput = () => { rep.opacity = +slider.value / 100; pct.textContent = `${slider.value}%`; app.rebuild(); };
    slider.onchange = () => app.repsChanged(m);
    box.append(el("div", { className: "lab" }, "Material"), mat,
               el("div", { className: "row", style: "margin-top:5px" }, el("span", { className: "dots-label" }, "opacity"), slider, pct));
    numberRows(m, box);
    const n = app.repIds(rep).length;
    box.append(el("div", { className: "note" }, `${n} atom${n === 1 ? "" : "s"}${rep.tinted && m.color && m.color !== "default" ? ` · the model's ${m.color} over it` : ""}`));
  }

  // --- filter by a number on the atoms -------------------------------------------------
  // A model can carry per-atom numbers (what each pose of a merged docking set did in
  // MD, a B-factor).  One slider a number keeps the atoms whose value is in range.  An
  // atom with no value is left as it was: a filter has nothing to say about a number
  // that was never measured, and hiding those atoms would take the one protein a merged
  // set of ligands sits on with them.  The filter is the model's, not one
  // representation's, since it decides which of its atoms are drawn at all.
  function numberTests(m) {
    return Object.entries(m.numberFilters || {})
      .map(([name, f]) => [m.topo.numbers?.[name], f])
      .filter(([vals]) => vals);  // fmt: skip
  }
  // Which atoms a filter is about at all: those carrying a value for any of the
  // model's numbers.  The rest -- the one protein a merged set of ligands sits on --
  // is never filtered.  A part that should have had a number and has none is not
  // exempt though: it fails, or a cutoff of 0.5 keeps the 16 poses under it together
  // with the 195 that were never measured, which is not what anyone asked for.
  function inScope(m) {
    if (m._scope) return m._scope;
    const scope = new Uint8Array(m.topo.natoms);
    for (const vals of Object.values(m.topo.numbers || {})) {
      for (let a = 0; a < scope.length; a++) {
        if (!scope[a] && vals[a] !== null && vals[a] !== undefined) scope[a] = 1;
      }
    }
    m._scope = scope;
    return scope;
  }
  function passes(tests, scope, a) {
    if (!scope[a]) return true;  // not a part these numbers describe
    for (const [vals, f] of tests) {
      const v = vals[a];
      if (v === null || v === undefined) return false;  // should have had one; has none
      if (f.above ? !(v >= f.cut) : !(v <= f.cut)) return false;
    }
    return true;
  }
  function applyNumbers(m) {
    const tests = numberTests(m), scope = inScope(m);
    for (const rep of m.reps) {
      if (!rep.full) rep.full = rep.mask.slice();  // what it was before any filter
      for (let a = 0; a < rep.mask.length; a++) {
        rep.mask[a] = rep.full[a] && passes(tests, scope, a) ? 1 : 0;
      }
      rep.version++;
      app.colorRep(m, rep);
    }
    app.aggregate(m);
    app.rebuild();
    // What the count means: the residues the filters actually judge, not every
    // residue left drawn.  An atom with no value passes, so counting those would
    // add the protein and every unmeasured part to the number and leave it
    // nothing like the same selection typed out.
    const kept = new Set(), parts = new Set(), R = m.topo.atoms.residue;
    for (let a = 0; a < R.length; a++) {
      if (!scope[a]) continue;  // not a part the numbers describe
      parts.add(R[a]);
      if (passes(tests, scope, a)) kept.add(R[a]);
    }
    return [kept.size, parts.size];
  }
  //: one row a number, appended to the representations panel under Opacity
  function numberRows(m, box) {
    const nums = m.topo?.numbers || {};
    const names = Object.keys(nums);
    if (!names.length) return;
    m.numberFilters = m.numberFilters || {};
    const count = el("div", { className: "note" });
    const rows = [];
    const show = () => {
      const [kept, measured] = applyNumbers(m);
      count.textContent = measured
        ? `${kept} of ${measured} drawn \u00b7 the rest of the model is untouched`
        : "nothing here carries these numbers";  // fmt: skip
      for (const f of rows) f();
    };
    box.append(el("div", { className: "lab" }, "Filter the model by a number"));
    for (const name of names) {
      // a loop, not Math.min(...vals): spreading an array of 40,000 numbers into a
      // call is enough to overflow the stack
      let lo = Infinity, hi = -Infinity, seen = 0;
      for (const v of nums[name]) {
        if (v === null || v === undefined) continue;
        seen++;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (!seen) continue;
      const f = m.numberFilters[name] || (m.numberFilters[name] = { above: false, cut: hi });
      const dir = button(f.above ? "\u2265" : "\u2264", () => {
        f.above = !f.above;
        show();
      });
      const slider = el("input", { type: "range", min: lo, max: hi,
                                   step: (hi - lo) / 200 || 1, value: f.cut, className: "opacity" });  // fmt: skip
      // typed as well as dragged: a slider cannot hit 1.5 exactly on a range of
      // 0.314 to 16.703, and a cutoff worth using is usually a round number
      const read = el("input", { className: "cut", type: "number", step: (hi - lo) / 200 || 1 });
      const set = v => {
        if (!Number.isFinite(v)) return;
        f.cut = Math.min(hi, Math.max(lo, v));
        show();
      };
      slider.oninput = () => set(+slider.value);
      slider.onchange = () => app.repsChanged(m);
      read.onchange = () => { set(+read.value); app.repsChanged(m); };
      read.onkeydown = e => { if (e.key === "Enter") read.blur(); e.stopPropagation(); };
      rows.push(() => {
        dir.textContent = f.above ? "\u2265" : "\u2264";
        slider.value = f.cut;
        if (document.activeElement !== read) read.value = String(+f.cut.toFixed(3));
        const off = f.above ? f.cut <= lo : f.cut >= hi;
        dir.classList.toggle("on", !off);
      });
      box.append(el("div", { className: "row filter", title: name },
                    el("span", { className: "dots-label" }, name), dir, slider, read));  // fmt: skip
    }
    box.append(count);
    show();
  }

  async function setRepSelection(m, rep, text) {
    if (!text) return;
    let ids;
    try { ids = Array.from(await app.selectIds(m, text)); } catch (e) { app.message(e.message, true); return; }
    rep.mask = new Uint8Array(m.topo.natoms);
    for (const a of ids) rep.mask[a] = 1;
    rep.sel = text; rep.label = text; rep.version++;
    app.colorRep(m, rep);
    app.repsChanged(m);
    app.message(`${text}: ${ids.length} atom${ids.length === 1 ? "" : "s"}`);
  }
  function setRepStyle(m, rep, style) {
    const was = rep.style;
    rep.style = style;
    // a color that means nothing in the new style: that style's usual one
    if (style === "surface" && !["electrostatic", "hydrophobic"].includes(rep.color) && !named(rep.color)) rep.color = "electrostatic";
    else if (was === "surface" && ["electrostatic", "hydrophobic"].includes(rep.color)) rep.color = app.isRibbon(rep) ? "ss" : "element";
    rep.version++;
    app.colorRep(m, rep);
    if (style === "surface" && rep.color === "electrostatic") forceFieldCharges(m);
    app.repsChanged(m);
  }
  function setRepColor(m, rep, spec) {
    rep.color = spec; rep.tinted = false;
    app.colorRep(m, rep);
    if (rep.style === "surface" && spec === "electrostatic") forceFieldCharges(m);
    app.repsChanged(m);
  }
  function setRepMaterial(m, rep, name) {
    rep.material = name;
    rep.opacity = null;  // the material's own
    app.repsChanged(m);
  }
  function deleteRep(m, rep) {
    m.reps = m.reps.filter(r => r !== rep);
    if (m.currentRep === rep.id) m.currentRep = m.reps[Math.max(0, m.reps.length - 1)]?.id;
    app.repsChanged(m);
  }
  // + New: the selected atoms of this model, else a copy of the current representation's atoms
  function newRepFromSelection() {
    const m = app.selection()[0]?.[0] || app.currentModel();
    if (!m?.state) return;
    const ids = app.selected(m);
    if (ids.length) { addRep([[m, ids]], { style: "sticks" }); return; }
    const cur = currentRep(m);
    if (cur) addRep([[m, app.repIds(cur)]], { style: "lines", sel: cur.sel, label: cur.label });
    else addRep([[m, app.range(m.topo.natoms)]], { style: "lines", sel: "all" });
  }
  $("rep-new").onclick = newRepFromSelection;
  $("rep-dup").onclick = () => {
    const m = app.currentModel(), cur = currentRep(m);
    if (!cur) return;
    const copy = app.makeRep(m, { mask: cur.mask.slice(), sel: cur.sel, label: cur.label, style: cur.style, color: cur.color,
                                  tinted: cur.tinted, material: cur.material, opacity: cur.opacity });
    if (cur.custom) { copy.rgb = cur.rgb.slice(); copy.custom = true; }
    m.reps.splice(m.reps.indexOf(cur) + 1, 0, copy);
    m.currentRep = copy.id;
    app.repsChanged(m);
  };
  $("rep-del").onclick = () => { const m = app.currentModel(), cur = currentRep(m); if (cur) deleteRep(m, cur); };
  $("rep-presets").onclick = () => {
    const m = app.currentModel();
    if (!m?.state) return;
    const r = $("rep-presets").getBoundingClientRect();
    const box = menu(PRESETS.map(([k, label]) => ({ label, run: () => preset(m, k) })), r.left, r.bottom + 4);
    box.style.left = `${Math.max(0, r.right - box.getBoundingClientRect().width)}px`;
  };
  app.onReps = m => { if (!m || m === app.currentModel()) repsPanel(); };

  // the side bar: shown or put away; its split between models and representations dragged
  function toggleSide() {
    $("window").classList.toggle("nomodels");
    $("btn-reps").classList.toggle("on", !$("window").classList.contains("nomodels"));
    app.draw();
  }
  $("btn-reps").onclick = toggleSide;
  $("btn-reps").classList.add("on");
  {
    const split = $("side-split"), sec = $("models-sec");
    let start = null;
    split.onpointerdown = e => { start = [e.clientY, sec.getBoundingClientRect().height]; split.setPointerCapture(e.pointerId); };
    split.onpointermove = e => {
      if (!start) return;
      const h = Math.max(70, Math.min($("rdock").clientHeight - 160, start[1] + e.clientY - start[0]));
      sec.style.setProperty("--models-h", `${h}px`);
    };
    split.onpointerup = () => { start = null; };
  }

  // a choice on a model's row: a button and a menu, so that choosing what it
  // already says does it again (a <select> only acts on a change)
  function picker(title, options, current, pick) {
    const label = () => options.find(([v]) => v === current())?.[1] ?? current();
    const b = el("button", { className: "pick", title }, label());
    b.onclick = () => {
      const r = b.getBoundingClientRect();
      menu(options.map(([v, l]) => ({ label: l, checked: () => current() === v,
                                      run: async () => { await pick(v); b.textContent = label(); } })),
           r.left, r.bottom);
    };
    return b;
  }

  function swatchColor(spec) {
    if (spec === "default" || !spec) return "linear-gradient(135deg,#e53935 50%,#ffd54f 50%)";
    if (spec === "chain") return "linear-gradient(135deg,#5c5 33%,#3bd 33% 66%,#e5c 66%)";
    const c = named(spec) || [0.5, 0.5, 0.5];
    return `rgb(${c.map(x => Math.round(x * 255)).join(",")})`;
  }

  function modelsDock() {
    const list = $("rdock-list");
    list.innerHTML = "";
    const sort = $("rdock-sort");
    if (!sort.options.length || sort.dataset.columns !== state.columns.join("|")) {
      sort.innerHTML = "";
      sort.add(new Option("as listed", ""));
      for (const c of ["name", ...state.columns.filter(c => c !== "name")]) {
        sort.add(new Option(`by ${c} ↑`, `${c}|0`)); sort.add(new Option(`by ${c} ↓`, `${c}|1`));
      }
      sort.dataset.columns = state.columns.join("|");
    }
    let group = null;
    for (const m of sortedModels()) {
      if (m.group !== group && !state.sort?.column) {
        group = m.group;
        if (group) list.append(groupHeader(group));
      }
      const row = el("div", { className: `mrow${m.shown ? " on" : ""}${m.index === state.current ? " current" : ""}` });
      const box = el("input", { type: "checkbox", checked: m.shown, title: "show" });
      box.onchange = () => { state.current = m.index; app.show([m], box.checked); };
      const sw = el("span", { className: "swatch" });
      sw.style.background = swatchColor(m.swatch || m.color);
      const name = el("span", { className: "name", title: `${m.name} (model ${m.index + 1}, for "in ${m.index + 1}" `
        + "on the command line): click to show it alone and look at it" }, el("span", { className: "num" }, `${m.index + 1}`), m.name);
      name.onclick = () => app.solo(m);
      const colors = m.color && !MODEL_COLORS.some(([v]) => v === m.color) ? [...MODEL_COLORS, [m.color, "its own"]] : MODEL_COLORS;
      const color = picker("color", colors, () => m.color || "default", async v => {
        await app.load(m); colorModel(m, v); sw.style.background = swatchColor(m.swatch || v);
      });
      const x = el("button", { className: "x", title: "close this model" }, "✕");
      x.onclick = () => closeModel(m);
      row.append(el("div", { className: "top" }, box, sw, name, color, x));
      // a click on the row (not on its controls): this model's representations below
      row.onclick = async e => {
        if (e.target.closest("button, input, .name")) return;
        state.current = m.index;
        await app.load(m);
        modelsDock(); repsPanel();
      };
      row.oncontextmenu = e => {
        e.preventDefault();
        menu([
          { label: "Show Only This", run: () => { for (const o of app.live()) o.shown = o === m; app.solo(m); } },
          { label: m.shown ? "Hide" : "Show", run: () => { state.current = m.index; app.show([m], !m.shown); } },
          { label: "Center", run: async () => { await app.show([m]); const [c, r] = app.bounds(m, app.range(m.topo.natoms)); app.flyTo(c, Math.max(r, 6)); } },
          "-",
          { label: "Select All of It", run: async () => { await app.show([m]); app.select(m, app.range(m.topo.natoms)); } },
          { label: "Copy SMILES", run: async () => { await app.show([m]); app.clearSelection(); state.current = m.index; copySmiles(); } },
          ...(isSeries(m) ? [...seriesMemberItems(m), "-"] : []),
          { label: "Align On…", run: () => alignPanel(m.index), off: () => app.live().length < 2 },
          { label: "Copy Into", sub: copyIntoItems(m, []) },
          { label: "Download", sub: () => DOWNLOADS.map(([fmt, label]) => ({ label, run: async () => { await app.load(m); download(fmt, m, []); } })) },
          { label: "Save…", run: async () => { await app.show([m]); app.clearSelection(); state.current = m.index; app.rebuild(); savePanel(); } },
          "-",
          { label: "Close", run: () => closeModel(m) },
        ], e.clientX, e.clientY);
      };
      const data = Object.entries(m.row || {}).filter(([k]) => k !== "name" && k !== "SMILES");  // a series' SMILES: in its table
      if (data.length) row.append(el("div", { className: "data", title: data.map(([k, v]) => `${k} ${cell(v)}`).join("\n") },
        data.map(([k, v]) => `${k} ${cell(v)}`).join(" · ")));
      list.append(row);
    }
  }
  $("rdock-sort").onchange = () => {
    const [column, desc] = $("rdock-sort").value.split("|");
    state.sort = column ? { column, descending: desc === "1" } : null;
    modelsDock();
  };
  $("rdock-all").onclick = () => app.show(app.live(), true);
  $("rdock-none").onclick = () => app.show(app.live().filter(m => m.index !== state.current), false);
  app.onModels = () => {
    modelsDock(); repsPanel();
    for (const p of Object.values(panels)) p.redraw?.();  // a series' table
  };
  // a group's header: its box shows or hides every model in it (all the pockets, say)
  function groupHeader(group) {
    const members = state.models.filter(m => m.group === group && !m.closed);
    const on = members.filter(m => m.shown).length;
    const box = el("input", { type: "checkbox", checked: on === members.length, title: `show or hide every ${group} model` });
    box.indeterminate = on > 0 && on < members.length;
    box.onchange = async () => {
      if (box.checked && members.length > 50 && !confirm(`Show all ${members.length} ${group} models? Each one loads.`)) {
        box.checked = false;
        return;
      }
      await app.show(members, box.checked);
    };
    const header = el("label", { className: "mgroup" }, box, ` ${group} (${members.length})`);
    if (isSeries({ group }) && members.length) {  // its table: a row per member, a column per field
      const t = el("button", { className: "table-btn", title: `${group} as a table: its SDF fields, sortable, to plot` }, "table");
      t.onclick = e => { e.preventDefault(); e.stopPropagation(); seriesTablePanel(group); };
      header.append(t);
    }
    if (isSeries({ group }) && members.length) header.oncontextmenu = e => {
      e.preventDefault();
      menu(seriesMemberItems(members.find(m => m.index === state.current) || members[members.length - 1]), e.clientX, e.clientY);
    };
    return header;
  }


  const cell = v => typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")) : v ?? "";
  function sortedModels() {
    const list = app.live();
    const { column, descending } = state.sort || {};
    if (!column) return list;
    return [...list].sort(byColumn(column, descending));
  }
  // models in order of a column of their rows; those without it last
  function byColumn(column, descending) {
    const key = m => column === "name" ? m.name : value(m, column);
    return (a, b) => {
      const x = key(a), y = key(b);
      if (x === y) return a.index - b.index;
      if (x === undefined || x === null) return 1;
      if (y === undefined || y === null) return -1;
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return descending ? -c : c;
    };
  }
  function stepModel(by) {
    const now = app.currentModel();
    const list = sortedModels().filter(m => m.group === now.group);
    const next = list[Math.min(Math.max(list.indexOf(now) + by, 0), list.length - 1)];
    if (next && next !== now) app.solo(next);
  }
  $("btn-render").onclick = () => {  // a second click puts it away, as Sequence does the sequence
    if (panels.picture) { panels.picture.remove(); delete panels.picture; } else renderPanel();
  };
  $("btn-min").onclick = minimizePanel;

  // --- the sequence, above the view: click a residue to select it (drag or Shift: a
  // run of them; Ctrl: add or take away), right click to mutate, center or show it ------

  const ONE = { ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
    LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
    HID: "H", HIE: "H", HIP: "H", HSD: "H", HSE: "H", HSP: "H", CYX: "C", CYM: "C", ASH: "D", GLH: "E", LYN: "K",
    MSE: "M", SEP: "S", TPO: "T", PTR: "Y", DA: "A", DC: "C", DG: "G", DT: "T", DU: "U", A: "A", C: "C", G: "G",
    U: "U", T: "T", RA: "A", RC: "C", RG: "G", RU: "U", ADE: "A", CYT: "C", GUA: "G", THY: "T", URA: "U" };
  const letter = name => ONE[name.toUpperCase()] || (name.length === 1 ? name : "X");
  let seqKey = "", seqAnchor = null, seqDrag = null;

  function toggleSequence(on = $("seqbar").hidden) {
    $("seqbar").hidden = !on;
    try { localStorage.setItem("boonza.sequence", on ? "1" : "0"); } catch (e) { /* not kept */ }
    seqKey = "";
    paintSequence();
    app.draw();
  }
  $("btn-seq").onclick = () => toggleSequence();

  // per shown model, its chains' traced residues (amino acids, nucleotides, Martini's BB)
  function sequenceChains() {
    const out = [];
    for (const m of app.shownModels()) {
      const R = m.topo.residues, C = m.topo.chains, chains = new Map();
      for (let r = 0; r < R.name.length; r++) {
        // traced residues, and amino acids and nucleotides the trace missed (a broken backbone)
        if (!R.kind[r] && !(R.name[r].toUpperCase() in ONE && m.topo.residueAtoms[r].length > 3)) continue;
        if (!chains.has(R.chain[r])) chains.set(R.chain[r], []);
        chains.get(R.chain[r]).push(r);
      }
      for (const [c, rs] of chains) out.push({ m, chain: C.name[c] || C.segid[c] || "-", rs });
    }
    return out;
  }

  function buildSequence() {
    const bar = $("seqbar");
    bar.innerHTML = "";
    const lines = sequenceChains();
    bar.classList.toggle("none", !lines.length);  // nothing to show: no room taken
    if (!lines.length) return;
    const many = new Set(lines.map(l => l.m)).size > 1;
    for (const { m, chain, rs } of lines) {
      const R = m.topo.residues, ss = m.ss || [];
      const label = el("span", { className: "seqlabel", title: `${m.name}, chain ${chain}: ${rs.length} residues` },
                       many ? `${m.name.slice(0, 8)} ${chain}` : chain);
      const res = el("div", { className: "seqres" });
      let prev = null;
      for (const r of rs) {
        if (prev !== null && R.resid[r] - R.resid[prev] > 1)
          res.append(el("span", { className: "gap", title: `${R.resid[r] - R.resid[prev] - 1} residues not in the structure` }, "·"));
        const s = el("span", { className: `aa ss-${ss[r] || "C"}` }, letter(R.name[r]));
        s.dataset.m = m.index; s.dataset.r = r;
        if (R.resid[r] % 10 === 0) s.dataset.n = R.resid[r];
        s.title = `${R.name[r]} ${R.resid[r]}${R.insertion[r].trim()} · chain ${chain}`;
        res.append(s);
        prev = r;
      }
      bar.append(el("div", { className: "seqline" }, label, res));
    }
  }

  // build when the models or their atoms change; mark the selected residues
  function paintSequence() {
    const bar = $("seqbar");
    if (bar.hidden) return;
    const key = app.shownModels().map(m => `${m.index}:${m.topo.natoms}:${m.topo.version}`).join("|");
    if (key !== seqKey) { seqKey = key; buildSequence(); }
    const picked = new Map();
    for (const [m, ids] of app.selection()) picked.set(m.index, new Set(ids.map(a => m.topo.atoms.residue[a])));
    for (const s of bar.querySelectorAll(".aa")) s.classList.toggle("sel", !!picked.get(+s.dataset.m)?.has(+s.dataset.r));
  }

  // residues from one to another along a chain's line
  function residueRun(m, r0, r1) {
    const line = sequenceChains().find(l => l.m === m && l.rs.includes(r0) && l.rs.includes(r1));
    if (!line) return [r1];
    const [i, j] = [line.rs.indexOf(r0), line.rs.indexOf(r1)].sort((a, b) => a - b);
    return line.rs.slice(i, j + 1);
  }
  const residueAtoms = (m, rs) => rs.flatMap(r => m.topo.residueAtoms[r]);
  const seqTarget = e => e.target.closest?.(".aa");

  $("seqbar").addEventListener("pointerdown", e => {
    const s = seqTarget(e);
    if (!s || e.button !== 0) return;
    e.preventDefault();
    const m = state.models[+s.dataset.m], r = +s.dataset.r;
    const add = e.ctrlKey || e.metaKey;
    if (e.shiftKey && seqAnchor && seqAnchor.m === m) {
      app.select(m, residueAtoms(m, residueRun(m, seqAnchor.r, r)), add ? "add" : "set");
    } else {
      app.select(m, residueAtoms(m, [r]), add ? "toggle" : "set");
      seqAnchor = { m, r };
    }
    seqDrag = { m, r, add, before: add ? new Set(app.selected(m)) : null };
    $("seqbar").setPointerCapture(e.pointerId);
  });
  $("seqbar").addEventListener("pointermove", e => {
    if (!seqDrag) return;
    const s = document.elementFromPoint(e.clientX, e.clientY)?.closest?.(".aa");
    if (!s || +s.dataset.m !== seqDrag.m.index || +s.dataset.r === seqDrag.last) return;
    seqDrag.last = +s.dataset.r;
    const m = seqDrag.m, run = residueAtoms(m, residueRun(m, seqDrag.r, +s.dataset.r));
    app.select(m, seqDrag.add ? [...seqDrag.before, ...run] : run, "set");
  });
  $("seqbar").addEventListener("pointerup", () => { seqDrag = null; });
  $("seqbar").addEventListener("dblclick", () => app.centerView());
  $("seqbar").addEventListener("contextmenu", e => {
    e.preventDefault();
    const s = seqTarget(e);
    if (s) {
      const m = state.models[+s.dataset.m], r = +s.dataset.r;
      if (!app.selected(m).some(a => m.topo.atoms.residue[a] === r)) app.select(m, residueAtoms(m, [r]), "set");
    }
    const sel = app.selection();
    if (!sel.length) return;
    const [m, ids] = sel[0];
    const names = [...new Set(ids.map(a => m.topo.atoms.residue[a]))].map(r => `${m.topo.residues.name[r]}${m.topo.residues.resid[r]}`);
    const what = names.length > 3 ? `${names.length} residues` : names.join(", ");
    menu([
      { label: `Mutate ${what} to`, sub: () => AMINO.map(aa => ({ label: `${aa} (${ONE[aa]})`, run: () => mutate(aa) })),
        off: () => !m.topo.editable },
      { label: "Rename or Renumber…", run: relabelPanel, off: () => !m.topo.editable },
      "-",
      { label: "Center", run: () => app.centerView() },
      { label: "Show as Sticks", run: () => addRep([[m, ids]], { style: "sticks" }) },
      { label: "Hide Atoms", run: () => hideAtoms([[m, ids.filter(a => !m.topo.residues.trace.includes(a))]]) },
      { label: "Color", sub: () => SINGLE.map(c => ({ label: c, run: () => colorIn([[m, ids]], c) })) },
      "-",
      { label: "Delete", run: deleteSelected },
    ], e.clientX, e.clientY);
  });

  // the selection's SMILES in the footer, for a small molecule
  let footTicket = 0;
  async function footerSmiles() {
    const ticket = ++footTicket;
    const sel = app.selection();
    if (sel.length !== 1) return;
    const [m, ids] = sel[0];
    if (!m.topo || !ids.every(a => m.topo.atoms.class[a] === 1)) return;
    let mols;
    try { mols = (await smilesOf(m, ids)).filter(x => x.smiles); } catch (e) { return; }
    if (ticket !== footTicket || !mols.length) return;
    const text = mols.map(x => x.smiles).join(".");
    $("footer-edit").append(el("span", { className: "what" }, "SMILES"), el("code", { className: "smiles", title: text }, text),
      button("Copy", async () => app.message((await copyText(text)) ? `copied ${text}` : text)));
  }

  // an empty view (nothing given, or the last model closed): the builder, to start
  app.onEmpty = () => {
    openDock("builder");
    setHydrogens("all");
    app.message("An empty view: pick an atom or a ring in the Builder and click in the view to start a molecule, "
      + "or type its SMILES. File | Open and File | Fetch from PDB bring in structures.");
  };


  function mousePanel() {
    panel("mouse", "Mouse", body => {
      const rows = [
        ["Left click", "select an atom (a bond: its two atoms; a ribbon: its residue)"],
        ["Shift / Ctrl click", "toggle / the whole residue"], ["Double / triple click", "the residue / the molecule"],
        ["Left drag", "select a box (Shift adds, Ctrl takes residues)"], ["Click away", "clear the selection"],
        ["Middle drag", "rotate (in the perimeter: about z)"], ["Shift + middle", "pan"], ["Ctrl + middle", "zoom"],
        ["Middle click", "center of rotation (away from atoms: reset)"], ["Wheel", "zoom"],
        ["Alt + middle", "rotate the selected atoms"], ["Shift + Alt + middle", "translate the selected atoms"],
        ["Alt + left drag", "turn the selected bond"], ["Right click", "the popup menu"], ["Right drag", "zoom"],
        ["Shift (hover)", "what an atom is"], ["Space", "command line"],
      ];
      const t = el("table", { className: "mouse-table" });
      for (const [k, v] of rows) { const tr = t.insertRow(); tr.insertCell().textContent = k; tr.insertCell().textContent = v; }
      body.append(t);
      const two = el("input", { type: "checkbox", checked: state.twoButton });
      two.onchange = () => { state.twoButton = two.checked; localStorage.setItem("boonza.twoButton", two.checked ? "1" : "0"); };
      body.append(row(el("label", {}, two, " two-button mouse: the right button is the middle one; Alt alone opens the popup")));
      const pad = el("input", { type: "checkbox", checked: state.trackpad });
      pad.onchange = () => { state.trackpad = pad.checked; localStorage.setItem("boonza.trackpad", pad.checked ? "1" : "0"); };
      body.append(row(el("label", {}, pad, " trackpad: a two-finger swipe rotates (in the perimeter: about z), "
        + "Shift + swipe pans, a pinch zooms")));
      body.append(el("p", { className: "help" }, "On a Mac, Ctrl is Cmd and Alt is Option. Without a middle button, "
        + "use the trackpad or two-button setting."));
    });
  }

  function fogPanel() {
    panel("fog", "Fog", body => {
      const f = app.renderer.fog;
      const on = el("input", { type: "checkbox", checked: f.on });
      const start = el("input", { type: "range", min: 0, max: 1, step: 0.01, value: f.start });
      const strength = el("input", { type: "range", min: 0, max: 1, step: 0.01, value: f.strength });
      const apply = () => { f.on = on.checked; f.start = +start.value; f.strength = +strength.value; app.draw(); $("vt-fog").classList.toggle("on", f.on); };
      on.onchange = start.oninput = strength.oninput = apply;
      body.append(row(el("label", {}, on, " depth cue")), row(el("span", { className: "what" }, "Start"), start),
                  row(el("span", { className: "what" }, "Strength"), strength));
    }, [innerWidth - 700, innerHeight - 230]);
  }
  function zclipPanel() {
    panel("zclip", "Clipping", body => {
      const c = app.camera.zclip;
      const front = el("input", { type: "range", min: 0.02, max: 1, step: 0.01, value: c[0] });
      const back = el("input", { type: "range", min: 0.02, max: 1, step: 0.01, value: c[1] });
      const apply = () => { c[0] = +front.value; c[1] = +back.value; app.draw(); $("vt-clip").classList.toggle("on", c[0] < 1 || c[1] < 1); };
      front.oninput = back.oninput = apply;
      body.append(row(el("span", { className: "what" }, "Front"), front), row(el("span", { className: "what" }, "Back"), back),
                  row(button("Reset", () => { front.value = back.value = 1; apply(); })));
    }, [innerWidth - 640, innerHeight - 230]);
  }

  // boonza on the models: fetch a PDB entry, superpose one model on another
  function fetchPanel() {
    panel("fetch", "Fetch from PDB", body => {
      const input = el("input", { placeholder: "PDB ID, e.g. 1IRK", size: 12, spellcheck: false });
      const go = button("Fetch", async () => {
        const id = input.value.trim();
        if (!id) return;
        try {
          const out = await app.busy(`downloading ${id.toUpperCase()} from RCSB`, () => app.post("api/fetch", { id }));
          await addModels(out);
          panels.fetch?.remove(); delete panels.fetch;
          app.message(`${id.toUpperCase()}: opened (saved as ${id.toUpperCase()}.cif where boonza view runs)`);
        } catch (e) { app.message(e.message, true); }
      }, "blue");
      input.onkeydown = e => { if (e.key === "Enter") go.click(); };
      body.append(row(input, go));
      body.append(el("p", { className: "help" }, "mmCIF from files.rcsb.org, kept in the folder boonza view was started in."));
      setTimeout(() => input.focus(), 0);
    }, [innerWidth / 2 - 160, 80]);
  }

  // superpose one model on another: on all of each, or on chains, residue ranges and
  // atoms chosen for each side (or any selection), by sequence, structure or atom for atom
  const ALIGN_METHODS = [["matchmaker", "Sequence alignment (matchmaker, whole chains)"],
                         ["superpose", "Fit matched residues (paired by sequence)"],
                         ["pair", "Fit atom for atom (in order)"],
                         ["cealign", "Structure only (CE, remote homologs)"]];
  const ALIGN_ATOMS = [["ca", "alpha carbons"], ["backbone", "backbone"], ["heavy", "all heavy atoms"]];
  async function alignPanel(mobileIndex = null) {
    const models = app.live();
    if (models.length < 2) { app.message("superposing needs two models: open another", true); return; }
    await Promise.all(models.map(m => app.load(m)));
    panel("align", "Superpose", body => {
      const mob = mobileIndex ?? app.currentModel().index;
      // onto: a shown model with chains of the same kind (protein onto protein), else any other
      const kinds = m => new Set(m.topo.residues.kind.filter(k => k));
      const same = m => [...kinds(m)].some(k => kinds(state.models[mob]).has(k));
      const others = models.filter(m => m.index !== mob);
      const other = others.find(m => m.shown && same(m)) || others.find(same) || others.find(m => m.shown) || others[0];
      const side = (chosen, label) => {
        const model = el("select");
        for (const m of models) model.add(new Option(`${m.index + 1}  ${m.name}`, m.index, false, m.index === chosen));
        const chain = el("select");
        const residues = el("input", { placeholder: "all, or 10-120 150-160", spellcheck: false, style: "width:150px" });
        const sel = el("input", { placeholder: "or any selection (replaces the rest)", spellcheck: false, style: "width:290px" });
        const count = el("span", { className: "what", style: "width:auto" });
        const chains = () => {
          const m = state.models[+model.value], C = m.topo.chains;
          chain.innerHTML = "";
          chain.add(new Option("all chains", ""));
          for (const c of [...new Set(C.name.map((n, k) => n || C.segid[k]).filter(Boolean))]) chain.add(new Option(`chain ${c}`, c));
        };
        chains();
        model.onchange = () => { chains(); refresh(); };
        chain.onchange = residues.oninput = sel.oninput = () => refresh();
        body.append(el("h4", {}, label), row(model, chain), row(el("span", { className: "what" }, "residues"), residues),
                    row(sel), row(count));
        return { model, chain, residues, sel, count };
      };
      const a = side(mob, "Move"), b = side(other.index, "Onto");
      const method = el("select"), atoms = el("select");
      for (const [v, l] of ALIGN_METHODS) method.add(new Option(l, v));
      for (const [v, l] of ALIGN_ATOMS) atoms.add(new Option(l, v));
      method.onchange = atoms.onchange = () => refresh();
      body.append(el("h4", {}, "How"), row(el("span", { className: "what" }, "method"), method),
                  row(el("span", { className: "what" }, "atoms"), atoms));
      // the selection a side's choices make
      const selection = (s, kind) => {
        if (s.sel.value.trim()) return s.sel.value.trim();
        const m = state.models[+s.model.value], parts = [];
        if (s.chain.value) parts.push(`chain ${s.chain.value}`);
        const ranges = s.residues.value.trim().split(/[\s,]+/).filter(Boolean).map(r => r.replace(/^(-?\d+)-(-?\d+)$/, "$1 to $2"));
        if (ranges.length) parts.push(`resid ${ranges.join(" ")}`);
        if (kind !== "cealign") {
          const at = atoms.value;
          parts.push(at === "heavy" ? "not hydrogen" : at === "backbone" ? (m.topo.cg ? "name BB" : "backbone")
            : m.topo.cg ? "name BB GC" : "name CA");
        }
        return parts.join(" and ");
      };
      const restricted = s => !!(s.sel.value.trim() || s.residues.value.trim());
      // matchmaker takes whole chains: a residue range or a selection fits matched residues instead
      const how = () => method.value === "matchmaker" && (restricted(a) || restricted(b)) ? "superpose" : method.value;
      let ticket = 0;
      async function refresh() {
        const t0 = ++ticket, kind = how();
        atoms.disabled = kind === "matchmaker" || kind === "cealign";
        for (const s of [a, b]) {
          if (kind === "matchmaker") { s.count.textContent = s.chain.value ? `chain ${s.chain.value}` : "every chain tried"; continue; }
          const q = selection(s, kind);
          try {
            const ids = await app.selectIds(state.models[+s.model.value], q || "all");
            if (t0 === ticket) s.count.textContent = `${ids.length} atoms: ${q || "all"}`;
          } catch (e) { if (t0 === ticket) s.count.textContent = e.message; }
        }
      }
      const go = button("Superpose", async () => {
        const m = state.models[+a.model.value], r = state.models[+b.model.value];
        if (m === r) { app.message("pick two different models", true); return; }
        const kind = how();
        const req = { m: m.index, ref: r.index, method: kind };
        if (kind === "matchmaker") Object.assign(req, { chain: a.chain.value || null, ref_chain: b.chain.value || null });
        else Object.assign(req, { sel: selection(a, kind) || null, ref_sel: selection(b, kind) || null });
        try {
          const out = await app.busy("superposing", () => app.post("api/align", req));
          m.shown = r.shown = true;
          await app.reloadModel(m, false);
          await app.show([m, r]);
          app.centerView();
          const id = out.identity != null ? `, sequence identity ${(100 * out.identity).toFixed(0)}%` : "";
          const all = out.full_rmsd != null ? ` (all ${out.paired} pairs: ${out.full_rmsd} Å)` : "";
          const swap = kind !== method.value ? " (residues chosen: fitted matched residues)" : "";
          app.message(`${m.name} onto ${r.name} by ${out.method}${swap}: RMSD ${out.rmsd} Å over ${out.pairs} atom pairs${all}${id}. Undo puts it back.`);
        } catch (e) { app.message(e.message, true); }
      }, "blue");
      body.append(row(go));
      body.append(el("p", { className: "help" }, "Residues: numbers and ranges (10-120 150-160). Fit matched residues "
        + "pairs one atom a residue by aligning the two sequences and drops the pairs far apart; atom for atom pairs "
        + "the two selections in order, as many on each side. The moving model's coordinates change: "
        + "Edit | Undo restores them, File | Save Model keeps them."));
      refresh();
    }, [innerWidth / 2 - 220, 70]);
  }

  async function addModels(out, look = true) {
    const empty = !app.shownModels().length;
    for (const info of out.models.slice(state.models.length)) state.models.push(app.newModel(info, state.models.length));
    const first = state.models[out.added[0]];
    state.current = first.index;
    await app.show([first]);
    if (empty) app.resetView();
    else if (look) app.centerView();
  }

  // open and save: paths on the server (where the files are)
  function pathPanel(id, title, value, run) {
    return panel(id, title, body => {
      const input = el("input", { value, style: "width:360px", spellcheck: false,
                                  placeholder: id === "open" ? "a path, or a PDB ID (e.g. 1IRK)" : "" });
      const go = button(title.split(" ")[0], () => run(input.value.trim()), "blue");
      input.onkeydown = e => { if (e.key === "Enter") go.click(); };
      body.append(row(input, go));
      body.append(el("p", { className: "help" }, "A path on the machine boonza view runs on: a structure, a saved session (.bview), "
        + "or a folder of boonza modes, triage or cavity; or a PDB ID (e.g. 1IRK), downloaded from RCSB."));
      setTimeout(() => input.focus(), 0);
    }, [innerWidth / 2 - 220, 80]);
  }
  function openPanel() {
    const p = pathPanel("open", "Open File", "", async path => { if (await openPath(path)) panels.open?.remove(); });
    p.querySelector(".body").append(row(button("Browse this computer…", browse)),
      el("p", { className: "help" }, "Or drop files anywhere on the window."));
  }
  function savePanel() {
    const m = actingModel();
    if (!m) return;
    const base = (m.topo?.path || m.name).replace(/(\.[A-Za-z0-9]+)?(\.gz|\.bz2)?$/, "_edited$1");
    pathPanel("save", "Save Model", base, async p => { if (await savePath(p)) panels.save?.remove(); });
  }
  function saveSessionPanel() {
    const base = (state.title || "session").replace(/[\s:+]+/g, "_").replace(/\.[A-Za-z0-9]+$/, "");
    pathPanel("session", "Save Session", `${base}.bview`, async p => {
      try {
        const out = await app.busy("saving the session", () => app.post("api/save_session", { path: p, page: app.pageState() }));
        app.message(`saved the session: ${out.path} (open it with boonza view, or File | Open)`);
        panels.session?.remove(); delete panels.session;
      } catch (e) { app.message(e.message, true); }
    });
  }
  async function openPath(path) {
    try {
      const id = /^[0-9][A-Za-z0-9]{3}$/.test(path);  // a PDB ID, unless a file is named so
      const out = id ? await app.busy(`opening ${path.toUpperCase()} (from RCSB unless it is a file here)`,
                                      () => app.post("api/open", { path }))
                     : await app.post("api/open", { path });
      await addModels(out);
      if (path.endsWith(".bview") && out.page) await app.restorePage(out.page);
      app.message(`opened ${path}: ${out.added.length} model(s)`);
      return true;
    } catch (e) { app.message(e.message, true); return false; }
  }
  // A structure assembled from parts -- every pose of a docking run on one protein,
  // say -- can record where each part came from, as a `source` atom property.  This
  // opens that file: the whole complex the clicked molecule is a piece of.  It comes
  // from its own run in its own frame, so it is put onto the model it came from.
  //: which part Open Source of… acts on: the selected residue, else the one clicked
  function sourceTarget(c) {
    const R = c.m.topo.residues, of = c.m.topo.atoms.residue;
    const named = k => `${R.name[of[k]]}${R.resid[of[k]]}`;
    for (const [model, ids] of app.selection()) {
      if (model !== c.m || !ids.length) continue;
      const res = new Set();
      for (const k of ids) res.add(of[k]);
      if (res.size === 1) return { a: ids[0], name: named(ids[0]) };
      return { a: null, name: `${res.size} selected residues` };
    }
    return { a: c.a, name: c.molName };
  }

  async function openSource(m, a) {
    if (a === null || a === undefined) {
      app.message("select one residue, or right-click the one you want", true);
      return;
    }
    let path;
    try {
      path = (await app.getJSON(`api/atom?m=${m.index}&a=${a}`)).properties?.source;
    } catch (e) { app.message(e.message, true); return; }
    if (!path) { app.message("this atom records no source structure", true); return; }
    // the same file asked for twice is the model already open, not a second copy:
    // opening one a click piles up superposed copies of the same structure
    const name = path.split("/").pop();
    let opened = state.models.find(x => !x.closed && x !== m && x.name === name);
    if (opened) {
      app.message(`${name} is already open`);
    } else {
      const before = state.models.length;
      if (!await openPath(path)) return;
      opened = state.models[before];
      try {
        const out = await app.post("api/align", { m: opened.index, ref: m.index, method: "matchmaker" });
        await app.reloadModel(opened, false);
        app.message(`${name} onto ${m.name}: RMSD ${out.rmsd} Å over ${out.pairs} atom pairs`
          + " \u00b7 its ligand is where the run ended, not where it started");  // fmt: skip
      } catch (e) {  // it is open either way, in its own frame
        app.message(`${name} is open, but not superposed: ${e.message}`, true);
      }
    }
    // The complex alone.  solo would hide only the other files opened here and leave
    // the model they came from drawn -- but that model has a protein of its own, and
    // two proteins superposed, two cofactors, and this ligand lost among a thousand
    // others is not a structure anybody can read.  The cloud comes back from its box
    // in the model list.
    for (const x of app.live()) x.shown = x === opened;
    await app.show([opened]);
    app.centerView();
    app.message(`${opened.name} alone \u00b7 tick ${m.name} in the list to bring it back`);
  }
  async function savePath(path) {
    const m = actingModel();
    if (!m) return false;
    try { const out = await app.post("api/save", { m: m.index, path }); app.message(`saved ${out.path}`); return true; }
    catch (e) { app.message(e.message, true); return false; }
  }
  async function closeModel(which = null) {
    const m = which || app.currentModel();
    if (!m || m.closed) return;
    try {
      await app.post("api/close", { m: m.index });
      m.closed = true; m.shown = false;
      state.meters = state.meters.filter(x => x.model !== m);
      const rest = app.live();
      if (rest.length) {
        state.current = rest[0].index;
        if (!app.shownModels().length) await app.show([rest[0]]);
      }
      app.rebuild(); app.onModels(); app.onSelection();
      app.message(`closed ${m.name}`);
      if (!rest.length) app.onEmpty();
    } catch (e) { app.message(e.message, true); }
  }

  theme(localStorage.getItem("boonza.theme") || "dark");  // black by default
  setHydrogens(state.hydrogens);
  $("seqbar").hidden = localStorage.getItem("boonza.sequence") === "0";
  const onModels = app.onModels;
  app.onModels = () => { onModels(); paintSequence(); };
}

// --- the trackball: turn the view by dragging a ball, without a mouse's middle button --------

function trackball(app) {
  const svg = $("ball"), box = $("trackball");
  const R = 38, RING = 48;  // the ball and the ring around it, in the 100 x 100 viewBox
  let translate = false;
  // the ball's great circles turn with the view; the ring around it turns about z
  function paint() {
    if (box.hidden) return;
    const r = app.camera.rot, lines = [];
    for (const axis of [0, 1, 2]) {
      const pts = [];
      for (let k = 0; k <= 48; k++) {
        const t = 2 * Math.PI * k / 48;
        const u = axis === 0 ? [0, Math.cos(t), Math.sin(t)] : axis === 1 ? [Math.cos(t), 0, Math.sin(t)] : [Math.cos(t), Math.sin(t), 0];
        const v = [0, 1, 2].map(i => r[3 * i] * u[0] + r[3 * i + 1] * u[1] + r[3 * i + 2] * u[2]);
        pts.push([50 + (R - 1) * v[0], 50 - (R - 1) * v[1], v[2]]);
      }
      for (let k = 0; k < pts.length - 1; k++) {
        const front = pts[k][2] + pts[k + 1][2] > 0;
        lines.push(`<line x1="${pts[k][0].toFixed(1)}" y1="${pts[k][1].toFixed(1)}" x2="${pts[k + 1][0].toFixed(1)}" y2="${pts[k + 1][1].toFixed(1)}" `
          + `style="stroke: var(${front ? "--accent" : "--muted"})" stroke-width="${front ? 1.8 : 0.9}" opacity="${front ? 1 : 0.45}"/>`);
      }
    }
    svg.innerHTML = `<defs><radialGradient id="ball-shade" cx="38%" cy="32%" r="70%">
        <stop offset="0" style="stop-color: var(--hover)"/><stop offset="1" style="stop-color: var(--field)"/></radialGradient></defs>
      <circle cx="50" cy="50" r="${(R + RING) / 2}" fill="none" style="stroke: var(--line)" stroke-width="${RING - R}" opacity="0.7"/>
      <circle cx="50" cy="50" r="${R}" fill="url(#ball-shade)" style="stroke: var(--line)"/>${lines.join("")}`;
  }
  const draw = app.draw;
  app.draw = () => { draw(); paint(); };
  new MutationObserver(paint).observe(box, { attributes: true, attributeFilter: ["hidden"] });
  paint();

  // a point relative to the ball's centre, in ball radii, y up
  function where(e) {
    const b = svg.getBoundingClientRect(), px = Math.min(b.width, b.height) / 100 * R;
    return [(e.clientX - b.left - b.width / 2) / px, -(e.clientY - b.top - b.height / 2) / px];
  }
  // on the ball's front half (outside it: on its rim)
  function onBall([x, y]) {
    const d = Math.hypot(x, y);
    return d < 1 ? [x, y, Math.sqrt(1 - d * d)] : [x / d, y / d, 0];
  }

  $("ball-mode").onclick = () => {
    translate = !translate;
    $("ball-mode").textContent = translate ? "✥" : "⟳";
    $("ball-mode").classList.toggle("on", translate);
  };
  let drag = null;
  svg.addEventListener("contextmenu", e => e.preventDefault());
  svg.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    app.stopFlight();
    const p = where(e);
    drag = { x: e.clientX, y: e.clientY, p, z: Math.hypot(...p) > 1, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey };
    if (drag.alt && !app.beginMove()) drag.alt = false;
    svg.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  svg.addEventListener("pointermove", e => {
    if (!drag) { svg.style.cursor = Math.hypot(...where(e)) > 1 ? "alias" : ""; return; }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    const p0 = drag.p, p1 = where(e);
    drag.x = e.clientX; drag.y = e.clientY; drag.p = p1;
    if (drag.ctrl) { app.camera.zoom(Math.exp(dy * 0.01)); app.draw(); return; }
    if (translate) {
      if (drag.alt) app.translateSelected(dx * 3, dy * 3);
      else {
        const s = 2 * app.camera.distance * Math.tan(app.camera.fov / 2) / 300;
        app.camera.pan(dx * s, -dy * s); app.draw();
      }
      return;
    }
    let turn;
    if (drag.z) {  // the ring: the angle swept about the ball's centre
      turn = [0, 0, 1, Math.atan2(p1[1], p1[0]) - Math.atan2(p0[1], p0[0])];
    } else {       // the ball: the point grabbed stays under the pointer
      const a = onBall(p0), b = onBall(p1);
      const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      turn = [...c, Math.atan2(Math.hypot(...c), a[0] * b[0] + a[1] * b[1] + a[2] * b[2])];
    }
    if (drag.alt) app.rotateSelected(...turn);
    else { app.camera.rotate(...turn); app.draw(); }
  });
  const end = () => {
    if (drag?.alt) app.endMove();
    drag = null;
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  box.addEventListener("wheel", e => { e.preventDefault(); app.camera.zoom(Math.exp(e.deltaY * 0.0015)); app.draw(); }, { passive: false });
}
