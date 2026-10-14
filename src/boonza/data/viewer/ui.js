// The MOE window around the rendering area: the menu bar, the right button bar,
// the popup menu, the footer (editing area, Atoms, Ribbon, Contacts, Fog, ZClip),
// the corner (theme, R S L, trackball, dial), the command line and the panels.
// Render commands act on the selected atoms, or on all of them if none are (MOE).

import { colorAtoms, colorResidues, isColor, named } from "./colors.js";
import { STYLE_NAMES, STYLES } from "./geometry.js";

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
          menu(sub, r.right - 2, r.top - 3, depth + 1);
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
    if (!e.target.closest(".menu") && !e.target.closest(".menutitle") && !e.target.closest(".rbtn.sub")
        && !e.target.closest("#theme-button") && !e.target.closest("#f-atoms") && !e.target.closest("#f-ribbon")) closeMenus();
  });

  // --- what render commands act on ----------------------------------------------------

  function forTargets(fn) {
    for (const [m, ids] of app.targets()) fn(m, ids);
    app.rebuild();
  }
  const classIds = (m, k) => app.range(m.topo.natoms).filter(a => m.topo.atoms.class[a] === k);
  const hydrogenIds = m => app.range(m.topo.natoms).filter(a => m.topo.atoms.element[a] === "H");

  // the atoms of a category in every shown model: [[m, ids]]
  function category(what) {
    return app.shownModels().map(m => {
      const n = m.topo.natoms, sel = m.state.sel;
      const ids = {
        all: () => app.range(n), receptor: () => classIds(m, 0), ligand: () => classIds(m, 1),
        solvent: () => classIds(m, 2), hydrogens: () => hydrogenIds(m),
        selected: () => app.selected(m), unselected: () => app.range(n).filter(a => !sel[a]),
      }[what]();
      return [m, ids];
    });
  }

  function setShown(what, on) {
    for (const [m, ids] of category(what)) for (const a of ids) m.state.show[a] = on ? 1 : 0;
    if (on && what === "hydrogens" && state.hydrogens !== "all") { state.hydrogens = "all"; }
    app.rebuild();
  }

  function setStyle(style) {
    forTargets((m, ids) => { for (const a of ids) { m.state.style[a] = style; m.state.show[a] = 1; } });
  }

  function colorBy(spec) {
    forTargets((m, ids) => colorAtoms(spec, ids, m.topo, m.ss, m.state.rgb));
  }

  // ribbons: of the residues of the target atoms
  function targetResidues(m, ids) {
    const out = new Set();
    for (const a of ids) out.add(m.topo.atoms.residue[a]);
    return [...out].filter(r => m.topo.residues.trace[r] >= 0);
  }
  function ribbon(on) {
    forTargets((m, ids) => { for (const r of targetResidues(m, ids)) m.state.ribbon[r] = on ? 1 : 0; });
  }
  function ribbonStyle(style) {
    forTargets(m => { m.state.ribbonStyle = style; });
  }
  function ribbonColor(spec) {
    forTargets((m, ids) => colorResidues(spec, targetResidues(m, ids), m.topo, m.ss, m.state.ribbonRGB));
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

  // MOE's SiteView: the ligand and its pocket, everything else as ribbon, looked at
  async function siteView() {
    const m = app.currentModel();
    if (!m.topo) return;
    const lig = await serverSelect(m, `same residue as (${m.topo.ligand})`);
    if (!lig.length) { app.message("no ligand: Ligand | Choose Ligand picks one", true); return; }
    const pocket = await serverSelect(m, pocketSelection(m));
    const st = m.state;
    if (!m.beforeSite) m.beforeSite = { show: st.show.slice(), style: st.style.slice(), rgb: st.rgb.slice(), ribbon: st.ribbon.slice() };
    st.show.fill(0);
    for (let r = 0; r < st.ribbon.length; r++) st.ribbon[r] = m.topo.residues.trace[r] >= 0 ? 1 : 0;
    for (const a of pocket) { st.show[a] = 1; st.style[a] = app.STYLE.stick; }
    // a ligand of points (a site's) stays spheres, coloured as it was
    const molecule = lig.filter(a => m.topo.bondsOf[a].length);
    for (const a of lig) { st.show[a] = 1; st.style[a] = m.topo.bondsOf[a].length ? app.STYLE.stick : app.STYLE.space; }
    colorAtoms(`element/${m.topo.carbon || "green"}`, molecule, m.topo, m.ss, st.rgb);
    app.rebuild();
    const [c, r] = app.bounds(m, [...lig, ...pocket]);
    app.flyTo(c, Math.max(r, 8));
  }

  // WholeView, what MOE lacks: the display from before SiteView back, and everything in view
  function wholeView() {
    for (const m of app.shownModels()) {
      const b = m.beforeSite;
      if (!b || !m.state || b.show.length !== m.state.show.length) { m.beforeSite = null; continue; }
      Object.assign(m.state, { show: b.show, style: b.style, rgb: b.rgb, ribbon: b.ribbon });
      m.beforeSite = null;
    }
    app.rebuild();
    app.viewAll();
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
  async function selectServer(make) {
    app.clearSelection();
    for (const m of app.shownModels()) {
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

  const selectItems = () => [
    { label: "All", run: () => selectWhere(() => true), key: "Ctrl A" },
    { label: "Clear", run: () => { app.clearSelection(); app.rebuild(); app.onSelection(); } },
    { label: "Invert", run: invert },
    "-",
    { label: "Ligand", run: () => selectServer(m => `same residue as (${m.topo.ligand})`) },
    { label: "Pocket", run: () => selectServer(pocketSelection) },
    { label: "Receptor", run: () => selectWhere((m, a) => m.topo.atoms.class[a] === 0) },
    { label: "Solvent", run: () => selectWhere((m, a) => m.topo.atoms.class[a] === 2) },
    { label: "Hydrogens", run: () => selectWhere((m, a) => m.topo.atoms.element[a] === "H") },
    { label: "Element", sub: [
      ...["C", "N", "O", "S", "P"].map(e => ({ label: e, run: () => selectWhere((m, a) => m.topo.atoms.element[a] === e) })),
      { label: "Halogens", run: () => selectWhere((m, a) => ["F", "Cl", "Br", "I"].includes(m.topo.atoms.element[a])) },
    ] },
    "-",
    { label: "Selector…", run: () => openCLI("select ") },
  ];
  const extendItems = () => [
    { label: "Residues", run: () => extend("residues") },
    { label: "Chains", run: () => extend("chains") },
    { label: "Molecules", run: () => extend("molecules") },
    { label: "Hydrogens", run: () => extend("hydrogens") },
    { label: "Near (4.5 Å)", run: () => extendNear() },
  ];

  // --- render items ------------------------------------------------------------------

  const categories = [["All", "all"], ["Receptor", "receptor"], ["Ligand", "ligand"], ["Solvent", "solvent"],
                      ["Hydrogens", "hydrogens"], "-", ["Selected", "selected"], ["Unselected", "unselected"]];
  const hideItems = () => categories.map(c => c === "-" ? c : { label: c[0], run: () => setShown(c[1], false) });
  const showItems = () => categories.map(c => c === "-" ? c : { label: c[0], run: () => setShown(c[1], true) });
  const atomItems = () => [
    ...STYLE_NAMES.map((name, k) => ({ label: name, run: () => setStyle(k) })),
    "-",
    { label: "Color", sub: atomColorItems },
    { label: "Hide", run: () => forTargets((m, ids) => { for (const a of ids) m.state.show[a] = 0; }) },
  ];
  const atomColorItems = () => [
    { label: "Element", run: () => colorBy("element") },
    { label: "Carbon", sub: CARBONS.map(c => ({ label: c, run: () => colorBy(`element/${c}`) })) },
    { label: "Chain", run: () => colorBy("chain") },
    { label: "Residue Type", run: () => colorBy("restype") },
    { label: "Residue Name", run: () => colorBy("resname") },
    { label: "Secondary Structure", run: () => colorBy("ss") },
    { label: "B-factor", run: () => colorBy("bfactor") },
    { label: "Single", sub: SINGLE.map(c => ({ label: c, run: () => colorBy(c) })) },
  ];
  const ribbonItems = () => [
    { label: "Show", run: () => ribbon(true) },
    { label: "Hide", run: () => ribbon(false) },
    "-",
    { label: "Cartoon", run: () => ribbonStyle("cartoon"), checked: () => app.currentModel()?.state?.ribbonStyle === "cartoon" },
    { label: "Tube", run: () => ribbonStyle("tube"), checked: () => app.currentModel()?.state?.ribbonStyle === "tube" },
    "-",
    { label: "Color", sub: [
      { label: "Secondary Structure", run: () => ribbonColor("ss") },
      { label: "Chain", run: () => ribbonColor("chain") },
      { label: "Spectrum (N to C)", run: () => ribbonColor("spectrum") },
      { label: "Residue Type", run: () => ribbonColor("restype") },
      { label: "Single", sub: SINGLE.map(c => ({ label: c, run: () => ribbonColor(c) })) },
    ] },
  ];
  const labelItems = () => [
    ...[["Name", "name"], ["Element", "element"], ["Residue", "residue"], ["Charge", "charge"]]
      .map(([l, mode]) => ({ label: l, run: () => labels(mode) })),
    "-",
    { label: "Off", run: () => labels("off") },
  ];
  function cycleHydrogens() {
    state.hydrogens = { polar: "all", all: "none", none: "polar" }[state.hydrogens];
    app.message(`hydrogens: ${{ all: "all shown", polar: "polar ones shown", none: "hidden" }[state.hydrogens]}`);
    app.rebuild();
  }
  function toggleContacts() {
    state.contacts = !state.contacts;
    $("f-contacts").classList.toggle("on", state.contacts);
    app.refreshContacts().then(app.rebuild);
  }
  const renderItems = () => [
    { label: "Center", run: () => app.centerView() },
    { label: "SiteView", run: siteView },
    { label: "WholeView", run: wholeView, key: "W" },
    { label: "Reset View", run: () => app.resetView(), key: "R" },
    "-",
    { label: "Hide", sub: hideItems },
    { label: "Show", sub: showItems },
    { label: "Atoms", sub: atomItems },
    { label: "Ribbon", sub: ribbonItems },
    { label: "Atom Labels", sub: labelItems },
    { label: "Surface", sub: surfaceItems },
    { label: "Hydrogens", sub: [
      { label: "All", run: () => { state.hydrogens = "all"; app.rebuild(); }, checked: () => state.hydrogens === "all" },
      { label: "Polar", run: () => { state.hydrogens = "polar"; app.rebuild(); }, checked: () => state.hydrogens === "polar" },
      { label: "None", run: () => { state.hydrogens = "none"; app.rebuild(); }, checked: () => state.hydrogens === "none" },
    ] },
    { label: "Contacts", run: toggleContacts, checked: () => state.contacts },
    "-",
    { label: "Unit Cell", run: () => { state.cell = !state.cell; app.rebuild(); }, checked: () => !!state.cell },
    { label: "Secondary Structure of Each Frame", run: () => { state.ssEachFrame = !state.ssEachFrame; app.setFrame(state.frame); },
      checked: () => !!state.ssEachFrame },
    { label: "Projection", sub: [
      { label: "Orthographic", run: () => projection(true), checked: () => app.camera.ortho },
      { label: "Perspective", run: () => projection(false), checked: () => !app.camera.ortho },
    ] },
    { label: "Theme", sub: themeItems },
    { label: "Lighting", sub: [
      { label: "Standard", run: () => lighting(false), checked: () => !app.renderer.publication },
      { label: "Publication (outlines, occlusion)", run: () => lighting(true), checked: () => app.renderer.publication },
    ] },
    { label: "Full Screen", run: () => document.documentElement.requestFullscreen?.(), key: "F12" },
  ];
  // Render | Surface: of the selected atoms, else of the current model's receptor
  const SURFACES = [["plain", "Molecular"], ["electrostatic", "Electrostatic"], ["hydrophobic", "Hydrophobic"]];
  const surfaceItems = () => [
    ...SURFACES.map(([kind, label]) => ({ label, run: () => setSurface(kind),
      checked: () => app.currentModel()?.surfaceSpec?.kind === kind })),
    "-",
    { label: "Off", run: () => setSurface(null) },
  ];
  let surfaceVersion = 0;
  function setSurface(kind) {
    const sel = app.selection();
    const targets = sel.length ? sel : [[app.currentModel(), null]];
    for (const [m, ids] of targets) {
      if (!m?.topo) continue;
      if (!kind) { m.surfaceSpec = null; m.surfaceMesh = null; continue; }
      let atoms = ids;
      if (!atoms) {
        const A = m.topo.atoms, all = app.range(m.topo.natoms);
        atoms = all.filter(a => A.class[a] === 0);
        if (!atoms.length) atoms = all.filter(a => A.class[a] !== 2);
      }
      m.surfaceSpec = { kind, ids: atoms, version: ++surfaceVersion };
    }
    if (kind) app.message(`${kind === "plain" ? "molecular" : kind} surface${kind === "electrostatic" ? ": red −5 to blue +5 kT/e (Coulomb, ε = 4r)" : kind === "hydrophobic" ? ": dark cyan hydrophilic, goldenrod hydrophobic" : ""}`);
    app.rebuild();
  }

  const themeItems = () => [
    { label: "Blanc", run: () => theme("blanc"), checked: () => document.body.classList.contains("blanc") },
    { label: "Noir", run: () => theme("noir"), checked: () => document.body.classList.contains("noir") },
    "-",
    { label: "Save Picture…", run: savePicture },
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

  app.setTheme = name => theme(name);
  function theme(name) {
    document.body.className = name;
    app.renderer.background = name === "noir" ? [0, 0, 0] : [1, 1, 1];
    localStorage.setItem("boonza.theme", name);
    app.draw();
  }
  // File | Save Picture: the view as it is, larger and finished (outlines, occlusion)
  function savePicture() {
    panel("picture", "Save Picture", body => {
      const w0 = app.canvas.clientWidth, h0 = app.canvas.clientHeight;
      const size = el("select");
      for (const k of [1, 2, 3, 4]) size.add(new Option(`${k}× · ${k * w0} × ${k * h0}`, k, false, k === 3));
      const back = el("select");
      for (const [v, l] of [["view", "as the view"], ["white", "white"], ["black", "black"], ["none", "transparent"]]) back.add(new Option(l, v));
      const check = (label, on) => {
        const box = el("input", { type: "checkbox", checked: on });
        return [box, el("label", {}, box, ` ${label}`)];
      };
      const [outline, outlineLabel] = check("outlines", true);
      const [ao, aoLabel] = check("ambient occlusion (shadows in crevices)", true);
      const [fog, fogLabel] = check("fog (depth cue)", app.renderer.fog.on);
      const go = button("Save PNG", () => {
        const k = +size.value;
        const bg = { white: [1, 1, 1], black: [0, 0, 0] }[back.value] || null;
        app.message("drawing the picture…");
        setTimeout(() => {  // let the message show first
          try {
            const pic = app.renderer.picture(app.camera, k * w0, k * h0, { outline: outline.checked, ao: ao.checked,
              background: bg, transparent: back.value === "none", fog: fog.checked });
            pic.toBlob(blob => {
              const name = (state.title || "boonza").replace(/[\s:/·+]+/g, "_").replace(/_+$/, "");
              el("a", { href: URL.createObjectURL(blob), download: `${name}-${state.frame + 1}.png` }).click();
              app.message(`saved a ${pic.width} × ${pic.height} picture`);
            });
          } catch (e) { app.message(e.message, true); }
          app.draw();
        }, 30);
      }, "blue");
      body.append(row(el("span", { className: "what" }, "size"), size));
      body.append(row(el("span", { className: "what" }, "background"), back));
      body.append(row(outlineLabel), row(aoLabel), row(fogLabel), row(go));
      body.append(el("p", { className: "help" }, "Labels are not drawn into the picture. Render | Lighting | Publication "
        + "shows outlines and occlusion on screen as well."));
    });
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
  async function serverEdit(m, url, body, topology = true) {
    try {
      const out = await app.post(url, { m: m.index, ...body });
      await app.reloadModel(m, out.topology ?? topology);
      return out;
    } catch (e) { app.message(e.message, true); return null; }
  }
  const autoH = () => localStorage.getItem("boonza.autoH") !== "0";
  async function edit(op, value) {
    const t = editTarget(op === "bond" ? 2 : 1);
    if (!t) return;
    const [m, atoms] = t;
    if (op === "bond" && atoms.length !== 2) { app.message("select the bond's two atoms", true); return; }
    await serverEdit(m, "api/edit", { op, atoms, value, hydrogens: op === "name" ? false : autoH() });
  }
  async function undo(redo = false) {
    const m = app.selection()[0]?.[0] || app.currentModel();
    const out = await serverEdit(m, redo ? "api/redo" : "api/undo", {});
    if (out) app.message(redo ? "redone" : "undone");
  }
  async function deleteSelected() {
    const t = editTarget();
    if (t) await serverEdit(t[0], "api/edit", { op: "delete", atoms: t[1], hydrogens: autoH() });
  }
  async function hydrogens(what) {
    const m = app.selection()[0]?.[0] || app.currentModel();
    const sel = app.selected(m);
    const A = m.topo.atoms;
    const pool = sel.length ? sel : app.range(m.topo.natoms);
    if (what === "delete") {
      const hs = pool.filter(a => A.element[a] === "H");
      if (hs.length) await serverEdit(m, "api/edit", { op: "delete", atoms: hs, hydrogens: false });
    } else {
      const heavy = pool.filter(a => A.element[a] !== "H");
      if (heavy.length) await serverEdit(m, "api/edit", { op: "hydrogens", atoms: heavy });
    }
  }

  // constraints, as MOE's Constrain: 0 free, 1 tethered, 2 fixed
  function constrain(kind) {
    const sel = app.selection();
    if (!sel.length && kind !== "freeall") { app.message("select atoms first", true); return; }
    if (kind === "freeall") for (const m of app.shownModels()) m.state.constraint.fill(0);
    for (const [m, ids] of sel) for (const a of ids) {
      if (kind === "unfix") { if (m.state.constraint[a] === 2) m.state.constraint[a] = 0; }
      else m.state.constraint[a] = { tether: 1, fix: 2, free: 0 }[kind];
    }
    const counts = constraintCounts(app.currentModel());
    app.message(`fixed ${counts.fixed}, tethered ${counts.tethered}`);
  }
  function constraintCounts(m) {
    const c = m?.state?.constraint || [];
    let fixed = 0, tethered = 0;
    for (const x of c) { if (x === 2) fixed++; else if (x === 1) tethered++; }
    return { fixed, tethered };
  }

  async function minimize({ ligandOnly = false, restraint = null } = {}) {
    const m = app.selection()[0]?.[0] || app.currentModel();
    if (!m.topo?.editable) { app.message("this model cannot be minimized", true); return; }
    const k = restraint ?? +(localStorage.getItem("boonza.restraint") || 10);
    const c = m.state.constraint, body = { m: m.index, restraint: k };
    if (ligandOnly) {
      body.fixed = app.range(m.topo.natoms).filter(a => m.topo.atoms.class[a] !== 1);
    } else if (c.some(x => x)) {
      body.fixed = app.range(c.length).filter(a => c[a] === 2);
      body.tethered = app.range(c.length).filter(a => c[a] === 1);
    }
    try {
      const out = await app.busy("minimizing (the first time sets up the force field)", () => app.post("api/minimize", body));
      await app.reloadModel(m, false);
      const how = out.mobile === "constraints" ? `${out.fixed_atoms} fixed, ${out.held_atoms} tethered`
        : `the ligand and its pocket free, ${out.held_atoms} heavy atoms tethered`;
      app.message(`Minimized: E ${out.energy_before} → ${out.energy_after} kJ/mol; ${how}; `
        + `moved atoms RMSD ${out.mobile_rmsd} Å (largest ${out.mobile_max} Å)`);
    } catch (e) { app.message(e.message, true); }
  }

  function measureMeter() {
    const t = editTarget(2);
    if (!t) return;
    const [m, atoms] = t;
    if (atoms.length > 4) { app.message("a meter is 2, 3 or 4 atoms", true); return; }
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
      { label: "Save…", run: savePanel, key: "Ctrl S" },
      { label: "Save Session…", run: saveSessionPanel },
      { label: "Close", run: () => closeModel() },
      "-",
      { label: "Save Picture…", run: savePicture },
    ],
    Edit: () => [
      { label: "Undo", run: () => undo(), key: "Ctrl Z" },
      { label: "Redo", run: () => undo(true), key: "Ctrl Y" },
      "-",
      { label: "Delete", run: deleteSelected, key: "Del" },
      { label: "Build", sub: [
        { label: "Builder…", run: builderPanel },
        { label: "Hydrogens", sub: [
          { label: "Add Hydrogens", run: () => hydrogens("add") },
          { label: "Adjust Hydrogens", run: () => hydrogens("add") },
          { label: "Delete Hydrogens", run: () => hydrogens("delete") },
        ] },
      ] },
      { label: "Constrain", sub: constrainItems },
      { label: "Minimize…", run: minimizePanel },
      { label: "Measure", sub: measureItems },
    ],
    Select: selectItems,
    Render: renderItems,
    Protein: () => [
      { label: "Fetch from PDB…", run: fetchPanel },
      { label: "Protein Builder…", run: () => openDock("protein") },
      { label: "Align/Superpose…", run: alignPanel, off: () => app.live().length < 2 },
    ],
    Compute: () => [
      { label: "Assign Force Field", run: assignForceField },
      { label: "Energy Minimize…", run: minimizePanel },
      { label: "Conformations", sub: [{ label: "Torsion Profile…", run: () => profilePanel() }] },
      { label: "Ligand Interactions", run: toggleContacts, checked: () => state.contacts },
      { label: "Dynamics Player", run: () => { $("player").hidden = app.nframes() < 2; },
        off: () => app.nframes() < 2 },
    ],
    Window: () => [
      { label: "Models", run: modelsPanel, checked: () => !$("window").classList.contains("nomodels") },
      { label: "Command Line", run: () => openCLI(), key: "Space" },
      { label: "Python Console", run: togglePython, key: "`", off: () => !app.PYTHON },
    ],
    Help: () => [
      { label: "Mouse…", run: mousePanel },
      { label: "Keys…", run: () => app.message("Space: command line · Ctrl Z / Ctrl Y: undo, redo · Del: delete · ←/→: frames · ↑/↓: next model · R: reset view · F12: full screen") },
    ],
  };
  const constrainItems = () => [
    { label: "Tether", run: () => constrain("tether") },
    { label: "Fix", run: () => constrain("fix") },
    { label: "Unfix", run: () => constrain("unfix") },
    { label: "Free", run: () => constrain("free") },
    "-",
    { label: "Free All", run: () => constrain("freeall") },
  ];
  const measureItems = () => [
    { label: "Meter (2-4 selected atoms)", run: measureMeter },
    { label: "Clear Meters", run: () => { state.meters = []; app.rebuild(); } },
  ];

  for (const [name, items] of Object.entries(menus)) {
    const t = el("div", { className: "menutitle" });
    t.innerHTML = `<u>${name[0]}</u>${name.slice(1)}`;
    t.onclick = () => {
      const open = t.classList.contains("open");
      closeMenus();
      if (open) return;
      t.classList.add("open");
      const r = t.getBoundingClientRect();
      menu(items, r.left, r.bottom);
    };
    t.onmouseenter = () => { if (document.querySelector(".menutitle.open") && !t.classList.contains("open")) t.click(); };
    $("menus").append(t);
  }

  // --- right button bar (RHS) --------------------------------------------------------------

  const rhs = [
    { label: "System", blue: true, sub: systemItems },
    "-",
    { label: "Open", run: openPanel },
    { label: "Constrain", sub: constrainItems },
    { label: "Close", red: true, run: () => closeModel() },
    "-",
    { label: "Center", blue: true, run: () => app.centerView() },
    { label: "SiteView", run: siteView },
    { label: "WholeView", run: wholeView },
    "-",
    { label: "Hydrogens", run: cycleHydrogens },
    { label: "Hide", sub: hideItems },
    { label: "Show", sub: showItems },
    "-",
    { label: "Ligand", sub: () => [
      { label: "Ligand Interactions", run: toggleContacts, checked: () => state.contacts },
      "-",
      { label: "Choose Ligand", run: chooseLigand },
      { label: "Ligand Torsion Profile", run: () => profilePanel() },
      "-",
      { label: "Minimize (Vac)", run: () => minimize({ ligandOnly: true }) },
    ] },
    { label: "Measure", sub: measureItems },
    "-",
    { label: "Builder", blue: true, run: builderPanel },
    { label: "Minimize", run: () => minimize() },
    "-",
    { label: "Select", sub: selectItems },
    { label: "Extend", sub: extendItems },
    "-",
    { label: "Delete", red: true, run: deleteSelected },
  ];
  function systemItems() {
    return [
      ...app.live().map(m => ({ label: m.name, checked: () => m.shown,
                                run: () => { app.state.current = m.index; app.show([m], !m.shown); } })),
      "-",
      { label: "Models…", run: modelsPanel },
    ];
  }
  // a static page (an export) shows models: what would change them is left off
  const EDITS = new Set(["Open", "Constrain", "Close", "Builder", "Minimize", "Delete"]);
  for (const b of rhs) {
    if (app.STATIC && EDITS.has(b.label)) continue;
    if (b === "-") { $("rhs").append(el("div", { className: "sep" })); continue; }
    const btn = el("button", { className: `rbtn${b.sub ? " sub" : ""}${b.blue ? " blue" : ""}${b.red ? " red" : ""}` }, b.label);
    btn.onclick = () => {
      if (b.sub) {
        const r = btn.getBoundingClientRect();
        const items = typeof b.sub === "function" ? b.sub() : b.sub;
        const m = menu(items, 0, r.top);
        m.style.left = `${r.left - m.getBoundingClientRect().width - 2}px`;
      } else b.run();
    };
    $("rhs").append(btn);
  }

  async function chooseLigand() {
    const t = editTarget();
    if (!t) return;
    const [m, atoms] = t;
    try {
      const out = await app.post("api/set_ligand", { m: m.index, atoms });
      m.topo.ligand = out.ligand;
      app.message("ligand chosen: SiteView, Pocket, contacts and Minimize use it");
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
        ? "  · no force field yet: Compute | Assign Force Field"
        : `  · q ${info.charge.toFixed(4)} e · σ ${info.sigma?.toFixed(3) ?? "–"} Å · ε ${info.epsilon?.toFixed(4) ?? "–"} kcal/mol`
          + `${info.type ? ` · type ${info.type}` : ""}${info.force_field === "assigned" ? " (Amber ff19SB/GAFF2)" : ""}`;
      app.message(`${app.atomLabel(m, a)}${ff}`);
    } catch (e) { /* the label alone */ }
  };
  async function assignForceField() {
    const m = app.selection()[0]?.[0] || app.currentModel();
    if (!m.topo?.editable) { app.message("this model has no atoms to assign a force field to", true); return; }
    try {
      await app.busy("assigning the force field (Amber ff19SB, GAFF2 for ligands)", () => app.post("api/assign_ff", { m: m.index }));
      const sel = app.selected(m);
      if (sel.length === 1) await app.describeAtom(m, sel[0]);
      else app.message("force field assigned: click an atom to see its charge, σ and ε");
    } catch (e) { app.message(e.message, true); }
  }

  app.popup = (x, y) => menu(() => [
    ...renderItems().slice(0, 9),
    "-",
    { label: "Select", sub: selectItems },
    { label: "Extend", sub: extendItems },
    { label: "System", sub: systemItems },
  ], x, y);

  // --- footer ---------------------------------------------------------------------------------

  $("f-atoms").onclick = () => { const r = $("f-atoms").getBoundingClientRect(); const m = menu(atomItems, r.left, 0); m.style.top = `${r.top - m.getBoundingClientRect().height}px`; };
  $("f-ribbon").onclick = () => { const r = $("f-ribbon").getBoundingClientRect(); const m = menu(ribbonItems, r.left, 0); m.style.top = `${r.top - m.getBoundingClientRect().height}px`; };
  $("f-contacts").onclick = toggleContacts;
  $("f-fog").onclick = fogPanel;
  $("f-zclip").onclick = zclipPanel;
  $("f-2d").onclick = () => {
    const on = $("lig2d").hidden;
    $("lig2d").hidden = !on;
    $("f-2d").classList.toggle("on", on);
    if (on) refresh2D();
  };
  let pending2D = 0;
  async function refresh2D() {
    if ($("lig2d").hidden) return;
    const ticket = ++pending2D;
    const sel = app.selection();
    const m = sel[0]?.[0] || app.currentModel();
    if (!m?.topo) return;
    const ids = sel[0]?.[1] || [];
    const dark = document.body.classList.contains("noir") ? 1 : 0;
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
  $("ff-button").onclick = () => app.message("Force field for Minimize and the energy profiles: Amber ff19SB "
    + "(with its water and ion partners) through viparr, GAFF2 with AM1-BCC charges for anything else (AmberTools)");
  $("pbc-button").onclick = () => { state.cell = !state.cell; app.rebuild(); $("pbc-button").classList.toggle("on", state.cell); };

  // the footer editing area: what the selection is, and its geometry to edit
  app.onSelection = () => {
    setTimeout(refresh2D, 0);
    const box = $("footer-edit");
    box.innerHTML = "";
    const sel = app.selection();
    const m = sel.length === 1 ? sel[0][0] : app.currentModel();
    $("pbc-button").classList.toggle("lit", !!(m?.box && m.box.flat().some(x => x)));
    if (!m || !m.topo) return;
    const n = sel.reduce((k, [, ids]) => k + ids.length, 0);
    if (!n) {
      const R = m.topo.residues;
      box.append(el("span", { className: "what" },
        `${m.name}: ${m.topo.natoms} atoms, ${R.name.length} residues, ${m.topo.chains.name.length} chains`));
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
      box.append(el("span", {}, app.atomLabel(m, a, false).split(" ")[0]), name,
                 el("span", { className: "what" }, `[${m.topo.atoms.element[a]}]`));
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
  };

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

  // --- corner: theme, R S L, trackball, dial ----------------------------------------------------

  $("theme-button").onclick = () => {
    const r = $("theme-button").getBoundingClientRect();
    const m = menu(themeItems, 0, 0);
    const b = m.getBoundingClientRect();
    m.style.left = `${r.right - b.width}px`; m.style.top = `${r.top - b.height}px`;
  };
  for (const b of document.querySelectorAll("#rsl button")) {
    const k = +b.dataset.mask;
    b.onclick = () => { state.masks[k] = !state.masks[k]; b.classList.toggle("off", !state.masks[k]); };
    b.ondblclick = () => {  // only this class
      for (const o of document.querySelectorAll("#rsl button")) {
        const j = +o.dataset.mask;
        if (j === 3) continue;
        state.masks[j] = j === k;
        o.classList.toggle("off", !state.masks[j]);
      }
    };
  }
  trackball(app);
  dial(app);

  // --- command line -------------------------------------------------------------------------------

  function openCLI(text = "") {
    $("cli").hidden = false;
    $("cli-input").value = text;
    $("cli-input").focus();
  }
  const closeCLI = () => { $("cli").hidden = true; $("cli-input").blur(); };
  $("btn-cli").onclick = () => openCLI();
  $("cli-close").onclick = closeCLI;
  $("cli").onsubmit = async e => {
    e.preventDefault();
    const keep = e.submitter === undefined && window.event?.shiftKey;
    await command($("cli-input").value.trim());
    if (!keep) closeCLI();
  };
  $("cli-input").onkeydown = e => {
    if (e.key === "Escape") closeCLI();
    if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); command($("cli-input").value.trim()); }
  };

  async function command(text) {
    if (!text) return;
    const [word, ...rest] = text.split(/\s+/);
    const arg = rest.join(" ");
    const on = async (sel, fn) => {  // a selection in each shown model, or the targets
      if (!sel) { forTargets(fn); return; }
      for (const m of app.shownModels()) fn(m, await serverSelect(m, sel));
      app.rebuild();
    };
    switch (word) {
      case "help":
        app.message("a selection selects atoms (boonza's language, e.g. resname LIG or within 5 of resname LIG)\n"
          + "commands: select SEL · show SEL · hide SEL · center [SEL] · style line|stick|ballline|ballstick|space [SEL] · "
          + "color SCHEME [SEL] · label name|element|residue|charge|off [SEL] · minimize · undo · redo · "
          + "open PATH · save PATH · siteview · wholeview · surface [electrostatic|hydrophobic|off] · reset");
        return;
      case "select": await selectServer(() => arg); return;
      case "show": await on(arg, (m, ids) => { for (const a of ids) m.state.show[a] = 1; }); return;
      case "hide": await on(arg, (m, ids) => { for (const a of ids) m.state.show[a] = 0; }); return;
      case "center": if (arg) await selectServer(() => arg); app.centerView(); return;
      case "style": {
        const [s, ...sel] = rest;
        const k = STYLES.indexOf(s);
        if (k < 0) { app.message(`styles: ${STYLES.join(", ")}`, true); return; }
        await on(sel.join(" "), (m, ids) => { for (const a of ids) { m.state.style[a] = k; m.state.show[a] = 1; } });
        return;
      }
      case "color": {
        const [c, ...sel] = rest;
        if (!isColor(c)) { app.message("colors: element, element/green, chain, restype, resname, ss, bfactor, or a name", true); return; }
        await on(sel.join(" "), (m, ids) => colorAtoms(c, ids, m.topo, m.ss, m.state.rgb));
        return;
      }
      case "label": {
        const [mode, ...sel] = rest;
        if (mode !== "off") state.labelMode = mode;
        await on(sel.join(" "), (m, ids) => { for (const a of ids) m.state.labels[a] = mode === "off" ? 0 : 1; });
        return;
      }
      case "minimize": await minimize(); return;
      case "undo": await undo(); return;
      case "redo": await undo(true); return;
      case "open": await openPath(arg); return;
      case "save": await savePath(arg); return;
      case "siteview": await siteView(); return;
      case "wholeview": wholeView(); return;
      case "surface": setSurface(arg === "off" ? null : ["electrostatic", "hydrophobic"].includes(arg) ? arg : "plain"); return;
      case "reset": app.resetView(); return;
      default: await selectServer(() => text);  // a selection
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
      else if (a.do === "show") for (const k of a.ids) { st.show[k] = 1; if (a.style) st.style[k] = STYLES.indexOf(a.style); }
      else if (a.do === "hide") for (const k of a.ids) st.show[k] = 0;
      else if (a.do === "color") colorAtoms(a.scheme, a.ids, m.topo, m.ss, st.rgb);
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
    if (e.key === "Escape") { closeMenus(); if (!$("cli").hidden) closeCLI(); return; }
    if (typing) return;
    const ctrl = e.ctrlKey || e.metaKey;
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
    else if (e.key === "w" || e.key === "W") wholeView();
    else if (e.key === "F12") document.documentElement.requestFullscreen?.();
  });

  // --- the player -----------------------------------------------------------------------------------

  $("slider").oninput = () => app.setFrame(parseInt($("slider").value));
  $("play").onclick = () => app.play();
  window.addEventListener("resize", app.draw);

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
    app.draw();
  }
  function closeDock() { $("ldock").hidden = true; $("window").classList.remove("docked"); app.draw(); }
  function toggleDock(tab) { if (!$("ldock").hidden && dockTab === tab) closeDock(); else openDock(tab); }
  for (const b of document.querySelectorAll("#ldock-tabs [data-tab]")) b.onclick = () => openDock(b.dataset.tab);
  $("ldock-close").onclick = closeDock;
  const grid = (cls = "grid") => el("div", { className: cls });

  function atomBuilder(body) {
    body.append(el("h3", {}, "Element"));
    const g = grid();
    for (const e of ELEMENTS) g.append(button(e, () => edit("element", e)));
    body.append(g);
    const other = el("input", { placeholder: "other element", spellcheck: false });
    body.append(row(other, button("Set", () => other.value && edit("element", other.value))));
    body.append(el("h3", {}, "Charge and bond"));
    body.append(row(el("span", { className: "what" }, "charge"), button("−1", () => edit("charge", -1)),
                    button("0", () => edit("charge", 0)), button("+1", () => edit("charge", 1))));
    body.append(row(el("span", { className: "what" }, "bond"), button("none", () => edit("bond", 0)),
                    button("single", () => edit("bond", 1)), button("double", () => edit("bond", 2)),
                    button("triple", () => edit("bond", 3))));
    body.append(el("h3", {}, "Functional groups"));
    body.append(el("p", { className: "help" }, "Select a hydrogen (or the atom carrying it): the group takes its place."));
    const fg = grid("grid wide");
    for (const [n, smi] of Object.entries(app.state.groups || {})) {  // the name, its SMILES below
      const b = button("", () => edit("attach", n), "group");
      b.title = `${n}: ${smi} (the first atom bonds)`;
      b.append(el("span", {}, n), el("small", {}, smi));
      fg.append(b);
    }
    body.append(fg);
    const smiles = el("input", { placeholder: "a group as SMILES, first atom bonds", spellcheck: false });
    body.append(row(smiles, button("Attach", () => smiles.value && edit("attach", smiles.value.trim()))));
    body.append(el("h3", {}, "New molecule"));
    const mol = el("input", { placeholder: "SMILES, e.g. c1ccccc1", spellcheck: false });
    const add = button("Add", async () => {
      const m = app.currentModel();
      if (!mol.value.trim() || !m?.topo?.editable) return;
      await serverEdit(m, "api/edit", { op: "molecule", value: { smiles: mol.value.trim(), centre: app.camera.center } });
    });
    mol.onkeydown = e => { if (e.key === "Enter") add.click(); };
    body.append(row(mol, add));
    body.append(el("h3", {}, "Hydrogens and atoms"));
    body.append(row(button("Add / Adjust H", () => hydrogens("add")), button("Delete H", () => hydrogens("delete")),
                    button("Delete atoms", deleteSelected)));
    body.append(row(button("Undo", () => undo()), button("Redo", () => undo(true))));
    const auto = el("input", { type: "checkbox", checked: autoH() });
    auto.onchange = () => localStorage.setItem("boonza.autoH", auto.checked ? "1" : "0");
    body.append(row(el("label", {}, auto, " adjust hydrogens after each edit")));
    body.append(el("p", { className: "help" }, "Acts on the selected atoms. Alt + middle drag moves them, "
      + "Shift + Alt + middle translates them, Alt + left drag turns a selected bond."));
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

  function minimizePanel() {
    panel("minimize", "Energy Minimize", body => {
      const m = app.currentModel();
      const c = constraintCounts(m);
      const k = el("input", { type: "number", value: localStorage.getItem("boonza.restraint") || 10, min: 0, step: 1, style: "width:60px" });
      k.onchange = () => localStorage.setItem("boonza.restraint", k.value);
      const what = c.fixed || c.tethered
        ? `Constraints: ${c.fixed} fixed, ${c.tethered} tethered (Edit | Constrain); the rest is free.`
        : "No constraints: the ligand and its pocket move, the other heavy atoms are tethered. "
          + "Constrain sets your own.";
      body.append(el("p", { className: "help" }, what));
      body.append(row(el("span", { className: "what" }, "Tether"), k, el("span", {}, "kcal/mol/Å²")));
      body.append(el("p", { className: "help" }, "Hydrogens are never held. Force field: Amber ff19SB, "
        + "GAFF2/AM1-BCC for ligands; vacuum with a 12 Å reaction field."));
      body.append(row(button("Minimize", () => minimize({ restraint: +k.value }), "blue"),
                      button("Ligand only (Vac)", () => minimize({ ligandOnly: true, restraint: +k.value }))));
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
  // two curves, as MOE's: in context (the rest rigid; angles that clash are forbidden,
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

  // --- the models, docked on the right: on/off, colour, representation, the table's data --

  const PRESETS = [["site", "SiteView"], ["ribbon", "Ribbon"], ["ribbonlig", "Ribbon + ligand"],
                   ["ribbonlines", "Ribbon + lines"], ["line", "Lines"], ["stick", "Sticks"],
                   ["ballstick", "Ball and Stick"], ["space", "Space filling"],
                   ["ribbonspace", "Ribbon + Space Filling"]];
  const MODEL_COLORS = [["default", "Default"], ["chain", "By chain"], ...["green", "cyan", "magenta", "yellow",
    "orange", "salmon", "purple", "slate", "white", "gray"].map(c => [c, c])];

  async function preset(m, kind) {
    await app.load(m);
    const st = m.state, n = m.topo.natoms, R = m.topo.residues;
    m.preset = kind;
    if (kind === "site") {  // the defaults: cartoon, pocket, ligand
      const reps = await Promise.all(m.topo.reps.map(async r => ({ ...r, ids: await serverSelect(m, r.sel) })));
      st.show.fill(0); st.ribbon.fill(0);
      app.applyReps(m, reps);
    } else {
      const ribbon = kind.startsWith("ribbon");
      for (let r = 0; r < st.ribbon.length; r++) st.ribbon[r] = ribbon && R.trace[r] >= 0 ? 1 : 0;
      const style = { line: 0, stick: 1, ballstick: 3, space: 4, ribbonlines: 0, ribbonlig: 1, ribbonspace: 4 }[kind];
      st.show.fill(0);
      let ids = app.range(n);
      if (kind === "ribbonlig" || kind === "ribbonspace") ids = await serverSelect(m, `same residue as (${m.topo.ligand})`);
      if (kind === "ribbonlines") ids = ids.filter(a => m.topo.atoms.class[a] !== 2);
      if (kind !== "ribbon") for (const a of ids) { st.show[a] = 1; st.style[a] = style; }
    }
    app.rebuild();
  }

  function colorModel(m, spec) {
    if (!m.state) return;
    m.color = spec;
    const all = app.range(m.topo.natoms), res = app.range(m.topo.residues.name.length);
    if (spec === "default") {
      colorAtoms("element", all, m.topo, m.ss, m.state.rgb);
      colorResidues(m.topo.cg ? "chain" : "ss", res, m.topo, m.ss, m.state.ribbonRGB);
    } else if (spec === "chain") {
      colorAtoms("element", all, m.topo, m.ss, m.state.rgb);
      const carbons = all.filter(a => m.topo.atoms.element[a] === "C");
      colorAtoms("chain", carbons, m.topo, m.ss, m.state.rgb);
      colorResidues("chain", res, m.topo, m.ss, m.state.ribbonRGB);
    } else {  // one colour: the carbons and the ribbon
      colorAtoms(`element/${spec}`, all, m.topo, m.ss, m.state.rgb);
      colorResidues(spec, res, m.topo, m.ss, m.state.ribbonRGB);
    }
    m.surfaceMesh = null;  // a plain surface takes the atoms' colours
    app.rebuild();
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
        if (group) list.append(el("div", { className: "mgroup" }, group));
      }
      const row = el("div", { className: `mrow${m.shown ? " on" : ""}${m.index === state.current ? " current" : ""}` });
      const box = el("input", { type: "checkbox", checked: m.shown, title: "show" });
      box.onchange = () => { state.current = m.index; app.show([m], box.checked); };
      const sw = el("span", { className: "swatch" });
      sw.style.background = swatchColor(m.color);
      const name = el("span", { className: "name", title: `${m.name}: click to show it alone and look at it` }, m.name);
      name.onclick = () => app.solo(m);
      const color = el("select", { title: "colour" });
      for (const [v, l] of MODEL_COLORS) color.add(new Option(l, v, false, (m.color || "default") === v));
      if (m.color && !MODEL_COLORS.some(([v]) => v === m.color)) color.add(new Option("its own", m.color, false, true));
      color.onchange = async () => { await app.load(m); colorModel(m, color.value); sw.style.background = swatchColor(color.value); };
      const rep = el("select", { title: "representation" });
      for (const [v, l] of PRESETS) rep.add(new Option(l, v, false, (m.preset || "site") === v));
      rep.onchange = () => preset(m, rep.value);
      const x = el("button", { className: "x", title: "close this model" }, "✕");
      x.onclick = () => closeModel(m);
      row.append(el("div", { className: "top" }, box, sw, name, color, rep, x));
      const data = Object.entries(m.row || {}).filter(([k]) => k !== "name");
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
  app.onModels = modelsDock;
  function modelsPanel() { $("window").classList.toggle("nomodels"); app.draw(); }

  const cell = v => typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")) : v ?? "";
  function sortedModels() {
    const list = app.live();
    const { column, descending } = state.sort || {};
    if (!column) return list;
    const key = m => column === "name" ? m.name : m.row?.[column];
    return [...list].sort((a, b) => {
      const x = key(a), y = key(b);
      if (x === y) return a.index - b.index;
      if (x === undefined) return 1;
      if (y === undefined) return -1;
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return descending ? -c : c;
    });
  }
  function stepModel(by) {
    const now = app.currentModel();
    const list = sortedModels().filter(m => m.group === now.group);
    const next = list[Math.min(Math.max(list.indexOf(now) + by, 0), list.length - 1)];
    if (next && next !== now) app.solo(next);
  }
  $("btn-models").onclick = modelsPanel;

  function mousePanel() {
    panel("mouse", "Mouse", body => {
      const rows = [
        ["Left click", "select an atom (a bond: its two atoms; a ribbon: its residue)"],
        ["Shift / Ctrl click", "toggle / the whole residue"], ["Double click", "the molecule"],
        ["Left drag", "select a box (Shift adds, Ctrl takes residues)"], ["Click away", "clear the selection"],
        ["Middle drag", "rotate (in the perimeter: about z)"], ["Shift + middle", "pan"], ["Ctrl + middle", "zoom"],
        ["Middle click", "centre of rotation (away from atoms: reset)"], ["Wheel", "zoom"],
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
      body.append(el("p", { className: "help" }, "On a Mac, Ctrl is Cmd and Alt is Option. Without a middle button, "
        + "use the two-button setting or the trackball in the corner."));
    });
  }

  function fogPanel() {
    panel("fog", "Fog", body => {
      const f = app.renderer.fog;
      const on = el("input", { type: "checkbox", checked: f.on });
      const start = el("input", { type: "range", min: 0, max: 1, step: 0.01, value: f.start });
      const strength = el("input", { type: "range", min: 0, max: 1, step: 0.01, value: f.strength });
      const apply = () => { f.on = on.checked; f.start = +start.value; f.strength = +strength.value; app.draw(); $("f-fog").classList.toggle("on", f.on); };
      on.onchange = start.oninput = strength.oninput = apply;
      body.append(row(el("label", {}, on, " depth cue")), row(el("span", { className: "what" }, "Start"), start),
                  row(el("span", { className: "what" }, "Strength"), strength));
    }, [innerWidth - 400, innerHeight - 220]);
  }
  function zclipPanel() {
    panel("zclip", "ZClip", body => {
      const c = app.camera.zclip;
      const front = el("input", { type: "range", min: 0.02, max: 1, step: 0.01, value: c[0] });
      const back = el("input", { type: "range", min: 0.02, max: 1, step: 0.01, value: c[1] });
      const apply = () => { c[0] = +front.value; c[1] = +back.value; app.draw(); $("f-zclip").classList.toggle("on", c[0] < 1 || c[1] < 1); };
      front.oninput = back.oninput = apply;
      body.append(row(el("span", { className: "what" }, "Front"), front), row(el("span", { className: "what" }, "Back"), back),
                  row(button("Reset", () => { front.value = back.value = 1; apply(); })));
    }, [innerWidth - 400, innerHeight - 220]);
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

  function alignPanel() {
    panel("align", "Align/Superpose", body => {
      const models = app.live();
      const pickModel = (chosen) => {
        const s = el("select");
        for (const m of models) s.add(new Option(m.name, m.index, false, m.index === chosen));
        return s;
      };
      const current = app.currentModel().index;
      const other = models.find(m => m.index !== current)?.index ?? current;
      const mobile = pickModel(current), ref = pickModel(other);
      const method = el("select");
      for (const [v, l] of [["matchmaker", "Matchmaker (sequence + pruning, as ChimeraX)"],
                            ["cealign", "CE (structure only, for remote homologs)"],
                            ["superpose", "Superpose alpha carbons (by sequence)"]]) method.add(new Option(l, v));
      body.append(row(el("span", { className: "what" }, "Move"), mobile),
                  row(el("span", { className: "what" }, "Onto"), ref),
                  row(el("span", { className: "what" }, "Method"), method));
      const go = button("Superpose", async () => {
        const m = state.models[+mobile.value], r = state.models[+ref.value];
        if (m === r) { app.message("pick two different models", true); return; }
        try {
          await Promise.all([app.load(m), app.load(r)]);
          const out = await app.busy("superposing", () => app.post("api/align", { m: m.index, ref: r.index, method: method.value }));
          m.shown = r.shown = true;
          await app.reloadModel(m, false);
          await app.show([m, r]);
          app.centerView();
          const id = out.identity != null ? `, sequence identity ${(100 * out.identity).toFixed(0)}%` : "";
          const all = out.full_rmsd != null ? ` (all ${out.paired} pairs: ${out.full_rmsd} Å)` : "";
          app.message(`${m.name} onto ${r.name} by ${out.method}: RMSD ${out.rmsd} Å over ${out.pairs} atom pairs${all}${id}. Undo puts it back.`);
        } catch (e) { app.message(e.message, true); }
      }, "blue");
      body.append(row(go));
      body.append(el("p", { className: "help" }, "boonza.matchmaker, boonza.cealign or boonza.superpose; "
        + "the moving model's coordinates change (Edit | Undo restores them, File | Save keeps them)."));
    }, [innerWidth / 2 - 200, 80]);
  }

  async function addModels(out) {
    for (const info of out.models.slice(state.models.length)) state.models.push(app.newModel(info, state.models.length));
    const first = state.models[out.added[0]];
    state.current = first.index;
    await app.show([first]);
    app.centerView();
  }

  // open and save: paths on the server (where the files are)
  function pathPanel(id, title, value, run) {
    panel(id, title, body => {
      const input = el("input", { value, style: "width:360px", spellcheck: false });
      const go = button(title.split(" ")[0], () => run(input.value.trim()), "blue");
      input.onkeydown = e => { if (e.key === "Enter") go.click(); };
      body.append(row(input, go));
      body.append(el("p", { className: "help" }, "A path on the machine boonza view runs on: a structure, a saved session (.bview), "
        + "or a folder of boonza modes, triage or sitemap."));
      setTimeout(() => input.focus(), 0);
    }, [innerWidth / 2 - 220, 80]);
  }
  function openPanel() { pathPanel("open", "Open File", "", async p => { if (await openPath(p)) panels.open?.remove(); }); }
  function savePanel() {
    const m = app.selection()[0]?.[0] || app.currentModel();
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
      const out = await app.post("api/open", { path });
      await addModels(out);
      if (path.endsWith(".bview") && out.page) await app.restorePage(out.page);
      app.message(`opened ${path}: ${out.added.length} model(s)`);
      return true;
    } catch (e) { app.message(e.message, true); return false; }
  }
  async function savePath(path) {
    const m = app.selection()[0]?.[0] || app.currentModel();
    try { const out = await app.post("api/save", { m: m.index, path }); app.message(`saved ${out.path}`); return true; }
    catch (e) { app.message(e.message, true); return false; }
  }
  async function closeModel(which = null) {
    const m = which || app.currentModel();
    if (!m || app.live().length < 2) { app.message("the last model stays open", true); return; }
    try {
      await app.post("api/close", { m: m.index });
      m.closed = true; m.shown = false;
      state.meters = state.meters.filter(x => x.model !== m);
      state.current = app.live()[0].index;
      if (!app.shownModels().length) await app.show([app.live()[0]]);
      app.rebuild(); app.onModels(); app.onSelection();
    } catch (e) { app.message(e.message, true); }
  }

  theme(localStorage.getItem("boonza.theme") || "noir");  // black by default
}

// --- the trackball and the dial --------------------------------------------------------------

function trackball(app) {
  const svg = $("ball"), box = $("trackball");
  let translate = false;
  // a ball drawn as its great circles, turning with the view
  function paint() {
    const r = app.camera.rot, lines = [];
    const circle = (axis) => {
      const pts = [];
      for (let k = 0; k <= 48; k++) {
        const t = 2 * Math.PI * k / 48;
        const u = axis === 0 ? [0, Math.cos(t), Math.sin(t)] : axis === 1 ? [Math.cos(t), 0, Math.sin(t)] : [Math.cos(t), Math.sin(t), 0];
        const v = [0, 1, 2].map(i => r[3 * i] * u[0] + r[3 * i + 1] * u[1] + r[3 * i + 2] * u[2]);
        pts.push([50 + 30 * v[0], 50 - 30 * v[1], v[2]]);
      }
      return pts;
    };
    for (const axis of [0, 1, 2]) {
      const pts = circle(axis);
      for (let k = 0; k < pts.length - 1; k++) {
        const front = pts[k][2] + pts[k + 1][2] > 0;
        lines.push(`<line x1="${pts[k][0].toFixed(1)}" y1="${pts[k][1].toFixed(1)}" x2="${pts[k + 1][0].toFixed(1)}" y2="${pts[k + 1][1].toFixed(1)}" stroke="${front ? "#fff" : "rgba(255,255,255,0.35)"}" stroke-width="${front ? 1.6 : 1}"/>`);
      }
    }
    svg.innerHTML = `<defs><radialGradient id="g" cx="40%" cy="35%" r="65%"><stop offset="0" stop-color="#7fb6ff"/><stop offset="1" stop-color="#1e5fc8"/></radialGradient></defs>
      <circle cx="50" cy="50" r="31" fill="url(#g)"/>${lines.join("")}`;
  }
  const draw = app.draw;
  app.draw = () => { draw(); paint(); };
  paint();

  $("ball-mode").onclick = () => { translate = !translate; $("ball-mode").textContent = translate ? "✥" : "⟳"; };
  svg.addEventListener("contextmenu", e => {
    e.preventDefault();
    app.popup?.(e.clientX, e.clientY);
  });
  let drag = null;
  svg.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    const r = svg.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const edgeX = fx < 0.15 || fx > 0.85, edgeY = fy < 0.15 || fy > 0.85;
    drag = { x: e.clientX, y: e.clientY, mode: edgeX && edgeY ? "z" : edgeY ? "x" : edgeX ? "y" : "free",
             alt: e.altKey, ctrl: e.ctrlKey || e.metaKey };
    if (drag.alt && !app.beginMove()) drag.alt = false;
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.ctrl) { app.camera.zoom(Math.exp(dy * 0.01)); app.draw(); return; }
    const k = 0.02;
    if (translate) {
      if (drag.alt) app.translateSelected(dx * 3, dy * 3);
      else {
        const s = 2 * app.camera.distance * Math.tan(app.camera.fov / 2) / 300;
        app.camera.pan(dx * s, -dy * s); app.draw();
      }
      return;
    }
    const [ax, ay, az, ang] = drag.mode === "z" ? [0, 0, 1, -dx * k] : drag.mode === "x" ? [1, 0, 0, dy * k]
      : drag.mode === "y" ? [0, 1, 0, dx * k] : [dy, dx, 0, Math.hypot(dx, dy) * k];
    if (drag.alt) app.rotateSelected(ax, ay, az, ang);
    else { app.camera.rotate(ax, ay, az, ang); app.draw(); }
  });
  svg.addEventListener("pointerup", () => {
    if (drag?.alt) app.endMove();
    drag = null;
  });
  box.addEventListener("wheel", e => { e.preventDefault(); app.camera.zoom(Math.exp(e.deltaY * 0.0015)); app.draw(); }, { passive: false });
}

function dial(app) {
  const d = $("dial"), marks = $("dial-marks");
  let drag = null, offset = 0;
  d.addEventListener("pointerdown", e => { drag = { x: e.clientX, fine: e.ctrlKey || e.metaKey }; d.setPointerCapture(e.pointerId); });
  d.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    drag.x = e.clientX;
    app.camera.zoom(Math.exp(-dx * (drag.fine ? 0.002 : 0.01)));
    offset += dx;
    marks.style.backgroundPosition = `${offset}px 0`;
    app.draw();
  });
  d.addEventListener("pointerup", () => { drag = null; });
}
