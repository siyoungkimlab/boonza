// The page's core: the models (each a structure, pose or set of spheres the server
// holds), each drawn by its representations, the scene, picking, moving atoms, and
// frames.  mouse.js binds the mouse; ui.js builds the menus, side bar and panels.
//
// A representation is a set of a model's atoms drawn one way: a style (lines, sticks,
// ball and stick, spheres, cartoon, tube, surface), a color scheme and a switch.  A
// model has as many as it likes and they add up: the same atom can be a line in one
// and a sphere in another.

import { Camera } from "./camera.js";
import { colorAtoms, colorResidues, named, RIBBON_SCHEMES } from "./colors.js";
import { buildScene, setDetail, STYLES } from "./geometry.js";
import { Renderer } from "./gl.js";
import { setupMouse } from "./mouse.js";
import { surface } from "./surface.js";
import { setupUI } from "./ui.js";

const $ = id => document.getElementById(id);
const canvas = $("view");
const renderer = new Renderer(canvas);
const camera = new Camera();
const CACHE_BYTES = 256e6;  // frames kept in the page, over all models
const CONTACT_COLOR = [0.2, 0.75, 1];
const METER_COLOR = [0.95, 0.75, 0.1];
const HBOND_COLOR = [0.3, 0.9, 0.45];
const STYLE = Object.fromEntries(STYLES.map((s, k) => [s, k]));

// the representations' styles, as the page names them, and the atom style each draws
const REP_STYLES = [["lines", "Lines"], ["sticks", "Sticks"], ["ballstick", "Ball & stick"],
                    ["spheres", "Spheres"], ["cartoon", "Cartoon"], ["tube", "Tube"], ["surface", "Surface"]];
const ATOM_STYLE = { lines: STYLE.line, sticks: STYLE.stick, ballstick: STYLE.ballstick, spheres: STYLE.space };
const RIBBON = new Set(["cartoon", "tube"]);
// other names a style goes by (the server's, the command line's)
const STYLE_ALIAS = { line: "lines", stick: "sticks", ballline: "ballstick", space: "spheres", sphere: "spheres",
                      picked: "ballstick", vdw: "spheres", ribbon: "cartoon" };
const repStyle = s => (REP_STYLES.some(([k]) => k === s) ? s : STYLE_ALIAS[s]) || null;
// a representation's material: how opaque it is and how it shines; its opacity can be
// set apart from the material's own
const MATERIALS = [["opaque", "Opaque", 1, 1], ["glossy", "Glossy", 1, 2.5], ["matte", "Matte", 1, 0],
                   ["transparent", "Transparent", 0.5, 1], ["ghost", "Ghost", 0.2, 0.3]];
const material = name => MATERIALS.find(([k]) => k === name) || MATERIALS[0];
const repAlpha = rep => Math.round(100 * (rep.opacity ?? material(rep.material)[2])) / 100;
const repShine = rep => material(rep.material)[3];

const state = {
  title: "", columns: [], models: [], current: 0, frame: 0,
  playing: false, pending: null, loading: false,
  masks: [true, true, true, true],  // pickable: receptor, ligand, solvent, ribbons
  hydrogens: "polar",               // shown: all, polar, none
  contacts: false, meters: [], labelMode: "name", pocketNumbers: true, busy: 0,
  surfaceGrid: +(localStorage.getItem("boonza.surfaceGrid") || 0.5),  // A, View | Add Surface | Resolution
};

// --- server ----------------------------------------------------------------

// a static page (boonza.viewer.export): what the server would answer, as files in data/
const STATIC = document.querySelector('meta[name="boonza-static"]')?.content === "1";
function apiURL(url) {
  if (!STATIC) return url;
  const u = new URL(url, location.href);
  const name = u.pathname.split("/").pop();
  const keys = ["m", "i"].filter(k => u.searchParams.has(k)).map(k => `-${k}${u.searchParams.get(k)}`).join("");
  return `data/${name}${keys}${name === "frame" ? ".bin" : ".json"}`;
}

async function getJSON(url) {
  const r = await fetch(apiURL(url));
  if (STATIC && !r.ok) throw new Error("not on this page: it needs boonza view");
  const out = await r.json();
  if (!r.ok) throw new Error(out.error || r.statusText);
  return out;
}

// the token the server wrote into this page: every request that changes something carries it
const TOKEN = document.querySelector('meta[name="boonza-token"]')?.content || "";
const PYTHON = document.querySelector('meta[name="boonza-python"]')?.content === "1";

async function post(url, body = {}) {
  if (STATIC) throw new Error("this page shows models only: editing, minimizing and selections typed in need boonza view");
  const r = await fetch(url, { method: "POST", body: JSON.stringify(body),
                               headers: { "Content-Type": "application/json", "X-Boonza-Token": TOKEN } });
  const out = await r.json();
  if (!r.ok) throw new Error(out.error || r.statusText);
  return out;
}

async function getFrame(model, i) {
  if (model.frames.has(i)) return model.frames.get(i);
  const r = await fetch(apiURL(`api/frame?m=${model.index}&i=${i}`));
  if (!r.ok) throw new Error(STATIC ? `frame ${i + 1} is not on this page` : (await r.json()).error);
  let f;
  if (STATIC) {  // positions, then the box's nine numbers
    const all = new Float32Array(await r.arrayBuffer()), n = all.length - 9, b = all.subarray(n);
    f = { pos: all.slice(0, n), box: [[b[0], b[1], b[2]], [b[3], b[4], b[5]], [b[6], b[7], b[8]]] };
  } else {
    f = { pos: new Float32Array(await r.arrayBuffer()), box: JSON.parse(r.headers.get("X-Box")) };
  }
  const loaded = state.models.filter(m => m.topo).reduce((n, m) => n + m.topo.natoms, 0);
  const keep = Math.max(2, Math.floor(CACHE_BYTES / (Math.max(loaded, 1) * 12)));
  if (model.frames.size >= keep) model.frames.delete(model.frames.keys().next().value);
  model.frames.set(i, f);
  return f;
}

// a long call (minimize, a profile): the Cancel button lights while it runs
async function busy(what, call) {
  state.busy++;
  $("btn-cancel").disabled = false;
  message(`${what}…`);
  try {
    return await call();
  } finally {
    state.busy--;
    $("btn-cancel").disabled = state.busy === 0;
  }
}

// --- messages -------------------------------------------------------------------

function message(text, error = false) {
  const el = $("message");
  el.innerHTML = "";
  if (!text) return;
  const span = document.createElement("span");
  span.textContent = text;
  if (error) span.className = "err";
  el.appendChild(span);
}

function atomLabel(model, a, long = true) {
  const { atoms: A, residues: R, chains: C } = model.topo;
  const r = A.residue[a];
  const chain = C.name[R.chain[r]] || C.segid[R.chain[r]] || "-";
  const q = A.charge && A.charge[a] ? ` ${A.charge[a] > 0 ? "+" : ""}${A.charge[a]}` : "";
  const where = state.models.filter(m => !m.closed).length > 1 ? `${model.name}  ` : "";
  const res = `${chain}:${R.name[r]}${R.resid[r]}${R.insertion[r].trim()}`;
  return long ? `${where}${res} ${A.name[a]}  (${A.element[a]}${q}, atom ${a})` : `${res} ${A.name[a]}`;
}

// --- models and their state ---------------------------------------------------------

const live = () => state.models.filter(m => !m.closed);
const shownModels = () => state.models.filter(m => m.shown && m.topo && !m.closed);
const currentModel = () => state.models[state.current];

function newModel(info, index) {
  return { ...info, index, topo: null, loading: null, frames: new Map(), contacts: [], reps: [] };
}

// what every atom carries apart from its representations: selected, constrained,
// labeled; and what the representations add up to (worked out by aggregate)
function initialState(model) {
  const topo = model.topo, n = topo.natoms, nres = topo.residues.name.length;
  return {
    sel: new Uint8Array(n), constraint: new Uint8Array(n), labels: new Uint8Array(n),
    show: new Uint8Array(n), style: new Uint8Array(n), ribbon: new Uint8Array(nres),
  };
}

// --- representations ----------------------------------------------------------------

let repSerial = 0;

// a representation of these atoms (ids, or a mask): {sel, label, style, color, on}
function makeRep(model, { ids = null, mask = null, sel = null, label = null, style = "lines", color = "element",
                          on = true, tinted = true, material = "opaque", opacity = null }) {
  const n = model.topo.natoms;
  if (!mask) { mask = new Uint8Array(n); for (const a of ids || []) mask[a] = 1; }
  const rep = { id: ++repSerial, sel, label: label || sel || "picked atoms", mask, style: repStyle(style) || "lines",
                color, on, tinted, material, opacity, custom: false, version: 0, rgb: null };
  colorRep(model, rep);
  return rep;
}

const repIds = rep => { const out = []; for (let a = 0; a < rep.mask.length; a++) if (rep.mask[a]) out.push(a); return out; };
const isRibbon = rep => RIBBON.has(rep.style);

// the residues a ribbon representation draws: those of its atoms with a trace
function repResidues(model, rep) {
  const R = model.topo.residues, A = model.topo.atoms, on = new Uint8Array(R.name.length);
  for (let a = 0; a < rep.mask.length; a++) if (rep.mask[a] && R.trace[A.residue[a]] >= 0) on[A.residue[a]] = 1;
  const out = [];
  for (let r = 0; r < on.length; r++) if (on[r]) out.push(r);
  return out;
}

// a ribbon's color per residue: the ribbon schemes, a color, else the scheme of its traced atoms
function ribbonColors(model, spec, residues, out) {
  const topo = model.topo;
  if (RIBBON_SCHEMES.includes(spec) || named(spec)) { colorResidues(spec, residues, topo, model.ss, out); return; }
  const [scheme, carbon] = spec.split("/");
  if (scheme === "element") { colorResidues(carbon && named(carbon) ? carbon : "white", residues, topo, model.ss, out); return; }
  const trace = residues.map(r => topo.residues.trace[r]);
  const tmp = new Float32Array(topo.natoms * 3);
  colorAtoms(spec, trace, topo, model.ss, tmp);
  residues.forEach((r, k) => { for (let d = 0; d < 3; d++) out[3 * r + d] = tmp[3 * trace[k] + d]; });
}

// a representation's colors, from its scheme, then the model's own color over them
// (its receptor's carbons and its ribbon -- beads, whose protein a class does not tell
// from a ligand, too -- or, chosen in the list, every carbon), so models of a group are
// told apart at a glance
function colorRep(model, rep) {
  const topo = model.topo, n = topo.natoms, nres = topo.residues.name.length;
  if (rep.style === "surface") { rep.version++; return; }
  if (isRibbon(rep)) {
    if (!rep.rgb || rep.rgb.length !== nres * 3) rep.rgb = new Float32Array(nres * 3);
    const residues = repResidues(model, rep);
    ribbonColors(model, rep.color, residues, rep.rgb);
    const tint = rep.tinted && model.color && model.color !== "default" ? model.color : null;
    if (tint) {
      const own = topo.cg || model.tintAll ? residues
        : residues.filter(r => topo.residueAtoms[r].some(a => topo.atoms.class[a] === 0));
      colorResidues(tint, own, topo, model.ss, rep.rgb);
    }
    return;
  }
  if (!rep.rgb || rep.rgb.length !== n * 3) rep.rgb = new Float32Array(n * 3);
  const ids = repIds(rep);
  colorAtoms(rep.color, ids, topo, model.ss, rep.rgb);
  const tint = rep.tinted && rep.color.startsWith("element") && model.color && model.color !== "default" ? model.color : null;
  if (tint) {
    const own = model.tintAll ? ids : ids.filter(a => topo.atoms.class[a] === 0);
    colorAtoms(`element/${tint}`, own, topo, model.ss, rep.rgb);
  }
  rep.custom = false;
}

// the representations drawn: what they add up to per atom (shown, the widest style,
// a ribbon), for picking, fitting the view and the labels
function aggregate(model) {
  const st = model.state, topo = model.topo;
  st.show.fill(0); st.style.fill(0); st.ribbon.fill(0);
  const RADIUS = [1, 2, 3, 3, 4];  // line, stick, ball and line, ball and stick, space: which is widest
  for (const rep of model.reps) {
    if (!rep.on || rep.style === "surface") continue;
    if (isRibbon(rep)) { for (const r of repResidues(model, rep)) st.ribbon[r] = 1; continue; }
    const k = ATOM_STYLE[rep.style];
    for (let a = 0; a < rep.mask.length; a++) {
      if (!rep.mask[a]) continue;
      if (!st.show[a] || RADIUS[k] > RADIUS[st.style[a]]) st.style[a] = k;
      st.show[a] = 1;
    }
  }
  return st;
}

function applyReps(model, reps) {
  for (const r of reps) model.reps.push(makeRep(model, { ids: r.ids, sel: r.sel, label: r.label, style: r.style, color: r.color }));
}

// the model's representations changed: drawn again, and the side bar told
function repsChanged(model) {
  if (model?.state) aggregate(model);
  rebuild();
  app.onReps(model);
}

// atoms worth knowing apart: hydrogens, and those on carbon (hidden when only polar ones show)
function hydrogenFlags(topo) {
  const n = topo.natoms, h = new Uint8Array(n), nonpolar = new Uint8Array(n);
  const B = topo.bonds, el = topo.atoms.element;
  for (let a = 0; a < n; a++) h[a] = el[a] === "H" ? 1 : 0;
  for (let k = 0; k < B.i.length; k++) {
    const i = B.i[k], j = B.j[k];
    if (h[i] && el[j] === "C") nonpolar[i] = 1;
    if (h[j] && el[i] === "C") nonpolar[j] = 1;
  }
  return { h, nonpolar };
}

async function load(model) {
  if (model.topo) return model;
  if (!model.loading) {
    model.loading = (async () => {
      const topo = await getJSON(`api/topology?m=${model.index}`);
      topo.spectrum = spectrum(topo);
      topo.residueAtoms = topo.residues.name.map(() => []);
      topo.atoms.residue.forEach((r, a) => topo.residueAtoms[r].push(a));
      topo.flags = hydrogenFlags(topo);
      topo.bondsOf = bondIndex(topo);
      const f = await getFrame(model, Math.min(state.frame, topo.nframes - 1));
      model.pos = f.pos; model.box = f.box; model.ss = topo.ss;
      model.topo = topo;
      const old = model.keep;  // the state before a reload, to carry over by atom name
      model.state = initialState(model);
      if (old) carryOver(old, model);
      else {
        model.reps = [];
        applyReps(model, await Promise.all(topo.reps.map(async r => ({ ...r, ids: r.ids || await selectIds(model, r.sel) }))));
      }
      aggregate(model);
      // a numbered model (a pocket, a site) is labeled at its ligand -- the site's points
      // -- when it has one, not at the middle of the protein drawn with it
      if (model.label) model.labelIds = await selectIds(model, `same residue as (${topo.ligand})`).catch(() => null);
      model.keep = null;
      return model;
    })().catch(e => { model.loading = null; throw e; });
  }
  return model.loading;
}

function bondIndex(topo) {
  const out = topo.atoms.name.map(() => []);
  const B = topo.bonds;
  for (let k = 0; k < B.i.length; k++) { out[B.i[k]].push(B.j[k]); out[B.j[k]].push(B.i[k]); }
  return out;
}

const range = n => Array.from({ length: n }, (_, k) => k);

function atomKey(topo, a) {
  const R = topo.residues, r = topo.atoms.residue[a];
  return `${R.chain[r]}|${R.resid[r]}|${R.insertion[r]}|${R.name[r]}|${topo.atoms.name[a]}`;
}

// after an edit that changed the atoms: every atom keeps its state and its place in
// each representation, found by name; new atoms (added hydrogens) take their bonded
// neighbor's
function carryOver(old, model) {
  const topo = model.topo, st = model.state, n = topo.natoms;
  const from = new Map();
  old.keys.forEach((k, a) => from.set(k, a));
  const found = new Int32Array(n).fill(-1);
  for (let a = 0; a < n; a++) {
    const o = from.get(atomKey(topo, a));
    if (o === undefined) continue;
    found[a] = o;
    st.sel[a] = old.sel[o]; st.constraint[a] = old.constraint[o]; st.labels[a] = old.labels[o];
  }
  const masks = old.reps.map(rep => { const m = new Uint8Array(n); for (let a = 0; a < n; a++) if (found[a] >= 0) m[a] = rep.mask[found[a]]; return m; });
  // new atoms (a group, a side chain, hydrogens) join what the atoms they hang on are
  // in: spread outward through the bonds, never selected
  const known = found.map(o => o >= 0 ? 1 : 0);
  let grew = true;
  while (grew) {
    grew = false;
    for (let a = 0; a < n; a++) {
      if (known[a]) continue;
      // a neighbor of its own residue first: a metal put back joins its heme, not the histidine
      const r = topo.atoms.residue[a];
      const nb = topo.bondsOf[a].find(b => known[b] && topo.atoms.residue[b] === r) ?? topo.bondsOf[a].find(b => known[b]);
      if (nb === undefined) continue;
      for (const m of masks) m[a] = m[nb];
      known[a] = 1; grew = true;
    }
  }
  model.reps = old.reps.map((rep, k) => {
    const out = { ...rep, mask: masks[k], rgb: null, version: rep.version + 1 };
    const custom = rep.custom && !isRibbon(rep) && rep.style !== "surface";
    colorRep(model, out);
    if (custom) {  // its own colors, atom by atom, where the atom was there
      for (let a = 0; a < n; a++) {
        const o = found[a];
        if (o >= 0 && topo.atoms.element[a] === old.element[o]) for (let d = 0; d < 3; d++) out.rgb[3 * a + d] = rep.rgb[3 * o + d];
      }
      out.custom = true;
    }
    return out;
  });
  // a new molecule, bonded to nothing that was there: sticks (built ones: ball and
  // stick), colored by element, in a representation of its own
  const fresh = range(n).filter(a => !known[a]);
  if (fresh.length) {
    const style = model.group === "built" ? "ballstick" : "sticks";
    const into = model.reps.find(r => r.fresh && r.style === style && r.on);
    if (into) { for (const a of fresh) into.mask[a] = 1; colorRep(model, into); }
    else model.reps.push(Object.assign(makeRep(model, { ids: fresh, label: "new atoms", style }), { fresh: true }));
  }
}

// after an edit: positions again, and with new atoms the topology too
async function reloadModel(model, topology) {
  model.frames.clear();
  if (topology) {
    const st = model.state, topo = model.topo;
    model.keep = { ...st, reps: model.reps, keys: range(topo.natoms).map(a => atomKey(topo, a)), element: topo.atoms.element };
    model.topo = null;
    model.loading = null;
    await load(model);
  } else {
    const f = await getFrame(model, 0);
    model.pos = f.pos; model.box = f.box;
  }
  await refreshContacts();
  rebuild();
  app.onSelection();
}

async function selectIds(model, sel) {
  if (STATIC) {  // only those the export worked out
    const ids = model.topo?.selections?.[sel];
    if (!ids) throw new Error("selections typed in need boonza view");
    return Int32Array.from(ids);
  }
  const frame = model.topo ? Math.min(state.frame, model.topo.nframes - 1) : 0;
  return Int32Array.from((await post("api/select", { m: model.index, sel, frame })).ids);
}

function nframes() {
  return Math.max(1, ...shownModels().map(m => m.topo.nframes));
}

async function show(models, on = true) {
  for (const m of models) m.shown = on;
  // a model tied to a frame of another (a site and the MD frame it was found in):
  // that model comes along, at that frame
  const linked = on ? models.filter(m => m.frame) : [];
  for (const m of linked) {
    const target = state.models.find(t => t.name === m.frame.model && !t.closed);
    if (target && !models.includes(target)) { target.shown = true; models = [...models, target]; }
  }
  try {
    await Promise.all(models.filter(() => on).map(load));
  } catch (e) {
    message(e.message, true);
  }
  const last = linked[linked.length - 1];
  if (last) { await setFrame(last.frame.i); app.onModels(); return; }
  await refreshContacts();
  updatePlayer();
  rebuild();
  app.onModels();
  app.onSelection();
}

// one model of its group shown, made current, looked at (its ligand and pocket)
async function solo(model) {
  for (const m of live()) if (m.group === model.group) m.shown = m === model;
  state.current = model.index;
  await show([model]);
  const ids = interesting(model);
  if (ids.length) { const [c, r] = bounds(model, ids); flyTo(c, r + (model.topo.cg ? 10 : 6)); }
}

function interesting(model) {
  const st = model.state, out = [];
  const many = model.topo.natoms;
  for (let a = 0; a < many; a++) if (st.show[a]) out.push(a);
  return out.length && out.length < many / 2 ? out : range(many);
}

// --- selection ---------------------------------------------------------------------

function selected(model) {
  const out = [];
  if (!model.state) return out;
  const s = model.state.sel;
  for (let a = 0; a < s.length; a++) if (s[a]) out.push(a);
  return out;
}

// every selected atom, model by model: [[model, [atoms]], ...]
function selection() {
  return shownModels().map(m => [m, selected(m)]).filter(([, ids]) => ids.length);
}

// the atoms a render command acts on: the selected ones, or all if none are
function targets() {
  const sel = selection();
  return sel.length ? sel : shownModels().map(m => [m, range(m.topo.natoms)]);
}

function clearSelection() {
  for (const m of shownModels()) m.state.sel.fill(0);
}

// mode: set (only these), add, toggle, remove
function select(model, ids, mode = "set") {
  if (mode === "set") clearSelection();
  const s = model.state.sel;
  if (mode === "toggle") {
    const all = ids.every(a => s[a]);
    for (const a of ids) s[a] = all ? 0 : 1;
  } else {
    for (const a of ids) s[a] = mode === "remove" ? 0 : 1;
  }
  if (state.current !== model.index && ids.length) state.current = model.index;
  rebuild();
  app.onSelection();
}

// --- render state ----------------------------------------------------------------

function visibleAtom(model, a) {
  const st = model.state, f = model.topo.flags;
  if (!st.show[a]) return false;
  if (f.h[a] && (state.hydrogens === "none" || state.hydrogens === "polar" && f.nonpolar[a])) return false;
  return true;
}

let drawQueued = false;
function draw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; renderer.draw(camera); placeLabels(); });
}

function cellLines(box) {
  if (!box || !box.flat().some(x => x)) return [];
  const [a, b, c] = box, out = [];
  const corner = (i, j, k) => [0, 1, 2].map(d => i * a[d] + j * b[d] + k * c[d]);
  for (const [i, j, k] of [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    for (const [di, dj, dk] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      if (i + di > 1 || j + dj > 1 || k + dk > 1) continue;
      out.push(...corner(i, j, k), ...corner(i + di, j + dj, k + dk));
    }
  return out;
}

// a surface representation's mesh: made again when its atoms move, or its coloring,
// atoms or the resolution change.  Electrostatic and hydrophobic color it by property;
// a color name paints it that color, anything else leaves it plain
function surfaceOf(m, rep) {
  const grid = state.surfaceGrid, s = rep.mesh;
  const kind = ["electrostatic", "hydrophobic"].includes(rep.color) ? rep.color : "plain";
  if (s && s.pos === m.pos && s.version === rep.version && s.grid === grid && s.color === rep.color) return s.mesh;
  const t0 = performance.now();
  const ids = repIds(rep);
  if (!ids.length) return null;
  const mesh = surface(m, ids, kind, grid);
  const c = kind === "plain" ? named(rep.color) : null;
  if (c) for (let v = 0; v < mesh.color.length; v += 3) { mesh.color[v] = c[0]; mesh.color[v + 1] = c[1]; mesh.color[v + 2] = c[2]; }
  rep.mesh = { pos: m.pos, version: rep.version, grid, color: rep.color, mesh };
  if (performance.now() - t0 > 2000) message(`surface: ${mesh.index.length / 3} triangles in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  return mesh;
}

// the scene: each shown model's representations, one drawing each, then the
// selection's highlight where no representation draws a selected atom
function modelItems(m) {
  const st = m.state, topo = m.topo, n = topo.natoms, R = topo.residues;
  const nres = R.name.length;
  aggregate(m);
  const none = new Uint8Array(n), noRibbon = new Uint8Array(nres);
  const selected = state.picture ? none : st.sel;  // a picture leaves out the selection's highlight
  const drawn = new Uint8Array(n), traced = new Uint8Array(n), items = [];
  const base = { topo, pos: m.pos, ss: m.ss };
  const look = rep => ({ alpha: repAlpha(rep), shine: repShine(rep) });
  for (const rep of m.reps) {
    if (!rep.on) continue;
    if (rep.style === "surface") {
      const mesh = surfaceOf(m, rep);
      if (mesh) items.push({ ...base, ...look(rep), state: { show: none, style: none, rgb: null, sel: none, ribbon: noRibbon }, surface: mesh });
      continue;
    }
    if (isRibbon(rep)) {
      const ribbon = new Uint8Array(nres), sel = new Uint8Array(n);
      for (const r of repResidues(m, rep)) { ribbon[r] = 1; const t = R.trace[r]; traced[t] = 1; sel[t] = selected[t]; }
      items.push({ ...base, ...look(rep), state: { show: none, style: none, rgb: null, sel, ribbon, ribbonRGB: rep.rgb, ribbonStyle: rep.style } });
      continue;
    }
    // shown atoms, through the hydrogen filter
    const show = new Uint8Array(n), sel = new Uint8Array(n);
    for (let a = 0; a < n; a++) if (rep.mask[a] && visibleAtom(m, a)) { show[a] = 1; drawn[a] = 1; sel[a] = selected[a]; }
    items.push({ ...base, ...look(rep), state: { show, style: new Uint8Array(n).fill(ATOM_STYLE[rep.style]), rgb: rep.rgb, sel, ribbon: noRibbon } });
  }
  // the selected atoms nothing draws: a pink ball each; contacts, bonds and meters dashed
  const loose = new Uint8Array(n);
  for (let a = 0; a < n; a++) loose[a] = selected[a] && !drawn[a] && !traced[a] ? 1 : 0;
  const dashes = [];
  if (state.contacts) for (const [i, j] of m.contacts || []) dashes.push([...xyz(m, i), ...xyz(m, j), ...CONTACT_COLOR]);
  if (state.hbonds) for (const [i, j] of m.hbonds || []) dashes.push([...xyz(m, i), ...xyz(m, j), ...HBOND_COLOR]);
  for (const meter of state.meters) {
    if (meter.model !== m) continue;
    const a = meter.atoms;
    for (let k = 0; k + 1 < a.length; k++) dashes.push([...xyz(m, a[k]), ...xyz(m, a[k + 1]), ...METER_COLOR]);
  }
  items.push({ ...base, state: { show: none, style: none, rgb: null, sel: loose, ribbon: noRibbon }, dashes });
  m.drawn = drawn;
  return items;
}

function rebuild() {
  const models = shownModels();
  const scene = buildScene(models.flatMap(modelItems));
  showSummary();
  if (state.cell) {
    const extra = models.flatMap(m => cellLines(m.box));
    if (extra.length) {
      const base = scene.layers[0];  // opaque, plain
      const p = new Float32Array(base.lines.position.length + extra.length);
      p.set(base.lines.position); p.set(extra, base.lines.position.length);
      const c = new Float32Array(p.length).fill(0.6);
      c.set(base.lines.color);
      base.lines = { position: p, color: c };
    }
  }
  renderer.setScene(scene);
  makeLabels();
  draw();
}

const xyz = (m, a) => [m.pos[3 * a], m.pos[3 * a + 1], m.pos[3 * a + 2]];

// a picture: the scene drawn in detail (rounder sticks and ribbons, lines as sticks),
// without the selection's highlight; ``draw(renderer)`` makes it, then the screen's is back
function withPictureScene(draw) {
  state.picture = true;
  setDetail(2); renderer.detail = 2;
  try {
    rebuild();
    return draw(renderer);
  } finally {
    state.picture = false;
    setDetail(1); renderer.detail = 1;
    rebuild();
  }
}

// --- labels: atom labels, meters, pocket numbers ------------------------------------

function labelText(m, a) {
  const { atoms: A, residues: R } = m.topo, r = A.residue[a];
  switch (state.labelMode) {
    case "element": return A.element[a];
    case "residue": return `${R.name[r]}${R.resid[r]}`;
    case "charge": return A.charge[a] ? `${A.charge[a] > 0 ? "+" : ""}${A.charge[a]}` : "";
    default: return A.name[a];
  }
}

function addLabel(at, text, cls = "") {
  const div = document.createElement("div");
  div.className = `label ${cls}`;
  div.textContent = text;
  div.dataset.at = JSON.stringify(at);
  $("labels").appendChild(div);
}

function makeLabels() {
  $("labels").innerHTML = "";
  let count = 0;
  for (const m of shownModels()) {
    if (m.label && state.pocketNumbers) addLabel(bounds(m, m.labelIds?.length ? m.labelIds : range(m.topo.natoms))[0], m.label, "pocket");
    const lab = m.state.labels;
    for (let a = 0; a < lab.length && count < 600; a++) {
      if (lab[a] && m.drawn[a]) { addLabel(xyz(m, a), labelText(m, a)); count++; }
    }
  }
  for (const meter of state.meters) {
    if (!meter.model.shown || meter.model.closed) continue;
    const { value, at } = measure(meter.model, meter.atoms);
    addLabel(at, meter.atoms.length === 2 ? value.toFixed(2) : `${value.toFixed(1)}°`, "meter");
  }
}

function placeLabels() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  for (const div of $("labels").children) {
    const v = camera.toView(JSON.parse(div.dataset.at));
    if (v[2] >= -0.5) { div.style.display = "none"; continue; }
    const [x, y] = camera.screen(v, w, h);
    div.style.display = "";
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
  }
}

// a distance (2 atoms), angle (3) or dihedral (4) and where to write it
function measure(m, atoms) {
  const p = atoms.map(a => xyz(m, a));
  const sub = (u, v) => u.map((x, d) => x - v[d]);
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const crs = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const len = u => Math.hypot(...u);
  const mid = (u, v) => u.map((x, d) => (x + v[d]) / 2);
  if (p.length === 2) return { value: len(sub(p[1], p[0])), at: mid(p[0], p[1]) };
  if (p.length === 3) {
    const u = sub(p[0], p[1]), v = sub(p[2], p[1]);
    return { value: Math.acos(Math.max(-1, Math.min(1, dot(u, v) / len(u) / len(v)))) * 180 / Math.PI, at: p[1] };
  }
  const b0 = sub(p[0], p[1]), b1 = sub(p[2], p[1]), b2 = sub(p[3], p[2]);
  const n1 = b1.map(x => x / len(b1));
  const v = sub(b0, n1.map(x => x * dot(b0, n1))), w = sub(b2, n1.map(x => x * dot(b2, n1)));
  return { value: Math.atan2(dot(crs(n1, v), w), dot(v, w)) * 180 / Math.PI, at: mid(p[1], p[2]) };
}

// --- camera ---------------------------------------------------------------------

function bounds(model, ids) {
  const { pos } = model;
  const c = [0, 0, 0];
  for (const a of ids) for (let d = 0; d < 3; d++) c[d] += pos[3 * a + d] / ids.length;
  let r = 0;
  for (const a of ids) r = Math.max(r, Math.hypot(pos[3 * a] - c[0], pos[3 * a + 1] - c[1], pos[3 * a + 2] - c[2]));
  return [c, r];
}

let flight = 0;
function flyTo(center, radius, ms = 400) {
  const from = { center: camera.center.slice(), distance: camera.distance, radius: camera.radius };
  const to = new Camera();
  to.fit(center, radius, canvas.clientWidth / canvas.clientHeight);
  const start = performance.now(), id = ++flight;
  camera.pivot = null;
  const step = now => {
    if (id !== flight) return;
    const x = Math.min((now - start) / ms, 1), e = x * x * (3 - 2 * x);
    camera.center = from.center.map((v, d) => v + (to.center[d] - v) * e);
    camera.distance = from.distance * Math.pow(to.distance / from.distance, e);
    camera.radius = from.radius + (to.radius - from.radius) * e;
    renderer.draw(camera);
    placeLabels();
    if (x < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
const stopFlight = () => { flight++; };

// the atoms drawn, as flat xyz (subsampled), for fitting the view
function drawnXYZ(only = null) {
  const out = [];
  for (const m of shownModels()) {
    const n = m.topo.natoms, every = Math.max(1, Math.floor(n / 5000));
    const ribbonAtoms = new Set(m.topo.residues.trace.filter((t, r) => t >= 0 && m.state.ribbon[r]));
    for (let a = 0; a < n; a += only ? 1 : every) {
      if (only ? !only.has(`${m.index}:${a}`) : !(m.drawn?.[a] || ribbonAtoms.has(a))) continue;
      out.push(...xyz(m, a));
    }
  }
  if (!out.length && !only) for (const m of shownModels()) out.push(...m.pos);
  return out;
}

function sphereOf(flat) {
  const n = flat.length / 3;
  if (!n) return [[0, 0, 0], 10];
  const c = [0, 0, 0];
  for (let a = 0; a < n; a++) for (let d = 0; d < 3; d++) c[d] += flat[3 * a + d] / n;
  let r = 0;
  for (let a = 0; a < n; a++) r = Math.max(r, Math.hypot(flat[3 * a] - c[0], flat[3 * a + 1] - c[1], flat[3 * a + 2] - c[2]));
  return [c, r + 2];
}

// Center: the selected atoms, else everything drawn
function centerView(animate = true) {
  const sel = selection();
  const keys = sel.length ? new Set(sel.flatMap(([m, ids]) => ids.map(a => `${m.index}:${a}`))) : null;
  const [c, r] = sphereOf(drawnXYZ(keys));
  if (animate) flyTo(c, Math.max(r, 6));
  else { camera.fit(c, Math.max(r, 6), canvas.clientWidth / canvas.clientHeight); draw(); }
}

// everything drawn in view, turned as it is (Fit All)
function viewAll() {
  const [c, r] = sphereOf(drawnXYZ());
  flyTo(c, Math.max(r, 6));
}

function resetView() {
  stopFlight();
  camera.orient(drawnXYZ());
  camera.pivot = null;
  camera.fit(...sphereOf(drawnXYZ()), canvas.clientWidth / canvas.clientHeight);
  draw();
}

// the point under screen (x, y), in the plane through the view's center facing the eye
function unproject(x, y) {
  const r = camera.rot, scale = 2 * camera.distance * Math.tan(camera.fov / 2) / canvas.clientHeight;
  const vx = (x - canvas.clientWidth / 2) * scale, vy = -(y - canvas.clientHeight / 2) * scale;
  return [0, 1, 2].map(d => camera.center[d] + r[d] * vx + r[3 + d] * vy);
}

// of a bond's two atoms, the one nearer screen (x, y)
function nearerAtom(m, atoms, x, y) {
  const project = projector();
  const d = a => { const p = project(m, a); return p ? Math.hypot(p[0] - x, p[1] - y) : Infinity; };
  return d(atoms[0]) <= d(atoms[1]) ? atoms[0] : atoms[1];
}

// --- picking -----------------------------------------------------------------------

// a function giving an atom's [screen x, screen y, view z, pixels per Å], or null
function projector() {
  const w = canvas.clientWidth, h = canvas.clientHeight, [near] = camera.clip();
  const r = camera.rot, c = camera.center, dist = camera.distance;
  return (m, a) => {
    const px = m.pos[3 * a] - c[0], py = m.pos[3 * a + 1] - c[1], pz = m.pos[3 * a + 2] - c[2];
    const vz = r[6] * px + r[7] * py + r[8] * pz - dist;
    if (vz > -near) return null;
    const v = [r[0] * px + r[1] * py + r[2] * pz, r[3] * px + r[4] * py + r[5] * pz, vz];
    const [sx, sy, scale] = camera.screen(v, w, h);
    return [sx, sy, vz, scale];
  };
}

const pickable = (m, a) => state.masks[m.topo.atoms.class[a]];

// what is under (x, y): {model, atom}, {model, atoms: [i, j]} for a bond, {model,
// residue} for a ribbon, or null.  An atom wins only inside what is drawn of it (its
// ball, or a stick's joint), so a click along a bond picks the bond: the torsion
const SIZE = { atoms: [0, 0.2, 0.25, 0.25, 1], beads: [0, 0.6, 0.5, 0.5, 1] };  // per style; x radius for balls, space
function drawnRadius(m, a) {
  const st = m.state.style[a], s = (m.topo.cg ? SIZE.beads : SIZE.atoms)[st];
  if (st === STYLE.ballline || st === STYLE.ballstick || st === STYLE.space) return s * m.topo.atoms.radius[a];
  return m.state.sel[a] ? Math.max(s, 0.35) : s;
}
function pick(x, y) {
  const project = projector();
  let atom = null, atomZ = -Infinity, loose = null, looseD = 7, ribbon = null, ribbonZ = -Infinity;
  for (const m of shownModels()) {
    const R = m.topo.residues;
    for (let a = 0; a < m.topo.natoms; a++) {
      const drawn = m.drawn[a], r = m.topo.atoms.residue[a];
      const onRibbon = !drawn && state.masks[3] && R.trace[r] === a && m.state.ribbon[r];
      if (!drawn && !onRibbon) continue;
      if (drawn && !pickable(m, a)) continue;
      const p = project(m, a);
      if (!p) continue;
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (onRibbon) {
        if (d < Math.max(6, 1.0 * p[3]) && p[2] > ribbonZ) { ribbon = { model: m, residue: r }; ribbonZ = p[2]; }
        continue;
      }
      const inside = Math.max(3, drawnRadius(m, a) * p[3]);
      if (d < inside && p[2] > atomZ) { atom = { model: m, atom: a }; atomZ = p[2]; }
      if (d < looseD) { loose = { model: m, atom: a }; looseD = d; }
    }
  }
  if (atom) return atom;
  // a bond: the click near its drawn line or stick
  let bond = null, bestD = Infinity;
  for (const m of shownModels()) {
    const B = m.topo.bonds;
    for (let k = 0; k < B.i.length; k++) {
      const i = B.i[k], j = B.j[k];
      if (!m.drawn[i] || !m.drawn[j] || !pickable(m, i) || !pickable(m, j)) continue;
      const p = project(m, i), q = project(m, j);
      if (!p || !q) continue;
      const dx = q[0] - p[0], dy = q[1] - p[1], len2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((x - p[0]) * dx + (y - p[1]) * dy) / len2));
      const d = Math.hypot(p[0] + t * dx - x, p[1] + t * dy - y);
      const reach = Math.max(4, 0.3 * (p[3] + q[3]) / 2);
      if (d < reach && d < bestD) { bestD = d; bond = { model: m, atoms: [i, j] }; }
    }
  }
  return bond || loose || ribbon;
}

// atoms (and ribbon residues) drawn inside a screen rectangle: [[model, ids], ...]
function inRect(x0, y0, x1, y1) {
  const project = projector(), out = [];
  for (const m of shownModels()) {
    const ids = [], R = m.topo.residues;
    for (let a = 0; a < m.topo.natoms; a++) {
      const r = m.topo.atoms.residue[a];
      const ribbon = state.masks[3] && R.trace[r] === a && m.state.ribbon[r];
      if (!(m.drawn[a] && pickable(m, a)) && !ribbon) continue;
      const p = project(m, a);
      if (!p || p[0] < x0 || p[0] > x1 || p[1] < y0 || p[1] > y1) continue;
      if (m.drawn[a]) ids.push(a);
      else ids.push(...m.topo.residueAtoms[r]);
    }
    if (ids.length) out.push([m, ids]);
  }
  return out;
}

// --- moving the selected atoms (Alt + mouse) ------------------------------------------

let moving = null;

function beginMove() {
  const sel = selection();
  if (!sel.length) return false;
  const [m, ids] = sel[0];
  if (!m.topo.editable) { message("this model cannot be edited", true); return false; }
  m.pos = m.pos.slice();  // a working copy: the cached frame stays as it was
  moving = { model: m, ids, start: m.pos.slice() };
  return true;
}

// rotate the selected atoms about their center: view-space axis and angle
function rotateSelected(ax, ay, az, angle) {
  if (!moving) return;
  const m = moving.model, ids = moving.ids, r = camera.rot;
  // the axis in world space
  const w = [0, 1, 2].map(d => r[d] * ax + r[3 + d] * ay + r[6 + d] * az);
  const n = Math.hypot(...w) || 1;
  const u = w.map(x => x / n);
  const c = [0, 0, 0];
  for (const a of ids) for (let d = 0; d < 3; d++) c[d] += m.pos[3 * a + d] / ids.length;
  const cs = Math.cos(angle), sn = Math.sin(angle);
  for (const a of ids) {
    const v = [0, 1, 2].map(d => m.pos[3 * a + d] - c[d]);
    const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    for (let d = 0; d < 3; d++) m.pos[3 * a + d] = c[d] + v[d] * cs + cr[d] * sn + u[d] * dot * (1 - cs);
  }
  rebuild();
}

// translate the selected atoms by (dx, dy) screen pixels, or dz along the view
function translateSelected(dx, dy, dz = 0) {
  if (!moving) return;
  const m = moving.model, r = camera.rot;
  const scale = 2 * camera.distance * Math.tan(camera.fov / 2) / canvas.clientHeight;
  const v = [dx * scale, -dy * scale, dz * scale];
  const w = [0, 1, 2].map(d => r[d] * v[0] + r[3 + d] * v[1] + r[6 + d] * v[2]);
  for (const a of moving.ids) for (let d = 0; d < 3; d++) m.pos[3 * a + d] += w[d];
  rebuild();
}

// set the moving side of a dihedral to `angle` (a preview: commit with set_dihedral)
function turnDihedral(d, angle) {
  const m = d.model, p = d.base, [, b, c] = d.atoms;
  const B = xyzOf(p, b), C = xyzOf(p, c);
  let ax = C.map((x, k) => x - B[k]);
  const n = Math.hypot(...ax);
  ax = ax.map(x => x / n);
  const t = d.sign * (angle - d.angle) * Math.PI / 180, cs = Math.cos(t), sn = Math.sin(t);
  const out = p.slice();
  for (const a of d.moving) {
    const v = [0, 1, 2].map(k => p[3 * a + k] - C[k]);
    const dot = ax[0] * v[0] + ax[1] * v[1] + ax[2] * v[2];
    const cr = [ax[1] * v[2] - ax[2] * v[1], ax[2] * v[0] - ax[0] * v[2], ax[0] * v[1] - ax[1] * v[0]];
    for (let k = 0; k < 3; k++) out[3 * a + k] = C[k] + v[k] * cs + cr[k] * sn + ax[k] * dot * (1 - cs);
  }
  m.pos = out;
  rebuild();
}
const xyzOf = (p, a) => [p[3 * a], p[3 * a + 1], p[3 * a + 2]];

async function endMove() {
  if (!moving) return;
  const { model: m, ids } = moving;
  moving = null;
  const flat = [];
  for (const a of ids) flat.push(...xyz(m, a));
  try {
    await post("api/set_positions", { m: m.index, atoms: ids, xyz: flat });
    await reloadModel(m, false);
  } catch (e) {
    message(e.message, true);
  }
}

// --- frames ----------------------------------------------------------------------

function updatePlayer() {
  const n = nframes();
  $("player").hidden = n < 2;
  $("slider").max = n - 1;
  if (state.frame >= n) state.frame = n - 1;
  $("slider").value = state.frame;
  $("frame-label").textContent = `frame ${state.frame + 1} / ${n}`;
}

async function refreshContacts() {
  if (state.hbonds) await refreshHbonds();
  if (!state.contacts) return;
  await Promise.all(shownModels().map(async m => {
    const i = Math.min(state.frame, m.topo.nframes - 1);
    try { m.contacts = (await getJSON(`api/contacts?m=${m.index}&i=${i}`)).contacts; }
    catch (e) { m.contacts = []; }
  }));
}

// hydrogen bonds between two selections (Analyze | Hydrogen Bonds), in each shown model
async function refreshHbonds() {
  const { b, sel } = state.hbonds;
  let n = 0, how = "";
  const found = new Set();  // the methods of the models that found some
  await Promise.all(shownModels().map(async m => {
    const frame = Math.min(state.frame, m.topo.nframes - 1);
    const a = sel ? selected(m) : state.hbonds.a;  // the selection: in the models that have one
    if (sel && !a.length) { m.hbonds = []; return; }
    try {
      const out = await post("api/hbonds", { m: m.index, a, b, frame });
      m.hbonds = out.pairs; n += out.pairs.length;
      if (out.pairs.length) found.add(out.method); else how = how || out.method;
    } catch (e) { m.hbonds = []; how = e.message; }
  }));
  return { n, how: found.size ? [...found].join("; ") : how };
}

async function setFrame(i) {
  const n = nframes();
  i = ((i % n) + n) % n;
  if (state.loading) { state.pending = i; return; }
  state.loading = true;
  try {
    await Promise.all(shownModels().map(async m => {
      const k = Math.min(i, m.topo.nframes - 1);
      const f = await getFrame(m, k);
      if (state.ssEachFrame) m.ss = (await getJSON(`api/ss?m=${m.index}&i=${k}`)).ss;
      m.pos = f.pos; m.box = f.box;
    }));
    state.frame = i;
    await refreshContacts();
    updatePlayer();
    rebuild();
    app.onFrame();
  } catch (e) {
    message(e.message, true);
    state.playing = false;
  } finally {
    state.loading = false;
  }
  if (state.pending !== null) {
    const p = state.pending;
    state.pending = null;
    await setFrame(p);
  }
}

function stride() { return Math.max(1, parseInt($("stride").value) || 1); }

async function play() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "⏸" : "▶";
  while (state.playing) {
    const t = performance.now();
    await setFrame(state.frame + stride());
    await new Promise(ok => setTimeout(ok, Math.max(0, 33 - (performance.now() - t))));
  }
  $("play").textContent = "▶";
}

// per residue, 0..1 along its chain's traced residues (spectrum colors)
function spectrum(topo) {
  const R = topo.residues, out = new Float32Array(R.chain.length);
  const byChain = new Map();
  R.chain.forEach((c, r) => { if (!byChain.has(c)) byChain.set(c, []); byChain.get(c).push(r); });
  for (const rs of byChain.values()) {
    const traced = rs.filter(r => R.trace[r] >= 0);
    const along = traced.length > 1 ? traced : rs;
    along.forEach((r, k) => { out[r] = along.length > 1 ? k / (along.length - 1) : 0; });
    let last = 0;
    const set = new Set(along);
    for (const r of rs) { if (set.has(r)) last = out[r]; else out[r] = last; }
  }
  return out;
}

// --- sessions: the page's state, saved with the models and put back ---------------------

const ARRAYS = { sel: Uint8Array, constraint: Uint8Array, labels: Uint8Array };
function toB64(a) {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = "";
  for (let k = 0; k < bytes.length; k += 0x8000) s += String.fromCharCode(...bytes.subarray(k, k + 0x8000));
  return btoa(s);
}
function fromB64(text, Type) {
  const s = atob(text), bytes = new Uint8Array(s.length);
  for (let k = 0; k < s.length; k++) bytes[k] = s.charCodeAt(k);
  return new Type(bytes.buffer);
}

function pageState() {
  const models = {};
  for (const m of live()) {
    if (!m.state) { models[m.index] = { shown: m.shown }; continue; }
    const st = {};
    for (const k of Object.keys(ARRAYS)) st[k] = toB64(m.state[k]);
    const reps = m.reps.map(r => ({ sel: r.sel, label: r.label, style: r.style, color: r.color, on: r.on,
                                    tinted: r.tinted, material: r.material, opacity: r.opacity,
                                    mask: toB64(r.mask), ...(r.custom ? { rgb: toB64(r.rgb) } : {}) }));
    models[m.index] = { shown: m.shown, color: m.color, tintAll: !!m.tintAll, natoms: m.topo.natoms, state: st, reps };
  }
  return {
    camera: { rot: camera.rot, center: camera.center, distance: camera.distance, radius: camera.radius,
              ortho: camera.ortho, zclip: camera.zclip, pivot: camera.pivot },
    theme: document.body.className, current: state.current, frame: state.frame,
    hydrogens: state.hydrogens, contacts: state.contacts, labelMode: state.labelMode, masks: state.masks,
    fog: renderer.fog, cell: !!state.cell,
    meters: state.meters.map(x => ({ m: x.model.index, atoms: x.atoms })), models,
  };
}

// a session saved when every atom had one style and one color: a representation
// for each style used, with the atoms' colors, and one for the ribbon
function repsFromAtoms(m, saved) {
  const show = fromB64(saved.show, Uint8Array), style = fromB64(saved.style, Uint8Array);
  const rgb = fromB64(saved.rgb, Float32Array), out = [];
  const names = { [STYLE.line]: "lines", [STYLE.stick]: "sticks", [STYLE.ballline]: "ballstick",
                  [STYLE.ballstick]: "ballstick", [STYLE.space]: "spheres" };
  for (const style_ of new Set(Object.values(names))) {
    const ids = range(show.length).filter(a => show[a] && names[style[a]] === style_);
    if (!ids.length) continue;
    const rep = makeRep(m, { ids, label: "atoms", style: style_ });
    rep.rgb.set(rgb.subarray(0, rep.rgb.length)); rep.custom = true;
    out.push(rep);
  }
  const ribbon = fromB64(saved.ribbon, Uint8Array), R = m.topo.residues;
  const traced = range(ribbon.length).filter(r => ribbon[r] && R.trace[r] >= 0);
  if (traced.length) {
    const rep = makeRep(m, { ids: traced.flatMap(r => m.topo.residueAtoms[r]), label: "ribbon",
                             style: saved.ribbonStyle === "tube" ? "tube" : "cartoon", color: "ss" });
    if (saved.ribbonRGB) rep.rgb.set(fromB64(saved.ribbonRGB, Float32Array).subarray(0, rep.rgb.length));
    out.unshift(rep);
  }
  return out;
}

let appliedPage = null;
async function restorePage(page) {
  if (!page || page === appliedPage) return;
  appliedPage = page;
  const offset = page.offset || 0;
  const where = old => offset + page.indices.indexOf(Number(old));
  for (const [old, saved] of Object.entries(page.models || {})) {
    const m = state.models[where(old)];
    if (!m) continue;
    m.shown = saved.shown;
    if (!saved.state) continue;
    await load(m);
    if (m.topo.natoms !== saved.natoms) continue;
    for (const [k, Type] of Object.entries(ARRAYS)) m.state[k] = fromB64(saved.state[k], Type);
    m.color = saved.color; m.tintAll = !!saved.tintAll;
    if (saved.reps) {
      m.reps = saved.reps.map(r => {
        const rep = makeRep(m, { mask: fromB64(r.mask, Uint8Array), sel: r.sel, label: r.label, style: r.style,
                                 color: r.color, on: r.on, tinted: r.tinted, material: r.material || "opaque",
                                 opacity: r.opacity ?? null });
        if (r.rgb) { rep.rgb = fromB64(r.rgb, Float32Array); rep.custom = true; }
        return rep;
      });
    } else if (saved.state.show) {
      m.reps = repsFromAtoms(m, saved.state);  // a session saved before representations
    }
    aggregate(m);
  }
  state.meters = (page.meters || []).map(x => ({ model: state.models[where(x.m)], atoms: x.atoms })).filter(x => x.model);
  Object.assign(state, { hydrogens: page.hydrogens ?? state.hydrogens, contacts: !!page.contacts,
                         labelMode: page.labelMode || state.labelMode, masks: page.masks || state.masks,
                         cell: !!page.cell });
  if (page.fog) Object.assign(renderer.fog, page.fog);
  if (page.theme) app.setTheme?.(page.theme);
  const c = page.camera;
  if (c) Object.assign(camera, { rot: c.rot, center: c.center, distance: c.distance, radius: c.radius,
                                 ortho: c.ortho, zclip: c.zclip, pivot: c.pivot });
  if (page.current !== undefined && state.models[where(page.current)]) state.current = where(page.current);
  await show(live().filter(m => m.shown));
  if (page.frame) await setFrame(page.frame);
}

// --- the app, as mouse.js and ui.js see it ------------------------------------------------

const app = {
  state, camera, renderer, canvas,
  getJSON, post, busy, message, atomLabel,
  live, shownModels, currentModel, load, show, solo, reloadModel, selectIds, newModel, applyReps,
  selected, selection, targets, select, clearSelection, visibleAtom,
  rebuild, draw, flyTo, stopFlight, centerView, resetView, viewAll, bounds, measure, xyz,
  pick, inRect, unproject, nearerAtom, beginMove, rotateSelected, translateSelected, turnDihedral, endMove,
  setFrame, play, nframes, updatePlayer, refreshContacts, refreshHbonds, TOKEN,
  range, STYLE, pageState, restorePage, PYTHON, withPictureScene,
  REP_STYLES, MATERIALS, makeRep, colorRep, repIds, isRibbon, repStyle, repsChanged, aggregate, repAlpha,
  onSelection: () => {}, onModels: () => {}, onEmpty: () => {}, onReps: () => {}, onFrame: () => {},  // ui.js fills these in
};
app.STATIC = STATIC;
window.boonzaView = app;  // for scripting the page (and testing it)

// a few numbers about the whole view (pocket RMSD, burial, ...), top right, and
// under them the current model's own (a site's distance to the ligand, ...)
function showSummary() {
  const m = state.models[state.current];
  const own = m && m.shown ? m.summary || [] : [];
  const key = `${state.current}:${!!m?.shown}:${(state.summary || []).length}`;
  const box = $("summary");
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.innerHTML = "";
  const add = (cls, text, title = "") => {
    const d = document.createElement("div");
    d.className = cls; d.textContent = text; d.title = title;
    box.append(d);
  };
  for (const { label, value, title } of state.summary || []) { add("k", label, title); add("v", value, title); }
  if (own.length) {
    add("h", m.name, m.group);
    for (const { label, value, title } of own) { add("k", label, title); add("v", value, title); }
  }
  box.hidden = !box.children.length;
}

async function main() {
  if (STATIC) document.documentElement.classList.add("static");  // (the theme sets body's class)
  setupUI(app);
  setupMouse(app);
  try {
    const listing = await getJSON("api/models");
    state.title = listing.title;
    state.columns = listing.columns;
    state.summary = listing.summary || [];
    state.groups = listing.groups || {};  // the builder's functional groups: name -> SMILES
    state.models = listing.models.map((m, index) => newModel(m, index));
    document.title = listing.title ? `${listing.title} · boonza` : "boonza";
    if (!state.models.length) {  // an empty view: build something, or open it
      rebuild();
      app.onEmpty();
      return;
    }
    const first = state.models.find(m => m.shown) || state.models[0];
    first.shown = true;
    state.current = first.index;
    await show(state.models.filter(m => m.shown));
    resetView();
    showSummary();
    if (listing.page) await restorePage(listing.page);
    app.onSelection();
    const n = shownModels().reduce((k, m) => k + m.topo.natoms, 0);
    const intro = `${listing.title || "boonza"}: ${n} atoms${live().length > 1 ? `, ${live().length} models` : ""}`;
    message(intro);
    setTimeout(() => { if ($("message").textContent === intro) message(""); }, 4000);
  } catch (e) {
    message(e.message, true);
    throw e;
  }
}

main();
