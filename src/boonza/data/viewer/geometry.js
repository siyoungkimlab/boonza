// A model's render state -> a scene of typed arrays for gl.js.
// Every atom has its own state: shown or not, a style, a color; every
// residue its ribbon.  Selected atoms are drawn pink, with a ball (hidden ones
// too), and a selected residue's ribbon pink.

export const STYLES = ["line", "stick", "ballline", "ballstick", "space"];
export const STYLE_NAMES = ["Line", "Stick", "Ball and Line", "Ball and Stick", "Space Filling"];
const [LINE, STICK, BALLLINE, BALLSTICK, SPACE] = [0, 1, 2, 3, 4];
export const RIBBONS = ["cartoon", "tube"];
const PINK = [1, 0.45, 0.85];

class Grow {  // a growable Float32Array
  constructor() { this.a = new Float32Array(1024); this.n = 0; }
  push(...v) {
    if (this.n + v.length > this.a.length) {
      const b = new Float32Array(Math.max(this.a.length * 2, this.n + v.length));
      b.set(this.a);
      this.a = b;
    }
    for (const x of v) this.a[this.n++] = x;
  }
  append(v) {  // a whole array (a surface's): no spread, which overflows the stack
    if (this.n + v.length > this.a.length) {
      const b = new Float32Array(Math.max(this.a.length * 2, this.n + v.length));
      b.set(this.a.subarray(0, this.n));
      this.a = b;
    }
    this.a.set(v, this.n);
    this.n += v.length;
  }
  get array() { return this.a.subarray(0, this.n); }
}

// sizes (Å) of the styles; beads are about twice as wide as atoms
const SIZES = {
  atoms: { stick: 0.2, ball: 0.25, bond: 0.1, cross: 0.3, picked: 0.35 },
  beads: { stick: 0.6, ball: 0.5, bond: 0.35, cross: 1.0, picked: 1.1 },
};

// models: [{topo, pos, ss, state, dashes, surface, alpha, shine}]; state: {show, style,
// rgb, sel, ribbon, ribbonRGB, ribbonStyle}; dashes [[x, y, z, x, y, z, r, g, b], ...].
// Drawings of one material (opacity and shine) share a layer; the opaque, plain one first
export function buildScene(models) {
  const layers = new Map();
  const layerOf = (alpha, shine) => {
    const key = `${alpha}|${shine}`;
    if (!layers.has(key)) {
      layers.set(key, { alpha, shine, s: { center: new Grow(), radius: new Grow(), color: new Grow() },
                        c: { a: new Grow(), b: new Grow(), radius: new Grow(), color: new Grow() },
                        l: { position: new Grow(), color: new Grow() },
                        m: { position: new Grow(), normal: new Grow(), color: new Grow(), index: [] } });
    }
    return layers.get(key);
  };
  layerOf(1, 1);
  for (const model of models) {
    const L = layerOf(model.alpha ?? 1, model.shine ?? 1);
    addModel(model, L.s, L.c, L.l, L.m);
    if (model.surface) addSurface(model.surface, L.m);
  }
  const out = [...layers.values()].sort((a, b) => b.alpha - a.alpha).map(({ alpha, shine, s, c, l, m }) => ({
    alpha, shine,
    spheres: { center: s.center.array, radius: s.radius.array, color: s.color.array },
    cylinders: { a: c.a.array, b: c.b.array, radius: c.radius.array, color: c.color.array },
    mesh: { position: m.position.array, normal: m.normal.array, color: m.color.array, index: new Uint32Array(m.index) },
    lines: { position: l.position.array, color: l.color.array },
  }));
  return { layers: out };
}

// a surface (surface.js) joins the meshes, its vertices after theirs
function addSurface(surf, m) {
  const first = m.position.n / 3;
  m.position.append(surf.position);
  m.normal.append(surf.normal);
  m.color.append(surf.color);
  for (let k = 0; k < surf.index.length; k++) m.index.push(surf.index[k] + first);
}

function addModel({ topo, pos, ss, state, dashes }, s, c, l, m) {
  const size = topo.cg ? SIZES.beads : SIZES.atoms;
  const { show, style, rgb, sel } = state;
  const A = topo.atoms, B = topo.bonds, n = topo.natoms;
  for (const d of dashes || []) dash(d, l, c);
  const P = a => [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
  const C = a => sel[a] ? PINK : [rgb[3 * a], rgb[3 * a + 1], rgb[3 * a + 2]];

  // bonds: a line where either end is drawn as lines, a stick where both are sticks
  const bonded = new Uint8Array(n);
  // a residue drawn as a ribbon shows no atoms, so a bond out of a sticks selection
  // into it was drawn by nobody and the selection looked detached: one residue of a
  // chain chosen as the ligand floated beside the cartoon.  Such a bond is drawn as a
  // stub, from the atom that is shown to half way
  const ribbonOn = new Uint8Array(n);
  for (let r = 0; r < topo.residues.trace.length; r++) {
    if (state.ribbon[r]) for (const a of topo.residueAtoms[r]) ribbonOn[a] = 1;
  }
  for (let k = 0; k < B.i.length; k++) {
    const i = B.i[k], j = B.j[k];
    if (!show[i] && !show[j]) continue;
    if (!show[i] || !show[j]) {  // a stub into a ribbon, and only into a ribbon
      const vis = show[i] ? i : j, hid = show[i] ? j : i;
      if (!ribbonOn[hid] || style[vis] === SPACE) continue;
      const pv = P(vis), ph = P(hid);
      const mid = pv.map((x, d) => (x + ph[d]) / 2);
      bonded[vis] = 1;
      if (style[vis] === LINE || style[vis] === BALLLINE) {
        segment(l, c, pv, mid, C(vis), topo.cg ? LINE_RADIUS.beads : LINE_RADIUS.atoms);
      } else {
        const r0 = style[vis] === BALLSTICK ? size.bond : size.stick;
        c.a.push(...pv); c.b.push(...mid);
        c.radius.push(r0); c.color.push(...C(vis));
      }
      continue;
    }
    const si = style[i], sj = style[j];
    if (si === SPACE || sj === SPACE) continue;
    bonded[i] = bonded[j] = 1;
    const pi = P(i), pj = P(j);
    const line = si === LINE || sj === LINE || si === BALLLINE || sj === BALLLINE;
    const order = topo.cg ? 1 : Math.min(B.order[k] || 1, 3);
    const r0 = si === BALLSTICK || sj === BALLSTICK ? size.bond : size.stick;
    // a double bond is two strands, a triple three, side by side in the plane of a neighbor
    const w = order > 1 ? offsetDirection(topo, pos, i, j) : null;
    const strands = order === 1 ? [[0, 1]] : order === 2 ? [[-1, 0.5], [1, 0.5]] : [[-1, 0.4], [0, 0.4], [1, 0.4]];
    const gap = line ? (order === 2 ? 0.11 : 0.15) : r0 * (order === 2 ? 0.62 : 0.85);
    for (const [k2, scale] of strands) {
      const o = w ? w.map(x => x * gap * k2) : [0, 0, 0];
      const a0 = pi.map((x, d) => x + o[d]), b0 = pj.map((x, d) => x + o[d]);
      const mid = a0.map((x, d) => (x + b0[d]) / 2);
      if (line) {
        const lr = (topo.cg ? LINE_RADIUS.beads : LINE_RADIUS.atoms) * (order === 1 ? 1 : 0.7);
        segment(l, c, a0, mid, C(i), lr);
        segment(l, c, mid, b0, C(j), lr);
      } else {
        const r = r0 * (order === 1 ? 1 : scale);
        c.a.push(...a0, ...mid); c.b.push(...mid, ...b0);
        c.radius.push(r, r); c.color.push(...C(i), ...C(j));
      }
    }
  }
  // the traced atom of each residue whose ribbon is on
  const R = topo.residues;
  const traced = new Uint8Array(n);
  for (let r = 0; r < R.trace.length; r++) if (R.trace[r] >= 0 && state.ribbon[r]) traced[R.trace[r]] = 1;
  // atoms: joints of sticks, balls, spheres; lone atoms of lines as small crosses
  for (let a = 0; a < n; a++) {
    if (!show[a]) {
      // a hidden atom shows only when selected: a pink ball, or its ribbon turns pink
      if (sel[a] && !traced[a]) { s.center.push(...P(a)); s.radius.push(size.picked); s.color.push(...PINK); }
      continue;
    }
    const st = style[a];
    let r = 0;
    if (st === STICK) r = size.stick;
    else if (st === BALLLINE || st === BALLSTICK) r = size.ball * A.radius[a];
    else if (st === SPACE) r = A.radius[a];
    else if (DETAIL > 1 && bonded[a]) r = topo.cg ? LINE_RADIUS.beads : LINE_RADIUS.atoms;  // a picture's line joints
    if (sel[a]) r = Math.max(r, size.picked);  // a selected atom shows as a pink ball
    if (r > 0) {
      s.center.push(...P(a)); s.radius.push(r); s.color.push(...C(a));
    } else if (!bonded[a]) {
      const p = P(a);
      for (let d = 0; d < 3; d++) {
        const u = [0, 0, 0];
        u[d] = size.cross;
        segment(l, c, [p[0] - u[0], p[1] - u[1], p[2] - u[2]], [p[0] + u[0], p[1] + u[1], p[2] + u[2]], C(a),
                (topo.cg ? LINE_RADIUS.beads : LINE_RADIUS.atoms));
      }
    }
  }
  // ribbons in the residue's color, pink where the traced atom is selected
  const color = new Float32Array(n * 3);
  let any = false;
  for (let r = 0; r < R.trace.length; r++) {
    const t = R.trace[r];
    if (t < 0 || !traced[t]) continue;
    any = true;
    for (let d = 0; d < 3; d++) color[3 * t + d] = sel[t] ? PINK[d] : state.ribbonRGB[3 * r + d];
  }
  if (any) {
    const shape = topo.cg ? CG_SHAPE : state.ribbonStyle === "tube" ? TUBE : SHAPE;
    cartoon(topo, pos, traced, ss, color, m, shape);
    bases(topo, pos, traced, color, c, m);
  }
}

// a nucleotide's base in its ribbon: a filled slab of each ring, outlined, and a rod
// from the backbone through the sugar to it, in the residue's colour
const RINGS = { six: ["N1", "C2", "N3", "C4", "C5", "C6"], five: ["C4", "C5", "N7", "C8", "N9"] };
function bases(topo, pos, traced, color, c, m) {
  const R = topo.residues, A = topo.atoms;
  const P = a => [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
  for (let r = 0; r < R.trace.length; r++) {
    const t = R.trace[r];
    if (t < 0 || !traced[t] || R.kind[r] !== 2) continue;
    const named = {};
    for (const a of topo.residueAtoms[r]) named[A.name[a].replace("*", "'")] = a;
    const col = [color[3 * t], color[3 * t + 1], color[3 * t + 2]];
    const rings = [RINGS.six, ...(named.N9 !== undefined ? [RINGS.five] : [])]
      .map(ring => ring.map(n => named[n])).filter(ring => ring.every(a => a !== undefined));
    if (!rings.length) continue;  // no base atoms (beads, a backbone alone)
    const rod = (a, b, radius) => { c.a.push(...a); c.b.push(...b); c.radius.push(radius); c.color.push(...col); };
    // backbone to sugar to the base's glycosidic nitrogen (purines N9, pyrimidines N1)
    const sugar = named["C1'"], glyco = named.N9 ?? named.N1;
    if (sugar !== undefined && glyco !== undefined) { rod(P(t), P(sugar), 0.32); rod(P(sugar), P(glyco), 0.32); }
    else if (glyco !== undefined) rod(P(t), P(glyco), 0.32);
    for (const ring of rings) {
      const pts = ring.map(P);
      const ctr = [0, 1, 2].map(d => pts.reduce((s, q) => s + q[d], 0) / pts.length);
      let n = [0, 0, 0];  // the ring's plane: summed cross products around it
      for (let k = 0; k < pts.length; k++) n = n.map((x, d) => x + cross(sub(pts[k], ctr), sub(pts[(k + 1) % pts.length], ctr))[d]);
      n = norm(n);
      // both faces, a little apart, so the slab has a body
      for (const side of [1, -1]) {
        const off = n.map(x => x * 0.12 * side), first = m.position.n / 3;
        m.position.push(...ctr.map((x, d) => x + off[d]));
        m.normal.push(...n.map(x => x * side)); m.color.push(...col);
        for (const q of pts) { m.position.push(...q.map((x, d) => x + off[d])); m.normal.push(...n.map(x => x * side)); m.color.push(...col); }
        for (let k = 0; k < pts.length; k++) m.index.push(first, first + 1 + k, first + 1 + (k + 1) % pts.length);
      }
      for (let k = 0; k < pts.length; k++) rod(pts[k], pts[(k + 1) % pts.length], 0.13);
    }
  }
}

// a line segment of one color: a GL line, or (a picture) a thin cylinder
function segment(l, c, p, q, color, r) {
  if (DETAIL > 1) { c.a.push(...p); c.b.push(...q); c.radius.push(r); c.color.push(...color); return; }
  l.position.push(...p, ...q);
  l.color.push(...color, ...color);
}

// the side a multiple bond's strands spread to: toward a neighbor of either end
function offsetDirection(topo, pos, i, j) {
  const p = a => [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
  const pi = p(i), pj = p(j), v = pj.map((x, d) => x - pi[d]);
  const vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] || 1;
  const nb = topo.bondsOf ? (topo.bondsOf[i].find(x => x !== j) ?? topo.bondsOf[j].find(x => x !== i)) : undefined;
  let r = nb !== undefined ? p(nb).map((x, d) => x - pi[d]) : [v[1], -v[0], 0];
  const s = (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) / vv;
  r = r.map((x, d) => x - v[d] * s);
  let n = Math.hypot(...r);
  if (n < 1e-6) { r = Math.abs(v[0]) < 0.9 * Math.sqrt(vv) ? [0, v[2], -v[1]] : [v[2], 0, -v[0]]; n = Math.hypot(...r) || 1; }
  return r.map(x => x / n);
}

// a dashed line: 0.25 Å dashes, 0.15 Å gaps
function dash([x0, y0, z0, x1, y1, z1, r, g, b], l, c) {
  const len = Math.hypot(x1 - x0, y1 - y0, z1 - z0);
  for (let t = 0; t < len; t += 0.4) {
    const u = t / len, v = Math.min(t + 0.25, len) / len;
    segment(l, c, [x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, z0 + (z1 - z0) * u],
            [x0 + (x1 - x0) * v, y0 + (y1 - y0) * v, z0 + (z1 - z0) * v], [r, g, b], DASH_RADIUS);
  }
}

// --- cartoon -------------------------------------------------------------

let SUB = 8;     // spline samples per residue
let RING = 12;   // vertices around the cross-section
// 1: the screen's; 2: a picture's (smoother ribbons, lines drawn as thin sticks, which
// a picture scaled down from a larger drawing would otherwise lose)
let DETAIL = 1;
export function setDetail(k) { DETAIL = k; SUB = k > 1 ? 18 : 8; RING = k > 1 ? 32 : 12; }
const LINE_RADIUS = { atoms: 0.07, beads: 0.2 }, DASH_RADIUS = 0.06;
const MAXGAP = { 1: 4.6, 2: 8.5 };  // trace distance (Å) above which a chain breaks
// half width, half thickness of the cross-section; coarse-grained chains (no
// secondary structure) are a thicker tube, in scale with their beads
const SHAPE = { H: [1.1, 0.25], E: [1.0, 0.3], C: [0.3, 0.3], N: [0.45, 0.45] };
// beads: secondary structure from the trace (the server's P-SEA), a thicker coil
const CG_SHAPE = { H: [1.3, 0.3], E: [1.2, 0.35], C: [0.45, 0.45], N: [1.0, 1.0] };
const TUBE = { H: [0.3, 0.3], E: [0.3, 0.3], C: [0.3, 0.3], N: [0.45, 0.45] };
const ARROW = 1.7;  // half width of a strand's arrow head

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };

// consecutive traced residues of the selection, split at chain changes and gaps
function segments(topo, pos, inRep) {
  const R = topo.residues, out = [];
  let seg = [];
  const P = a => [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
  for (let r = 0; r < R.trace.length; r++) {
    const t = R.trace[r];
    const ok = t >= 0 && inRep[t];
    if (ok && seg.length) {
      const q = seg[seg.length - 1];
      const gap = Math.hypot(...sub(P(t), P(R.trace[q])));
      if (R.chain[q] !== R.chain[r] || R.kind[q] !== R.kind[r] || q !== r - 1 || gap > MAXGAP[R.kind[r]]) {
        if (seg.length > 1) out.push(seg);
        seg = [];
      }
    }
    if (ok) seg.push(r);
    else if (seg.length) { if (seg.length > 1) out.push(seg); seg = []; }
  }
  if (seg.length > 1) out.push(seg);
  return out;
}

// any unit vector perpendicular to t
function perpendicular(t) {
  return norm(cross(t, Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
}

function cartoon(topo, pos, inRep, ss, color, m, SHAPE) {
  const R = topo.residues;
  const P = a => [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
  for (const seg of segments(topo, pos, inRep)) {
    const n = seg.length;
    const code = seg.map(r => R.kind[r] === 2 ? "N" : (ss[r] === "H" || ss[r] === "E" ? ss[r] : "C"));
    let pts = seg.map(r => P(R.trace[r]));
    // strands: average out the CA zig-zag so sheets are flat
    pts = pts.map((p, k) => code[k] === "E" && k > 0 && k < n - 1 && code[k - 1] === "E" && code[k + 1] === "E"
      ? p.map((x, d) => (pts[k - 1][d] + 2 * x + pts[k + 1][d]) / 4) : p);
    // guides: trace -> guide atom (carbonyl O); a trace without one (beads) is
    // oriented by its curvature: the binormal runs along a helix's axis and lies in
    // a sheet's plane, as the O would
    const raw = seg.map((r, k) => {
      const g = R.guide[r];
      if (g >= 0) return sub(P(g), P(R.trace[r]));
      if (k === 0 || k === n - 1) return null;
      const t = sub(pts[k + 1], pts[k - 1]);
      const c = sub(sub(pts[k], pts[k - 1]), sub(pts[k + 1], pts[k]));
      const b = cross(t, c);
      return Math.hypot(...b) > 1e-6 ? b : null;
    });
    for (let k = 0; k < n; k++) if (!raw[k]) raw[k] = raw[k + 1] || raw[k - 1] || null;
    const guides = [];
    for (let k = 0; k < n; k++) {
      const t = sub(pts[Math.min(k + 1, n - 1)], pts[Math.max(k - 1, 0)]);
      let v = raw[k] || (k ? guides[k - 1] : perpendicular(norm(t)));
      if (k && dot(v, guides[k - 1]) < 0) v = v.map(x => -x);
      guides.push(norm(v));
    }
    const cols = seg.map(r => [color[3 * R.trace[r]], color[3 * R.trace[r] + 1], color[3 * R.trace[r] + 2]]);
    const at = k => pts[Math.min(Math.max(k, 0), n - 1)];

    // samples along a Catmull-Rom spline through the trace
    const first = m.position.n / 3;
    let rings = 0;
    for (let k = 0; k < n - 1; k++) {
      for (let s = 0; s < (k === n - 2 ? SUB + 1 : SUB); s++) {
        const u = s / SUB;
        const p0 = at(k - 1), p1 = at(k), p2 = at(k + 1), p3 = at(k + 2);
        const u2 = u * u, u3 = u2 * u;
        const p = [0, 1, 2].map(d => 0.5 * (2 * p1[d] + (p2[d] - p0[d]) * u
          + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * u2 + (3 * p1[d] - p0[d] - 3 * p2[d] + p3[d]) * u3));
        const tan = norm([0, 1, 2].map(d => 0.5 * ((p2[d] - p0[d])
          + 2 * (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * u + 3 * (3 * p1[d] - p0[d] - 3 * p2[d] + p3[d]) * u2)));
        const res = u < 0.5 ? k : k + 1;
        // the arrow keeps its strand's orientation rather than turning toward the coil
        const arrow = code[k] === "E" && code[k + 1] === "E" && code[k + 2] !== "E";
        const g = arrow ? guides[k] : norm(guides[k].map((x, d) => x * (1 - u) + guides[k + 1][d] * u));
        let nrm = sub(g, tan.map(x => x * dot(g, tan)));
        nrm = Math.hypot(...nrm) > 1e-3 ? norm(nrm) : perpendicular(tan);
        const bin = cross(tan, nrm);
        // width: blend between residues; strands end in an arrow
        const [w0, h0] = SHAPE[code[k]], [w1, h1] = SHAPE[code[k + 1]];
        let w = w0 + (w1 - w0) * u, h = h0 + (h1 - h0) * u;
        if (arrow && SHAPE.E[0] !== SHAPE.E[1]) { w = ARROW * (1 - u) + 0.3 * u; h = SHAPE.E[1]; }
        else if (code[k] === "E" && code[k + 1] !== "E") { w = SHAPE.C[0]; h = SHAPE.C[1]; }
        const col = cols[res];
        for (let j = 0; j < RING; j++) {
          const a = 2 * Math.PI * j / RING, ca = Math.cos(a), sa = Math.sin(a);
          m.position.push(...[0, 1, 2].map(d => p[d] + nrm[d] * w * ca + bin[d] * h * sa));
          m.normal.push(...norm([0, 1, 2].map(d => nrm[d] * ca / w + bin[d] * sa / h)));
          m.color.push(...col);
        }
        rings++;
      }
    }
    for (let q = 0; q < rings - 1; q++) {
      for (let j = 0; j < RING; j++) {
        const a = first + q * RING + j, b = first + q * RING + (j + 1) % RING;
        m.index.push(a, a + RING, b, b, a + RING, b + RING);
      }
    }
  }
}
