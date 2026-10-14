// A model's render state -> a scene of typed arrays for gl.js.
// Every atom has its own state, as in MOE: shown or not, a style, a colour; every
// residue its ribbon.  Selected atoms are drawn pink, with a ball.

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

// models: [{topo, pos, ss, state, dashes}]; state: {show, style, rgb, sel, ribbon,
// ribbonRGB, ribbonStyle}; dashes [[x, y, z, x, y, z, r, g, b], ...]
export function buildScene(models) {
  const s = { center: new Grow(), radius: new Grow(), color: new Grow() };
  const c = { a: new Grow(), b: new Grow(), radius: new Grow(), color: new Grow() };
  const l = { position: new Grow(), color: new Grow() };
  const m = { position: new Grow(), normal: new Grow(), color: new Grow(), index: [] };
  for (const model of models) {
    addModel(model, s, c, l, m);
    if (model.surface) addSurface(model.surface, m);
  }
  return {
    spheres: { center: s.center.array, radius: s.radius.array, color: s.color.array },
    cylinders: { a: c.a.array, b: c.b.array, radius: c.radius.array, color: c.color.array },
    mesh: { position: m.position.array, normal: m.normal.array, color: m.color.array,
            index: new Uint32Array(m.index) },
    lines: { position: l.position.array, color: l.color.array },
  };
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
  for (const d of dashes || []) dash(d, l);
  const P = a => [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]];
  const C = a => sel[a] ? PINK : [rgb[3 * a], rgb[3 * a + 1], rgb[3 * a + 2]];

  // bonds: a line where either end is drawn as lines, a stick where both are sticks
  const bonded = new Uint8Array(n);
  for (let k = 0; k < B.i.length; k++) {
    const i = B.i[k], j = B.j[k];
    if (!show[i] || !show[j]) continue;
    const si = style[i], sj = style[j];
    if (si === SPACE || sj === SPACE) continue;
    bonded[i] = bonded[j] = 1;
    const pi = P(i), pj = P(j);
    const line = si === LINE || sj === LINE || si === BALLLINE || sj === BALLLINE;
    const order = topo.cg ? 1 : Math.min(B.order[k] || 1, 3);
    const r0 = si === BALLSTICK || sj === BALLSTICK ? size.bond : size.stick;
    // a double bond is two strands, a triple three, side by side in the plane of a neighbour
    const w = order > 1 ? offsetDirection(topo, pos, i, j) : null;
    const strands = order === 1 ? [[0, 1]] : order === 2 ? [[-1, 0.5], [1, 0.5]] : [[-1, 0.4], [0, 0.4], [1, 0.4]];
    const gap = line ? (order === 2 ? 0.11 : 0.15) : r0 * (order === 2 ? 0.62 : 0.85);
    for (const [k2, scale] of strands) {
      const o = w ? w.map(x => x * gap * k2) : [0, 0, 0];
      const a0 = pi.map((x, d) => x + o[d]), b0 = pj.map((x, d) => x + o[d]);
      const mid = a0.map((x, d) => (x + b0[d]) / 2);
      if (line) {
        l.position.push(...a0, ...mid, ...mid, ...b0);
        l.color.push(...C(i), ...C(i), ...C(j), ...C(j));
      } else {
        const r = r0 * (order === 1 ? 1 : scale);
        c.a.push(...a0, ...mid); c.b.push(...mid, ...b0);
        c.radius.push(r, r); c.color.push(...C(i), ...C(j));
      }
    }
  }
  // atoms: joints of sticks, balls, spheres; lone atoms of lines as small crosses
  for (let a = 0; a < n; a++) {
    if (!show[a]) {
      continue;  // hidden atoms stay hidden, selected or not
    }
    const st = style[a];
    let r = 0;
    if (st === STICK) r = size.stick;
    else if (st === BALLLINE || st === BALLSTICK) r = size.ball * A.radius[a];
    else if (st === SPACE) r = A.radius[a];
    if (sel[a]) r = Math.max(r, size.picked);  // a selected atom shows as a pink ball
    if (r > 0) {
      s.center.push(...P(a)); s.radius.push(r); s.color.push(...C(a));
    } else if (!bonded[a]) {
      const p = P(a);
      for (let d = 0; d < 3; d++) {
        const u = [0, 0, 0];
        u[d] = size.cross;
        l.position.push(p[0] - u[0], p[1] - u[1], p[2] - u[2], p[0] + u[0], p[1] + u[1], p[2] + u[2]);
        l.color.push(...C(a), ...C(a));
      }
    }
  }
  // ribbons: the traced atom of each residue whose ribbon is on, in the residue's colour
  const R = topo.residues;
  const traced = new Uint8Array(n);
  const color = new Float32Array(n * 3);
  let any = false;
  for (let r = 0; r < R.trace.length; r++) {
    const t = R.trace[r];
    if (t < 0 || !state.ribbon[r]) continue;
    traced[t] = 1;
    any = true;
    for (let d = 0; d < 3; d++) color[3 * t + d] = state.ribbonRGB[3 * r + d];
  }
  if (any) {
    const shape = topo.cg ? CG_SHAPE : state.ribbonStyle === "tube" ? TUBE : SHAPE;
    cartoon(topo, pos, traced, ss, color, m, shape);
  }
}

// the side a multiple bond's strands spread to: toward a neighbour of either end
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
function dash([x0, y0, z0, x1, y1, z1, r, g, b], l) {
  const len = Math.hypot(x1 - x0, y1 - y0, z1 - z0);
  for (let t = 0; t < len; t += 0.4) {
    const u = t / len, v = Math.min(t + 0.25, len) / len;
    l.position.push(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, z0 + (z1 - z0) * u,
                    x0 + (x1 - x0) * v, y0 + (y1 - y0) * v, z0 + (z1 - z0) * v);
    l.color.push(r, g, b, r, g, b);
  }
}

// --- cartoon -------------------------------------------------------------

const SUB = 8;     // spline samples per residue
const RING = 12;   // vertices around the cross-section
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
