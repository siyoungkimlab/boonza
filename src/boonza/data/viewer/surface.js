// Molecular surfaces: a smooth (Gaussian) surface around atoms, coloured plain,
// by electrostatic potential or by hydrophobicity.
//
// The surface is where a sum of blobs, one per atom, equals 1:
//     rho(x) = sum_i exp(-B (|x - x_i|^2 / r_i^2 - 1)),
// so an atom alone is a sphere of its radius r_i (van der Waals, or the bead's)
// and neighbours blend into one skin that bridges gaps narrower than a water.
// B sets how much they blend; it is triangulated by marching tetrahedra.
//
// Potential: Coulomb with a distance-dependent dielectric (eps = 4r), summed
// within 15 A; shown from -5 (red) through 0 (white) to +5 kT/e (blue).
// Hydrophobicity: the atoms' signs (properties.py) averaged with Gaussian
// weights (sigma 1.5 A) at each point; dark cyan (hydrophilic), white, goldenrod
// (hydrophobic), as ChimeraX colours it.

const B = 2.0;            // blending
const SPACING = { atoms: 0.75, beads: 1.2 };  // grid (A)
const KT = 0.596;         // kcal/mol at 300 K
const COULOMB = 332.06;   // kcal A / (mol e^2)
const REACH = 15;         // A, the potential's
const SCALE = 5;          // kT/e at full colour
const SIGMA = 1.5;        // A, hydrophobicity's smoothing

const RED = [0.84, 0.19, 0.15], WHITE = [1, 1, 1], BLUE = [0.19, 0.38, 0.86];
const CYAN = [0, 0.55, 0.55], GOLD = [0.85, 0.65, 0.13];

// the six tetrahedra of a cube, by its corners (x + 2y + 4z)
const TETS = [[0, 5, 1, 7], [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7]];

export function surface(model, ids, kind = "plain") {
  const { topo, pos } = model;
  const R = topo.atoms.radius;
  const h = topo.cg ? SPACING.beads : SPACING.atoms;
  // the grid: the atoms' box, padded by twice the largest radius
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], rmax = 0;
  for (const a of ids) {
    for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], pos[3 * a + d]); hi[d] = Math.max(hi[d], pos[3 * a + d]); }
    rmax = Math.max(rmax, R[a]);
  }
  lo = lo.map(x => x - 2 * rmax - h); hi = hi.map(x => x + 2 * rmax + h);
  const nx = Math.ceil((hi[0] - lo[0]) / h) + 1, ny = Math.ceil((hi[1] - lo[1]) / h) + 1, nz = Math.ceil((hi[2] - lo[2]) / h) + 1;
  const rho = new Float32Array(nx * ny * nz);
  const at = (i, j, k) => i + nx * (j + ny * k);
  for (const a of ids) {
    const r = R[a], reach = 2 * r, x = pos[3 * a], y = pos[3 * a + 1], z = pos[3 * a + 2];
    const i0 = Math.max(0, Math.floor((x - reach - lo[0]) / h)), i1 = Math.min(nx - 1, Math.ceil((x + reach - lo[0]) / h));
    const j0 = Math.max(0, Math.floor((y - reach - lo[1]) / h)), j1 = Math.min(ny - 1, Math.ceil((y + reach - lo[1]) / h));
    const k0 = Math.max(0, Math.floor((z - reach - lo[2]) / h)), k1 = Math.min(nz - 1, Math.ceil((z + reach - lo[2]) / h));
    const inv = 1 / (r * r);
    for (let k = k0; k <= k1; k++) {
      const dz = lo[2] + k * h - z;
      for (let j = j0; j <= j1; j++) {
        const dy = lo[1] + j * h - y, dyz = dy * dy + dz * dz;
        for (let i = i0; i <= i1; i++) {
          const dx = lo[0] + i * h - x, d2 = dx * dx + dyz;
          if (d2 < reach * reach) rho[at(i, j, k)] += Math.exp(-B * (d2 * inv - 1));
        }
      }
    }
  }

  // marching tetrahedra; a vertex per crossed grid edge, shared by its triangles
  const position = [], index = [], edges = new Map();
  const corner = c => [c & 1, (c >> 1) & 1, (c >> 2) & 1];
  const vertex = (p, q) => {
    const key = p < q ? p * rho.length + q : q * rho.length + p;
    let v = edges.get(key);
    if (v !== undefined) return v;
    const t = (1 - rho[p]) / (rho[q] - rho[p]);
    const P = grid(p), Q = grid(q);
    v = position.length / 3;
    position.push(P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1]), P[2] + t * (Q[2] - P[2]));
    edges.set(key, v);
    return v;
  };
  const grid = p => {
    const i = p % nx, j = Math.floor(p / nx) % ny, k = Math.floor(p / (nx * ny));
    return [lo[0] + i * h, lo[1] + j * h, lo[2] + k * h];
  };
  const cube = new Int32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    let inside = 0;
    for (let c = 0; c < 8; c++) {
      const [a, b, d] = corner(c);
      cube[c] = at(i + a, j + b, k + d);
      if (rho[cube[c]] >= 1) inside++;
    }
    if (inside === 0 || inside === 8) continue;
    for (const tet of TETS) {
      const v = tet.map(c => cube[c]);
      const ins = v.filter(p => rho[p] >= 1), out = v.filter(p => rho[p] < 1);
      if (ins.length === 0 || ins.length === 4) continue;
      if (ins.length === 1 || ins.length === 3) {
        const [one, rest] = ins.length === 1 ? [ins[0], out] : [out[0], ins];
        const tri = rest.map(p => vertex(one, p));
        orient(tri, ins.length === 1 ? one : null, ins.length === 3 ? one : null);
      } else {
        const [a, b] = ins, [c, d] = out;
        const q = [vertex(a, c), vertex(a, d), vertex(b, d), vertex(b, c)];
        orient([q[0], q[1], q[2]], a, null);
        orient([q[0], q[2], q[3]], a, null);
      }
    }
  }
  // triangles wound so their normal points out (away from an inside corner)
  function orient(tri, inside, outside) {
    const [a, b, c] = tri.map(v => position.slice(3 * v, 3 * v + 3));
    const n = cross(sub(b, a), sub(c, a));
    const ref = grid(inside ?? outside);
    const s = dot(n, sub(a, ref)) * (inside !== null ? 1 : -1);
    if (s >= 0) index.push(tri[0], tri[1], tri[2]); else index.push(tri[0], tri[2], tri[1]);
  }

  const nv = position.length / 3;
  const normal = new Float32Array(3 * nv), color = new Float32Array(3 * nv);
  const q = topo.atoms.q, lipo = topo.atoms.lipo, rgb = model.state.rgb;
  // the atoms near a point (its normal, its colour), and the charged ones in reach
  const near = neighbours(pos, ids, Math.max(2 * rmax, 3 * SIGMA));
  const charged = kind === "electrostatic" && q ? neighbours(pos, ids.filter(a => q[a]), REACH) : null;
  for (let v = 0; v < nv; v++) {
    const x = position[3 * v], y = position[3 * v + 1], z = position[3 * v + 2];
    // the normal: the density's gradient
    let gx = 0, gy = 0, gz = 0, phi = 0, wsum = 0, lsum = 0, best = Infinity, nearest = -1;
    near(x, y, z, a => {
      const dx = x - pos[3 * a], dy = y - pos[3 * a + 1], dz = z - pos[3 * a + 2];
      const d2 = dx * dx + dy * dy + dz * dz, r = R[a];
      if (d2 < 4 * r * r) {
        const g = Math.exp(-B * (d2 / (r * r) - 1)) * 2 * B / (r * r);
        gx += g * dx; gy += g * dy; gz += g * dz;
      }
      if (d2 - r * r < best) { best = d2 - r * r; nearest = a; }
      if (kind === "hydrophobic" && lipo && d2 < 9 * SIGMA * SIGMA) {
        const w = Math.exp(-d2 / (2 * SIGMA * SIGMA));
        wsum += w; lsum += w * lipo[a];
      }
    });
    if (charged) charged(x, y, z, a => {
      const dx = x - pos[3 * a], dy = y - pos[3 * a + 1], dz = z - pos[3 * a + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < REACH * REACH) phi += COULOMB * q[a] / (4 * Math.max(d2, 1));
    });
    const len = Math.hypot(gx, gy, gz) || 1;
    normal[3 * v] = gx / len; normal[3 * v + 1] = gy / len; normal[3 * v + 2] = gz / len;
    let c;
    if (kind === "electrostatic") {
      const t = Math.max(-1, Math.min(1, phi / KT / SCALE));
      c = t < 0 ? mix(WHITE, RED, -t) : mix(WHITE, BLUE, t);
    } else if (kind === "hydrophobic") {
      const t = wsum ? Math.max(-1, Math.min(1, lsum / wsum)) : 0;
      c = t < 0 ? mix(WHITE, CYAN, -t) : mix(WHITE, GOLD, t);
    } else {
      c = nearest >= 0 ? [rgb[3 * nearest], rgb[3 * nearest + 1], rgb[3 * nearest + 2]] : WHITE;
    }
    color[3 * v] = c[0]; color[3 * v + 1] = c[1]; color[3 * v + 2] = c[2];
  }
  return { position: Float32Array.from(position), normal, color, index: Uint32Array.from(index) };
}

// each atom within reach of a point (and some a little further), from cells reach wide
function neighbours(pos, ids, reach) {
  const cells = new Map(), key = (i, j, k) => ((i + 512) * 1024 + (j + 512)) * 1024 + (k + 512);
  for (const a of ids) {
    const k = key(Math.floor(pos[3 * a] / reach), Math.floor(pos[3 * a + 1] / reach), Math.floor(pos[3 * a + 2] / reach));
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(a);
  }
  return (x, y, z, visit) => {
    const i = Math.floor(x / reach), j = Math.floor(y / reach), k = Math.floor(z / reach);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const cell = cells.get(key(i + a, j + b, k + c));
      if (cell) for (const atom of cell) visit(atom);
    }
  };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
