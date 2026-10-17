// Molecular surfaces: the solvent-excluded surface (SES) around atoms, coloured
// by electrostatic potential, by hydrophobicity, or one colour.
//
// The SES is what a probe sphere (a water, 1.4 A) rolling over the atoms
// leaves: the solvent-accessible surface (each atom's sphere grown by the
// probe) pulled back in by the probe's radius.  On a grid: the solvent points
// next to the molecule, moved onto the grown spheres, are where the probe's
// centre can be; a point is inside the molecule if it is farther than the
// probe from all of them, and the surface, triangulated by marching
// tetrahedra, is where that distance equals the probe's.  The grid's spacing
// is the resolution (0.5 A standard).
//
// Potential: Coulomb with a distance-dependent dielectric (eps = 4r), read 1.4 A
// out from the surface along its normal, as ChimeraX colours it: from -10 (red)
// through 0 (white) to +10 kcal/(mol e) (blue).  Every charge counts, unless
// that would be slow (a large system): then those within 15 A.  It is worked
// out on a 1 A grid where the surface reads it, and interpolated.
// Hydrophobicity: the atoms' signs (properties.py) averaged with Gaussian
// weights (sigma 1.5 A) at each point; dark cyan (hydrophilic), white, goldenrod
// (hydrophobic), as ChimeraX colours it.

export const GRIDS = [[1.0, "Coarse (1.0 Å)"], [0.5, "Standard (0.5 Å)"], [0.3, "Fine (0.3 Å)"]];
const PROBE = 1.4;        // A, a water
const BEAD_GRID = 2;      // beads: a grid this much coarser
const COULOMB = 332.06;   // kcal A / (mol e^2)
const OFFSET = 1.4;       // A, off the surface, where the potential is read
const SCALE = 10;         // kcal/(mol e) at full colour
const REACH = 15;         // A, the potential's, for a large system
const ALL_PAIRS = 3e8;    // points x charges above which REACH applies
const PHI_GRID = 1.0;     // A, the potential's grid, interpolated to the vertices
const SIGMA = 1.5;        // A, hydrophobicity's smoothing

const RED = [0.84, 0.19, 0.15], WHITE = [1, 1, 1], BLUE = [0.19, 0.38, 0.86];
const CYAN = [0, 0.55, 0.55], GOLD = [0.85, 0.65, 0.13], PLAIN = [0.86, 0.86, 0.88];

// the six tetrahedra of a cube, by its corners (x + 2y + 4z)
const TETS = [[0, 5, 1, 7], [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7]];

export function surface(model, ids, kind = "electrostatic", grid = 0.5) {
  const { topo, pos } = model;
  const R = topo.atoms.radius;
  const h = topo.cg ? grid * BEAD_GRID : grid;
  // the grid: the atoms' box, padded by the largest grown radius
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], rmax = 0;
  for (const a of ids) {
    for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], pos[3 * a + d]); hi[d] = Math.max(hi[d], pos[3 * a + d]); }
    rmax = Math.max(rmax, R[a]);
  }
  const pad = rmax + PROBE + 2 * h;
  lo = lo.map(x => x - pad); hi = hi.map(x => x + pad);
  const nx = Math.ceil((hi[0] - lo[0]) / h) + 1, ny = Math.ceil((hi[1] - lo[1]) / h) + 1, nz = Math.ceil((hi[2] - lo[2]) / h) + 1;
  const N = nx * ny * nz;
  const at = (i, j, k) => i + nx * (j + ny * k);

  // e: how far each point is outside the nearest grown sphere (negative inside;
  // left large where no sphere comes near)
  const INF = 1e20, band = h;
  const e = new Float32Array(N).fill(INF);
  for (const a of ids) {
    const r = R[a] + PROBE, reach = r + band, x = pos[3 * a], y = pos[3 * a + 1], z = pos[3 * a + 2];
    const i0 = Math.max(0, Math.ceil((x - reach - lo[0]) / h)), i1 = Math.min(nx - 1, Math.floor((x + reach - lo[0]) / h));
    const j0 = Math.max(0, Math.ceil((y - reach - lo[1]) / h)), j1 = Math.min(ny - 1, Math.floor((y + reach - lo[1]) / h));
    const k0 = Math.max(0, Math.ceil((z - reach - lo[2]) / h)), k1 = Math.min(nz - 1, Math.floor((z + reach - lo[2]) / h));
    for (let k = k0; k <= k1; k++) {
      const dz = lo[2] + k * h - z;
      for (let j = j0; j <= j1; j++) {
        const dy = lo[1] + j * h - y, dyz = dy * dy + dz * dz;
        for (let i = i0; i <= i1; i++) {
          const dx = lo[0] + i * h - x, p = at(i, j, k);
          const d = Math.sqrt(dx * dx + dyz) - r;
          if (d < e[p]) e[p] = d;
        }
      }
    }
  }
  // probe centres: the solvent points next to the molecule, each moved onto the
  // grown sphere nearest it (unless that lands inside another)
  const grown = neighbours(pos, ids, rmax + PROBE + band);
  const centres = [];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const p = at(i, j, k);
    if (e[p] < 0 || e[p] >= band) continue;
    const x = lo[0] + i * h, y = lo[1] + j * h, z = lo[2] + k * h;
    let best = Infinity, b = -1;
    grown(x, y, z, a => {
      const d = Math.hypot(x - pos[3 * a], y - pos[3 * a + 1], z - pos[3 * a + 2]) - R[a] - PROBE;
      if (d < best) { best = d; b = a; }
    });
    if (b < 0) continue;
    const dx = x - pos[3 * b], dy = y - pos[3 * b + 1], dz = z - pos[3 * b + 2];
    const f = (R[b] + PROBE) / (Math.hypot(dx, dy, dz) || 1);
    const c = [pos[3 * b] + dx * f, pos[3 * b + 1] + dy * f, pos[3 * b + 2] + dz * f];
    let clash = false;
    grown(c[0], c[1], c[2], a => {
      if (a !== b && Math.hypot(c[0] - pos[3 * a], c[1] - pos[3 * a + 1], c[2] - pos[3 * a + 2]) < R[a] + PROBE - 1e-3) clash = true;
    });
    if (!clash) centres.push(c[0], c[1], c[2]);
  }
  // rho: a point's distance to the nearest probe centre over the probe's radius
  // (0 in the solvent, large deep inside); the molecule is where it is at least 1
  const rho = e;
  for (let p = 0; p < N; p++) rho[p] = rho[p] < 0 ? INF : 0;
  const reach = PROBE + 1.5 * h;
  for (let c = 0; c < centres.length; c += 3) {
    const x = centres[c], y = centres[c + 1], z = centres[c + 2];
    const i0 = Math.max(0, Math.ceil((x - reach - lo[0]) / h)), i1 = Math.min(nx - 1, Math.floor((x + reach - lo[0]) / h));
    const j0 = Math.max(0, Math.ceil((y - reach - lo[1]) / h)), j1 = Math.min(ny - 1, Math.floor((y + reach - lo[1]) / h));
    const k0 = Math.max(0, Math.ceil((z - reach - lo[2]) / h)), k1 = Math.min(nz - 1, Math.floor((z + reach - lo[2]) / h));
    for (let k = k0; k <= k1; k++) {
      const dz = lo[2] + k * h - z;
      for (let j = j0; j <= j1; j++) {
        const dy = lo[1] + j * h - y, dyz = dy * dy + dz * dz;
        for (let i = i0; i <= i1; i++) {
          const p = at(i, j, k);
          if (rho[p] === 0) continue;
          const dx = lo[0] + i * h - x, d = Math.sqrt(dx * dx + dyz) / PROBE;
          if (d < rho[p]) rho[p] = d;
        }
      }
    }
  }
  for (let p = 0; p < N; p++) if (rho[p] > 2) rho[p] = 2;

  // marching tetrahedra; a vertex per crossed grid edge, shared by its triangles,
  // its normal the field's gradient (rho grows inward)
  const position = [], normals = [], index = [], edges = new Map();
  const OFF = [0, 1, nx, nx + 1, nx * ny, nx * ny + 1, nx * ny + nx, nx * ny + nx + 1];
  const gridPoint = p => {
    const i = p % nx, j = Math.floor(p / nx) % ny, k = Math.floor(p / (nx * ny));
    return [i, j, k];
  };
  const gradient = p => {
    const [i, j, k] = gridPoint(p);
    const g = (q, m) => rho[q] - rho[m];
    return [g(at(Math.min(i + 1, nx - 1), j, k), at(Math.max(i - 1, 0), j, k)),
            g(at(i, Math.min(j + 1, ny - 1), k), at(i, Math.max(j - 1, 0), k)),
            g(at(i, j, Math.min(k + 1, nz - 1)), at(i, j, Math.max(k - 1, 0)))];  // fmt: skip
  };
  const vertex = (p, q) => {
    const key = p < q ? p * N + q : q * N + p;
    let v = edges.get(key);
    if (v !== undefined) return v;
    const t = (1 - rho[p]) / (rho[q] - rho[p]);
    const P = gridPoint(p), Q = gridPoint(q), gp = gradient(p), gq = gradient(q);
    v = position.length / 3;
    for (let d = 0; d < 3; d++) {
      position.push(lo[d] + h * (P[d] + t * (Q[d] - P[d])));
      normals.push(-(gp[d] + t * (gq[d] - gp[d])));
    }
    edges.set(key, v);
    return v;
  };
  const cube = new Int32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const base = at(i, j, k);
    let inside = 0;
    for (let c = 0; c < 8; c++) {
      cube[c] = base + OFF[c];
      if (rho[cube[c]] >= 1) inside++;
    }
    if (inside === 0 || inside === 8) continue;
    for (const tet of TETS) {
      const v = tet.map(c => cube[c]);
      const ins = v.filter(p => rho[p] >= 1), out = v.filter(p => rho[p] < 1);
      if (ins.length === 0 || ins.length === 4) continue;
      if (ins.length === 1 || ins.length === 3) {
        const [one, rest] = ins.length === 1 ? [ins[0], out] : [out[0], ins];
        orient(rest.map(p => vertex(one, p)));
      } else {
        const [a, b] = ins, [c, d] = out;
        const q = [vertex(a, c), vertex(a, d), vertex(b, d), vertex(b, c)];
        orient([q[0], q[1], q[2]]);
        orient([q[0], q[2], q[3]]);
      }
    }
  }
  // triangles wound so their normal agrees with the vertices' (outward)
  function orient(tri) {
    const [a, b, c] = tri.map(v => position.slice(3 * v, 3 * v + 3));
    const n = cross(sub(b, a), sub(c, a));
    const m = normals.slice(3 * tri[0], 3 * tri[0] + 3);
    if (dot(n, m) >= 0) index.push(tri[0], tri[1], tri[2]); else index.push(tri[0], tri[2], tri[1]);
  }

  const nv = position.length / 3;
  const normal = new Float32Array(3 * nv), color = new Float32Array(3 * nv);
  const q = topo.atoms.q, lipo = topo.atoms.lipo;
  const near = neighbours(pos, ids, 3 * SIGMA);
  // the potential: on a grid of its own (it varies slowly), made where it is read
  let potential = null;
  if (kind === "electrostatic" && q) {
    const charges = ids.filter(a => q[a]);
    const inReach = nv / 8 * charges.length > ALL_PAIRS ? neighbours(pos, charges, REACH) : null;
    const g = PHI_GRID, plo = lo.map(x => x - OFFSET - g);
    const mx = Math.ceil((hi[0] - lo[0] + 2 * OFFSET + 2 * g) / g) + 1;
    const my = Math.ceil((hi[1] - lo[1] + 2 * OFFSET + 2 * g) / g) + 1;
    const mz = Math.ceil((hi[2] - lo[2] + 2 * OFFSET + 2 * g) / g) + 1;
    const phi = new Float32Array(mx * my * mz).fill(NaN);
    const node = (i, j, k) => {
      const p = i + mx * (j + my * k);
      if (!Number.isNaN(phi[p])) return phi[p];
      const x = plo[0] + i * g, y = plo[1] + j * g, z = plo[2] + k * g;
      let sum = 0;
      const add = a => {
        const dx = x - pos[3 * a], dy = y - pos[3 * a + 1], dz = z - pos[3 * a + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (!inReach || r2 < REACH * REACH) sum += COULOMB * q[a] / (4 * Math.max(r2, 1));
      };
      if (inReach) inReach(x, y, z, add); else for (const a of charges) add(a);
      return (phi[p] = sum);
    };
    potential = (x, y, z) => {  // trilinear
      const u = (x - plo[0]) / g, v = (y - plo[1]) / g, w = (z - plo[2]) / g;
      const i = Math.floor(u), j = Math.floor(v), k = Math.floor(w), fu = u - i, fv = v - j, fw = w - k;
      let sum = 0;
      for (let c = 0; c < 8; c++) {
        const a = c & 1, b = (c >> 1) & 1, d = (c >> 2) & 1;
        sum += node(i + a, j + b, k + d) * (a ? fu : 1 - fu) * (b ? fv : 1 - fv) * (d ? fw : 1 - fw);
      }
      return sum;
    };
  }
  for (let v = 0; v < nv; v++) {
    let nx_ = normals[3 * v], ny_ = normals[3 * v + 1], nz_ = normals[3 * v + 2];
    const len = Math.hypot(nx_, ny_, nz_) || 1;
    nx_ /= len; ny_ /= len; nz_ /= len;
    normal[3 * v] = nx_; normal[3 * v + 1] = ny_; normal[3 * v + 2] = nz_;
    const x = position[3 * v], y = position[3 * v + 1], z = position[3 * v + 2];
    let c = PLAIN;
    if (potential) {
      const phi = potential(x + OFFSET * nx_, y + OFFSET * ny_, z + OFFSET * nz_);
      const t = Math.max(-1, Math.min(1, phi / SCALE));
      c = t < 0 ? mix(WHITE, RED, -t) : mix(WHITE, BLUE, t);
    } else if (kind === "electrostatic") {
      c = WHITE;
    } else if (kind === "hydrophobic" && lipo) {
      let wsum = 0, lsum = 0;
      near(x, y, z, a => {
        const dx = x - pos[3 * a], dy = y - pos[3 * a + 1], dz = z - pos[3 * a + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 9 * SIGMA * SIGMA) { const w = Math.exp(-r2 / (2 * SIGMA * SIGMA)); wsum += w; lsum += w * lipo[a]; }
      });
      const t = wsum ? Math.max(-1, Math.min(1, lsum / wsum)) : 0;
      c = t < 0 ? mix(WHITE, CYAN, -t) : mix(WHITE, GOLD, t);
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
