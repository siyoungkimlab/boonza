// An orbit camera: rotation R about a center, at a distance, with perspective.

export class Camera {
  constructor() {
    this.rot = [1, 0, 0, 0, 1, 0, 0, 0, 1];  // row-major 3x3, world -> view
    this.center = [0, 0, 0];
    this.distance = 50;
    this.radius = 20;  // of the scene, for clipping and fog
    this.fov = 0.6;    // vertical, radians
    this.pivot = null; // the center of rotation (middle-click an atom), else the view's
    this.zclip = [1, 1]; // front and back clipping, as fractions of the default depth
    this.ortho = true;   // orthographic by default; perspective on request
  }

  // the half height of the view (Å) at the center: what zooming changes
  halfHeight() { return this.distance * Math.tan(this.fov / 2); }

  // a view-space point on the screen: [x, y, pixels per Å there] for a w x h canvas
  screen(v, w, h) {
    if (this.ortho) {
      const hh = this.halfHeight();
      return [(v[0] / (hh * w / h) * 0.5 + 0.5) * w, (0.5 - v[1] / hh * 0.5) * h, h / (2 * hh)];
    }
    const k = 1 / Math.tan(this.fov / 2);
    return [(k * h / w * v[0] / -v[2] * 0.5 + 0.5) * w, (0.5 - k * v[1] / -v[2] * 0.5) * h, k * h / 2 / -v[2]];
  }

  // see a sphere whole, in the narrower of the two directions (aspect = width / height)
  fit(center, radius, aspect = 1) {
    this.center = center.slice();
    this.radius = Math.max(radius, 2);
    const half = Math.min(this.fov / 2, Math.atan(Math.tan(this.fov / 2) * aspect));
    this.distance = this.radius / Math.sin(half);
  }

  // look along the smallest principal axis of points (flat xyz), the largest across
  orient(xyz) {
    const n = xyz.length / 3;
    if (n < 3) return;
    const m = [0, 0, 0];
    for (let a = 0; a < n; a++) for (let d = 0; d < 3; d++) m[d] += xyz[3 * a + d] / n;
    const c = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let a = 0; a < n; a++)
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) c[3 * i + j] += (xyz[3 * a + i] - m[i]) * (xyz[3 * a + j] - m[j]);
    const [values, vectors] = eigen(c);
    const order = [0, 1, 2].sort((i, j) => values[j] - values[i]);
    const x = order[0], y = order[1];
    const ex = [vectors[x], vectors[3 + x], vectors[6 + x]];
    const ey = [vectors[y], vectors[3 + y], vectors[6 + y]];
    const ez = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
    this.rot = [...ex, ...ey, ...ez];
  }

  // rotate about view-space axis (x, y, z) by angle, about the pivot when there is one
  rotate(x, y, z, angle) {
    const before = this.rot.slice();
    this._rotate(x, y, z, angle);
    if (!this.pivot) return;
    // keep the pivot where it is on the screen: c' = c + R'^T (q - D^T q)... done in world
    // space as: the pivot's view position is unchanged
    const p = this.pivot, c = this.center, r0 = before, r1 = this.rot;
    const q = [0, 1, 2].map(i => r0[3 * i] * (p[0] - c[0]) + r0[3 * i + 1] * (p[1] - c[1]) + r0[3 * i + 2] * (p[2] - c[2]));
    // the new center puts p at the same view coordinates q under the new rotation
    for (let d = 0; d < 3; d++) c[d] = p[d] - (r1[d] * q[0] + r1[3 + d] * q[1] + r1[6 + d] * q[2]);
  }

  _rotate(x, y, z, angle) {
    const n = Math.hypot(x, y, z);
    if (!n || !angle) return;
    x /= n; y /= n; z /= n;
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const m = [t * x * x + c, t * x * y - s * z, t * x * z + s * y,
               t * x * y + s * z, t * y * y + c, t * y * z - s * x,
               t * x * z - s * y, t * y * z + s * x, t * z * z + c];
    const r = this.rot, out = new Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        out[3 * i + j] = m[3 * i] * r[j] + m[3 * i + 1] * r[3 + j] + m[3 * i + 2] * r[6 + j];
    this.rot = out;
  }

  // move the center by (dx, dy) in view-space Å
  pan(dx, dy) {
    const r = this.rot;
    for (let d = 0; d < 3; d++) this.center[d] -= r[d] * dx + r[3 + d] * dy;
  }

  zoom(factor) {
    this.distance = Math.min(Math.max(this.distance * factor, 2), 5000);
  }

  // column-major 4x4 matrices for WebGL
  view() {
    const r = this.rot, c = this.center;
    const t = [0, 1, 2].map(i => -(r[3 * i] * c[0] + r[3 * i + 1] * c[1] + r[3 * i + 2] * c[2]));
    return new Float32Array([
      r[0], r[3], r[6], 0,
      r[1], r[4], r[7], 0,
      r[2], r[5], r[8], 0,
      t[0], t[1], t[2] - this.distance, 1,
    ]);
  }

  clip() {  // near, far planes around the scene (ZClip moves them in)
    const near = Math.max(this.distance - this.radius * 1.5 * this.zclip[0], 0.5);
    return [near, Math.max(this.distance + this.radius * 1.5 * this.zclip[1], near + 0.5)];
  }

  projection(aspect) {
    const [n, f] = this.clip();
    if (this.ortho) {
      const hh = this.halfHeight();
      return new Float32Array([
        1 / (hh * aspect), 0, 0, 0,
        0, 1 / hh, 0, 0,
        0, 0, -2 / (f - n), 0,
        0, 0, -(f + n) / (f - n), 1,
      ]);
    }
    const k = 1 / Math.tan(this.fov / 2);
    return new Float32Array([
      k / aspect, 0, 0, 0,
      0, k, 0, 0,
      0, 0, (f + n) / (n - f), -1,
      0, 0, 2 * f * n / (n - f), 0,
    ]);
  }

  // a world point in view space
  toView(p) {
    const r = this.rot, c = this.center;
    const x = p[0] - c[0], y = p[1] - c[1], z = p[2] - c[2];
    return [r[0] * x + r[1] * y + r[2] * z, r[3] * x + r[4] * y + r[5] * z,
            r[6] * x + r[7] * y + r[8] * z - this.distance];
  }
}

// eigenvalues and eigenvectors (columns) of a symmetric 3x3 matrix, by Jacobi rotations
function eigen(a) {
  a = a.slice();
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) off += a[3 * p + q] ** 2;
    if (off < 1e-18) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[3 * p + q];
      if (Math.abs(apq) < 1e-30) continue;
      const theta = (a[3 * q + q] - a[3 * p + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {  // a = J^T a J
        const akp = a[3 * k + p], akq = a[3 * k + q];
        a[3 * k + p] = c * akp - s * akq; a[3 * k + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[3 * p + k], aqk = a[3 * q + k];
        a[3 * p + k] = c * apk - s * aqk; a[3 * q + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[3 * k + p], vkq = v[3 * k + q];
        v[3 * k + p] = c * vkp - s * vkq; v[3 * k + q] = s * vkp + c * vkq;
      }
    }
  }
  return [[a[0], a[4], a[8]], v];
}
