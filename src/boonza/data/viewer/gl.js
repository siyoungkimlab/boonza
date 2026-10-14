// WebGL2 renderer: impostor spheres, instanced cylinders, triangle meshes and lines.
// A scene is {spheres, cylinders, mesh, lines}, plain typed arrays made by geometry.js.

const SHADE = `
uniform vec3 uBg;
uniform vec2 uFog;  // fog from depth uFog.x to uFog.y (view-space distance)
uniform float uFogStrength;
vec3 shade(vec3 color, vec3 n, vec3 p) {
  vec3 v = normalize(-p);
  if (dot(n, v) < 0.0) n = -n;
  vec3 key = normalize(vec3(-0.4, 0.5, 1.0));
  float diff = max(dot(n, key), 0.0) * 0.7 + max(dot(n, v), 0.0) * 0.25;
  float spec = pow(max(dot(n, normalize(key + v)), 0.0), 40.0) * 0.3;
  vec3 c = color * (0.22 + diff) + vec3(spec);
  float fog = clamp((-p.z - uFog.x) / (uFog.y - uFog.x), 0.0, 1.0);
  return mix(c, uBg, fog * uFogStrength);
}`;

const SPHERE_VS = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 aCenter;
layout(location=2) in float aRadius;
layout(location=3) in vec3 aColor;
uniform mat4 uView, uProj;
out vec3 vPos, vCenter, vColor;
out float vRadius;
void main() {
  vec3 c = (uView * vec4(aCenter, 1.0)).xyz;
  // a quad facing the camera, in front of the sphere, large enough under perspective
  vec3 p = c + vec3(aCorner * aRadius * 1.6, aRadius);
  vPos = p; vCenter = c; vRadius = aRadius; vColor = aColor;
  gl_Position = uProj * vec4(p, 1.0);
}`;

const SPHERE_FS = `#version 300 es
precision highp float;
in vec3 vPos, vCenter, vColor;
in float vRadius;
uniform mat4 uProj;
uniform float uOrtho;  // 1: parallel rays along -z; 0: rays from the eye
out vec4 frag;
${SHADE}
void main() {
  vec3 p;
  if (uOrtho > 0.5) {
    vec2 q = vPos.xy - vCenter.xy;
    float h2 = vRadius * vRadius - dot(q, q);
    if (h2 < 0.0) discard;
    p = vec3(vPos.xy, vCenter.z + sqrt(h2));
  } else {
    vec3 d = normalize(vPos);
    float b = dot(d, vCenter);
    float disc = b * b - dot(vCenter, vCenter) + vRadius * vRadius;
    if (disc < 0.0) discard;
    p = d * (b - sqrt(disc));
  }
  vec4 clip = uProj * vec4(p, 1.0);
  gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
  frag = vec4(shade(vColor, normalize(p - vCenter), p), 1.0);
}`;

const CYLINDER_VS = `#version 300 es
layout(location=0) in vec3 aLocal;  // (cos, sin, 0 or 1)
layout(location=1) in vec3 aA;
layout(location=2) in vec3 aB;
layout(location=3) in float aRadius;
layout(location=4) in vec3 aColor;
uniform mat4 uView, uProj;
out vec3 vPos, vNormal, vColor;
void main() {
  vec3 axis = aB - aA;
  vec3 t = normalize(axis);
  vec3 h = abs(t.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0);
  vec3 u = normalize(cross(t, h)), v = cross(t, u);
  vec3 n = u * aLocal.x + v * aLocal.y;
  vec4 p = uView * vec4(aA + axis * aLocal.z + n * aRadius, 1.0);
  vPos = p.xyz; vNormal = mat3(uView) * n; vColor = aColor;
  gl_Position = uProj * p;
}`;

const MESH_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
uniform mat4 uView, uProj;
out vec3 vPos, vNormal, vColor;
void main() {
  vec4 p = uView * vec4(aPos, 1.0);
  vPos = p.xyz; vNormal = mat3(uView) * aNormal; vColor = aColor;
  gl_Position = uProj * p;
}`;

const SURFACE_FS = `#version 300 es
precision highp float;
in vec3 vPos, vNormal, vColor;
out vec4 frag;
${SHADE}
void main() { frag = vec4(shade(vColor, normalize(vNormal), vPos), 1.0); }`;

const LINE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uView, uProj;
out vec3 vPos, vColor;
void main() {
  vec4 p = uView * vec4(aPos, 1.0);
  vPos = p.xyz; vColor = aColor;
  gl_Position = uProj * p;
}`;

const LINE_FS = `#version 300 es
precision highp float;
in vec3 vPos, vColor;
uniform vec3 uBg;
uniform vec2 uFog;
uniform float uFogStrength;
out vec4 frag;
void main() {
  float fog = clamp((-vPos.z - uFog.x) / (uFog.y - uFog.x), 0.0, 1.0);
  frag = vec4(mix(vColor, uBg, fog * uFogStrength), 1.0);
}`;

// the finishing pass of a picture (and of Render | Lighting | Publication):
// silhouettes where depth jumps, and ambient occlusion from the depths around
// each pixel -- a point with much nearer surface around it is in a crevice
const POST_VS = `#version 300 es
void main() {  // one triangle over the whole target
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const POST_FS = `#version 300 es
precision highp float;
uniform sampler2D uColor, uDepth;
uniform vec2 uClip;        // near, far
uniform float uOrtho;
uniform float uPx;         // pixels per A at the scene's centre
uniform float uOutline;    // line width in pixels (0: none)
uniform float uAO;         // occlusion strength (0: none)
uniform vec4 uBack;        // background (alpha 0: transparent)
out vec4 frag;
float viewZ(ivec2 q) {     // distance in front of the eye (A)
  float d = texelFetch(uDepth, q, 0).r;
  if (d >= 1.0) return 1e6;
  return uOrtho > 0.5 ? mix(uClip.x, uClip.y, d)
                      : uClip.x * uClip.y / (uClip.y - d * (uClip.y - uClip.x));
}
void main() {
  ivec2 q = ivec2(gl_FragCoord.xy), size = textureSize(uDepth, 0);
  float z = viewZ(q);
  vec3 c = texelFetch(uColor, q, 0).rgb;
  float edge = 0.0;
  if (uOutline > 0.0) {
    int w = int(max(uOutline, 1.0));
    for (int k = 0; k < 4; k++) {
      ivec2 o = ivec2(k == 0 ? w : k == 1 ? -w : 0, k == 2 ? w : k == 3 ? -w : 0);
      float z2 = viewZ(clamp(q + o, ivec2(0), size - 1));
      if (z2 - z > 1.5) edge = 1.0;   // something 1.5 A or more behind it: an edge
    }
  }
  if (z > 1e5) {           // background, unless an outline runs over it
    frag = edge > 0.0 ? vec4(0.0, 0.0, 0.0, 1.0) : uBack;
    return;
  }
  float occ = 0.0;
  if (uAO > 0.0) {
    const int N = 24;
    float r = 6.0 * uPx;   // look 6 A around
    for (int k = 0; k < N; k++) {
      float a = float(k) * 2.39996;            // golden angle: a spiral of samples
      float s = sqrt((float(k) + 0.5) / float(N));
      ivec2 o = ivec2(cos(a) * s * r, sin(a) * s * r);
      float z2 = viewZ(clamp(q + o, ivec2(0), size - 1));
      float dz = z - z2;                        // how much nearer the surface there is
      occ += smoothstep(0.3, 2.5, dz) * (1.0 - smoothstep(6.0, 10.0, dz));
    }
    occ /= float(N);
  }
  c *= 1.0 - uAO * occ;
  frag = vec4(mix(c, vec3(0.0), edge), 1.0);
}`;

const SIDES = 12;

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true });
    if (!gl) throw new Error("this browser has no WebGL2");
    this.gl = gl;
    this.canvas = canvas;
    this.background = [1, 1, 1];
    this.fog = { on: true, start: 0.35, strength: 0.75 };
    this.programs = {
      sphere: this.program(SPHERE_VS, SPHERE_FS),
      cylinder: this.program(CYLINDER_VS, SURFACE_FS),
      mesh: this.program(MESH_VS, SURFACE_FS),
      line: this.program(LINE_VS, LINE_FS),
      post: this.program(POST_VS, POST_FS, ["uColor", "uDepth", "uClip", "uOrtho", "uPx", "uOutline", "uAO", "uBack"]),
    };
    this.publication = false;  // the finishing pass on screen too
    this.counts = { spheres: 0, cylinders: 0, mesh: 0, lines: 0 };

    // spheres: a quad, plus per-instance center, radius, color
    this.sphereVao = gl.createVertexArray();
    gl.bindVertexArray(this.sphereVao);
    this.attribute(0, 2, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
    this.sphereBuf = [this.attribute(1, 3, null, 1), this.attribute(2, 1, null, 1),
                      this.attribute(3, 3, null, 1)];

    // cylinders: an open tube of SIDES quads, plus per-instance ends, radius, color
    const local = [], index = [];
    for (let k = 0; k <= SIDES; k++) {
      const a = 2 * Math.PI * k / SIDES;
      local.push(Math.cos(a), Math.sin(a), 0, Math.cos(a), Math.sin(a), 1);
    }
    for (let k = 0; k < SIDES; k++) {
      const i = 2 * k;
      index.push(i, i + 2, i + 1, i + 1, i + 2, i + 3);
    }
    this.cylinderVao = gl.createVertexArray();
    gl.bindVertexArray(this.cylinderVao);
    this.attribute(0, 3, new Float32Array(local));
    this.cylinderBuf = [this.attribute(1, 3, null, 1), this.attribute(2, 3, null, 1),
                        this.attribute(3, 1, null, 1), this.attribute(4, 3, null, 1)];
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(index), gl.STATIC_DRAW);
    this.cylinderIndices = index.length;

    this.meshVao = gl.createVertexArray();
    gl.bindVertexArray(this.meshVao);
    this.meshBuf = [this.attribute(0, 3), this.attribute(1, 3), this.attribute(2, 3)];
    this.meshIndex = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.meshIndex);

    this.lineVao = gl.createVertexArray();
    gl.bindVertexArray(this.lineVao);
    this.lineBuf = [this.attribute(0, 3), this.attribute(1, 3)];
    gl.bindVertexArray(null);
  }

  program(vs, fs, more = []) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const uniforms = {};
    for (const name of ["uView", "uProj", "uBg", "uFog", "uFogStrength", "uOrtho", ...more]) uniforms[name] = gl.getUniformLocation(p, name);
    return { p, uniforms };
  }

  attribute(loc, size, data = null, divisor = 0) {
    const gl = this.gl, buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    if (divisor) gl.vertexAttribDivisor(loc, divisor);
    return buf;
  }

  upload(buffers, arrays) {
    const gl = this.gl;
    buffers.forEach((b, k) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, arrays[k], gl.DYNAMIC_DRAW);
    });
  }

  setScene(scene) {
    const gl = this.gl, { spheres, cylinders, mesh, lines } = scene;
    this.upload(this.sphereBuf, [spheres.center, spheres.radius, spheres.color]);
    this.upload(this.cylinderBuf, [cylinders.a, cylinders.b, cylinders.radius, cylinders.color]);
    this.upload(this.meshBuf, [mesh.position, mesh.normal, mesh.color]);
    gl.bindVertexArray(this.meshVao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.meshIndex);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.index, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    this.upload(this.lineBuf, [lines.position, lines.color]);
    this.counts = { spheres: spheres.radius.length, cylinders: cylinders.radius.length,
                    mesh: mesh.index.length, lines: lines.position.length / 3 };
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, c = this.canvas;
    const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }

  draw(camera) {
    const gl = this.gl;
    this.resize();
    const w = this.canvas.width, h = this.canvas.height;
    if (!this.publication) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.drawScene(camera, w, h);
      return;
    }
    const target = this.target(w, h);
    this.drawScene(camera, w, h, target.fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.finish(camera, target, w, h, { outline: Math.max(1, Math.round(w / this.canvas.clientWidth)), ao: 0.6 });
  }

  // a framebuffer with colour and depth textures, kept while its size holds
  target(w, h, key = "screen") {
    const gl = this.gl;
    this.targets = this.targets || {};
    const t = this.targets[key];
    if (t && t.w === w && t.h === h) return t;
    if (t) { gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.color); gl.deleteTexture(t.depth); }
    const tex = (internal, format, type) => {
      const x = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, x);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
      for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
      for (const k of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE);
      return x;
    };
    const color = tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    const depth = tex(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`no ${w} x ${h} picture on this GPU`);
    this.targets[key] = { fb, color, depth, w, h };
    return this.targets[key];
  }

  // the finishing pass, from a target into the bound framebuffer
  finish(camera, t, w, h, { outline = 1, ao = 0.6, back = null } = {}) {
    const gl = this.gl, { p, uniforms: u } = this.programs.post;
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t.color); gl.uniform1i(u.uColor, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, t.depth); gl.uniform1i(u.uDepth, 1);
    gl.uniform2fv(u.uClip, camera.clip());
    gl.uniform1f(u.uOrtho, camera.ortho ? 1 : 0);
    const px = camera.ortho ? h / (2 * camera.halfHeight()) : h / 2 / Math.tan(camera.fov / 2) / camera.distance;
    gl.uniform1f(u.uPx, px);
    gl.uniform1f(u.uOutline, outline);
    gl.uniform1f(u.uAO, ao);
    gl.uniform4fv(u.uBack, back || [...this.background, 1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
    gl.enable(gl.DEPTH_TEST);
  }

  // a picture of w x h pixels (sharper than the screen: drawn at up to twice that
  // and scaled down), finished with outlines and occlusion; a canvas to save
  picture(camera, w, h, { outline = true, ao = true, background = null, transparent = false, fog = true } = {}) {
    const gl = this.gl, max = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), 8192);
    const ss = Math.max(1, Math.min(2, Math.floor(max / Math.max(w, h))));
    const W = w * ss, H = h * ss;
    const keep = { bg: this.background, fog: this.fog.on };
    if (background) this.background = background;
    this.fog.on = fog && !transparent;
    const scene = this.target(W, H, "picture-scene");
    this.drawScene(camera, W, H, scene.fb);
    const out = this.target(W, H, "picture-out");
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.fb);
    this.finish(camera, scene, W, H, { outline: outline ? Math.max(1, Math.round(ss * h / 900)) : 0,
                                       ao: ao ? 0.6 : 0, back: transparent ? [0, 0, 0, 0] : null });
    const pixels = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    for (const k of ["picture-scene", "picture-out"]) {  // big: let them go
      const t = this.targets[k];
      gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.color); gl.deleteTexture(t.depth);
      delete this.targets[k];
    }
    this.background = keep.bg; this.fog.on = keep.fog;
    // rows come bottom up; turn them over, then scale down to w x h
    const big = document.createElement("canvas");
    big.width = W; big.height = H;
    const img = new ImageData(W, H), row = W * 4;
    for (let y = 0; y < H; y++) img.data.set(pixels.subarray((H - 1 - y) * row, (H - y) * row), y * row);
    big.getContext("2d").putImageData(img, 0, 0);
    if (ss === 1) return big;
    const small = document.createElement("canvas");
    small.width = w; small.height = h;
    const g = small.getContext("2d");
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(big, 0, 0, w, h);
    return small;
  }

  drawScene(camera, width, height, fb = null) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, width, height);
    gl.clearColor(...this.background, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const view = camera.view(), proj = camera.projection(width / height);
    // fog: from `start` (a fraction of the scene's depth in front of its centre) to its back
    const f = this.fog, depth = camera.radius * 1.5;
    const fog = f.on
      ? [camera.distance - depth * (1 - 2 * f.start), camera.distance + depth]
      : [1e9, 1e9 + 1];
    const use = ({ p, uniforms }) => {
      gl.useProgram(p);
      gl.uniformMatrix4fv(uniforms.uView, false, view);
      gl.uniformMatrix4fv(uniforms.uProj, false, proj);
      gl.uniform3fv(uniforms.uBg, this.background);
      gl.uniform2fv(uniforms.uFog, fog);
      gl.uniform1f(uniforms.uFogStrength, f.strength);
      gl.uniform1f(uniforms.uOrtho, camera.ortho ? 1 : 0);
    };
    const n = this.counts;
    if (n.spheres) {
      use(this.programs.sphere);
      gl.bindVertexArray(this.sphereVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.spheres);
    }
    if (n.cylinders) {
      use(this.programs.cylinder);
      gl.bindVertexArray(this.cylinderVao);
      gl.drawElementsInstanced(gl.TRIANGLES, this.cylinderIndices, gl.UNSIGNED_SHORT, 0, n.cylinders);
    }
    if (n.mesh) {
      use(this.programs.mesh);
      gl.bindVertexArray(this.meshVao);
      gl.drawElements(gl.TRIANGLES, n.mesh, gl.UNSIGNED_INT, 0);
    }
    if (n.lines) {
      use(this.programs.line);
      gl.bindVertexArray(this.lineVao);
      gl.drawArrays(gl.LINES, 0, n.lines);
    }
    gl.bindVertexArray(null);
  }
}
