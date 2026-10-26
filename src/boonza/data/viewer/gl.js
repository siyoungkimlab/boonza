// WebGL2 renderer: impostor spheres, instanced cylinders, triangle meshes and lines.
// A scene is {layers: [{alpha, spheres, cylinders, mesh, lines}]}, plain typed arrays
// made by geometry.js.  Opaque layers (alpha 1) are drawn first; each see-through layer
// then in two passes, its depth and then its color where it is nearest, so a
// transparent surface shows its front over what is behind it and never itself.

const SHADE = `
uniform vec3 uBg;
uniform float uAlpha;  // the layer's opacity
uniform float uShine;  // its highlights: 0 matte, 1 plain, more glossy
uniform vec2 uFog;  // fog from depth uFog.x to uFog.y (view-space distance)
uniform float uFogStrength;
vec3 shade(vec3 color, vec3 n, vec3 p) {
  vec3 v = normalize(-p);
  if (dot(n, v) < 0.0) n = -n;
  vec3 key = normalize(vec3(-0.4, 0.5, 1.0));
  float diff = max(dot(n, key), 0.0) * 0.7 + max(dot(n, v), 0.0) * 0.25;
  float spec = pow(max(dot(n, normalize(key + v)), 0.0), 40.0 + 40.0 * max(uShine - 1.0, 0.0)) * 0.3 * uShine;
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
  frag = vec4(shade(vColor, normalize(p - vCenter), p), uAlpha);
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
void main() { frag = vec4(shade(vColor, normalize(vNormal), vPos), uAlpha); }`;

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
uniform float uAlpha;
out vec4 frag;
void main() {
  float fog = clamp((-vPos.z - uFog.x) / (uFog.y - uFog.x), 0.0, 1.0);
  frag = vec4(mix(vColor, uBg, fog * uFogStrength), uAlpha);
}`;

// a finished picture onto the screen, as it is
const COPY_FS = `#version 300 es
precision highp float;
uniform sampler2D uColor;
out vec4 frag;
void main() { frag = texelFetch(uColor, ivec2(gl_FragCoord.xy), 0); }`;

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
uniform float uPx;         // pixels per A at the scene's center
uniform float uOutline;    // line width in pixels (0: none)
uniform float uAO;         // occlusion strength (0: none)
uniform float uRadius;     // how far occlusion looks, in pixels
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
    for (int k = 0; k < 8; k++) {   // the four sides and the four corners
      ivec2 o = ivec2(k < 4 ? (k == 0 ? w : k == 1 ? -w : 0) : (k < 6 ? w : -w),
                      k < 4 ? (k == 2 ? w : k == 3 ? -w : 0) : (k % 2 == 0 ? w : -w));
      if (k >= 4) o = ivec2(vec2(o) * 0.7071);
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
    const int N = 32;
    // each pixel turns the spiral of samples its own way: noise that averages out
    float turn = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    for (int k = 0; k < N; k++) {
      float a = float(k) * 2.39996 + turn;     // golden angle: a spiral of samples
      float s = sqrt((float(k) + 0.5) / float(N));
      ivec2 o = ivec2(cos(a) * s * uRadius, sin(a) * s * uRadius);
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
      post: this.program(POST_VS, POST_FS, ["uColor", "uDepth", "uClip", "uOrtho", "uPx", "uOutline", "uAO", "uRadius", "uBack"]),
    };
    this.programs.copy = this.program(POST_VS, COPY_FS, ["uColor"]);
    this.publication = false;  // the finishing pass on screen too

    // shared by every layer: a sphere's quad; a cylinder's tube of quads (a picture,
    // detail 2, draws them rounder from the same instances)
    this.quad = this.buffer(new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
    this.tubes = {};
    for (const [detail, sides] of [[1, SIDES], [2, 36]]) {
      const local = [], index = [];
      for (let k = 0; k <= sides; k++) {
        const a = 2 * Math.PI * k / sides;
        local.push(Math.cos(a), Math.sin(a), 0, Math.cos(a), Math.sin(a), 1);
      }
      for (let k = 0; k < sides; k++) {
        const i = 2 * k;
        index.push(i, i + 2, i + 1, i + 1, i + 2, i + 3);
      }
      const ix = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ix);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(index), gl.STATIC_DRAW);
      this.tubes[detail] = { local: this.buffer(new Float32Array(local)), index: ix, count: index.length };
    }
    this.detail = 1;
    this.layers = [];  // made as scenes need them, and kept
    this.drawn = [];   // the current scene's: [{layer, alpha, counts}]
  }

  buffer(data) {
    const gl = this.gl, b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }

  // one layer's buffers and vertex arrays: spheres, cylinders (at each detail), mesh, lines
  layer() {
    const gl = this.gl, L = {};
    L.sphereVao = gl.createVertexArray();
    gl.bindVertexArray(L.sphereVao);
    this.bind(this.quad, 0, 2);
    L.sphereBuf = [this.attribute(1, 3, null, 1), this.attribute(2, 1, null, 1), this.attribute(3, 3, null, 1)];
    L.cylinders = {};
    L.cylinderBuf = null;
    for (const [detail, tube] of Object.entries(this.tubes)) {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      this.bind(tube.local, 0, 3);
      if (!L.cylinderBuf) {
        L.cylinderBuf = [this.attribute(1, 3, null, 1), this.attribute(2, 3, null, 1),
                         this.attribute(3, 1, null, 1), this.attribute(4, 3, null, 1)];
      } else {
        [[1, 3], [2, 3], [3, 1], [4, 3]].forEach(([loc, size], k) => this.bind(L.cylinderBuf[k], loc, size, 1));
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, tube.index);
      L.cylinders[detail] = { vao, count: tube.count };
    }
    L.meshVao = gl.createVertexArray();
    gl.bindVertexArray(L.meshVao);
    L.meshBuf = [this.attribute(0, 3), this.attribute(1, 3), this.attribute(2, 3)];
    L.meshIndex = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, L.meshIndex);
    L.lineVao = gl.createVertexArray();
    gl.bindVertexArray(L.lineVao);
    L.lineBuf = [this.attribute(0, 3), this.attribute(1, 3)];
    gl.bindVertexArray(null);
    return L;
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
    for (const name of ["uView", "uProj", "uBg", "uFog", "uFogStrength", "uOrtho", "uAlpha", "uShine", ...more]) uniforms[name] = gl.getUniformLocation(p, name);
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

  bind(buf, loc, size, divisor = 0) {  // an existing buffer as an attribute of the bound VAO
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    if (divisor) gl.vertexAttribDivisor(loc, divisor);
  }

  upload(buffers, arrays) {
    const gl = this.gl;
    buffers.forEach((b, k) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, arrays[k], gl.DYNAMIC_DRAW);
    });
  }

  setScene(scene) {
    const gl = this.gl;
    const layers = scene.layers || [{ alpha: 1, ...scene }];
    while (this.layers.length < layers.length) this.layers.push(this.layer());
    this.drawn = layers.map(({ alpha, shine = 1, spheres, cylinders, mesh, lines }, k) => {
      const L = this.layers[k];
      this.upload(L.sphereBuf, [spheres.center, spheres.radius, spheres.color]);
      this.upload(L.cylinderBuf, [cylinders.a, cylinders.b, cylinders.radius, cylinders.color]);
      this.upload(L.meshBuf, [mesh.position, mesh.normal, mesh.color]);
      gl.bindVertexArray(L.meshVao);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, L.meshIndex);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.index, gl.DYNAMIC_DRAW);
      gl.bindVertexArray(null);
      this.upload(L.lineBuf, [lines.position, lines.color]);
      return { layer: L, alpha, shine, counts: { spheres: spheres.radius.length, cylinders: cylinders.radius.length,
                                          mesh: mesh.index.length, lines: lines.position.length / 3 } };
    });
  }

  get glass() { return this.drawn.some(d => d.alpha < 1); }

  // a target's depth into another of its size, for see-through layers drawn after the finish
  copyDepth(from, to, w, h) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, from.fb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, to.fb);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.DEPTH_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
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
    // outlines and occlusion of what is solid; see-through layers over the finished view
    const target = this.target(w, h);
    const finish = { outline: Math.max(1, Math.round(w / this.canvas.clientWidth)), ao: 0.6 };
    this.drawScene(camera, w, h, target.fb, null, "opaque");
    if (!this.glass) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.finish(camera, target, w, h, finish);
      return;
    }
    const out = this.target(w, h, "screen-out");
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.fb);
    this.finish(camera, target, w, h, finish);
    this.copyDepth(target, out, w, h);
    this.drawScene(camera, w, h, out.fb, null, "glass");
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.programs.copy.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, out.color);
    gl.uniform1i(this.programs.copy.uniforms.uColor, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);
  }

  // a framebuffer with color and depth textures, kept while its size holds
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

  // the finishing pass, from a target into the bound framebuffer; ``px``: pixels per A
  // at the scene's center (a picture's tiles pass the whole picture's)
  finish(camera, t, w, h, { outline = 1, ao = 0.6, back = null, px = null, radius = null } = {}) {
    const gl = this.gl, { p, uniforms: u } = this.programs.post;
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t.color); gl.uniform1i(u.uColor, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, t.depth); gl.uniform1i(u.uDepth, 1);
    gl.uniform2fv(u.uClip, camera.clip());
    gl.uniform1f(u.uOrtho, camera.ortho ? 1 : 0);
    px = px ?? (camera.ortho ? h / (2 * camera.halfHeight()) : h / 2 / Math.tan(camera.fov / 2) / camera.distance);
    gl.uniform1f(u.uPx, px);
    gl.uniform1f(u.uOutline, outline);
    gl.uniform1f(u.uAO, ao);
    gl.uniform1f(u.uRadius, radius ?? Math.min(5 * px, 0.08 * Math.min(w, h)));
    gl.uniform4fv(u.uBack, back || [...this.background, 1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
    gl.enable(gl.DEPTH_TEST);
  }

  // a picture of w x h pixels: drawn `quality` times larger in tiles (so any size fits
  // the GPU), finished with outlines and occlusion, and averaged down -- the edges come
  // out smooth.  Returns a canvas.
  picture(camera, w, h, { outline = true, ao = true, background = null, transparent = false, fog = true,
                          quality = 3 } = {}) {
    const gl = this.gl;
    const ss = Math.max(1, Math.min(4, Math.round(quality)));
    const vp = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
    const maxTex = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), vp[0], vp[1], 4096);
    const W = w * ss, H = h * ss;  // the whole drawing, never held at once
    const proj = camera.projection(W / H);
    const px = camera.ortho ? H / (2 * camera.halfHeight()) : H / 2 / Math.tan(camera.fov / 2) / camera.distance;
    const radius = ao ? Math.min(5 * px, 0.15 * maxTex) : 0;
    const line = outline ? Math.max(1, Math.round(ss * Math.max(1, Math.min(w, h) / 1100))) : 0;
    const pad = Math.ceil(radius + line + 2);
    const tile = Math.max(64, Math.floor((maxTex - 2 * pad) / ss));  // output pixels per tile side
    const keep = { bg: this.background, fog: this.fog.on };
    if (background) this.background = background;
    this.fog.on = fog && !transparent;
    const out = new ImageData(w, h), n = ss * ss;
    try {
      for (let y0 = 0; y0 < h; y0 += tile) {
        for (let x0 = 0; x0 < w; x0 += tile) {
          const tw = Math.min(tile, w - x0), th = Math.min(tile, h - y0);
          // the tile's part of the drawing, padded (y counts down from the top)
          const X0 = x0 * ss - pad, Y0 = y0 * ss - pad, TW = tw * ss + 2 * pad, TH = th * ss + 2 * pad;
          const xl = 2 * X0 / W - 1, xr = 2 * (X0 + TW) / W - 1;
          const yt = 1 - 2 * Y0 / H, yb = 1 - 2 * (Y0 + TH) / H;
          const sx = 2 / (xr - xl), sy = 2 / (yt - yb), tx = -(xr + xl) / (xr - xl), ty = -(yt + yb) / (yt - yb);
          const P = new Float32Array(16);  // [sx 0 0 tx; 0 sy 0 ty; 0 0 1 0; 0 0 0 1] * proj
          for (let c = 0; c < 4; c++) {
            const p0 = proj[4 * c], p1 = proj[4 * c + 1], p2 = proj[4 * c + 2], p3 = proj[4 * c + 3];
            P[4 * c] = sx * p0 + tx * p3; P[4 * c + 1] = sy * p1 + ty * p3; P[4 * c + 2] = p2; P[4 * c + 3] = p3;
          }
          const scene = this.target(TW, TH, "picture-scene");
          this.drawScene(camera, TW, TH, scene.fb, P, "opaque");
          const fin = this.target(TW, TH, "picture-out");
          gl.bindFramebuffer(gl.FRAMEBUFFER, fin.fb);
          this.finish(camera, scene, TW, TH, { outline: line, ao: ao ? 0.6 : 0, px, radius,
                                                back: transparent ? [0, 0, 0, 0] : null });
          if (this.glass) {  // see-through layers over the finished tile
            this.copyDepth(scene, fin, TW, TH);
            this.drawScene(camera, TW, TH, fin.fb, P, "glass");
          }
          const pixels = new Uint8Array(TW * TH * 4);
          gl.readPixels(0, 0, TW, TH, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          // average each ss x ss block: the tile holds color times coverage (a
          // transparent background is 0, a see-through layer over it a part), so the
          // sums over the block divided by its coverage are the block's color
          for (let oy = 0; oy < th; oy++) {
            for (let ox = 0; ox < tw; ox++) {
              let r = 0, g = 0, b = 0, a = 0;
              for (let dy = 0; dy < ss; dy++) {
                const row = TH - 1 - (pad + oy * ss + dy);  // GL rows count up from the bottom
                let k = 4 * (row * TW + pad + ox * ss);
                for (let dx = 0; dx < ss; dx++, k += 4) {
                  r += pixels[k] * 255; g += pixels[k + 1] * 255; b += pixels[k + 2] * 255; a += pixels[k + 3];
                }
              }
              const o = 4 * ((y0 + oy) * w + x0 + ox);
              if (a > 0) { out.data[o] = r / a; out.data[o + 1] = g / a; out.data[o + 2] = b / a; }
              out.data[o + 3] = a / n;
            }
          }
        }
      }
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      for (const k of ["picture-scene", "picture-out"]) {  // big: let them go
        const t = this.targets?.[k];
        if (!t) continue;
        gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.color); gl.deleteTexture(t.depth);
        delete this.targets[k];
      }
      this.background = keep.bg; this.fog.on = keep.fog;
    }
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    canvas.getContext("2d").putImageData(out, 0, 0);
    return canvas;
  }

  // part: "all"; "opaque" (cleared first); or "glass", the see-through layers over
  // what the target holds
  drawScene(camera, width, height, fb = null, projection = null, part = "all") {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, width, height);
    if (part !== "glass") {
      gl.clearColor(...this.background, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
    gl.enable(gl.DEPTH_TEST);
    const view = camera.view(), proj = projection || camera.projection(width / height);
    // fog: from `start` (a fraction of the scene's depth in front of its center) to its back
    const f = this.fog, depth = camera.radius * 1.5;
    const fog = f.on
      ? [camera.distance - depth * (1 - 2 * f.start), camera.distance + depth]
      : [1e9, 1e9 + 1];
    const use = ({ p, uniforms }, alpha, shine = 1) => {
      gl.useProgram(p);
      gl.uniformMatrix4fv(uniforms.uView, false, view);
      gl.uniformMatrix4fv(uniforms.uProj, false, proj);
      gl.uniform3fv(uniforms.uBg, this.background);
      gl.uniform2fv(uniforms.uFog, fog);
      gl.uniform1f(uniforms.uFogStrength, f.strength);
      gl.uniform1f(uniforms.uOrtho, camera.ortho ? 1 : 0);
      gl.uniform1f(uniforms.uAlpha, alpha);
      gl.uniform1f(uniforms.uShine, shine);
    };
    const layer = ({ layer: L, counts: n, shine }, alpha) => {
      if (n.spheres) {
        use(this.programs.sphere, alpha, shine);
        gl.bindVertexArray(L.sphereVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.spheres);
      }
      if (n.cylinders) {
        use(this.programs.cylinder, alpha, shine);
        const cyl = L.cylinders[this.detail] || L.cylinders[1];
        gl.bindVertexArray(cyl.vao);
        gl.drawElementsInstanced(gl.TRIANGLES, cyl.count, gl.UNSIGNED_SHORT, 0, n.cylinders);
      }
      if (n.mesh) {
        use(this.programs.mesh, alpha, shine);
        gl.bindVertexArray(L.meshVao);
        gl.drawElements(gl.TRIANGLES, n.mesh, gl.UNSIGNED_INT, 0);
      }
      if (n.lines) {
        use(this.programs.line, alpha, shine);
        gl.bindVertexArray(L.lineVao);
        gl.drawArrays(gl.LINES, 0, n.lines);
      }
    };
    if (part !== "glass") for (const d of this.drawn) if (d.alpha >= 1) layer(d, 1);
    if (part !== "opaque") {
      for (const d of this.drawn) {
        if (d.alpha >= 1) continue;
        // its depth alone, then its color only where it is that depth: the front of it
        gl.colorMask(false, false, false, false);
        layer(d, d.alpha);
        gl.colorMask(true, true, true, true);
        gl.depthFunc(gl.LEQUAL);
        gl.depthMask(false);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        layer(d, d.alpha);
        gl.disable(gl.BLEND);
        gl.depthMask(true);
        gl.depthFunc(gl.LESS);
      }
    }
    gl.bindVertexArray(null);
  }
}
