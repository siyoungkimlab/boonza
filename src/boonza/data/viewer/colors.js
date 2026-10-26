// Color schemes: per-atom RGB (0..1) for a representation.
//   element, element/<name>  CPK; carbons gray, or <name> (element/chain: by chain)
//                            (coarse-grained: backbone beads gray, the rest by restype)
//   chain                    one color per chain
//   restype                  amino acids: hydrophobic, polar, positive, negative;
//                            bases A, T/U, G, C; anything else by name
//   resname                  one color per residue name (lipids, ligands, ions)
//   ss                       helix, strand, coil
//   spectrum                 blue (N-terminus) to red (C-terminus) along each chain
//   bfactor                  blue-white-red over the selection
//   <name> or #rrggbb        one color

export const NAMED = {
  white: [1, 1, 1], gray: [0.6, 0.6, 0.6], black: [0.1, 0.1, 0.1],
  red: [0.9, 0.2, 0.2], orange: [1, 0.6, 0.1], yellow: [1, 0.9, 0.2],
  green: [0.35, 0.8, 0.3], cyan: [0.2, 0.8, 0.9], blue: [0.25, 0.45, 1],
  purple: [0.65, 0.4, 0.9], magenta: [0.95, 0.3, 0.8], pink: [1, 0.6, 0.75],
  salmon: [1, 0.55, 0.45], wheat: [0.96, 0.87, 0.7], slate: [0.5, 0.5, 1],
  lime: [0.6, 1, 0.2], teal: [0.2, 0.6, 0.6],
};

export const SCHEMES = ["element", "chain", "restype", "resname", "ss", "spectrum", "bfactor"];

// residue type, by three-letter name; SIRAH writes "s" and the one-letter code
const TYPES = {
  hydrophobic: ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"],
  polar: ["SER", "THR", "ASN", "GLN", "TYR", "CYS", "CYX", "HIS", "HID", "HIE", "HSD", "HSE"],
  positive: ["LYS", "ARG", "HIP", "HSP"],
  negative: ["ASP", "GLU"],
};
const ONE = { A: "ALA", V: "VAL", L: "LEU", I: "ILE", M: "MET", F: "PHE", W: "TRP", P: "PRO",
              G: "GLY", S: "SER", T: "THR", N: "ASN", Q: "GLN", Y: "TYR", C: "CYS", X: "CYS",
              H: "HIS", K: "LYS", R: "ARG", D: "ASP", E: "GLU" };
const TYPE_COLOR = { hydrophobic: [0.9, 0.9, 0.9], polar: [0.35, 0.8, 0.3],
                     positive: [0.25, 0.45, 1], negative: [0.95, 0.25, 0.25] };
const RESTYPE = {};
for (const [type, names] of Object.entries(TYPES)) for (const n of names) RESTYPE[n] = TYPE_COLOR[type];
const CG_BACKBONE = new Set(["BB", "GN", "GC", "GO", "PX"]);

// nucleotides: A, DA, DA5, RA3, and SIRAH's DAX, AX5, GX3
const BASE = /^[DR]?([ATGCU])X?[35]?$/;
const BASE_COLOR = { A: [0.95, 0.35, 0.35], T: [0.35, 0.55, 1], U: [0.35, 0.55, 1],
                     G: [0.35, 0.8, 0.3], C: [1, 0.85, 0.2] };

function restype(name) {
  if (name.length === 2 && name[0] === "s" && ONE[name[1]]) name = ONE[name[1]];
  const base = BASE.exec(name);
  return RESTYPE[name] || (base ? BASE_COLOR[base[1]] : null);
}

function byName(name) {  // a stable color per name
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return CHAINS[h % CHAINS.length];
}

const ELEMENT = {
  H: [0.9, 0.9, 0.9], C: [0.6, 0.6, 0.6], N: [0.2, 0.35, 1], O: [1, 0.15, 0.15],
  S: [1, 0.85, 0.2], P: [1, 0.55, 0.1], F: [0.55, 0.9, 0.3], Cl: [0.2, 0.9, 0.2],
  Br: [0.65, 0.2, 0.15], I: [0.55, 0.1, 0.6], Na: [0.67, 0.36, 0.95],
  K: [0.56, 0.25, 0.83], Mg: [0.54, 1, 0], Ca: [0.24, 1, 0], Zn: [0.49, 0.5, 0.69],
  Fe: [0.88, 0.4, 0.2], Cu: [0.78, 0.5, 0.2], Mn: [0.61, 0.48, 0.78], Se: [1, 0.63, 0],
};
const OTHER = [1, 0.4, 0.7];

const CHAINS = [
  [0.35, 0.8, 0.3], [0.2, 0.75, 0.9], [0.95, 0.35, 0.8], [1, 0.85, 0.2],
  [0.95, 0.55, 0.45], [0.6, 0.6, 1], [1, 0.6, 0.1], [0.4, 0.85, 0.7],
  [0.85, 0.85, 0.85], [0.7, 0.45, 0.95],
];
// ribbon colors by secondary structure: helix red, strand yellow, coil light
const SS = { H: [0.9, 0.12, 0.12], E: [1, 0.85, 0.1], C: [0.82, 0.82, 0.85], "": [0.82, 0.82, 0.85] };

function hex(text) {
  const m = /^#?([0-9a-f]{6})$/i.exec(text);
  if (!m) return null;
  const v = parseInt(m[1], 16);
  return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

export function named(text) {
  return NAMED[text] || hex(text);
}

export function isColor(text) {
  const [scheme, carbon] = text.split("/");
  if (scheme === "element") return carbon === undefined || carbon === "chain" || named(carbon) !== null;
  return SCHEMES.includes(text) || named(text) !== null;
}

function rainbow(t) {  // blue -> cyan -> green -> yellow -> red
  const stops = [[0.2, 0.3, 1], [0.2, 0.85, 1], [0.3, 0.9, 0.3], [1, 0.9, 0.2], [1, 0.25, 0.2]];
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const k = Math.min(Math.floor(x), stops.length - 2), f = x - k;
  return stops[k].map((a, d) => a + (stops[k + 1][d] - a) * f);
}

// Fill out[3*a..] for the atoms in ids; topo is the page's topology with helpers.
export function colorAtoms(spec, ids, topo, ss, out) {
  const [scheme, carbon] = spec.split("/");
  const A = topo.atoms, R = topo.residues;
  if (scheme === "element" && topo.cg) {
    for (const a of ids) {
      const name = R.name[A.residue[a]];
      put(out, a, CG_BACKBONE.has(A.name[a]) ? ELEMENT.C : restype(name) || byName(name));
    }
  } else if (scheme === "element") {
    const c = carbon && carbon !== "chain" ? named(carbon) : null;
    for (const a of ids) {
      const e = A.element[a];
      if (e === "C" && carbon === "chain") put(out, a, CHAINS[R.chain[A.residue[a]] % CHAINS.length]);
      else put(out, a, e === "C" && c ? c : ELEMENT[e] || OTHER);
    }
  } else if (scheme === "chain") {
    for (const a of ids) put(out, a, CHAINS[R.chain[A.residue[a]] % CHAINS.length]);
  } else if (scheme === "restype") {
    for (const a of ids) put(out, a, restype(R.name[A.residue[a]]) || byName(R.name[A.residue[a]]));
  } else if (scheme === "resname") {
    for (const a of ids) put(out, a, byName(R.name[A.residue[a]]));
  } else if (scheme === "ss") {
    for (const a of ids) put(out, a, SS[ss[A.residue[a]]] || SS.C);
  } else if (scheme === "spectrum") {
    for (const a of ids) put(out, a, rainbow(topo.spectrum[A.residue[a]]));
  } else if (scheme === "bfactor") {
    let lo = Infinity, hi = -Infinity;
    for (const a of ids) { lo = Math.min(lo, A.bfactor[a]); hi = Math.max(hi, A.bfactor[a]); }
    const span = hi > lo ? hi - lo : 1;
    for (const a of ids) {
      const t = (A.bfactor[a] - lo) / span;
      put(out, a, t < 0.5 ? [2 * t, 2 * t, 1] : [1, 2 - 2 * t, 2 - 2 * t]);
    }
  } else {
    const c = named(scheme) || NAMED.white;
    for (const a of ids) put(out, a, c);
  }
}

function put(out, a, c) {
  out[3 * a] = c[0]; out[3 * a + 1] = c[1]; out[3 * a + 2] = c[2];
}

// per residue colors for ribbons: ss, chain, spectrum, restype, or one color
export const RIBBON_SCHEMES = ["ss", "chain", "spectrum", "restype"];
export function colorResidues(spec, residues, topo, ss, out) {
  const R = topo.residues;
  const one = named(spec);
  for (const r of residues) {
    let c;
    if (spec === "ss") c = SS[ss[r]] || SS.C;
    else if (spec === "chain") c = CHAINS[R.chain[r] % CHAINS.length];
    else if (spec === "spectrum") c = rainbow(topo.spectrum[r]);
    else if (spec === "restype") c = restype(R.name[r]) || NAMED.gray;
    else c = one || NAMED.white;
    out[3 * r] = c[0]; out[3 * r + 1] = c[1]; out[3 * r + 2] = c[2];
  }
}
