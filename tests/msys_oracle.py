"""Dump msys results in the canonical form boonza's tests compare against.

Runs under the Python that can import msys (3.10), never under boonza:

    msys_oracle.py load PATH
    msys_oracle.py clone PATH IDS_JSON
    msys_oracle.py append PATH PATH
    msys_oracle.py time PATH OUT REPEAT
"""

import json
import sys
import time

import msys


def _prop(obj, name):
    v = getattr(obj, name)
    return v() if callable(v) else v


def canon(m):
    atoms = list(m.atoms)
    aidx = {a.id: k for k, a in enumerate(atoms)}
    residues = list(m.residues)
    ridx = {r.id: k for k, r in enumerate(residues)}
    chains = list(m.chains)
    cidx = {c.id: k for k, c in enumerate(chains)}
    cts = list(m.cts)
    ctidx = {c.id: k for k, c in enumerate(cts)}
    atom_props = list(_prop(m, "atom_props"))

    out = {}
    out["atoms"] = [
        [a.name, a.atomic_number, a.mass, a.charge, a.formal_charge]
        + [float(x) for x in a.pos]
        + [float(x) for x in a.vel]
        + [ridx[a.residue.id]]
        for a in atoms
    ]
    out["atom_props"] = {p: [a[p] for a in atoms] for p in atom_props}
    out["residues"] = [[cidx[r.chain.id], r.resid, r.name, r.insertion] for r in residues]
    out["chains"] = [[ctidx[c.ct.id], c.name, c.segid] for c in chains]
    out["cts"] = [[c.name, {k: c[k] for k in c.keys()}] for c in cts]
    out["bonds"] = sorted(
        sorted([aidx[b.first.id], aidx[b.second.id]]) + [b.order] for b in m.bonds
    )
    out["fragids"] = [a.fragid for a in atoms]

    tables = {}
    for t in _prop(m, "tables"):
        pprops = list(t.params.props)
        tprops = list(_prop(t, "term_props"))
        terms = []
        for term in t.terms:
            p = term.param
            terms.append(
                [
                    [aidx[a.id] for a in term.atoms],
                    None if p is None else {k: p[k] for k in pprops},
                    {k: term[k] for k in tprops},
                ]
            )
        oprops = list(t.override_params.props)
        ov = [
            [min(pi.id, pj.id), max(pi.id, pj.id), {k: op[k] for k in oprops}]
            for (pi, pj), op in t.overrides().items()
        ]
        tables[t.name] = {
            "category": str(t.category),
            "natoms": t.natoms,
            "terms": terms,
            "overrides": sorted(ov, key=lambda x: (x[0], x[1])),
        }
    out["tables"] = tables
    nb = m.nonbonded_info
    out["nbinfo"] = [nb.vdw_funct, nb.vdw_rule, nb.es_funct]
    out["cell"] = [[float(x) for x in row] for row in m.cell]
    aux = {}
    for name in _prop(m, "auxtable_names"):
        tab = m.auxtable(name)
        props = list(tab.props)
        aux[name] = {
            "props": sorted(props),
            "rows": [{k: tab.param(i)[k] for k in props} for i in range(tab.nparams)],
        }
    out["aux"] = aux
    return out


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "load":
        m = msys.Load(args[0])
        m.updateFragids()  # multi-model PDB/SDF loads keep per-model fragids
        result = canon(m)
    elif mode == "clone":
        m = msys.Load(args[0])
        with open(args[1]) as f:
            ids = json.load(f)
        result = canon(m.clone(ids))
    elif mode == "append":
        m = msys.Load(args[0])
        m.append(msys.Load(args[1]))
        m.updateFragids()  # msys append copies the appended atoms' stale fragids
        result = canon(m)
    elif mode == "select":
        m = msys.Load(args[0])
        with open(args[1]) as f:
            selections = json.load(f)
        result = []
        for sel in selections:
            try:
                result.append(m.selectIds(sel))
            except Exception as e:
                result.append("ERROR: " + str(e).strip().splitlines()[0])
    elif mode == "wrap":
        import numpy as np
        from msys.wrap import Wrapper

        m = msys.Load(args[0])
        m.setPositions(np.load(args[1]))
        Wrapper(m, center=args[2] or None, glue=json.loads(args[3])).wrap()
        result = m.getPositions().tolist()
    elif mode == "fragtime":
        m = msys.Load(args[0])
        t0 = time.perf_counter()
        frags = m.updateFragids()
        t1 = time.perf_counter()
        groups = msys.FindDistinctFragments(m)
        t2 = time.perf_counter()
        result = {"natoms": m.natoms, "nfragments": len(frags), "ngroups": len(groups),
                  "fragids": t1 - t0, "distinct": t2 - t1}  # fmt: skip
    elif mode == "distinct":
        m = msys.Load(args[0])
        result = [[k, list(v)] for k, v in sorted(msys.FindDistinctFragments(m).items())]
    elif mode == "sssr":
        m = msys.Load(args[0])
        result = [[a.id for a in ring] for ring in msys.GetSSSR(m.atoms, args[1] == "all")]
    elif mode == "knots":
        from msys import knot

        m = msys.Load(args[0])
        found = knot.FindKnots(m, max_cycle_size=int(args[1]) or None, selection=args[2],
                               ignore_excluded_knots=args[3] == "1")  # fmt: skip
        result = [[list(cycle), list(bond), idx] for cycle, bond, idx in found]
    elif mode == "prmtop":
        m = msys.LoadPrmTop(args[0])
        msys.ReadCrdCoordinates(m, args[1])
        msys.SaveDMS(m, args[2])
        result = {"natoms": m.natoms}
    elif mode == "time":
        path, out, repeat = args[0], args[1], int(args[2])
        m = msys.Load(path)
        result = {
            "natoms": m.natoms,
            "load": best(lambda: msys.Load(path), repeat),
            "save": best(lambda: msys.SaveDMS(m, out), repeat),
        }
    else:
        raise SystemExit(f"unknown mode {mode}")
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
