"""Canonical dump of a boonza System, matching tests/msys_oracle.py."""


def canon(s) -> dict:
    A, R, C, B = s.atoms, s.residues, s.chains, s.bonds
    cols = [A[c].tolist() for c in ("name", "anum", "mass", "charge", "formal_charge")]
    pos, vel, res = A["pos"].tolist(), A["vel"].tolist(), A["residue"].tolist()
    atoms = [[*(c[k] for c in cols), *pos[k], *vel[k], res[k]] for k in range(s.natoms)]
    tables = {}
    for name, t in s.tables.items():
        pp, tp = t.params.props, t.term_props
        pcols = {p: t.params[p].tolist() for p in pp}
        tcols = {p: t.values(p).tolist() for p in tp}
        tatoms, pids = t.atoms.tolist(), t.param_ids.tolist()
        terms = [
            [
                tatoms[k],
                None if pids[k] < 0 else {p: pcols[p][pids[k]] for p in pp},
                {p: tcols[p][k] for p in tp},
            ]
            for k in range(len(t))
        ]
        overrides = [[p1, p2, vals] for (p1, p2), vals in t.overrides.items()]
        tables[name] = {
            "category": t.category,
            "natoms": t.natoms,
            "terms": terms,
            "overrides": overrides,
        }
    nb = s.nonbonded_info
    return {
        "atoms": atoms,
        "atom_props": {p: A[p].tolist() for p in A.props},
        "residues": [
            list(r)
            for r in zip(
                R["chain"].tolist(),
                R["resid"].tolist(),
                R["name"].tolist(),
                R["insertion"].tolist(),
                strict=True,
            )
        ],  # fmt: skip
        "chains": [
            list(c)
            for c in zip(C["ct"].tolist(), C["name"].tolist(), C["segid"].tolist(), strict=True)
        ],
        "cts": [[c.name, {k: c[k] for k in c.keys()}] for c in s.cts],
        "bonds": sorted(
            [i, j, o]
            for i, j, o in zip(B["i"].tolist(), B["j"].tolist(), B["order"].tolist(), strict=True)
        ),
        "fragids": s.fragids.tolist(),
        "tables": tables,
        "nbinfo": [nb.vdw_funct, nb.vdw_rule, nb.es_funct],
        "cell": s.cell.tolist(),
        "aux": {
            name: {"props": sorted(tab.props), "rows": [tab.row(i) for i in range(len(tab))]}
            for name, tab in s.aux_tables.items()
        },
    }


def assert_same_content(a, b, atom_map=None, tables=None, rtol=1e-9) -> None:
    """Physical content of ``a`` is present in ``b`` (atoms mapped by ``atom_map``).

    Compares positions, velocities, masses, charges, names, bonds and the
    terms of each force-field table as (atoms, parameter values) sets.
    """
    import numpy as np

    m = np.arange(a.natoms) if atom_map is None else np.asarray(atom_map)
    assert b.natoms == a.natoms
    for col in ("pos", "vel", "mass", "charge", "anum"):
        np.testing.assert_array_equal(b.atoms[col][m], a.atoms[col], err_msg=col)
    assert b.atoms["name"][m].tolist() == a.atoms["name"].tolist()
    ra, rb = a.atoms["residue"], b.atoms["residue"][m]
    assert b.residues["resid"][rb].tolist() == a.residues["resid"][ra].tolist()
    assert b.residues["name"][rb].tolist() == a.residues["name"][ra].tolist()

    def bonds(s, mp):
        i, j = mp[s.bonds["i"]], mp[s.bonds["j"]]
        return sorted(zip(np.minimum(i, j).tolist(), np.maximum(i, j).tolist(), strict=True))

    inv = np.empty_like(m)
    inv[m] = np.arange(len(m))
    assert bonds(b, inv) == bonds(a, np.arange(a.natoms))
    for name in tables if tables is not None else a.tables:
        ta, tb = a.tables[name], b.tables[name]
        props = [p for p in ta.params.props if p in tb.params.props and p != "type"]
        props += [p for p in ta.term_props if p in tb.term_props]

        def rows(t, mp, props=props):
            atoms = [tuple(r) for r in mp[t.atoms].tolist()]
            vals = (
                np.column_stack([t.values(p).astype(float) for p in props])
                if props
                else (np.zeros((len(t), 0)))
            )
            order = sorted(range(len(t)), key=lambda k: (atoms[k], vals[k].tolist()))
            return [atoms[k] for k in order], vals[order]

        atoms_a, vals_a = rows(ta, np.arange(a.natoms))
        atoms_b, vals_b = rows(tb, inv)
        assert atoms_b == atoms_a, f"{name}: term atoms differ"
        np.testing.assert_allclose(vals_b, vals_a, rtol=rtol, atol=1e-12, err_msg=name)


def _first_diff(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            ours, ref = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
            return f"{path}: keys differ: only ours {ours}, only ref {ref}"
        for k in a:
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for k, (x, y) in enumerate(zip(a, b, strict=True)):
            d = _first_diff(x, y, f"{path}[{k}]")
            if d:
                return d
        return None
    if (
        a != b
        or type(a) is not type(b)
        and not (isinstance(a, int | float) and isinstance(b, int | float))
    ):
        return f"{path}: ours {a!r} != ref {b!r}"
    return None


def assert_same(ours: dict, ref: dict) -> None:
    diff = _first_diff(ours, ref, "system")
    assert diff is None, diff
