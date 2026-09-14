"""Dump mdtraj's DSSP assignment of a structure file for boonza's tests.

Runs under a Python that has mdtraj:

    mdtraj_oracle.py PATH              DSSP codes
    mdtraj_oracle.py dihedrals PATH    phi/psi/omega per residue (degrees; None if undefined)
    mdtraj_oracle.py hbonds BONDS.json PATH   baker_hubbard / wernet_nilsson triplets, with
                                              the topology's bonds replaced by BONDS.json
"""

import json
import sys
import warnings

import mdtraj as md
import numpy as np


def _dihedrals(t):
    nres = t.topology.n_residues
    out = {}
    # the residue each angle belongs to: N of phi, N of psi, the second N of omega
    for name, compute, owner in (("phi", md.compute_phi, 1), ("psi", md.compute_psi, 0),
                                 ("omega", md.compute_omega, 2)):  # fmt: skip
        idx, angles = compute(t)
        values = [None] * nres
        for quad, a in zip(idx, angles[0], strict=True):
            values[t.topology.atom(int(quad[owner])).residue.index] = float(np.degrees(a))
        out[name] = values
    return out


def _hbonds(bonds_path, path):
    t = md.load(path)
    top = t.topology
    top._bonds = []
    atoms = list(top.atoms)
    with open(bonds_path) as f:
        for i, j in json.load(f):
            top.add_bond(atoms[i], atoms[j])
    water = [a.index for a in atoms if a.residue.is_water]
    bh = md.baker_hubbard(t, freq=0.1, exclude_water=True, periodic=True)
    wn = md.wernet_nilsson(t, exclude_water=True, periodic=True)
    wn = [w.tolist() for w in wn]
    return {"water": water, "baker_hubbard": bh.tolist(), "wernet_nilsson": wn}


def _analysis(path):
    t = md.load(path)
    heavy, pairs_heavy = md.compute_contacts(t, scheme="closest-heavy")
    ca, pairs_ca = md.compute_contacts(t, scheme="ca")
    return {
        "natoms": t.n_atoms,
        "sasa": md.shrake_rupley(t, mode="atom")[0].tolist(),
        "residue_sasa": md.shrake_rupley(t, mode="residue")[0].tolist(),
        "contacts_heavy": heavy[0].tolist(),
        "pairs_heavy": pairs_heavy.tolist(),
        "contacts_ca": ca[0].tolist(),
        "pairs_ca": pairs_ca.tolist(),
        "rg": float(md.compute_rg(t)[0]),
    }


def main():
    warnings.simplefilter("ignore")
    if sys.argv[1] == "analysis":
        json.dump(_analysis(sys.argv[2]), sys.stdout)
        return
    if sys.argv[1] == "hbonds":
        json.dump(_hbonds(sys.argv[2], sys.argv[3]), sys.stdout)
        return
    mode = "dihedrals" if sys.argv[1] == "dihedrals" else "dssp"
    t = md.load(sys.argv[-1])
    residues = [[r.chain.chain_id, r.resSeq, r.name] for r in t.topology.residues]
    if mode == "dssp":
        codes = md.compute_dssp(t, simplified=False)
        json.dump({"residues": residues, "codes": codes.tolist()}, sys.stdout)
    else:
        json.dump({"residues": residues, **_dihedrals(t)}, sys.stdout)


if __name__ == "__main__":
    main()
