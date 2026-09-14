"""Dump gemmi's reading of an mmCIF file for boonza's tests.

Runs under a Python that has gemmi (not necessarily boonza's):

    gemmi_oracle.py PATH
    gemmi_oracle.py connections PATH      (LINK/SSBOND or _struct_conn bonds)
"""

import json
import sys

import gemmi


def connections(path):
    """Bonds from LINK/SSBOND (PDB) or _struct_conn (mmCIF), without hydrogen bonds and
    bonds to symmetry copies: [[chain, seq num, icode, atom name, altloc] x 2, type]."""
    st = gemmi.read_structure(path)
    out = []
    for con in st.connections:
        if con.type == gemmi.ConnectionType.Hydrog or con.asu == gemmi.Asu.Different:
            continue
        partners = [
            [
                p.chain_name,
                p.res_id.seqid.num,
                p.res_id.seqid.icode.strip(),
                p.atom_name,
                p.altloc.strip("\x00"),
            ]
            for p in (con.partner1, con.partner2)
        ]
        out.append(partners + [con.type.name])
    json.dump(out, sys.stdout)


def main():
    if sys.argv[1] == "connections":
        connections(sys.argv[2])
        return
    st = gemmi.read_structure(sys.argv[1], format=gemmi.CoorFormat.Mmcif)
    atoms = []
    for mi, model in enumerate(st):
        for chain in model:
            for res in chain:
                for a in res:
                    atoms.append([
                        mi, a.serial, chain.name, res.subchain, res.name, res.seqid.num,
                        res.seqid.icode.strip(), a.name, a.element.atomic_number,
                        a.altloc.strip("\x00"), a.pos.x, a.pos.y, a.pos.z, a.occ, a.b_iso,
                        a.charge,
                    ])  # fmt: skip
    atoms.sort(key=lambda r: (r[0], r[1]))  # file order
    json.dump(
        {"atoms": atoms, "cell": list(st.cell.parameters), "spacegroup": st.spacegroup_hm},
        sys.stdout,
    )


if __name__ == "__main__":
    main()
