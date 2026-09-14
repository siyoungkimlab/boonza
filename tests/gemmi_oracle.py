"""Dump gemmi's reading of an mmCIF file for boonza's tests.

Runs under a Python that has gemmi (not necessarily boonza's):

    gemmi_oracle.py PATH
"""

import json
import sys

import gemmi


def main():
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
