"""Run ChimeraX matchmaker for boonza's tests (headless ChimeraX script).

    ChimeraX --nogui --exit --script "chimerax_oracle.py REF CHAIN MOBILE CHAIN OUT.json"

SEQRES records are removed first so chain sequences hold only residues with
coordinates, as boonza's do.  Writes the secondary structure ChimeraX
computed for each chain (H helix, S strand, O other), the gapped alignment,
the paired and pruned residues, RMSDs and the transform.
"""

import json
import sys
import tempfile

from chimerax.core.commands import run


def _strip_seqres(path):
    lines = [ln for ln in open(path) if not ln.startswith("SEQRES")]
    f = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    f.writelines(lines)
    f.close()
    return f.name


def _chain(model, chain_id):
    for c in model.chains:
        if c.chain_id == chain_id:
            return c
    raise SystemExit(f"no chain {chain_id} in {model}")


def _ss(r):
    return "H" if r.is_helix else "S" if r.is_strand else "O"


def main():
    ref_path, ref_chain, mob_path, mob_chain, out = sys.argv[1:6]
    ref = run(session, f"open {_strip_seqres(ref_path)}")[0]  # noqa: F821 (ChimeraX global)
    mob = run(session, f"open {_strip_seqres(mob_path)}")[0]  # noqa: F821
    run(session, "dssp #1,2")  # noqa: F821  (what matchmaker computes internally)
    rc, mc = _chain(ref, ref_chain), _chain(mob, mob_chain)
    chains = {}
    for key, c in (("ref", rc), ("mobile", mc)):
        chains[key] = {
            "characters": c.characters,
            "residues": [[r.number, r.insertion_code.strip(), r.name, _ss(r)] for r in c.residues],
        }
    result = run(session, f"matchmaker #2/{mob_chain} to #1/{ref_chain}")[0]  # noqa: F821

    def resnums(atoms):
        return [[a.residue.number, a.residue.insertion_code.strip()] for a in atoms]

    json.dump({
        "chains": chains,
        "aligned_ref": result["aligned ref seq"].characters,
        "aligned_mobile": result["aligned match seq"].characters,
        "full_ref": resnums(result["full ref atoms"]),
        "full_mobile": resnums(result["full match atoms"]),
        "final_ref": resnums(result["final ref atoms"]),
        "final_mobile": resnums(result["final match atoms"]),
        "full_rmsd": result["full RMSD"],
        "final_rmsd": result["final RMSD"],
        "matrix": result["transformation matrix"].matrix.tolist(),
    }, open(out, "w"))  # fmt: skip


main()
