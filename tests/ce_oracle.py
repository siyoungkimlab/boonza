"""Run Biopython's CEAligner on two C-alpha coordinate arrays for boonza's tests.

Runs under a Python that has Biopython:

    ce_oracle.py REF.npy MOBILE.npy
"""

import json
import sys

import numpy as np
from Bio.PDB.cealign import CEAligner


def main():
    ref, mob = np.load(sys.argv[1]), np.load(sys.argv[2])
    aligner = CEAligner()
    # feed coordinates directly instead of parsed structures
    aligner.get_guide_coord_from_structure = lambda coords: coords.tolist()
    aligner.set_reference(ref)
    aligner.align(mob, transform=False)
    rot, tran = aligner._rigid_motion
    json.dump({"rms": aligner.rms, "rot": np.asarray(rot).tolist(),
               "tran": np.asarray(tran).tolist()}, sys.stdout)  # fmt: skip


if __name__ == "__main__":
    main()
