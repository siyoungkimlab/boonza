"""Biopython PairwiseAligner results for boonza's sequence alignment tests.

Runs under a Python that has Biopython:

    bio_align_oracle.py CASES.json
"""

import json
import sys

from Bio import Align
from Bio.Align import substitution_matrices


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)
    blosum = substitution_matrices.load("BLOSUM62")
    out = []
    for c in cases:
        al = Align.PairwiseAligner()
        al.substitution_matrix = blosum
        al.mode = c["mode"]
        al.open_gap_score = -c["open"]
        al.extend_gap_score = -c["extend"]
        if not c["end_gaps"]:
            al.end_gap_score = 0.0
        score = al.score(c["a"], c["b"])
        aln = al.align(c["a"], c["b"])[0]
        out.append({"score": float(score), "a": str(aln[0]), "b": str(aln[1])})
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
