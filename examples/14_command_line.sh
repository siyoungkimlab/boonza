#!/usr/bin/env bash
# The boonza command line on the example data.
#
#     bash examples/14_command_line.sh
#
# After `pip install -e .` the command is simply `boonza`; this script calls
# it through the Python that runs it so it also works uninstalled.
set -e
cd "$(dirname "$0")"
boonza() { "${PYTHON:-python}" -m boonza.cli "$@"; }
mkdir -p output

echo "\$ boonza info data/1LYZ.pdb"
boonza info data/1LYZ.pdb

echo; echo "\$ boonza select data/1LYZ.pdb \"resname TRP and name CA\""
boonza select data/1LYZ.pdb "resname TRP and name CA"

echo; echo "\$ boonza convert data/1LYZ.pdb output/lysozyme_protein.cif -s protein"
boonza convert data/1LYZ.pdb output/lysozyme_protein.cif -s protein

echo; echo "\$ boonza dssp data/1LYZ.pdb --simplified"
boonza dssp data/1LYZ.pdb --simplified

echo; echo "\$ boonza phipsi data/1LYZ.pdb | head -6"
boonza phipsi data/1LYZ.pdb | head -6

echo; echo "\$ boonza diff data/1LYZ.pdb output/lysozyme_protein.cif --no-positions | head -3"
boonza diff data/1LYZ.pdb output/lysozyme_protein.cif --no-positions | head -3 || true
