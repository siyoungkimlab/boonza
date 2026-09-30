#!/bin/bash
# Prepare structures with Schrodinger's prepwizard (Protein Preparation Wizard).  See -h.
# ./prepwizard.sh -j 8 -o mae boltz_results_input/predictions/*/*.mae

set -uo pipefail

# PREPWIZARD can come from the environment -- a cluster wrapper (bsub/sbatch
# script) can set it after loading the module:
#   ml use /path/to/modulefiles/
#   ml schrodinger/2026-3
# or point it at the binary:
#   PREPWIZARD=$SCHRODINGER/utilities/prepwizard
PREPWIZARD=${PREPWIZARD:-prepwizard}
JOBS=4
FORCE=0
OUTEXT=pdb
LOGDIR=${PREP_LOGDIR:-logs_prep}
CAPTERMINI=1
KEEPLIG=0

usage () {
    cat <<USAGE
usage: $(basename "$0") [-j N] [-o EXT] [-c] [-k] [-f] [-h] input [input ...]

Free N-/C-termini are capped with ACE/NMA by default (-c, always on).

Runs Schrodinger prepwizard over structure files, writing foo.pdb (or
foo.mae, foo.cif) -> foo.prepped.EXT beside each input (EXT is pdb by
default).

Inputs are .pdb, .mae or .cif files (Boltz --output_format mae writes .mae);
at least one is required.  Other files a glob picks up -- .json, .npz -- are
ignored.  Let the shell do the matching -- dir/*.mae, or a deeper glob --
rather than naming a directory.

  -j N   run N prepwizard jobs in parallel (default $JOBS)
  -o EXT output format, one of pdb mae cif (default $OUTEXT); prepwizard
         deduces the format from the extension it is handed
  -c     cap free N-/C-termini with ACE/NMA (prepwizard's -captermini).
         This is on by default -- unlike MOE's -ace/-nme, prepwizard has
         no per-chain selection, so every eligible free terminus in the
         structure is capped; there is currently no flag to leave them
         charged (NH3+ / COO-) instead.
  -k     keep ligand protonation and tautomer states as given in the
         input (-noepik, skips running Epik on het groups); by default
         Epik re-assigns them.  Only meaningful for inputs that carry
         charges (.mae), not PDB
  -f     re-prep inputs that already have a non-empty output; without it
         those are skipped, so an interrupted run just resumes.  Outputs do
         not record which options they were made with -- after changing -k,
         use -f
  -h     show this help and exit

examples:
  $(basename "$0") one.pdb two.pdb
  $(basename "$0") -j 8 boltz_results_config_A_1/predictions/*/*.pdb
  $(basename "$0") -j 8 */*/*/*.pdb          # everything below here
  $(basename "$0") -j 8 -o mae */*/*/*.pdb   # ... as Maestro files
  $(basename "$0") -j 8 -o mae boltz_results_*/predictions/*/*.mae

Files already named *.prepped.* are never used as inputs, whatever their
extension: preparation is not idempotent, and with -o mae the outputs sit
beside .mae inputs where the same glob would otherwise pick them up.

Every run also passes -fillloops -fillsidechains -addOXT -NOJOBID to
prepwizard (fill missing loops with Prime, rebuild incomplete side chains,
add terminal oxygens, and run in the foreground without job control).

Per-structure prepwizard output goes to $LOGDIR/<flattened path>.log, the
ok/skip/FAILED tally to $LOGDIR/summary.txt.  prepwizard also leaves a
<basename>-NNN job directory beside each input, holding per-stage
intermediates (Epik, H-bond optimization, refinement); this script removes
it only when it's empty (e.g. under -k with a short pipeline), otherwise it
is left for inspection.

  $PREPWIZARD
USAGE
}

while getopts "j:o:ckfh" opt; do
    case $opt in
        j) JOBS=$OPTARG ;;
        o) OUTEXT=$OPTARG ;;
        c) CAPTERMINI=1 ;;
        k) KEEPLIG=1 ;;
        f) FORCE=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[ "$#" -gt 0 ] || { echo "$(basename "$0"): no input given" >&2; usage >&2; exit 2; }

case $OUTEXT in
    pdb|mae|cif) ;;
    *) echo "unsupported output format: $OUTEXT (use pdb, mae or cif)" >&2; exit 2 ;;
esac

[ -x "$PREPWIZARD" ] || command -v "$PREPWIZARD" >/dev/null 2>&1 || {
    echo "no prepwizard at $PREPWIZARD (load the schrodinger module, or set \$PREPWIZARD to its utilities/prepwizard)" >&2
    exit 1
}

mkdir -p "$LOGDIR"

prep_one () {
    in=$1
    base=${in%.*}
    out=$base.prepped.$OUTEXT
    log=$LOGDIR/$(echo "$base" | tr / _).log

    if [ "$FORCE" -eq 0 ] && [ -s "$out" ]; then
        echo "skip   $in (output exists)"
        return 0
    fi

    prep_args=(-NOJOBID -fillloops -fillsidechains -addOXT)
    [ "$CAPTERMINI" -eq 1 ] && prep_args+=(-captermini)
    [ "$KEEPLIG" -eq 1 ] && prep_args+=(-noepik)

    # Run from the input's own directory so prepwizard's job-scratch
    # directory (named after the input's basename) lands beside it instead
    # of the invocation cwd, where same-named inputs from different
    # prediction folders could collide.
    indir=$(dirname "$in")
    inbase=$(basename "$in")
    outbase=$(basename "$out")

    if ( cd "$indir" && "$PREPWIZARD" "${prep_args[@]}" "$inbase" "$outbase" ) >"$log" 2>&1 \
        && [ -s "$out" ]; then
        rmdir "$indir/${inbase%.*}"-[0-9][0-9][0-9] 2>/dev/null
        echo "ok     $in"
    else
        echo "FAILED $in  (see $log)"
        return 1
    fi
}
export -f prep_one
export PREPWIZARD FORCE LOGDIR OUTEXT CAPTERMINI KEEPLIG

LIST=$LOGDIR/inputs.txt
: > "$LIST"

for arg in "$@"; do
    if [ -f "$arg" ]; then
        echo "$arg" >> "$LIST"
    elif [ -d "$arg" ]; then
        echo "$arg is a directory; pass the files, e.g. $arg/*.pdb" >&2
        exit 2
    else
        echo "no such file: $arg" >&2
        exit 2
    fi
done

# Structure files only, never our own outputs.
grep -E '\.(pdb|mae|cif)$' "$LIST" | grep -v '\.prepped\.' | sort -u > "$LIST.tmp"
mv "$LIST.tmp" "$LIST"

NIN=$(wc -l < "$LIST" | tr -d ' ')
[ "$NIN" -gt 0 ] || { echo "no .pdb/.mae/.cif inputs found" >&2; exit 1; }
echo "$NIN structures to consider, $JOBS job(s) in parallel"
if [ "$CAPTERMINI" -eq 1 ]; then
    echo "caps: free termini capped with ACE/NMA (-captermini); other chain ends NH3+/COO-"
else
    echo "caps: none; all chain ends NH3+/COO-"
fi
if [ "$KEEPLIG" -eq 1 ]; then
    echo "ligands: input protonation/tautomer states kept (-noepik)"
else
    echo "ligands: protonation/tautomer states re-assigned by Epik"
fi

tr '\n' '\0' < "$LIST" |
    xargs -0 -P "$JOBS" -I{} bash -c 'prep_one "$@"' _ {} |
    tee "$LOGDIR/summary.txt"

echo
printf 'done: %s prepared, %s skipped, %s failed\n' \
    "$(grep -c '^ok'     "$LOGDIR/summary.txt")" \
    "$(grep -c '^skip'   "$LOGDIR/summary.txt")" \
    "$(grep -c '^FAILED' "$LOGDIR/summary.txt")"
