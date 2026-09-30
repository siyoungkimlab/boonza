#!/bin/bash
# Prepare PDB structures with MOE StructurePreparation.  See -h.
# ./prep_moe.sh -j 8 -o mae -n B boltz_results_input/predictions/*/*.mae
# ./prep_moe.sh -j 8 -o mae boltz_*/structures/*.mae boltz_x/structures_prepped/
# ./prep_moe.sh -j 8 -o mae 8c70 6ron prepped/     # codes fetched from RCSB

set -uo pipefail

# The first three can come from the environment -- a cluster wrapper sets them
# to run this on a cluster.  The moebatch path is MOEBATCH rather than MOE
# because $MOE conventionally names MOE's install directory (a wrapper
# reads it that way); moebatch itself re-detects $MOE from its own location.
MOEBATCH=${MOEBATCH:-/Applications/moe2024.0604/bin/moebatch}
SVL=${PREP_SVL:-$HOME/Dropbox/Targets/prep_complex.svl}
JOBS=4
FORCE=0
OUTEXT=pdb
OUTDIR=
LOGDIR=${PREP_LOGDIR:-logs_prep}
PDBDIR=${PREP_PDBDIR:-pdb_rcsb}
ACE=
NME=
KEEPLIG=0

usage () {
    cat <<USAGE
usage: $(basename "$0") [-j N] [-o EXT] [-d DIR] [-p DIR] [-a CHAINS] [-n CHAINS] [-k] [-f] [-h] input [input ...] [OUTDIR]

Runs MOE StructurePreparation over structure files, writing foo.pdb (or
foo.mae, foo.cif, foo.dms) -> foo.prepped.EXT (EXT is pdb by default) beside
each input, or into an output directory if you name one --

  $(basename "$0") -j 8 -o mae boltz_*/structures/*.mae prepped/

A last argument that is not itself an input is that output directory: an
existing directory, a path written with a trailing /, or a name that does not
exist and has no structure extension or PDB-code shape.  It is created if
needed.  When that reading would be wrong -- an output directory named
runs.pdb, say -- use -d, which is the same thing said unambiguously.

Inputs are .pdb, .mae, .cif or .dms files (Boltz --output_format mae writes .mae);
at least one is required.  Other files a glob picks up -- .json, .npz -- are
ignored.  Let the shell do the matching -- dir/*.mae, or a deeper glob --
rather than naming an input directory.

An input can also be an RCSB PDB code -- 8c70, or the extended pdb_00008c70
form -- which is downloaded as mmCIF into $PDBDIR (see -p) and prepped from
there, so its output is $PDBDIR/<code>.prepped.EXT unless -d or a trailing
OUTDIR says otherwise.  A code already downloaded is reused, so re-running
costs no downloads; delete the .cif to fetch it again.  A code is told from a
path by shape alone -- four characters starting with 1-9 -- and only when
nothing by that name exists here, so a local file or directory named 8c70 is
read as that path, never as the code.

  -j N   run N moebatch jobs in parallel (default $JOBS)
  -o EXT output format, one of pdb mae moe mol2 mol (default $OUTEXT); MOE
         deduces the format from the extension it is handed
  -d DIR write the outputs into DIR (created if needed) as
         DIR/<basename>.prepped.EXT, instead of beside each input; the same
         thing a trailing OUTDIR does, and giving both is an error.  One
         directory for the whole run, so give inputs whose basenames are
         unique -- one boltz_*/structures per run, not several at once,
         since model_0.mae repeats in each
  -p DIR downloaded PDB entries are kept in DIR (default $PDBDIR), created if
         needed and reused across runs.  Only where the downloads live; the
         prepped outputs still follow -d or sit beside them
  -a CHAINS
         cap the N-terminus of these chains with ACE, e.g. -a A or -a A,B
  -n CHAINS
         cap the C-terminus of these chains with NME
         Chain ends not named are charged (NH3+ / COO-).  Chains are PDB
         chain letters and must be protein; an input lacking a named chain
         fails.  The same caps apply to every input in the run.
  -k     keep ligand protonation and tautomer states as given in the
         input (-keep_ligands); by default Protonate3D re-assigns them.
         Only meaningful for inputs that carry charges (.mae .dms), not PDB
  -f     re-prep inputs that already have a non-empty output; without it
         those are skipped, so an interrupted run just resumes.  Outputs do
         not record which caps they were made with -- after changing -a/-n,
         use -f
  -h     show this help and exit

examples:
  $(basename "$0") one.pdb two.pdb
  $(basename "$0") -j 8 boltz_results_config_A_1/predictions/*/*.pdb
  $(basename "$0") -j 8 */*/*/*.pdb          # everything below here
  $(basename "$0") -j 8 -o mae */*/*/*.pdb   # ... as Maestro files
  $(basename "$0") -o mae -n B */*/*/*.pdb    # chain B's C-terminus as NME
  $(basename "$0") -j 8 -o mae -n B boltz_results_*/predictions/*/*.mae
  $(basename "$0") 8c70                      # fetch 8c70.cif from RCSB, prep it
  $(basename "$0") -j 8 -o mae 8c70 6ron prepped/
  $(basename "$0") -j 8 \$(awk -F, 'NR>1{print \$11}' annotations.csv | sort -u)

  # a prediction run that writes a flat boltz_*/structures; prep each
  # such folder into its own structures_prepped sibling
  for d in boltz_*/structures; do
      $(basename "$0") -j 8 -o mae "\$d"/*.mae "\${d}_prepped"/
  done

Files already named *.prepped.* are never used as inputs, whatever their
extension: preparation is not idempotent, and with -o mae the outputs sit
beside .mae inputs where the same glob would otherwise pick them up.

Per-structure moebatch output goes to $LOGDIR/<flattened path>.log, the
ok/skip/FAILED tally to $LOGDIR/summary.txt, and MOE's own .structprep.txt
reports land beside each prepared structure.

  $MOEBATCH
  $SVL
USAGE
}

while getopts "j:o:d:p:a:n:kfh" opt; do
    case $opt in
        j) JOBS=$OPTARG ;;
        o) OUTEXT=$OPTARG ;;
        d) OUTDIR=$OPTARG ;;
        p) PDBDIR=$OPTARG ;;
        a) ACE=$OPTARG ;;
        n) NME=$OPTARG ;;
        k) KEEPLIG=1 ;;
        f) FORCE=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

# An argument shaped like an RCSB entry id and naming nothing on disk: 8c70,
# or the extended pdb_00008c70.  Codes are case-insensitive; RCSB files are
# named in lower case.
is_pdb_code () {
    [ -e "$1" ] && return 1
    case $1 in
        [1-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]) return 0 ;;
        [Pp][Dd][Bb]_[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9])
            return 0 ;;
        *) return 1 ;;
    esac
}

# Download one entry as mmCIF into $PDBDIR, and print the path to it.  An
# entry already downloaded is reused; a partial download is never left behind
# under that name, so an interrupted run does not poison the cache.
fetch_pdb () {
    code=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    cif=$PDBDIR/$code.cif
    if [ -s "$cif" ]; then
        echo "have   $code ($cif)" >&2
        echo "$cif"
        return 0
    fi
    command -v curl >/dev/null 2>&1 || { echo "no curl, needed to fetch $code" >&2; return 1; }
    mkdir -p "$PDBDIR" || return 1
    if curl -sfL -o "$cif.part" "https://files.rcsb.org/download/$code.cif" && [ -s "$cif.part" ]; then
        mv "$cif.part" "$cif"
        echo "fetch  $code ($cif)" >&2
        echo "$cif"
        return 0
    fi
    rm -f "$cif.part"
    return 1
}

[ "$#" -gt 0 ] || { echo "$(basename "$0"): no input given" >&2; usage >&2; exit 2; }

# A trailing OUTDIR: the last argument, when it cannot be an input itself.
# An existing file is always an input, so a plain list of files is untouched.
if [ "$#" -gt 1 ]; then
    last=
    for last in "$@"; do :; done
    takeit=0
    case $last in
        */) takeit=1 ;;
        *.pdb|*.mae|*.cif|*.dms) [ -d "$last" ] && takeit=1 ;;
        *) if ! is_pdb_code "$last" && { [ -d "$last" ] || [ ! -e "$last" ]; }; then
               takeit=1
           fi ;;
    esac
    if [ "$takeit" -eq 1 ]; then
        [ -z "$OUTDIR" ] || { echo "-d and a trailing output directory do the same thing; give one" >&2; exit 2; }
        OUTDIR=${last%/}
        [ -e "$OUTDIR" ] && [ ! -d "$OUTDIR" ] && { echo "$OUTDIR exists and is not a directory" >&2; exit 2; }
        set -- "${@:1:$(($# - 1))}"
        [ "$#" -gt 0 ] || { echo "$(basename "$0"): no input given, only an output directory" >&2; exit 2; }
    fi
fi

case $OUTEXT in
    pdb|mae|moe|mol2|mol) ;;
    *) echo "unsupported output format: $OUTEXT (use pdb, mae, moe, mol2 or mol)" >&2; exit 2 ;;
esac

[ -x "$MOEBATCH" ] || { echo "no moebatch at $MOEBATCH" >&2; exit 1; }
[ -r "$SVL" ] || { echo "no SVL script at $SVL" >&2; exit 1; }

mkdir -p "$LOGDIR"
[ -n "$OUTDIR" ] && mkdir -p "$OUTDIR"

prep_one () {
    in=$1
    base=${in%.*}
    if [ -n "$OUTDIR" ]; then
        out=$OUTDIR/$(basename "$base").prepped.$OUTEXT
    else
        out=$base.prepped.$OUTEXT
    fi
    log=$LOGDIR/$(echo "$base" | tr / _).log

    if [ "$FORCE" -eq 0 ] && [ -s "$out" ]; then
        echo "skip   $in (output exists)"
        return 0
    fi

    # macOS /bin/bash is 3.2: under set -u an empty "${svl_args[@]}" is an
    # unbound-variable error, hence the ${svl_args[@]+...} form.
    svl_args=()
    [ -n "$ACE" ] && svl_args+=(-ace "$ACE")
    [ -n "$NME" ] && svl_args+=(-nme "$NME")
    [ "$KEEPLIG" -eq 1 ] && svl_args+=(-keep_ligands)

    if "$MOEBATCH" -run "$SVL" ${svl_args[@]+"${svl_args[@]}"} "$in" "$out" >"$log" 2>&1 \
        && [ -s "$out" ]; then
        echo "ok     $in"
    else
        echo "FAILED $in  (see $log)"
        return 1
    fi
}
export -f prep_one
export MOEBATCH SVL FORCE LOGDIR OUTEXT OUTDIR ACE NME KEEPLIG

LIST=$LOGDIR/inputs.txt
: > "$LIST"

for arg in "$@"; do
    if [ -f "$arg" ]; then
        echo "$arg" >> "$LIST"
    elif is_pdb_code "$arg"; then
        cif=$(fetch_pdb "$arg") || { echo "could not fetch $arg from RCSB" >&2; exit 2; }
        echo "$cif" >> "$LIST"
    elif [ -d "$arg" ]; then
        echo "$arg is a directory; pass the files, e.g. $arg/*.pdb" >&2
        exit 2
    else
        echo "no such file or PDB code: $arg" >&2
        exit 2
    fi
done

# Structure files only, never our own outputs.
grep -E '\.(pdb|mae|cif|dms)$' "$LIST" | grep -v '\.prepped\.' | sort -u > "$LIST.tmp"
mv "$LIST.tmp" "$LIST"

NIN=$(wc -l < "$LIST" | tr -d ' ')
[ "$NIN" -gt 0 ] || { echo "no .pdb/.mae/.cif/.dms inputs found" >&2; exit 1; }
echo "$NIN structures to consider, $JOBS job(s) in parallel"
if [ -n "$OUTDIR" ]; then
    echo "outputs: $OUTDIR/<name>.prepped.$OUTEXT"
else
    echo "outputs: <name>.prepped.$OUTEXT beside each input"
fi
echo "caps: ACE on ${ACE:-none}, NME on ${NME:-none}; other chain ends NH3+/COO-"
if [ "$KEEPLIG" -eq 1 ]; then
    echo "ligands: input protonation/tautomer states kept"
else
    echo "ligands: protonation/tautomer states re-assigned by Protonate3D"
fi

tr '\n' '\0' < "$LIST" |
    xargs -0 -P "$JOBS" -I{} bash -c 'prep_one "$@"' _ {} |
    tee "$LOGDIR/summary.txt"

echo
printf 'done: %s prepared, %s skipped, %s failed\n' \
    "$(grep -c '^ok'     "$LOGDIR/summary.txt")" \
    "$(grep -c '^skip'   "$LOGDIR/summary.txt")" \
    "$(grep -c '^FAILED' "$LOGDIR/summary.txt")"
