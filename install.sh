#!/usr/bin/env bash
# Install boonza into a conda environment: Python, NumPy, OpenMM, AmberTools and
# the rest from conda-forge (environment.yml), then boonza itself with pip.
#
# Usage:
#   bash install.sh [NAME] [--cuda VERSION|none] [--openmm VERSION] [--dev] [--dry-run]
#
# NAME defaults to "boonza"; an existing environment of that name is updated.
#
# --cuda VERSION   OpenMM built for this CUDA release, such as 12.6 or 12. It
#                  pins cuda-version and stands in for the driver
#                  (CONDA_OVERRIDE_CUDA), so a login node without a GPU installs
#                  for the GPU nodes. Their driver must support the release:
#                  compare with "CUDA Version" in nvidia-smi on a GPU node.
#                  Without --cuda, conda picks a release from the driver of the
#                  machine running this script, or the newest one when it has
#                  none, and GPU nodes with an older driver then fail with
#                  CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
# --cuda none      OpenMM's build without CUDA: the CPU and OpenCL platforms.
#                  OpenCL runs on NVIDIA GPUs through the driver whatever CUDA
#                  release it supports (run boonza md with --platform OpenCL);
#                  ocl-icd-system is added so OpenCL finds the system's driver.
# --openmm VERSION this OpenMM release, such as 8.2 or 8.4.0 (8.1 or later).
# --dev            also pytest, ruff and MDAnalysis, for the test suite.
# --dry-run        show what conda would install, and change nothing.
#
# boonza needs NumPy 2. conda-forge builds OpenMM for CUDA releases before 12.6
# against NumPy 1 only, so a driver older than 12.6 needs --cuda none (OpenCL)
# rather than --cuda 12.2.
#
#   bash install.sh                               # env "boonza", conda's choice of CUDA
#   bash install.sh --cuda 12.6                   # OpenMM for GPU nodes with CUDA 12.6+
#   bash install.sh --cuda none --openmm 8.2      # OpenMM 8.2, CPU and OpenCL
#   bash install.sh md --cuda 12.9 --dev          # env "md", with the test tools
set -euo pipefail

ENV_NAME=""
CUDA=""
OPENMM=""
DEV=0
DRY_RUN=0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --cuda)     [ $# -ge 2 ] || { echo "--cuda needs a version, e.g. --cuda 12.6, or none" >&2; exit 2; }
                    CUDA="$2"; shift 2 ;;
        --cuda=*)   CUDA="${1#--cuda=}"; shift ;;
        --openmm)   [ $# -ge 2 ] || { echo "--openmm needs a version, e.g. --openmm 8.2" >&2; exit 2; }
                    OPENMM="$2"; shift 2 ;;
        --openmm=*) OPENMM="${1#--openmm=}"; shift ;;
        --dev)      DEV=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        -*)         echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
        *)          [ -z "$ENV_NAME" ] || { echo "Only one environment name (got '$ENV_NAME' and '$1')" >&2; exit 2; }
                    ENV_NAME="$1"; shift ;;
    esac
done
ENV_NAME="${ENV_NAME:-boonza}"

if [ -n "$CUDA" ] && [ "$CUDA" != none ] && ! [[ "$CUDA" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "--cuda takes a CUDA release such as 12.6 or 12, or none; not '$CUDA'" >&2
    exit 2
fi
if [ -n "$OPENMM" ]; then
    [[ "$OPENMM" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]] || { echo "--openmm takes a release such as 8.2 or 8.4.0; not '$OPENMM'" >&2; exit 2; }
    IFS=. read -r major minor _ <<<"$OPENMM"
    if [ "$major" -lt 8 ] || { [ "$major" -eq 8 ] && [ "${minor:-0}" -lt 1 ]; }; then
        echo "boonza needs OpenMM 8.1 or later; not $OPENMM" >&2
        exit 2
    fi
fi

command -v conda >/dev/null 2>&1 || { echo "conda not found: install Miniforge first" >&2; exit 1; }
# conda activate is a shell function, missing from a script until this is sourced
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda throughout, not mamba: mamba 2 keeps environments under its own root,
# where conda activate does not find them (conda 23.10+ solves with libmamba)
SOLVER=conda

# ----------------------------------------------------------------------
# 1. environment.yml with the pins asked for
# ----------------------------------------------------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
OPENMM_SPEC="openmm>=8.1"
[ -n "$OPENMM" ] && OPENMM_SPEC="openmm=$OPENMM"
EXTRA=()

if [ "$CUDA" = none ]; then
    # A CUDA build and a build without CUDA differ only by an opaque build
    # string, so pick the newest build without a cuda-version dependency that
    # suits Python 3.12 and NumPy 2, and pin it exactly.
    echo "==> Finding OpenMM ${OPENMM:-(newest)} without CUDA"
    conda search -c conda-forge --override-channels "$OPENMM_SPEC" --json >"$WORK/search.json" || {
        echo "conda search found no OpenMM ${OPENMM:-release}" >&2; exit 1; }
    # a file rather than a here-document inside $( ): older bash misparses those
    cat >"$WORK/pick.py" <<'PY'
import json
import re
import sys

builds = json.load(open(sys.argv[1])).get("openmm", [])

def fits(p):
    deps = p.get("depends", [])
    if any(d.startswith("cuda-version") for d in deps):
        return False
    if not any(d.startswith("python_abi 3.12") for d in deps):
        return False
    numpy = [d for d in deps if d.split()[0] == "numpy"]
    return not numpy or "<2.0" not in numpy[0]  # NumPy 2 allowed

def key(p):
    return ([int(x) for x in re.findall(r"\d+", p["version"])], p.get("build_number", 0),
            p.get("timestamp", 0))

good = sorted((p for p in builds if fits(p)), key=key)
if not good:
    sys.exit("no OpenMM build without CUDA for Python 3.12 and NumPy 2")
best = good[-1]
print(f"openmm={best['version']}={best['build']}")
PY
    OPENMM_SPEC="$("$(conda info --base)/bin/python" "$WORK/pick.py" "$WORK/search.json")"
    echo "    $OPENMM_SPEC"
    EXTRA+=("ocl-icd-system")
elif [ -n "$CUDA" ]; then
    # Pinning cuda-version selects OpenMM's build for that CUDA release, and
    # CONDA_OVERRIDE_CUDA stands in for the driver conda would otherwise detect.
    EXTRA+=("cuda-version=$CUDA")
    export CONDA_OVERRIDE_CUDA="$CUDA"
fi
[ "$DEV" -eq 1 ] && EXTRA+=("pytest" "ruff" "mdanalysis")

ENV_FILE="$WORK/environment.yml"
awk -v openmm="  - $OPENMM_SPEC" -v extra="${EXTRA[*]-}" '
    /^ *- *openmm/ { print openmm; next }
    { print }
    /^dependencies:/ && extra != "" { n = split(extra, e, " "); for (i = 1; i <= n; i++) print "  - " e[i] }
' "$REPO_DIR/environment.yml" >"$ENV_FILE"

echo "==> Environment '$ENV_NAME':"
echo "    openmm        ${OPENMM_SPEC#openmm}"
echo "    cuda          ${CUDA:-chosen by conda from the driver here}"
[ "${#EXTRA[@]}" -gt 0 ] && echo "    also          ${EXTRA[*]}"

# ----------------------------------------------------------------------
# 2. Create or update it
# ----------------------------------------------------------------------
explain() {
    echo >&2
    echo "The solve failed." >&2
    if [ -n "$CUDA" ] && [ "$CUDA" != none ] && [[ "$CUDA" =~ ^12(\.[0-5])?$ ]]; then
        echo "conda-forge builds OpenMM for CUDA before 12.6 against NumPy 1 only, and" >&2
        echo "boonza needs NumPy 2: use --cuda 12.6 or later if the GPU nodes' driver" >&2
        echo "supports it, or --cuda none to run on the OpenCL platform." >&2
    elif [ -n "$OPENMM" ]; then
        echo "OpenMM $OPENMM may have no build for this CUDA release with NumPy 2;" >&2
        echo "try another --openmm, another --cuda, or --cuda none." >&2
    fi
    exit 1
}

if [ "$DRY_RUN" -eq 1 ]; then
    # conda, not mamba: mamba 2's env create goes ahead despite --dry-run
    echo "==> Solving (dry run; nothing is installed)"
    conda env create --dry-run -n "$ENV_NAME-dry-run" -f "$ENV_FILE" || explain
    exit 0
fi
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Updating environment '$ENV_NAME'"
    "$SOLVER" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune || explain
else
    echo "==> Creating environment '$ENV_NAME'"
    "$SOLVER" env create -n "$ENV_NAME" -f "$ENV_FILE" || explain
fi
conda activate "$ENV_NAME"

echo "==> Installing boonza"
python -m pip install --quiet --no-deps -e "$REPO_DIR"

# ----------------------------------------------------------------------
# 3. Say what was installed
# ----------------------------------------------------------------------
echo
echo "==> Installed"
BOONZA_CUDA="$CUDA" python - <<'PY'
import os
import sys

import numpy
import openmm

import boonza

print(f"  python        {sys.version.split()[0]}")
print(f"  numpy         {numpy.__version__}")
print(f"  openmm        {openmm.version.version}")
print(f"  boonza        {boonza.__version__}")
names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print(f"  platforms     {', '.join(names)}")
cuda = os.environ.get("BOONZA_CUDA")
if cuda and cuda != "none" and "CUDA" not in names:
    # the CUDA plugin loads only where there is an NVIDIA driver
    print("  !! CUDA did not load here (expected without an NVIDIA driver); on a GPU")
    print("     node, check it with: python -m openmm.testInstallation")
if cuda == "none" and "OpenCL" not in names:
    print("  !! OpenCL did not load here; on a GPU node, check it with:")
    print("     python -m openmm.testInstallation")
try:
    from boonza import gaff

    home = gaff.find_amber_tools()
    print(f"  ambertools    {home} (GAFF 2.11: {gaff.gaff_parameters(home).name})")
except Exception as error:  # noqa: BLE001 - report whatever is wrong
    print(f"  !! ambertools {error}")
try:
    from boonza import ffxml

    print(f"  openmm xml    bundled from {' '.join(ffxml.bundled_version().split()[:2])}")
except (ImportError, AttributeError):
    pass
PY

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
[ "$CUDA" = none ] && echo "Run on the GPU with: boonza md ... --platform OpenCL"
exit 0
