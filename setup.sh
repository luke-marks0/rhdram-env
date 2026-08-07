#!/usr/bin/env bash
# RowHammer-OpenEnv one-shot machine bootstrap.
#
# Consolidates the phase-by-phase setup documented in README.md into a single
# idempotent command: check the toolchain, install Python deps, fetch the pinned
# upstream sources, build the native Ramulator worker, (re)build the signed DDR4
# profile, and run a verification gate.
#
# Every stage is safe to re-run. Native builds are cache-aware; when the pinned
# Ramulator commit changes underneath a stale libramulator.so (an ABI-drift
# footgun, see the source-fetch-pin memo) the script forces a clean rebuild.
#
# Usage:
#   ./setup.sh                 # full setup + fast verify (P0/P1/P2/P3 gates)
#   ./setup.sh --train         # also install GRPO training extras
#   ./setup.sh --venv          # create/use ./.venv for the Python deps
#   ./setup.sh --clean         # force a clean native rebuild
#   ./setup.sh --full-verify   # run the complete release re-qualification gate
#   ./setup.sh --no-verify     # stop after building; skip all gates
#   ./setup.sh --skip-deps     # assume Python deps are already installed
#   ./setup.sh -h | --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- options ---------------------------------------------------------------
WANT_TRAIN=0
WANT_VENV=0
WANT_CLEAN=0
WANT_FULL_VERIFY=0
WANT_VERIFY=1
WANT_DEPS=1

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

for arg in "$@"; do
  case "$arg" in
    --train)       WANT_TRAIN=1 ;;
    --venv)        WANT_VENV=1 ;;
    --clean)       WANT_CLEAN=1 ;;
    --full-verify) WANT_FULL_VERIFY=1 ;;
    --no-verify)   WANT_VERIFY=0 ;;
    --skip-deps)   WANT_DEPS=0 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

# ---- pretty output ---------------------------------------------------------
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=; G=; Y=; R=; N=; fi
step() { echo; echo "${B}==> $*${N}"; }
ok()   { echo "${G}  ok${N} $*"; }
warn() { echo "${Y}  warn${N} $*"; }
die()  { echo "${R}error:${N} $*" >&2; exit 1; }

PY="${PYTHON:-python3}"

# ---- 0. toolchain ----------------------------------------------------------
step "Checking host toolchain"
command -v "$PY"  >/dev/null || die "python3 not found"
command -v git    >/dev/null || die "git not found"
command -v cmake  >/dev/null || die "cmake is required to build Ramulator 2.1"
command -v g++    >/dev/null || die "g++ (C++20) is required to build the worker"
"$PY" - <<'PY' || die "Python 3.10+ required"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
ok "$("$PY" --version), $(cmake --version | head -1), $(g++ --version | head -1)"
if command -v ninja >/dev/null; then ok "ninja $(ninja --version)"; else warn "ninja not found; cmake will use the default (slower) generator"; fi

# ---- 1. python virtualenv (optional) --------------------------------------
if [ "$WANT_VENV" = 1 ]; then
  step "Preparing virtualenv (./.venv)"
  [ -d .venv ] || "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
  ok "using $(command -v python)"
fi

# ---- 2. python dependencies ------------------------------------------------
if [ "$WANT_DEPS" = 1 ]; then
  step "Installing Python dependencies"
  "$PY" -m pip install --disable-pip-version-check -q -r requirements.txt
  ok "core runtime deps (requirements.txt)"
  if [ "$WANT_TRAIN" = 1 ]; then
    "$PY" -m pip install --disable-pip-version-check -q -r requirements-train.txt
    ok "GRPO training extras (requirements-train.txt)"
  fi
else
  step "Skipping Python dependency install (--skip-deps)"
fi

# ---- 3. fetch pinned upstream sources -------------------------------------
step "Fetching pinned upstream sources"
"$PY" -B scripts/fetch_phase1_sources.py    # ramulator2 + OpenEnv -> third_party/
"$PY" -B scripts/fetch_phase3_sources.py    # admitted DDR4 VTS25 read-disturbance CSVs
ok "sources present and pin-verified"

# ---- 4. detect a stale native build ---------------------------------------
# build_phase1.py only rebuilds libramulator.so when it is *absent*, so a changed
# Ramulator pin can leave a stale .so whose ABI no longer matches the worker's
# headers. Force a clean rebuild when the built commit no longer matches the
# checked-out source (or when --clean was passed).
FETCHED_COMMIT="$(git -C third_party/ramulator2 rev-parse HEAD)"
STAMP="build/phase1/ramulator_commit.txt"
BUILT_COMMIT=""
[ -f "$STAMP" ] && BUILT_COMMIT="$(tr -d '[:space:]' < "$STAMP")"
if [ -f third_party/ramulator2/libramulator.so ] && [ "$BUILT_COMMIT" != "$FETCHED_COMMIT" ]; then
  warn "libramulator.so was built from '${BUILT_COMMIT:-unknown}', source is now '$FETCHED_COMMIT'"
  WANT_CLEAN=1
fi
if [ "$WANT_CLEAN" = 1 ]; then
  step "Clearing cached native build"
  rm -f third_party/ramulator2/libramulator.so
  rm -rf build/ramulator2 build/phase1 build/phase2
  ok "cache cleared; Ramulator + worker will be rebuilt from scratch"
fi

# ---- 5. build native worker (Phase 1 + Phase 2) ---------------------------
step "Building Ramulator + simulator worker"
"$PY" -B scripts/build_phase2.py            # transitively runs build_phase1
ok "native worker built (build/phase2/ramulator_worker)"

# ---- 6. (re)build the signed DDR4 profile ---------------------------------
step "Building the signed DDR4 read-disturbance profile"
"$PY" -B -m profile_builder.package.build >/dev/null
ok "profile built and signature verified (profiles/ddr4_vts25_v1)"

# ---- 7. verify -------------------------------------------------------------
if [ "$WANT_VERIFY" = 0 ]; then
  step "Skipping verification (--no-verify)"
elif [ "$WANT_FULL_VERIFY" = 1 ]; then
  step "Running full release re-qualification gate (this is slow)"
  "$PY" -B scripts/verify_release.py
  "$PY" -B scripts/verify_phase20.py
  ok "release gate passed"
else
  step "Running fast verification gate (P0 policy, P1 build, P2 worker, P3 profile)"
  "$PY" -B scripts/verify_phase0.py
  "$PY" -B scripts/verify_phase1.py
  "$PY" -B scripts/verify_phase2.py
  "$PY" -B scripts/verify_phase3.py
  ok "fast gate passed (run './setup.sh --full-verify' for the complete release matrix)"
fi

# ---- done ------------------------------------------------------------------
step "Setup complete"
cat <<EOF
Next steps:
  - Serve the environment:   ${PY} -m rowhammer_env.server.app
  - GRPO training dry-run:   ${PY} -B scripts/train_grpo.py --config configs/training/grpo_qwen4b.yaml --dry-run
$( [ "$WANT_TRAIN" = 0 ] && echo "  - Add training extras:     ./setup.sh --train --skip-deps" )
$( [ "$WANT_VENV" = 1 ]  && echo "  - Reactivate the venv:     source .venv/bin/activate" )
EOF
