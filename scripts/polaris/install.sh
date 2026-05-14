#!/bin/bash
#
# Polaris installer: builds esm2_env and boltz_env at $SCION_ROOT.
#
# Run from a Polaris LOGIN NODE (outbound HTTP works there). Expects:
#   * `module load conda` is available (every ALCF Polaris account).
#   * Write access to $SCION_ROOT.
#
# Sets up four things:
#   1. uv (downloaded to ~/.local/bin if missing).
#   2. A personal venv for the scion CLI at ~/.venvs/scion-cli.
#   3. cluster.toml seed at $SCION_ROOT (login-node thread caps).
#   4. Two built envs:
#        - esm2_env   (~few minutes, ~2.5 GB)
#        - boltz_env  (~15-25 minutes, ~5 GB weights via Boltz CLI)
#
# Idempotent: re-running skips envs that already exist. Use
# `scion install <env> --force` for a from-scratch rebuild.
#
# Usage:
#   SCION_ROOT=$HOME/scion bash install.sh
#   SCION_ROOT=/lus/eagle/projects/<PROJECT>/scion bash install.sh
#
set -euo pipefail

: "${SCION_ROOT:?Set SCION_ROOT to the install root (e.g. \$HOME/scion)}"

SCION_GIT_URL="${SCION_GIT_URL:-git+https://github.com/blaiszik/scion.git}"
CLI_VENV="${CLI_VENV:-$HOME/.venvs/scion-cli}"

echo "=== Scion install for Polaris ==="
echo "  SCION_ROOT:    $SCION_ROOT"
echo "  CLI venv:      $CLI_VENV"
echo "  source:        $SCION_GIT_URL"
echo

# ---------------------------------------------------------------------------
# 1. Module + Python
# ---------------------------------------------------------------------------
echo "[1/5] Loading conda module..."
# `module` is a shell function; PBS login shells already source it. If
# we're invoked under sh/bash without it, source the modules init.
if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        . /etc/profile.d/modules.sh
    fi
fi
module load conda

py_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$py_version" in
    3.10|3.11|3.12|3.13) ;;
    *) echo "Error: Python $py_version is too old for scion (requires >=3.10)." >&2
       echo "Try: module load conda  (Polaris conda default ships >=3.11)" >&2
       exit 1 ;;
esac
echo "  Python: $py_version"

# ---------------------------------------------------------------------------
# 2. uv (worker-venv builder)
# ---------------------------------------------------------------------------
echo "[2/5] Ensuring uv is on PATH..."
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "  uv not found; installing to ~/.local/bin via astral.sh..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

# ---------------------------------------------------------------------------
# 3. Scion CLI in a personal venv
# ---------------------------------------------------------------------------
echo "[3/5] Installing scion CLI into $CLI_VENV..."
if [ ! -d "$CLI_VENV" ]; then
    python -m venv "$CLI_VENV"
fi
# shellcheck disable=SC1091
source "$CLI_VENV/bin/activate"
pip install --upgrade --quiet pip
pip install --upgrade --quiet "$SCION_GIT_URL"
scion --help >/dev/null

# ---------------------------------------------------------------------------
# 4. Root layout + cluster.toml
# ---------------------------------------------------------------------------
echo "[4/5] Seeding $SCION_ROOT layout..."
export SCION_ROOT
mkdir -p "$SCION_ROOT"/{environments,envs,cache,home,.python}

# Login-node thread caps avoid the libgomp RLIMIT_NPROC crash on shared
# Polaris login nodes. Compute jobs get whatever the user / Boltz set.
if [ ! -f "$SCION_ROOT/cluster.toml" ]; then
    cat > "$SCION_ROOT/cluster.toml" <<'TOML'
# Seeded by scripts/polaris/install.sh. Login-node thread caps only.
[login_env]
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
TOML
    echo "  Wrote $SCION_ROOT/cluster.toml"
fi

# ---------------------------------------------------------------------------
# 5. Build envs from the wheel-bundled files
# ---------------------------------------------------------------------------
# Locate the bundled env files via scion.resources, which handles both
# the wheel-install case (_bundled_environments under the package) and
# the source-checkout case (sibling environments/ next to the package).
# Run from /tmp to keep CWD out of sys.path[0] — otherwise a stray
# `scion/` directory in the user's CWD can shadow the installed package.
BUNDLED=$(cd /tmp && python -c "from scion.resources import find_bundled_environments_dir as f; p=f(); print(p or '')")
if [ -z "$BUNDLED" ] || [ ! -d "$BUNDLED" ]; then
    echo "Error: could not locate bundled env files." >&2
    echo "  Re-install scion from the current main branch:" >&2
    echo "    pip install --upgrade --force-reinstall $SCION_GIT_URL" >&2
    exit 1
fi
echo "  bundled envs: $BUNDLED"

build_if_missing() {
    local env_name="$1"
    if [ -d "$SCION_ROOT/envs/$env_name" ]; then
        echo "  $env_name: already built (skip — use 'scion install $env_name --force' to rebuild)."
        return
    fi
    echo "  building $env_name ..."
    scion install "$BUNDLED/${env_name}.py"
}

echo "[5/5] Building worker envs..."
echo "      (uv pip install + provider.preload() stream live below; expect minutes)"
echo

build_if_missing esm2_env
build_if_missing boltz_env

echo
echo "=== Install complete ==="
scion status
echo
echo "Next: submit the demo job from this directory:"
echo "    qsub -A <YOUR_PROJECT> -v SCION_ROOT scripts/polaris/submit_demo.sh"
echo
echo "If you switch shells, reactivate the CLI venv first:"
echo "    module load conda && source $CLI_VENV/bin/activate"
echo "    export SCION_ROOT=$SCION_ROOT"
