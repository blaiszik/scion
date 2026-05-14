#!/bin/bash -l
#
# Polaris PBS job: fold a peptide with Boltz-2 on a GPU node.
#
# Submit with:
#   qsub -A <YOUR_PROJECT> -v SCION_ROOT scripts/polaris/submit_demo.sh
#
# The `-v SCION_ROOT` forwards your shell's $SCION_ROOT into the job's
# environment (PBS does not inherit by default). `-A` is required by
# qsub on Polaris and selects the project the allocation bills.
#
# By default this script activates the venv that install.sh created at
# ~/.venvs/scion-cli. To use an existing conda env instead, forward its
# name with -v:
#   qsub -A <project> -v SCION_ROOT,SCION_CONDA_ENV=scion submit_demo.sh
#
#PBS -N scion_demo
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"

echo "=== job $PBS_JOBID on $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

# `module` is a shell function; under PBS's default shell it may not be
# initialized, and on compute nodes /soft/modulefiles may not be on the
# default MODULEPATH. Source the modules init and `module use` defensively.
if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        . /etc/profile.d/modules.sh
    fi
fi
module use /soft/modulefiles 2>/dev/null || true

# `module load conda` is best-effort. It's needed for `conda activate`
# under SCION_CONDA_ENV, but the default CLI-venv path is self-contained
# once activated. The compute-node Lmod cache occasionally goes stale
# and reports "module unknown: conda" even when the modulefile exists;
# --ignore_cache is the ALCF-documented workaround.
if ! module load conda 2>/dev/null; then
    module --ignore_cache load conda 2>/dev/null || \
        echo "Note: 'module load conda' unavailable on this node; relying on the activated venv for python."
fi

# Two ways to bring scion into scope:
#   SCION_CONDA_ENV=<name>   -> conda activate <name>     (requires conda on PATH)
#   (default)                -> source ~/.venvs/scion-cli/bin/activate
if [ -n "${SCION_CONDA_ENV:-}" ]; then
    if ! type conda >/dev/null 2>&1; then
        echo "Error: SCION_CONDA_ENV=$SCION_CONDA_ENV but conda is not on PATH." >&2
        echo "  Unset SCION_CONDA_ENV to fall back to the CLI venv, or make" >&2
        echo "  sure 'module load conda' works on this compute node." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    conda activate "$SCION_CONDA_ENV"
else
    CLI_VENV="${CLI_VENV:-$HOME/.venvs/scion-cli}"
    if [ ! -f "$CLI_VENV/bin/activate" ]; then
        echo "Error: $CLI_VENV/bin/activate not found." >&2
        echo "  Re-run scripts/polaris/install.sh from a login node." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$CLI_VENV/bin/activate"
fi

python -c "import scion; print(f'scion {scion.__version__} from {scion.__file__}')"

if [ -z "${SCION_ROOT:-}" ]; then
    echo "Error: SCION_ROOT not forwarded to the job." >&2
    echo "  Resubmit with: qsub -A <project> -v SCION_ROOT $0" >&2
    exit 1
fi

python "$PBS_O_WORKDIR/demo_fold.py"
