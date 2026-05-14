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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Same environment the install.sh laid down.
if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        . /etc/profile.d/modules.sh
    fi
fi
module load conda

CLI_VENV="${CLI_VENV:-$HOME/.venvs/scion-cli}"
# shellcheck disable=SC1091
source "$CLI_VENV/bin/activate"

if [ -z "${SCION_ROOT:-}" ]; then
    echo "Error: SCION_ROOT not forwarded to the job." >&2
    echo "  Resubmit with: qsub -A <project> -v SCION_ROOT $0" >&2
    exit 1
fi

python "$PBS_O_WORKDIR/demo_fold.py"
