"""
Static resources bundled with the ``scion`` package.

Two flavors:

* ``CLUSTER_TOML_TEMPLATE`` — the contents of ``cluster.toml.example``
  embedded as a string, so ``scion init`` can seed
  ``{root}/cluster.toml`` without depending on file-shipping. Keep this
  in sync with the example file at repo root by running
  ``tests/test_resources.py``.
* ``find_bundled_environments_dir()`` — locates the bundled
  ``environments/`` directory. Works for both editable installs
  (sibling of the scion package source tree) and wheel installs
  (a ``_bundled_environments`` subdir under the package, populated
  via ``[tool.hatch.build.targets.wheel.force-include]``).
"""

from __future__ import annotations

from pathlib import Path

CLUSTER_TOML_TEMPLATE = """\
# Per-cluster runtime overrides for Scion.
#
# This file is loaded by Scion automatically — worker spawn,
# `scion check`, and the `--models` prebuild step all apply the
# matching overlay.
#
# This file is only for environment-variable overlays. Put root paths,
# scheduler metadata, hostname patterns, and runtime-dir preferences in
# ~/.config/scion/clusters.toml or the file pointed to by SCION_CLUSTERS_FILE.
#
# Three tables, all optional:
#   [env]          always applied
#   [login_env]    only when *not* inside a PBS/SLURM batch job
#   [compute_env]  only when inside a detected batch job
#
# Job detection defaults to PBS_JOBID and SLURM_JOB_ID. Cluster profiles
# can override job_env_vars for other schedulers.

# Shared HPC login nodes apply tight RLIMIT_NPROC, so torch / MKL /
# OpenBLAS spinning up one thread per CPU core (64+) crashes with
# `libgomp: Thread creation failed: Resource temporarily unavailable`
# before the model can even load. Cap to 1 on the login node; let
# compute jobs use whatever the user sets.
[login_env]
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"

# Example: if compute nodes lose direct outbound and need a site proxy
# for HF Hub / torch.hub downloads, add it here. Leave commented unless
# you actually hit a download failure inside a job.
# [compute_env]
# HTTPS_PROXY = "http://proxy.example.org:3128"
# HTTP_PROXY  = "http://proxy.example.org:3128"
"""


def find_bundled_environments_dir() -> Path | None:
    """
    Return the directory containing the bundled environment files, or None.

    Search order:
      1. ``<scion-package>/_bundled_environments`` (wheel install).
      2. ``<scion-package>/../environments`` (editable / source install).
    """
    import scion

    pkg_dir = Path(scion.__file__).resolve().parent
    candidates = [
        pkg_dir / "_bundled_environments",
        pkg_dir.parent / "environments",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*.py")):
            return path
    return None
