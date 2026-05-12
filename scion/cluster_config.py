"""
Per-cluster runtime configuration.

Each Scion install root can carry a ``cluster.toml`` that declares the
environment variables to overlay onto every subprocess Scion spawns
(worker spawn, ``scion check``, the ``--models`` prebuild). Three tables:

  [env]          always applied
  [login_env]    applied only when *not* inside a batch job
  [compute_env]  applied only when inside a PBS or SLURM batch job

Job detection is by ``PBS_JOBID`` (PBS Pro: Polaris, ALCF) or
``SLURM_JOB_ID`` (SLURM: Della, Perlmutter, most academic clusters).

Example ``cluster.toml`` for Polaris::

    [login_env]
    OMP_NUM_THREADS = "1"
    MKL_NUM_THREADS = "1"
    OPENBLAS_NUM_THREADS = "1"

The file is optional; absence means "no overrides," and Scion behaves
exactly as before.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


CLUSTER_CONFIG_FILENAME = "cluster.toml"


@dataclass
class ClusterConfig:
    """In-memory representation of ``{root}/cluster.toml``."""

    env: dict[str, str] = field(default_factory=dict)
    login_env: dict[str, str] = field(default_factory=dict)
    compute_env: dict[str, str] = field(default_factory=dict)

    def resolved_env(self, in_job: bool | None = None) -> dict[str, str]:
        """
        Return the merged env dict appropriate for the current context.

        ``in_job=None`` (default) auto-detects via PBS_JOBID / SLURM_JOB_ID.
        Pass an explicit bool to force one branch (useful for tests).
        """
        if in_job is None:
            in_job = is_in_batch_job()
        merged: dict[str, str] = {}
        merged.update(self.env)
        merged.update(self.compute_env if in_job else self.login_env)
        return merged


def is_in_batch_job() -> bool:
    """True when invoked inside a PBS Pro or SLURM batch job."""
    return bool(os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"))


def load_cluster_config(root: Path | str) -> ClusterConfig:
    """
    Load ``{root}/cluster.toml`` if present, else return an empty config.

    Malformed TOML or unexpected types are logged to stderr and ignored
    — the goal is to never block worker spawn over a config-file issue.
    """
    path = Path(root) / CLUSTER_CONFIG_FILENAME
    if not path.exists():
        return ClusterConfig()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Warning: {path} is malformed TOML — ignoring: {e}", file=sys.stderr)
        return ClusterConfig()

    def _strdict(value) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in value.items()}

    return ClusterConfig(
        env=_strdict(data.get("env")),
        login_env=_strdict(data.get("login_env")),
        compute_env=_strdict(data.get("compute_env")),
    )


def get_cluster_env(root: Path | str, in_job: bool | None = None) -> dict[str, str]:
    """Convenience: load and resolve in one call. Empty dict if no config file."""
    return load_cluster_config(root).resolved_env(in_job=in_job)
