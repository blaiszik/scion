"""Tests for layered cluster profiles."""

from __future__ import annotations

import textwrap
from pathlib import Path

from scion.cluster_config import get_cluster_env
from scion.clusters import (
    DEFAULT_JOB_ENV_VARS,
    detect_current_cluster,
    list_cluster_profiles,
)


def test_builtin_profiles_remain_available():
    profiles = list_cluster_profiles(config_paths=[])
    assert "polaris" in profiles
    assert profiles["polaris"].scheduler == "pbs"
    assert profiles["polaris"].root_path.as_posix().endswith("/scion")


def test_user_profiles_override_builtins_and_add_clusters(tmp_path: Path):
    config = tmp_path / "clusters.toml"
    config.write_text(
        textwrap.dedent(
            """\
            [clusters.polaris]
            root = "/override/scion"
            scheduler = "pbs"
            job_env_vars = ["PBS_JOBID"]

            [clusters.perlmutter]
            root = "/global/common/software/scion"
            scheduler = "slurm"
            job_env_vars = ["SLURM_JOB_ID"]
            hostname_patterns = ["*.nersc.gov", "perlmutter*"]
            gpu_arch = "a100"
            cuda_driver_max = "12.4"
            runtime_dir = "$SCRATCH/.scion/run"
            """
        )
    )

    profiles = list_cluster_profiles(config_paths=[config])
    assert profiles["polaris"].root == "/override/scion"
    assert profiles["polaris"].source == str(config)
    assert profiles["perlmutter"].scheduler == "slurm"
    assert profiles["perlmutter"].runtime_dir == "$SCRATCH/.scion/run"


def test_hostname_detection_uses_profile_patterns(tmp_path: Path, monkeypatch):
    config = tmp_path / "clusters.toml"
    config.write_text(
        textwrap.dedent(
            """\
            [clusters.frontier]
            root = "/ccs/proj/scion"
            scheduler = "slurm"
            hostname_patterns = ["frontier*.olcf.ornl.gov"]
            """
        )
    )
    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(config))

    profile = detect_current_cluster(
        hostname="frontier123.olcf.ornl.gov",
    )
    assert profile is not None
    assert profile.name == "frontier"


def test_cluster_env_uses_profile_job_env_vars(tmp_path: Path, monkeypatch):
    root = tmp_path / "scion"
    root.mkdir()
    (root / "cluster.toml").write_text(
        '[login_env]\nWHERE = "login"\n\n[compute_env]\nWHERE = "compute"\n'
    )
    profiles_file = tmp_path / "clusters.toml"
    profiles_file.write_text(
        f'[clusters.lsfbox]\nroot = "{root}"\njob_env_vars = ["LSB_JOBID"]\n'
    )

    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(profiles_file))
    for name in DEFAULT_JOB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    assert get_cluster_env(root) == {"WHERE": "login"}

    monkeypatch.setenv("LSB_JOBID", "12345")
    assert get_cluster_env(root) == {"WHERE": "compute"}
