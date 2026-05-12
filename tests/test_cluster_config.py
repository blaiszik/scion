"""Tests for {root}/cluster.toml loading and overlay resolution."""

from __future__ import annotations

import textwrap
from pathlib import Path

from scion.cluster_config import (
    ClusterConfig,
    get_cluster_env,
    is_in_batch_job,
    load_cluster_config,
)


def test_missing_file_returns_empty_config(tmp_path: Path):
    cfg = load_cluster_config(tmp_path)
    assert isinstance(cfg, ClusterConfig)
    assert cfg.env == {} and cfg.login_env == {} and cfg.compute_env == {}
    assert cfg.resolved_env(in_job=False) == {}
    assert cfg.resolved_env(in_job=True) == {}


def test_login_env_applied_outside_a_job(tmp_path: Path):
    (tmp_path / "cluster.toml").write_text(
        textwrap.dedent(
            """\
            [login_env]
            OMP_NUM_THREADS = "1"
            MKL_NUM_THREADS = "1"

            [compute_env]
            HTTPS_PROXY = "http://proxy.example:3128"
            """
        )
    )
    assert get_cluster_env(tmp_path, in_job=False) == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    assert get_cluster_env(tmp_path, in_job=True) == {
        "HTTPS_PROXY": "http://proxy.example:3128",
    }


def test_always_env_merges_under_context_env(tmp_path: Path):
    (tmp_path / "cluster.toml").write_text(
        textwrap.dedent(
            """\
            [env]
            OMP_NUM_THREADS = "1"
            HF_HUB_OFFLINE = "0"

            [compute_env]
            OMP_NUM_THREADS = "16"   # job-time override
            """
        )
    )
    # On login: only [env] applies (with no [login_env] entries)
    assert get_cluster_env(tmp_path, in_job=False) == {
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "0",
    }
    # In job: [compute_env] overlays on top of [env]
    assert get_cluster_env(tmp_path, in_job=True) == {
        "OMP_NUM_THREADS": "16",
        "HF_HUB_OFFLINE": "0",
    }


def test_malformed_toml_is_ignored_with_warning(tmp_path: Path, capsys):
    (tmp_path / "cluster.toml").write_text("this is not = valid toml = at all")
    cfg = load_cluster_config(tmp_path)
    assert cfg.env == {} and cfg.login_env == {} and cfg.compute_env == {}
    err = capsys.readouterr().err
    assert "malformed" in err.lower()


def test_job_detection_via_env_vars(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert is_in_batch_job() is False

    monkeypatch.setenv("PBS_JOBID", "123.polaris-pbs-01")
    assert is_in_batch_job() is True

    monkeypatch.delenv("PBS_JOBID")
    monkeypatch.setenv("SLURM_JOB_ID", "456789")
    assert is_in_batch_job() is True


def test_non_dict_table_is_silently_dropped(tmp_path: Path):
    # If a maintainer writes `env = "oops"` instead of `[env]`, don't crash.
    (tmp_path / "cluster.toml").write_text('env = "not a table"\n[login_env]\nA = "1"\n')
    cfg = load_cluster_config(tmp_path)
    assert cfg.env == {}
    assert cfg.login_env == {"A": "1"}


def test_resolved_env_auto_detects_job_state(tmp_path: Path, monkeypatch):
    (tmp_path / "cluster.toml").write_text(
        '[login_env]\nA = "login"\n\n[compute_env]\nA = "compute"\n'
    )
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert get_cluster_env(tmp_path) == {"A": "login"}

    monkeypatch.setenv("PBS_JOBID", "1.polaris")
    assert get_cluster_env(tmp_path) == {"A": "compute"}
