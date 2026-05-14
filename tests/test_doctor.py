"""Tests for scion doctor diagnostics."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

from scion.commands.doctor import run_doctor


def _args(**kwargs):
    defaults = {"cluster": None, "root": None, "env_name": None, "json": False}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_root(path: Path) -> None:
    for name in ("environments", "envs", "cache", "home", ".python"):
        (path / name).mkdir(parents=True, exist_ok=True)


def test_doctor_reports_existing_root_without_errors(tmp_path: Path):
    root = tmp_path / "scion"
    _make_root(root)

    report = run_doctor(_args(root=str(root)))

    assert report["root"] == str(root)
    assert report["summary"]["error"] == 0
    assert any(c["name"] == "root" and c["status"] == "ok" for c in report["checks"])


def test_doctor_reports_missing_requested_env_as_error(tmp_path: Path):
    root = tmp_path / "scion"
    _make_root(root)

    report = run_doctor(_args(root=str(root), env_name="missing_env"))

    assert report["summary"]["error"] >= 1
    assert any(c["name"] == "env:missing_env" for c in report["checks"])


def test_doctor_flags_torch_cuda_mismatch(tmp_path: Path, monkeypatch):
    """Doctor must surface env torch pin > cluster cuda_driver_max as error."""
    root = tmp_path / "scion"
    _make_root(root)

    env_dir = root / "envs" / "loose_env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").write_text("")  # presence is enough
    env_source = (
        "# /// script\n"
        "# requires-python = '>=3.10'\n"
        "# dependencies = ['torch>=2.6']\n"
        "# ///\n"
        "CAPABILITIES = ['embed']\n"
        "def setup(model, device='cpu'):\n    return None\n"
    )
    (env_dir / "env_source.py").write_text(env_source)

    profiles_file = tmp_path / "clusters.toml"
    profiles_file.write_text(
        textwrap.dedent(
            f"""\
            [clusters.testcap]
            root = "{root}"
            cuda_driver_max = "12.8"
            """
        )
    )
    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(profiles_file))

    report = run_doctor(_args(cluster="testcap"))
    compat_checks = [c for c in report["checks"] if c["name"].startswith("cuda compat:")]
    assert compat_checks, report["checks"]
    assert any(c["status"] == "error" for c in compat_checks)


def test_doctor_passes_when_torch_pinned_below_cap(tmp_path: Path, monkeypatch):
    """When env caps torch at <2.9, cap=12.8 should match cleanly."""
    root = tmp_path / "scion"
    _make_root(root)

    env_dir = root / "envs" / "good_env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").write_text("")
    env_source = (
        "# /// script\n"
        "# requires-python = '>=3.10'\n"
        "# dependencies = ['torch>=2.6,<2.9']\n"
        "# ///\n"
        "CAPABILITIES = ['embed']\n"
        "def setup(model, device='cpu'):\n    return None\n"
    )
    (env_dir / "env_source.py").write_text(env_source)

    profiles_file = tmp_path / "clusters.toml"
    profiles_file.write_text(
        textwrap.dedent(
            f"""\
            [clusters.testcap2]
            root = "{root}"
            cuda_driver_max = "12.8"
            """
        )
    )
    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(profiles_file))

    report = run_doctor(_args(cluster="testcap2"))
    compat_checks = [c for c in report["checks"] if c["name"].startswith("cuda compat:")]
    assert compat_checks, report["checks"]
    assert all(c["status"] == "ok" for c in compat_checks)


def test_doctor_uses_layered_cluster_profile(tmp_path: Path, monkeypatch):
    root = tmp_path / "scion"
    _make_root(root)
    (root / "cluster.toml").write_text('[login_env]\nOMP_NUM_THREADS = "1"\n')
    profiles_file = tmp_path / "clusters.toml"
    profiles_file.write_text(
        textwrap.dedent(
            f"""\
            [clusters.testbox]
            root = "{root}"
            scheduler = "lsf"
            job_env_vars = ["LSB_JOBID"]
            runtime_dir = "{tmp_path}"
            """
        )
    )

    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(profiles_file))
    report = run_doctor(_args(cluster="testbox"))

    assert report["cluster"] == "testbox"
    assert report["root"] == str(root)
    profile_check = next(c for c in report["checks"] if c["name"] == "cluster profile")
    assert profile_check["status"] == "ok"
    assert profile_check["detail"]["scheduler"] == "lsf"
