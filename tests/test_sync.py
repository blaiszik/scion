"""
Tests for `scion sync` — refreshing env_source.py in built worker venvs.

Uses a fake worker venv layout (no Boltz install needed) and runs the
command via its `cmd_sync(args)` entry point with a SimpleNamespace.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from scion.commands.sync import cmd_sync

VALID_ENV_SOURCE_V1 = textwrap.dedent(
    """\
    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["numpy"]
    # ///

    CAPABILITIES = ["embed"]


    def setup(model, device="cpu"):
        return object()
    """
)

VALID_ENV_SOURCE_V2 = textwrap.dedent(
    """\
    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["numpy"]
    # ///

    CAPABILITIES = ["embed"]


    def setup(model, device="cpu"):
        # changed body
        return "v2"
    """
)


def _build_fake_root(tmp_path: Path, env_name: str = "fake_env") -> Path:
    """Lay out a {root}/ tree as if `scion install` had run."""
    root = tmp_path / "scion_root"
    (root / "environments").mkdir(parents=True)
    env_dir = root / "envs" / env_name
    (env_dir / "bin").mkdir(parents=True)
    # We don't actually exec the worker venv in these tests, but `sync`
    # checks that the venv directory exists, so make a placeholder
    # python symlink so the directory looks built.
    (env_dir / "bin" / "python").touch()
    return root


def test_file_mode_registers_and_syncs(tmp_path: Path):
    root = _build_fake_root(tmp_path)
    incoming = tmp_path / "fake_env.py"
    incoming.write_text(VALID_ENV_SOURCE_V1)

    rc = cmd_sync(SimpleNamespace(source=str(incoming), root=str(root), with_scion=False))
    assert rc == 0
    assert (root / "environments" / "fake_env.py").read_text() == VALID_ENV_SOURCE_V1
    assert (root / "envs" / "fake_env" / "env_source.py").read_text() == VALID_ENV_SOURCE_V1


def test_name_mode_syncs_registered_source(tmp_path: Path):
    root = _build_fake_root(tmp_path)
    # Maintainer edits the registered source directly.
    registered = root / "environments" / "fake_env.py"
    registered.write_text(VALID_ENV_SOURCE_V2)
    # Worker venv still has the stale (v1) copy.
    worker = root / "envs" / "fake_env" / "env_source.py"
    worker.write_text(VALID_ENV_SOURCE_V1)

    rc = cmd_sync(SimpleNamespace(source="fake_env", root=str(root), with_scion=False))
    assert rc == 0
    assert worker.read_text() == VALID_ENV_SOURCE_V2


def test_all_envs_mode(tmp_path: Path):
    root = _build_fake_root(tmp_path, env_name="env_a")
    # Add a second env to the same root.
    (root / "envs" / "env_b" / "bin").mkdir(parents=True)
    (root / "envs" / "env_b" / "bin" / "python").touch()
    (root / "environments" / "env_a.py").write_text(VALID_ENV_SOURCE_V2)
    (root / "environments" / "env_b.py").write_text(VALID_ENV_SOURCE_V2)
    # Make worker copies stale on both.
    (root / "envs" / "env_a" / "env_source.py").write_text(VALID_ENV_SOURCE_V1)
    (root / "envs" / "env_b" / "env_source.py").write_text(VALID_ENV_SOURCE_V1)

    rc = cmd_sync(SimpleNamespace(source=None, root=str(root), with_scion=False))
    assert rc == 0
    assert (root / "envs" / "env_a" / "env_source.py").read_text() == VALID_ENV_SOURCE_V2
    assert (root / "envs" / "env_b" / "env_source.py").read_text() == VALID_ENV_SOURCE_V2


def test_name_mode_unregistered_env_errors(tmp_path: Path, capsys):
    root = _build_fake_root(tmp_path)
    rc = cmd_sync(SimpleNamespace(source="never_registered", root=str(root), with_scion=False))
    assert rc == 1
    err = capsys.readouterr().err
    assert "not registered" in err


def test_missing_worker_venv_is_reported_and_fails(tmp_path: Path, capsys):
    root = tmp_path / "scion_root"
    (root / "environments").mkdir(parents=True)
    (root / "environments" / "fake_env.py").write_text(VALID_ENV_SOURCE_V1)
    # No envs/ dir at all.

    rc = cmd_sync(SimpleNamespace(source="fake_env", root=str(root), with_scion=False))
    assert rc == 1
    err = capsys.readouterr().err
    assert "worker venv missing" in err


def test_invalid_source_rejected(tmp_path: Path, capsys):
    root = _build_fake_root(tmp_path)
    bad = tmp_path / "broken.py"
    bad.write_text("def not_setup(): pass\n")  # no PEP 723, no setup, no CAPABILITIES

    rc = cmd_sync(SimpleNamespace(source=str(bad), root=str(root), with_scion=False))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error" in err


@pytest.mark.parametrize("env_count", [1, 3])
def test_partial_failure_returns_nonzero_but_continues(
    tmp_path: Path, env_count: int, capsys
):
    """If some envs are unbuildable, others should still sync."""
    root = tmp_path / "scion_root"
    (root / "environments").mkdir(parents=True)
    for i in range(env_count):
        name = f"env_{i}"
        (root / "environments" / f"{name}.py").write_text(VALID_ENV_SOURCE_V1)
        if i == 0:
            (root / "envs" / name / "bin").mkdir(parents=True)
            (root / "envs" / name / "bin" / "python").touch()
        # Other envs are registered but not built; sync should skip them.

    rc = cmd_sync(SimpleNamespace(source=None, root=str(root), with_scion=False))
    # If any envs are unbuilt, we expect rc=1; if only env_0 exists and it
    # syncs successfully, rc=0.
    if env_count == 1:
        assert rc == 0
    else:
        assert rc == 1
    # In either case, env_0's worker copy should exist.
    assert (root / "envs" / "env_0" / "env_source.py").exists()
