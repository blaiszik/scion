"""
Tests for the auto-preload validation step folded into ``scion install``.

We don't run a full ``scion install`` (it spawns uv, downloads wheels,
takes minutes). Instead we test ``_run_preload_validation`` directly
against a faked worker venv layout: a symlinked python and a hand-rolled
env_source.py. Mirrors the pattern used in test_check_thorough.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scion.commands.install import _run_preload_validation

ENV_WITH_PRELOAD = """\
CAPABILITIES = ["embed"]

class Provider:
    def embed(self, sequences, **kw):
        return None
    def preload(self):
        print("PRELOAD_RAN_OK", flush=True)

def setup(model, device="cpu"):
    return Provider()
"""

ENV_PRELOAD_RAISES = """\
CAPABILITIES = ["embed"]

class Provider:
    def embed(self, sequences, **kw):
        return None
    def preload(self):
        raise RuntimeError("validation failure inside preload")

def setup(model, device="cpu"):
    return Provider()
"""

ENV_NO_PRELOAD = """\
CAPABILITIES = ["embed"]

class Provider:
    def embed(self, sequences, **kw):
        return None

def setup(model, device="cpu"):
    return Provider()
"""


def _make_fake_env(tmp_path: Path, env_name: str, source: str) -> tuple[Path, Path, Path]:
    root = tmp_path / "scion_root"
    env_target = root / "envs" / env_name
    (env_target / "bin").mkdir(parents=True)
    env_python = env_target / "bin" / "python"
    env_python.symlink_to(sys.executable)
    (env_target / "env_source.py").write_text(source)
    return root, env_target, env_python


def test_install_preload_succeeds_when_provider_has_preload(tmp_path, capfd):
    root, env_target, env_python = _make_fake_env(tmp_path, "ok_env", ENV_WITH_PRELOAD)
    rc = _run_preload_validation(
        root=root,
        env_name="ok_env",
        env_target=env_target,
        env_python=env_python,
        verbose=True,
    )
    out = capfd.readouterr()
    assert rc == 0, out
    assert "PRELOAD_RAN_OK" in (out.out + out.err)


def test_install_preload_fails_when_preload_raises(tmp_path, capfd):
    root, env_target, env_python = _make_fake_env(
        tmp_path, "bad_env", ENV_PRELOAD_RAISES
    )
    rc = _run_preload_validation(
        root=root,
        env_name="bad_env",
        env_target=env_target,
        env_python=env_python,
        verbose=False,
    )
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert rc != 0
    # Failure path always echoes worker output, even when not verbose.
    assert "validation failure inside preload" in combined


def test_install_preload_notes_when_provider_lacks_preload(tmp_path, capfd):
    root, env_target, env_python = _make_fake_env(tmp_path, "noprl", ENV_NO_PRELOAD)
    rc = _run_preload_validation(
        root=root,
        env_name="noprl",
        env_target=env_target,
        env_python=env_python,
        verbose=True,
    )
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert rc == 0, combined
    assert "no provider.preload()" in combined
