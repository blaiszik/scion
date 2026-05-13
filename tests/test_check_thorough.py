"""
Tests for ``scion check --thorough``.

Verifies that the thorough flag actually invokes ``provider.preload()``
when present, returns a clear message when it's absent, and surfaces
the exception when preload() raises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _build_root_with_env(tmp_path: Path, env_name: str, source: str) -> Path:
    """Fake a built worker venv: real python, real env_source.py."""
    root = tmp_path / "scion_root"
    env_dir = root / "envs" / env_name
    (env_dir / "bin").mkdir(parents=True)
    # Use the system python as the worker's python so subprocess.run works.
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(source)
    return root


ENV_WITH_PRELOAD = """\
CAPABILITIES = ["fold"]

class Provider:
    def fold(self, sequence, **kw):
        return {"mmcif": "stub"}
    def preload(self):
        print("[provider] preload was called", flush=True)

def setup(model, device="cpu"):
    return Provider()
"""

ENV_WITHOUT_PRELOAD = """\
CAPABILITIES = ["embed"]

class Provider:
    def embed(self, sequences, **kw):
        return {"per_residue": None}

def setup(model, device="cpu"):
    return Provider()
"""

ENV_PRELOAD_RAISES = """\
CAPABILITIES = ["fold"]

class Provider:
    def fold(self, sequence, **kw):
        return {"mmcif": "stub"}
    def preload(self):
        raise RuntimeError("kaboom from provider.preload()")

def setup(model, device="cpu"):
    return Provider()
"""


def _run_check(tmp_path, env_name, source, thorough: bool) -> tuple[int, str]:
    """
    Run `scion check` as a subprocess against a freshly-built fake env.

    We invoke via the CLI rather than calling cmd_check() directly because
    cmd_check spawns a worker subprocess to exec the inline script — and
    capturing the worker's stdout/stderr cleanly through cmd_check's own
    print() + subprocess.run path is simplest via an outer subprocess.
    """
    root = _build_root_with_env(tmp_path, env_name, source)
    cmd = [
        sys.executable, "-m", "scion.cli", "check", env_name,
        "--root", str(root),
        "--device", "cpu",
    ]
    if thorough:
        cmd.append("--thorough")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_thorough_calls_preload_when_defined(tmp_path: Path):
    rc, out = _run_check(tmp_path, "fold_env", ENV_WITH_PRELOAD, thorough=True)
    assert rc == 0, out
    assert "calling provider.preload()" in out
    assert "[provider] preload was called" in out
    assert "provider.preload() OK" in out


def test_thorough_explains_when_preload_missing(tmp_path: Path):
    rc, out = _run_check(tmp_path, "embed_env", ENV_WITHOUT_PRELOAD, thorough=True)
    assert rc == 0, out
    assert "no provider.preload()" in out
    # Still confirms the setup-only check passed
    assert "OK: provider" in out


def test_thorough_surfaces_preload_exception(tmp_path: Path):
    rc, out = _run_check(tmp_path, "broken_env", ENV_PRELOAD_RAISES, thorough=True)
    assert rc != 0, out
    assert "kaboom" in out
    assert "provider.preload() raised" in out


def test_default_check_does_not_call_preload(tmp_path: Path):
    """Without --thorough, preload() must not be called even when defined."""
    rc, out = _run_check(tmp_path, "fold_env", ENV_WITH_PRELOAD, thorough=False)
    assert rc == 0, out
    assert "[provider] preload was called" not in out
    assert "calling provider.preload()" not in out
