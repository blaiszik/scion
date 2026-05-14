"""Tests for ``scion check --all-envs`` and ``scion doctor --all-envs``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _fake_env(root: Path, env_name: str, source: str) -> None:
    env_dir = root / "envs" / env_name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(source)


ENV_OK = """\
CAPABILITIES = ["embed"]
class P:
    def embed(self, sequences, **kw): return None
def setup(model, device="cpu"): return P()
"""

ENV_BROKEN = """\
CAPABILITIES = ["embed"]
def setup(model, device="cpu"):
    raise RuntimeError("intentionally broken env")
"""


def test_check_all_envs_runs_each_and_reports_failures(tmp_path: Path):
    root = tmp_path / "scion_root"
    for name in ("env_ok_a", "env_ok_b", "env_bad"):
        _fake_env(root, name, ENV_BROKEN if name == "env_bad" else ENV_OK)

    proc = subprocess.run(
        [
            sys.executable, "-m", "scion.cli", "check", "--all-envs",
            "--root", str(root), "--device", "cpu",
        ],
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    # Each env name appears in the summary.
    for name in ("env_ok_a", "env_ok_b", "env_bad"):
        assert name in out, out
    # Aggregate exit code reflects that env_bad failed.
    assert proc.returncode != 0
    assert "Failed: env_bad" in out


def test_doctor_all_envs_iterates_built_envs(tmp_path: Path):
    root = tmp_path / "scion_root"
    for sub in ("environments", "envs", "cache", "home", ".python"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in ("env_one", "env_two"):
        _fake_env(root, name, ENV_OK)

    proc = subprocess.run(
        [
            sys.executable, "-m", "scion.cli", "doctor", "--all-envs",
            "--root", str(root),
        ],
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    assert "env: env_one" in out
    assert "env: env_two" in out
    assert "Overall:" in out
