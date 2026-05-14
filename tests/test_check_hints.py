"""
Tests for the hint matcher attached to ``scion check`` failures.

Verifies the pure function ``_suggest_hints`` recognises each documented
paper-cut pattern, and that ``scion check`` emits the hint on a real
failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scion.commands.check import _suggest_hints


def test_suggest_hints_matches_nccl():
    hints = _suggest_hints("ImportError: ... undefined symbol: ncclGetXyz ...")
    assert len(hints) == 1
    assert "scion install" in hints[0]
    assert "--force" in hints[0]


def test_suggest_hints_matches_driver_too_old():
    hints = _suggest_hints(
        "RuntimeError: The NVIDIA driver on your system is too old (found version ...)"
    )
    assert any("torch<2.9" in h or "Cap torch" in h for h in hints)


def test_suggest_hints_matches_missing_cuequivariance():
    hints = _suggest_hints("ModuleNotFoundError: No module named 'cuequivariance_torch'")
    assert any("cuequivariance" in h for h in hints)


def test_suggest_hints_matches_libgomp():
    hints = _suggest_hints("libgomp: Thread creation failed: Resource temporarily unavailable")
    assert any("RLIMIT_NPROC" in h or "OMP_NUM_THREADS" in h for h in hints)


def test_suggest_hints_returns_empty_on_clean_output():
    assert _suggest_hints("setup() OK") == []


def test_suggest_hints_matches_hf_hub_failure():
    hints = _suggest_hints(
        "huggingface_hub.utils._errors.LocalEntryNotFoundError: An error happened"
    )
    assert any("scion preload" in h or "HTTPS_PROXY" in h for h in hints)


def test_suggest_hints_matches_disk_full():
    hints = _suggest_hints("OSError: [Errno 28] No space left on device")
    assert any("disk" in h or "cache" in h for h in hints)


def test_check_emits_hint_on_failing_env(tmp_path: Path):
    """End-to-end: a fake env whose setup raises the NCCL pattern triggers the hint."""
    root = tmp_path / "scion_root"
    env_dir = root / "envs" / "nccl_env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(
        "CAPABILITIES = ['embed']\n"
        "def setup(model, device='cpu'):\n"
        "    raise ImportError('libsomething.so: undefined symbol: ncclGet42')\n"
    )

    proc = subprocess.run(
        [
            sys.executable, "-m", "scion.cli", "check", "nccl_env",
            "--root", str(root), "--device", "cpu",
        ],
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "Hints:" in out
    assert "--force" in out
