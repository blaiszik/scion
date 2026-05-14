"""
Tests for structured ScionSession exceptions and worker stderr capture.

Verify that:

* When the worker subprocess crashes before completing the setup_done
  handshake, the session raises ``WorkerSetupFailed`` and the exception
  message includes the tail of the worker's stderr.
* When a session method raises inside the worker, the session raises
  ``WorkerMethodError``.

Both rely on a fake "env" (an env_source.py we hand-write at a faked
worker venv path) plus the real session machinery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scion
from scion.errors import WorkerMethodError, WorkerSetupFailed
from scion.session import ScionSession


def _build_fake_env(tmp_path: Path, env_name: str, source: str) -> Path:
    root = tmp_path / "scion_root"
    env_dir = root / "envs" / env_name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(source)
    return root


@pytest.fixture
def scion_on_pythonpath(monkeypatch):
    """
    Put the repo's scion package on the worker's PYTHONPATH.

    The worker's bin/python is a bare symlink to sys.executable; in CI it
    lives outside a venv with pyvenv.cfg, so site-packages from the dev
    venv aren't visible. A real ``scion install``-built venv has scion in
    its own site-packages, so this fixture only matters for tests using a
    symlinked python.
    """
    scion_parent = str(Path(scion.__file__).resolve().parent.parent)
    existing = ""
    import os

    if "PYTHONPATH" in os.environ:
        existing = os.pathsep + os.environ["PYTHONPATH"]
    monkeypatch.setenv("PYTHONPATH", scion_parent + existing)


ENV_THAT_FAILS_SETUP = """\
import sys
CAPABILITIES = ["fold"]

def setup(model, device="cpu"):
    print("HELLO_FROM_SETUP_STDERR_TAIL", file=sys.stderr, flush=True)
    raise RuntimeError("synthetic setup failure")
"""


ENV_WHOSE_METHOD_RAISES = """\
CAPABILITIES = ["fold"]

class Provider:
    def fold(self, sequence, **kw):
        raise ValueError("synthetic fold failure: " + sequence)

def setup(model, device="cpu"):
    return Provider()
"""


def test_setup_failure_raises_worker_setup_failed_with_stderr(
    tmp_path: Path, scion_on_pythonpath
):
    root = _build_fake_env(tmp_path, "broken_env", ENV_THAT_FAILS_SETUP)
    session = ScionSession(
        env_name="broken_env",
        required_capability="fold",
        root=root,
        device="cpu",
        timeout=15.0,
    )
    with pytest.raises(WorkerSetupFailed) as excinfo:
        session.start()
    err = excinfo.value
    # The structured exception should carry the worker exit code...
    assert err.returncode is not None and err.returncode != 0
    # ...and a tail of the worker's stderr, which includes the setup error.
    combined = str(err) + "\n" + err.stderr
    assert "HELLO_FROM_SETUP_STDERR_TAIL" in combined
    assert "synthetic setup failure" in combined
    session.stop()


def test_method_error_inside_worker_raises_worker_method_error(
    tmp_path: Path, scion_on_pythonpath
):
    root = _build_fake_env(tmp_path, "raise_env", ENV_WHOSE_METHOD_RAISES)
    with ScionSession(
        env_name="raise_env",
        required_capability="fold",
        root=root,
        device="cpu",
        timeout=15.0,
    ) as session:
        with pytest.raises(WorkerMethodError) as excinfo:
            session.call("fold", {"sequence": "MKT"})
    assert "synthetic fold failure: MKT" in str(excinfo.value)
