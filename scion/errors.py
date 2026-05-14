"""
Structured exceptions raised by ScionSession.

These distinguish three common failure modes:

* ``WorkerSetupFailed`` — the worker subprocess crashed during the
  initial ``setup(model, device)`` call and never produced the
  ``setup_done`` handshake. Usually a missing dep, a bad checkpoint,
  or a CUDA/driver mismatch.
* ``WorkerProcessDied`` — the worker subprocess exited unexpectedly
  *after* the handshake. Often OOM, a SIGKILL from the scheduler, or
  a segfault in a CUDA/numerical kernel.
* ``WorkerMethodError`` — a method call raised inside the worker. The
  worker captured the traceback and sent it across as a structured
  error reply.

All three subclass ``ScionWorkerError`` and ``RuntimeError`` so existing
callers that ``except RuntimeError`` keep working. Each carries the last
slice of worker stderr captured by the session and the exit code if the
process died.
"""

from __future__ import annotations


class ScionWorkerError(RuntimeError):
    """Base class for worker-side failures surfaced through the session."""

    def __init__(
        self,
        message: str,
        stderr: str = "",
        returncode: int | None = None,
    ):
        self.short_message = message
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(self._format(message, stderr, returncode))

    @staticmethod
    def _format(message: str, stderr: str, returncode: int | None) -> str:
        parts = [message]
        if returncode is not None:
            parts.append(f"(worker exit code: {returncode})")
        tail = stderr.rstrip()
        if tail:
            parts.append("--- worker stderr (tail) ---")
            parts.append(tail)
        return "\n".join(parts)


class WorkerSetupFailed(ScionWorkerError):
    """Worker died before completing the setup_done handshake."""


class WorkerProcessDied(ScionWorkerError):
    """Worker subprocess exited after handshake; mid-call socket closed."""


class WorkerMethodError(ScionWorkerError):
    """A method dispatched to the worker raised inside the worker."""
