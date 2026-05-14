"""
Shared session machinery for capability clients.

A ScionSession owns the Unix socket and the worker subprocess. It is
*not* an RPC server in the network sense — it listens on a local socket
just to accept exactly one worker connection. Once connected, the
session can call any method advertised in the worker's ``setup_done``
frame.

Each capability client (``Folder``, ``Embedder``, ...) holds a
ScionSession and exposes type-safe wrappers around ``.call()`` that
match each capability's contract.
"""

from __future__ import annotations

import collections
import itertools
import os
import socket
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from .clusters import get_root_for_cluster
from .errors import WorkerMethodError, WorkerProcessDied, WorkerSetupFailed
from .protocol import (
    Frame,
    SocketClosed,
    create_server_socket,
    create_unix_socket_path,
    recv_frame,
    send_frame,
)


class _StderrBuffer:
    """
    Bounded ring buffer fed by a background reader thread.

    When the session pipes worker stderr (no ``log`` file was provided),
    we drain that pipe continuously so a chatty worker can't block on a
    full kernel pipe buffer, and so the tail is available to attach to
    any exception we raise.
    """

    def __init__(self, max_lines: int = 200):
        self._lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stream = None

    def start(self, stream) -> None:
        self._stream = stream
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="scion-worker-stderr",
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            for line in iter(self._stream.readline, b""):
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    text = repr(line)
                with self._lock:
                    self._lines.append(text)
        except (ValueError, OSError):
            # Stream was closed mid-readline; normal at shutdown.
            pass

    def snapshot(self) -> str:
        with self._lock:
            return "".join(self._lines)

    def stop(self, timeout: float = 1.0) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._stream = None


class ScionSession:
    """Owns the worker subprocess and brokers RPC calls."""

    def __init__(
        self,
        env_name: str,
        required_capability: str,
        model: str | None = None,
        checkpoint: str | None = None,
        device: str = "cuda",
        cluster: str | None = None,
        root: Path | str | None = None,
        socket_name: str | None = None,
        log=None,
        timeout: float = 600.0,
    ):
        if cluster is None and root is None:
            raise ValueError("Either `cluster` or `root` is required")

        if root is None:
            root = get_root_for_cluster(cluster)

        self.root = Path(root)
        self.env_name = env_name
        self.required_capability = required_capability
        self.model = model or env_name.replace("_env", "")
        self.checkpoint = checkpoint or "default"
        self.device = device
        self.timeout = timeout
        self.log = log
        self._stderr_buffer = _StderrBuffer()

        self.socket_name = socket_name or f"{env_name}_{uuid.uuid4().hex[:8]}"
        runtime_dir = os.environ.get("SCION_RUNTIME_DIR")
        if runtime_dir is None:
            from .clusters import get_profile_for_root

            profile = get_profile_for_root(self.root)
            if profile is not None and profile.runtime_dir:
                runtime_dir = str(Path(os.path.expandvars(profile.runtime_dir)).expanduser())
        self.socket_path = create_unix_socket_path(self.socket_name, runtime_dir=runtime_dir)

        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._env_manager = None
        self._wrapper_path: Path | None = None

        self._counter = itertools.count(start=1)
        self._lock = threading.Lock()
        self._capabilities: list[str] = []
        self._connected = False

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Bind the socket, spawn the worker, accept its connection."""
        self._server_socket = create_server_socket(self.socket_path, timeout=self.timeout)
        self._server_socket.listen(1)

        if self.log:
            print(f"Session listening on {self.socket_path}", file=self.log, flush=True)

        self._spawn_worker()
        self._accept_connection()
        self._receive_setup_done()

        if self.required_capability not in self._capabilities:
            self.stop()
            raise RuntimeError(
                f"Environment {self.env_name!r} does not advertise capability "
                f"{self.required_capability!r}. Advertised: {self._capabilities}"
            )

        self._connected = True

    def _spawn_worker(self) -> None:
        from .environment import EnvironmentManager

        self._env_manager = EnvironmentManager(root=self.root)
        self._wrapper_path = self._env_manager.generate_wrapper(
            env_name=self.env_name,
            model=self.checkpoint,
            device=self.device,
            socket_path=self.socket_path,
        )

        cmd = self._env_manager.get_spawn_command(self.env_name, self._wrapper_path)
        env = self._env_manager.get_environment_variables()

        if self.log:
            print(f"Spawning: {' '.join(cmd)}", file=self.log, flush=True)

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE if not self.log else None,
            stderr=subprocess.PIPE if not self.log else None,
        )

        # Drain worker stderr in a background thread so it can't deadlock on
        # a full pipe and so the tail is available to attach to any
        # exception we raise. Skip when the caller passed a log file —
        # the worker is writing there directly.
        if not self.log and self._process.stderr is not None:
            self._stderr_buffer.start(self._process.stderr)

    def _accept_connection(self) -> None:
        self._server_socket.settimeout(1.0)
        while True:
            try:
                self._client_socket, _ = self._server_socket.accept()
                break
            except TimeoutError:
                if self._process.poll() is not None:
                    raise WorkerSetupFailed(
                        f"Worker exited before connecting to {self.socket_path}",
                        stderr=self._collect_stderr(),
                        returncode=self._process.returncode,
                    )

        self._server_socket.settimeout(self.timeout)
        self._client_socket.settimeout(self.timeout)

    def _collect_stderr(self) -> str:
        """
        Return everything currently buffered from worker stderr.

        If the worker has exited, drain the pipe to EOF first so the last
        traceback lines aren't missed.
        """
        if self._process is not None and self._process.poll() is not None:
            self._stderr_buffer.stop(timeout=2.0)
        return self._stderr_buffer.snapshot()

    def _receive_setup_done(self) -> None:
        try:
            frame = recv_frame(self._client_socket)
        except SocketClosed as e:
            raise WorkerSetupFailed(
                "Worker closed the socket before sending setup_done",
                stderr=self._collect_stderr(),
                returncode=self._process.poll() if self._process else None,
            ) from e
        if frame.header.get("method") != "setup_done":
            raise WorkerSetupFailed(
                f"Expected setup_done frame, got method={frame.header.get('method')!r}",
                stderr=self._collect_stderr(),
            )
        self._capabilities = list(frame.header.get("args", {}).get("capabilities", []))

    def stop(self) -> None:
        """Send shutdown, close sockets, terminate worker, cleanup files."""
        if self._connected and self._client_socket is not None:
            try:
                self._send({"id": 0, "method": "shutdown", "args": {}})
                # Best-effort drain of the shutdown reply.
                try:
                    self._client_socket.settimeout(2.0)
                    recv_frame(self._client_socket)
                except (SocketClosed, TimeoutError, OSError):
                    pass
            except (BrokenPipeError, OSError):
                pass

        if self._client_socket is not None:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

        # Stop the stderr reader after the process has exited so any final
        # output is captured before the pipe closes.
        self._stderr_buffer.stop(timeout=1.0)

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        if self._wrapper_path is not None:
            try:
                self._wrapper_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._wrapper_path = None

        if self._env_manager is not None:
            self._env_manager.cleanup()
            self._env_manager = None

        self._connected = False

    # --- RPC -------------------------------------------------------------

    @property
    def capabilities(self) -> list[str]:
        return list(self._capabilities)

    def _send(self, header: dict, blobs: list[bytes] | None = None) -> None:
        send_frame(self._client_socket, header, blobs)

    def call(self, method: str, args: dict[str, Any] | None = None,
             blobs: list[bytes] | None = None) -> Frame:
        """
        Send a request and return the reply frame.

        Args travel as JSON in the header; blobs travel as raw bytes
        after the header. The worker dispatch will route blobs back
        into kwargs by their declared name.
        """
        if not self._connected:
            raise RuntimeError("Session not started. Call start() first or use as context manager.")

        args = args or {}
        blobs = blobs or []

        # Encode blob names so the worker can attach them by kwarg.
        blob_specs = []
        # If the caller passed bytes inline in args, hoist them out.
        for key, value in list(args.items()):
            if isinstance(value, (bytes, bytearray)):
                blob_specs.append({"name": key, "size": len(value), "dtype": "bytes"})
                blobs.append(bytes(value))
                args.pop(key)

        with self._lock:
            request_id = next(self._counter)
            header = {
                "id": request_id,
                "method": method,
                "args": args,
                "blobs": blob_specs,
            }
            try:
                self._send(header, blobs)
                reply = recv_frame(self._client_socket)
            except SocketClosed as e:
                raise WorkerProcessDied(
                    f"Worker socket closed during call to {method!r}",
                    stderr=self._collect_stderr(),
                    returncode=self._process.poll() if self._process else None,
                ) from e
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                raise WorkerProcessDied(
                    f"I/O error talking to worker during {method!r}: {e}",
                    stderr=self._collect_stderr(),
                    returncode=self._process.poll() if self._process else None,
                ) from e

        if int(reply.header.get("id", -1)) != request_id:
            raise RuntimeError(
                f"Reply id mismatch: expected {request_id}, got {reply.header.get('id')}"
            )
        if not reply.header.get("ok", False):
            raise WorkerMethodError(
                reply.header.get("error") or "Unknown worker error",
                stderr=self._stderr_buffer.snapshot(),
            )
        return reply

    # --- context manager ------------------------------------------------

    def __enter__(self) -> ScionSession:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False


def decode_result(frame: Frame) -> Any:
    """
    Inverse of worker.py:_to_jsonable. Reconstructs numpy arrays from
    blobs using the metadata in ``result["arrays"]``.
    """
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    payload = frame.header.get("result") or {}
    arrays = payload.get("arrays") or {}
    blobs = frame.blobs

    def restore(obj):
        if isinstance(obj, str) and obj in arrays:
            meta = arrays[obj]
            blob = blobs[int(meta["blob"])]
            if meta.get("dtype") == "bytes":
                return blob
            if np is None:
                raise RuntimeError(
                    "numpy is required to decode array results but is not installed"
                )
            return np.frombuffer(blob, dtype=meta["dtype"]).reshape(meta["shape"]).copy()
        if isinstance(obj, dict):
            return {k: restore(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [restore(v) for v in obj]
        return obj

    return restore(payload.get("value"))
