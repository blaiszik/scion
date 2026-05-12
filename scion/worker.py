#!/usr/bin/env python
"""
Scion worker process.

Runs inside the isolated pre-built environment. The lifecycle:

1. Call the env's `setup(model, device)` once to get a *provider*.
2. Connect to the server's Unix socket.
3. Send a `setup_done` frame advertising supported capabilities.
4. Loop: receive a request frame, dispatch by `method` onto the provider,
   send a reply frame. Reply blobs carry binary returns (numpy arrays).
5. Exit cleanly on `shutdown` method or socket close.

The provider object is duck-typed: each capability is a method whose
name matches the capability id (`fold`, `embed`, ...) and which accepts
keyword arguments matching the capability contract in `capabilities.py`.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any

from .protocol import (
    Frame,
    SocketClosed,
    connect_unix_socket,
    recv_frame,
    send_frame,
)


def _to_jsonable(value: Any) -> tuple[Any, list[bytes]]:
    """
    Convert a provider return value into (json_result, blobs).

    Numpy arrays are serialized as blobs with shape/dtype metadata in a
    nested ``arrays`` dict keyed by their position in the result.
    Strings, numbers, bools, None, lists, and dicts pass through.
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        np = None  # type: ignore[assignment]

    blobs: list[bytes] = []
    arrays: dict[str, dict] = {}

    def encode(obj):
        if np is not None and isinstance(obj, np.ndarray):
            idx = len(blobs)
            blobs.append(obj.tobytes(order="C"))
            key = f"__array_{idx}__"
            arrays[key] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "blob": idx,
            }
            return key
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [encode(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, bytes):
            idx = len(blobs)
            blobs.append(obj)
            key = f"__bytes_{idx}__"
            arrays[key] = {"blob": idx, "dtype": "bytes"}
            return key
        # Last-resort: stringify (covers Path, enums, etc.). The client
        # will see a string instead of crashing the worker.
        return str(obj)

    encoded = encode(value)
    return {"value": encoded, "arrays": arrays}, blobs


class ScionWorker:
    """Worker that dispatches RPC method calls onto a capability provider."""

    def __init__(
        self,
        provider: Any,
        capabilities: list[str],
        socket_path: str,
        log=None,
    ):
        self.provider = provider
        self.capabilities = list(capabilities)
        self.socket_path = socket_path
        self.log = log
        self._socket = None

    def _log(self, msg: str) -> None:
        if self.log:
            print(f"[Worker] {msg}", file=self.log, flush=True)

    def _send_setup_done(self) -> None:
        send_frame(
            self._socket,
            {
                "id": 0,
                "method": "setup_done",
                "args": {"capabilities": self.capabilities},
            },
        )

    def _send_error(self, request_id: int, message: str) -> None:
        send_frame(
            self._socket,
            {"id": request_id, "ok": False, "result": None, "error": message},
        )

    def _send_result(self, request_id: int, value: Any) -> None:
        result, blobs = _to_jsonable(value)
        send_frame(
            self._socket,
            {"id": request_id, "ok": True, "result": result, "error": None},
            blobs=blobs,
        )

    def _dispatch(self, frame: Frame) -> bool:
        """Handle one request frame. Returns True to keep looping."""
        header = frame.header
        request_id = int(header.get("id", 0))
        method = header.get("method")
        args = header.get("args") or {}

        if method == "shutdown":
            self._send_result(request_id, {"ok": True})
            return False

        if method == "health":
            self._send_result(request_id, {"ok": True, "capabilities": self.capabilities})
            return True

        if method in self.capabilities:
            handler = getattr(self.provider, method, None)
            if handler is None:
                self._send_error(
                    request_id,
                    f"Provider missing handler for capability {method!r}",
                )
                return True

            # Re-attach incoming blobs by name into the args dict.
            blob_specs = header.get("blobs", [])
            for i, blob in enumerate(frame.blobs):
                name = blob_specs[i].get("name") if i < len(blob_specs) else f"blob_{i}"
                if name and name not in args:
                    args[name] = blob

            try:
                value = handler(**args)
            except Exception as e:
                self._send_error(request_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                return True

            self._send_result(request_id, value)
            return True

        self._send_error(request_id, f"Unknown method: {method!r}")
        return True

    def run(self) -> None:
        """Main loop: connect, advertise, dispatch."""
        self._log(f"Connecting to {self.socket_path}")
        self._socket = connect_unix_socket(self.socket_path)
        self._log("Connected")

        self._send_setup_done()
        self._log(f"setup_done sent; capabilities={self.capabilities}")

        try:
            while True:
                try:
                    frame = recv_frame(self._socket)
                except SocketClosed:
                    self._log("Server closed connection")
                    break

                keep_going = self._dispatch(frame)
                if not keep_going:
                    self._log("Received shutdown, exiting")
                    break
        finally:
            if self._socket:
                self._socket.close()
            self._log("Worker shutdown complete")


def run_worker(
    setup_fn: Callable[..., Any],
    capabilities: list[str],
    model: str,
    device: str,
    socket_path: str,
    log=None,
) -> None:
    """
    Entry point used by generated wrapper scripts.

    Calls ``setup_fn(model, device)`` once to build a capability provider,
    then enters the RPC dispatch loop.
    """
    log = log or sys.stderr
    print(f"[Worker] Calling setup({model!r}, {device!r})", file=log, flush=True)
    provider = setup_fn(model, device)
    print(f"[Worker] Provider loaded: {type(provider).__name__}", file=log, flush=True)

    worker = ScionWorker(
        provider=provider,
        capabilities=capabilities,
        socket_path=socket_path,
        log=log,
    )
    worker.run()
