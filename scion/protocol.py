"""
Scion wire protocol.

A small length-prefixed JSON+blob request/response framing over a Unix
domain socket. Replaces Rootstock's i-PI protocol; the latter is shaped
for tight numerical loops (energies/forces in atomic units), while
protein FM calls are one-shot and ship strings (FASTA, mmCIF) and
binary blobs (MSA, embeddings) of arbitrary size.

Frame layout
------------
    [4-byte big-endian length N]
    [header_size: 4-byte big-endian]
    [header_json: header_size bytes, UTF-8 JSON]
    [blob_0 bytes][blob_1 bytes]...

The outer length N covers header_size + header_json + all blob bytes,
so receivers can read the whole frame in one shot before parsing.

Header schema
-------------
Request:
    {
        "id":     <int>,       # client-assigned correlation id
        "method": <str>,        # e.g. "fold", "embed", "health", "shutdown"
        "args":   <obj>,        # JSON-serializable kwargs
        "blobs":  [             # ordered list of attached binary blobs
            {"name": "msa", "size": <int>, "dtype": "bytes"},
            ...
        ]
    }

Reply:
    {
        "id":     <int>,        # echoes the request id
        "ok":     <bool>,
        "result": <obj> | null, # JSON-serializable result
        "blobs":  [...],        # same shape as request
        "error":  <str> | null  # set when ok=false
    }

Numpy arrays travel as blobs with metadata in `result["arrays"]`:
    "arrays": {"per_residue": {"shape": [1,22,1280], "dtype": "float32", "blob": 0}}
where "blob" is an index into the blobs list.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any


class SocketClosed(Exception):
    """Raised when socket connection is closed mid-frame."""


# Frame size limits. The outer length is a 32-bit unsigned int so the
# protocol-imposed max is 4 GiB; we cap lower to avoid trivial DoS.
MAX_FRAME_BYTES = 1 << 32  # 4 GiB hard limit
MAX_HEADER_BYTES = 16 << 20  # 16 MiB for the JSON header


@dataclass
class Frame:
    """One parsed request or reply frame."""

    header: dict[str, Any]
    blobs: list[bytes] = field(default_factory=list)


def _recvall(sock: socket.socket, nbytes: int) -> bytes:
    """Receive exactly nbytes, handling partial reads."""
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if len(chunk) == 0:
            raise SocketClosed("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, header: dict, blobs: list[bytes] | None = None) -> None:
    """Encode and send one frame on the given socket."""
    blobs = blobs or []

    # Stamp blob metadata into the header so the receiver can split.
    blob_meta = []
    for i, blob in enumerate(blobs):
        existing = header.get("blobs", [])
        if i < len(existing) and isinstance(existing[i], dict):
            meta = dict(existing[i])
            meta["size"] = len(blob)
            meta.setdefault("name", f"blob_{i}")
            blob_meta.append(meta)
        else:
            blob_meta.append({"name": f"blob_{i}", "size": len(blob)})
    header = {**header, "blobs": blob_meta}

    header_bytes = json.dumps(header).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {len(header_bytes)} bytes")

    payload_size = 4 + len(header_bytes) + sum(len(b) for b in blobs)
    if payload_size > MAX_FRAME_BYTES:
        raise ValueError(f"Frame too large: {payload_size} bytes")

    sock.sendall(payload_size.to_bytes(4, "big"))
    sock.sendall(len(header_bytes).to_bytes(4, "big"))
    sock.sendall(header_bytes)
    for blob in blobs:
        sock.sendall(blob)


def recv_frame(sock: socket.socket) -> Frame:
    """Receive and decode one frame from the given socket."""
    size_bytes = _recvall(sock, 4)
    payload_size = int.from_bytes(size_bytes, "big")
    if payload_size > MAX_FRAME_BYTES:
        raise ValueError(f"Incoming frame too large: {payload_size} bytes")

    header_size_bytes = _recvall(sock, 4)
    header_size = int.from_bytes(header_size_bytes, "big")
    if header_size > MAX_HEADER_BYTES:
        raise ValueError(f"Incoming header too large: {header_size} bytes")

    header_bytes = _recvall(sock, header_size)
    header = json.loads(header_bytes.decode("utf-8"))

    blobs: list[bytes] = []
    for meta in header.get("blobs", []):
        size = int(meta.get("size", 0))
        if size > 0:
            blobs.append(_recvall(sock, size))
        else:
            blobs.append(b"")

    consumed = 4 + header_size + sum(len(b) for b in blobs)
    if consumed != payload_size:
        # Drain any unread payload to keep the stream aligned, then complain.
        excess = payload_size - consumed
        if excess > 0:
            _recvall(sock, excess)
        raise ValueError(
            f"Frame size mismatch: declared {payload_size}, consumed {consumed}"
        )

    return Frame(header=header, blobs=blobs)


# -----------------------------------------------------------------------------
# Socket helpers (Unix domain only — single-node operation)
# -----------------------------------------------------------------------------

def create_unix_socket_path(name: str) -> str:
    """Create path for Unix domain socket."""
    return f"/tmp/scion_{name}_{os.getpid()}"


def create_server_socket(socket_path: str, timeout: float | None = None) -> socket.socket:
    """Create and bind a Unix domain socket server."""
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    if timeout is not None:
        sock.settimeout(timeout)
    return sock


def connect_unix_socket(
    socket_path: str,
    timeout: float | None = None,
    max_retries: int = 50,
    retry_delay: float = 0.1,
) -> socket.socket:
    """Connect to a Unix domain socket server with retries."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout is not None:
        sock.settimeout(timeout)

    for attempt in range(max_retries):
        try:
            sock.connect(socket_path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    raise RuntimeError(f"Failed to connect to {socket_path} after {max_retries} attempts")
