"""
Round-trip tests for scion.protocol.

Uses ``socket.socketpair()`` so both ends live in-process; verifies that
a request frame (with blobs) and its reply (with a different blob shape)
serialize and deserialize cleanly, headers decode to identical dicts,
and blob bytes are bit-exact.
"""

from __future__ import annotations

import socket
import threading

from scion.protocol import recv_frame, send_frame


def _pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


def test_simple_request_no_blobs():
    a, b = _pair()
    try:
        send_frame(a, {"id": 1, "method": "health", "args": {}})
        frame = recv_frame(b)
        assert frame.header["id"] == 1
        assert frame.header["method"] == "health"
        assert frame.header["args"] == {}
        assert frame.blobs == []
    finally:
        a.close()
        b.close()


def test_request_with_named_blobs():
    a, b = _pair()
    msa_blob = b"# A3M\n>q\nACDEFGHIK\n" * 200
    cif_blob = b"# templates" + b"X" * 1024
    try:
        # Server-side: receive in a thread because socketpair is synchronous.
        result: dict = {}

        def server():
            result["frame"] = recv_frame(b)

        t = threading.Thread(target=server)
        t.start()

        send_frame(
            a,
            {
                "id": 42,
                "method": "fold",
                "args": {"sequence": "MKT...", "num_recycles": 3},
                "blobs": [
                    {"name": "msa", "size": 0, "dtype": "bytes"},
                    {"name": "templates", "size": 0, "dtype": "bytes"},
                ],
            },
            blobs=[msa_blob, cif_blob],
        )

        t.join(timeout=5.0)
        frame = result["frame"]

        assert frame.header["id"] == 42
        assert frame.header["method"] == "fold"
        assert frame.header["args"]["sequence"] == "MKT..."
        assert frame.header["args"]["num_recycles"] == 3
        # Blob sizes get filled in by send_frame
        assert frame.header["blobs"][0]["name"] == "msa"
        assert frame.header["blobs"][0]["size"] == len(msa_blob)
        assert frame.header["blobs"][1]["name"] == "templates"
        assert frame.header["blobs"][1]["size"] == len(cif_blob)
        assert frame.blobs == [msa_blob, cif_blob]
    finally:
        a.close()
        b.close()


def test_reply_with_array_blob():
    """Simulate a worker reply that ships a numpy-style array as a blob."""
    a, b = _pair()
    fake_array = b"\x00" * (4 * 22 * 1280)  # float32 (1, 22, 1280) ≈ 110 KiB
    try:
        result: dict = {}

        def server():
            result["frame"] = recv_frame(b)

        t = threading.Thread(target=server)
        t.start()

        send_frame(
            a,
            {
                "id": 7,
                "ok": True,
                "error": None,
                "result": {
                    "value": {
                        "per_residue": "__array_0__",
                    },
                    "arrays": {
                        "__array_0__": {
                            "shape": [1, 22, 1280],
                            "dtype": "float32",
                            "blob": 0,
                        }
                    },
                },
            },
            blobs=[fake_array],
        )

        t.join(timeout=5.0)
        frame = result["frame"]

        assert frame.header["id"] == 7
        assert frame.header["ok"] is True
        assert frame.header["result"]["value"]["per_residue"] == "__array_0__"
        meta = frame.header["result"]["arrays"]["__array_0__"]
        assert meta["shape"] == [1, 22, 1280]
        assert meta["dtype"] == "float32"
        assert len(frame.blobs[0]) == len(fake_array)
        assert frame.blobs[0] == fake_array
    finally:
        a.close()
        b.close()


def test_large_header_under_limit():
    """A multi-megabyte JSON header (think: long FASTA) round-trips fine."""
    a, b = _pair()
    long_seq = "M" + "A" * (1 << 20)  # 1 MiB sequence in the header
    try:
        result: dict = {}

        def server():
            result["frame"] = recv_frame(b)

        t = threading.Thread(target=server)
        t.start()

        send_frame(
            a,
            {"id": 1, "method": "fold", "args": {"sequence": long_seq}},
        )

        t.join(timeout=10.0)
        assert result["frame"].header["args"]["sequence"] == long_seq
    finally:
        a.close()
        b.close()
