"""
Tests for the hostname-based profile fallback.

The previous behavior was strict-match: a user with a per-project root
(not the maintainer's shared path) lost all cluster metadata —
including ``cuda_driver_max``, which is the entire point of the
torch/CUDA pre-flight. ``get_profile_for_root_or_host`` adds a hostname
fallback so site checks fire regardless of where the user keeps their
install.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from scion.clusters import get_profile_for_root, get_profile_for_root_or_host


def test_root_match_takes_precedence_over_hostname(tmp_path, monkeypatch):
    profiles_file = tmp_path / "clusters.toml"
    profiles_file.write_text(
        textwrap.dedent(
            f"""\
            [clusters.bymatch]
            root = "{tmp_path}/match"
            cuda_driver_max = "12.4"

            [clusters.byhost]
            root = "/nowhere/else"
            hostname_patterns = ["*"]
            cuda_driver_max = "12.8"
            """
        )
    )
    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(profiles_file))

    matched = get_profile_for_root_or_host(tmp_path / "match")
    assert matched is not None
    assert matched.name == "bymatch"
    assert matched.cuda_driver_max == "12.4"


def test_hostname_fallback_when_root_does_not_match(tmp_path: Path, monkeypatch):
    """Custom root + hostname-pattern profile → fallback wins."""
    profiles_file = tmp_path / "clusters.toml"
    profiles_file.write_text(
        textwrap.dedent(
            """\
            [clusters.fakelocal]
            root = "/nowhere/that/exists"
            hostname_patterns = ["*"]
            cuda_driver_max = "12.8"
            """
        )
    )
    monkeypatch.setenv("SCION_CLUSTERS_FILE", str(profiles_file))

    custom_root = tmp_path / "my_own_scion"
    custom_root.mkdir()

    # Strict lookup misses; fallback finds via hostname pattern "*".
    assert get_profile_for_root(custom_root) is None
    fallback = get_profile_for_root_or_host(custom_root)
    assert fallback is not None
    assert fallback.name == "fakelocal"
    assert fallback.cuda_driver_max == "12.8"
