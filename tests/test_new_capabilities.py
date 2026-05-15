"""
Tests for the new ``design_sequence`` and ``dock`` capabilities.

The model wiring for ligandmpnn_env and diffdock_env is scaffolded; these
tests pin the *call surface*: the capability registry, dataclass round-
trips, and the env files' PEP 723 + lint cleanliness. Inference-path
coverage waits on a real env build.
"""

from __future__ import annotations

from pathlib import Path

from scion.capabilities import CAPABILITIES, DesignResult, DockResult
from scion.env_lint import lint_environment_imports
from scion.pep723 import get_capabilities, parse_pep723_metadata

REPO_ENVIRONMENTS = Path(__file__).resolve().parent.parent / "environments"


def test_capabilities_registry_lists_new_methods():
    assert "design_sequence" in CAPABILITIES
    assert "dock" in CAPABILITIES


def test_design_result_round_trip():
    payload = {
        "sequence": "MKTA",
        "sequences": ["MKTA", "MKTAV"],
        "scores": [0.42, 0.31],
        "masked_positions": [12, 14],
        "extras_passthrough": "yes",
    }
    r = DesignResult.from_dict(payload)
    assert r.sequence == "MKTA"
    assert r.sequences == ["MKTA", "MKTAV"]
    assert r.scores == [0.42, 0.31]
    assert r.masked_positions == [12, 14]
    assert r.extras["extras_passthrough"] == "yes"


def test_design_result_falls_back_when_only_sequences_given():
    r = DesignResult.from_dict({"sequences": ["MKTA"]})
    assert r.sequence == "MKTA"


def test_dock_result_round_trip():
    payload = {
        "poses": ["sdf1", "sdf2"],
        "scores": [0.81, 0.62],
        "mmcif": "data_x\nloop_\n",
        "diffdock_extra": 1,
    }
    r = DockResult.from_dict(payload)
    assert r.poses == ["sdf1", "sdf2"]
    assert r.scores == [0.81, 0.62]
    assert r.mmcif.startswith("data_x")
    assert r.extras["diffdock_extra"] == 1


def test_ligandmpnn_env_file_declares_design_capability():
    path = REPO_ENVIRONMENTS / "ligandmpnn_env.py"
    assert get_capabilities(path) == ["design_sequence"]
    meta = parse_pep723_metadata(path.read_text())
    assert meta is not None
    deps = " ".join(meta.get("dependencies", []))
    assert "torch" in deps
    assert "ligandmpnn" in deps


def test_diffdock_env_file_declares_dock_capability():
    path = REPO_ENVIRONMENTS / "diffdock_env.py"
    assert get_capabilities(path) == ["dock"]
    meta = parse_pep723_metadata(path.read_text())
    assert meta is not None
    deps = " ".join(meta.get("dependencies", []))
    assert "torch" in deps
    assert "diffdock" in deps


def test_new_env_files_lint_clean():
    """The scaffold env files must lint clean (no undeclared imports)."""
    for env_file in ("ligandmpnn_env.py", "diffdock_env.py"):
        warnings = lint_environment_imports(REPO_ENVIRONMENTS / env_file)
        assert warnings == [], f"{env_file}: {warnings}"


def test_new_envs_cap_torch_for_polaris_drivers():
    """Same regression guard as the other shipped envs: torch < 2.9."""
    for env_file in ("ligandmpnn_env.py", "diffdock_env.py"):
        meta = parse_pep723_metadata((REPO_ENVIRONMENTS / env_file).read_text())
        deps = meta.get("dependencies", [])
        torch_spec = next((d for d in deps if d.startswith("torch")), None)
        assert torch_spec is not None, f"{env_file} missing torch dep"
        assert "<2.9" in torch_spec or "<= 2.8" in torch_spec, (
            f"{env_file}'s torch pin {torch_spec!r} missing the <2.9 cap"
        )
