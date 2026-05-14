"""
Tests for the seed-cluster.toml / bundled-env install paths in scion init.

We don't run the full ``cmd_init`` (it prompts interactively); we test the
two helper functions and the resources module directly.
"""

from __future__ import annotations

from pathlib import Path

from scion.commands.init import _check_root_writable, _maybe_seed_cluster_toml
from scion.resources import CLUSTER_TOML_TEMPLATE, find_bundled_environments_dir


def test_seed_cluster_toml_writes_when_user_accepts(tmp_path: Path, monkeypatch):
    """Default prompt is 'y'; an empty stdin accepts the default and writes."""
    monkeypatch.setattr("builtins.input", lambda *_: "")
    root = tmp_path / "scion"
    root.mkdir()
    _maybe_seed_cluster_toml(root)
    written = (root / "cluster.toml").read_text()
    assert written == CLUSTER_TOML_TEMPLATE
    assert "[login_env]" in written
    assert "OMP_NUM_THREADS" in written


def test_seed_cluster_toml_skips_when_file_exists(tmp_path: Path):
    root = tmp_path / "scion"
    root.mkdir()
    (root / "cluster.toml").write_text("# pre-existing config\n")
    _maybe_seed_cluster_toml(root)
    assert (root / "cluster.toml").read_text() == "# pre-existing config\n"


def test_seed_cluster_toml_honors_user_n(tmp_path: Path, monkeypatch):
    """User answers 'n' → no file written."""
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    root = tmp_path / "scion"
    root.mkdir()
    _maybe_seed_cluster_toml(root)
    assert not (root / "cluster.toml").exists()


def test_find_bundled_environments_dir_in_editable_install():
    """In the dev repo, bundled envs live in ../environments/ relative to scion/."""
    found = find_bundled_environments_dir()
    # The dev repo always has environments/, so this must succeed in tests.
    assert found is not None and found.is_dir()
    names = {p.stem for p in found.glob("*.py")}
    assert "boltz_env" in names
    assert "esm2_env" in names


def test_check_root_writable_accepts_user_home(tmp_path: Path):
    """A path under tmp_path is writable; the gate must return True."""
    ok, why = _check_root_writable(tmp_path / "nonexistent_yet")
    assert ok, why


def test_check_root_writable_rejects_unwritable_parent(tmp_path: Path):
    """A path under a directory the user can't write to should fail."""
    import os
    if os.geteuid() == 0:
        return  # root can write anywhere; skip
    ok, why = _check_root_writable(Path("/proc/scion_init_probe"))
    assert not ok, why


def test_pyproject_force_includes_bundled_resources():
    """
    Regression guard: pyproject.toml must force-include environments/ and
    cluster.toml.example into the wheel under scion/_bundled_*/, so
    pip-installed users get the bundled resources `scion init` looks for.
    """
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    force_include = (
        data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
        .get("wheel", {}).get("force-include", {})
    )
    assert force_include.get("environments") == "scion/_bundled_environments", force_include
    assert force_include.get("cluster.toml.example") == \
        "scion/_bundled_resources/cluster.toml.example", force_include


def test_cluster_toml_template_matches_example_file_intent():
    """
    The embedded template should preserve the Polaris-relevant guidance
    (login-node thread caps). It needn't match the example file byte-for-byte,
    but the load-bearing parts (table headers + the three thread caps) must
    survive any edits.
    """
    for needle in (
        "[login_env]",
        'OMP_NUM_THREADS = "1"',
        'MKL_NUM_THREADS = "1"',
        'OPENBLAS_NUM_THREADS = "1"',
    ):
        assert needle in CLUSTER_TOML_TEMPLATE, needle
