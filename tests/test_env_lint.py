"""
Tests for scion.env_lint.

Pin the behaviour for: (a) the shipped env files lint cleanly so
``scion install`` doesn't print spurious warnings; (b) an env that
imports something it didn't declare is flagged; (c) known mapped
packages (fair-esm → esm) are accepted.
"""

from __future__ import annotations

from pathlib import Path

from scion.env_lint import lint_environment_imports

REPO_ENVIRONMENTS = Path(__file__).resolve().parent.parent / "environments"


def test_shipped_envs_lint_clean():
    """boltz_env and esm2_env must lint clean — they're the regression baseline."""
    for env_file in ("esm2_env.py", "boltz_env.py"):
        warnings = lint_environment_imports(REPO_ENVIRONMENTS / env_file)
        assert warnings == [], f"{env_file}: {warnings}"


def test_lint_flags_undeclared_import(tmp_path: Path):
    env = tmp_path / "leaky_env.py"
    env.write_text(
        "# /// script\n"
        "# requires-python = '>=3.10'\n"
        "# dependencies = ['numpy']\n"
        "# ///\n"
        "CAPABILITIES = ['embed']\n"
        "def setup(model, device='cpu'):\n"
        "    import boltz\n"
        "    return None\n"
    )
    warnings = lint_environment_imports(env)
    assert any("boltz" in w for w in warnings), warnings


def test_lint_resolves_mapped_pypi_to_import(tmp_path: Path):
    """fair-esm declared in deps; env imports `esm` — must not warn."""
    env = tmp_path / "esm_env.py"
    env.write_text(
        "# /// script\n"
        "# dependencies = ['fair-esm']\n"
        "# ///\n"
        "CAPABILITIES = ['embed']\n"
        "def setup(model, device='cpu'):\n"
        "    import esm\n"
        "    return None\n"
    )
    warnings = lint_environment_imports(env)
    assert warnings == [], warnings


def test_lint_ignores_stdlib_and_scion(tmp_path: Path):
    env = tmp_path / "trivial_env.py"
    env.write_text(
        "# /// script\n"
        "# dependencies = []\n"
        "# ///\n"
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        "import scion\n"
        "CAPABILITIES = ['embed']\n"
        "def setup(model, device='cpu'):\n"
        "    return None\n"
    )
    warnings = lint_environment_imports(env)
    assert warnings == [], warnings
