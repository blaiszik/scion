"""
Tests for `scion install`'s handling of [tool.uv] config inside PEP 723
metadata — specifically the find-links and extra-index-url options that
let env files pin CUDA-specific torch wheel indexes.

We don't actually run `scion install` here (it'd build a real venv);
we just verify the metadata is parsed correctly and the env files we
ship declare the expected uv config.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from scion.pep723 import parse_pep723_metadata


def test_tool_uv_block_parses_cleanly(tmp_path: Path):
    p = tmp_path / "env_with_uv_config.py"
    p.write_text(
        textwrap.dedent(
            """\
            # /// script
            # requires-python = ">=3.10"
            # dependencies = ["torch>=2.0"]
            #
            # [tool.uv]
            # extra-index-url = ["https://download.pytorch.org/whl/cu126"]
            # find-links = ["https://example.invalid/wheels"]
            # ///
            CAPABILITIES = ["embed"]
            def setup(model, device="cpu"): return object()
            """
        )
    )
    meta = parse_pep723_metadata(p.read_text())
    assert meta is not None
    uv = meta.get("tool", {}).get("uv", {})
    assert uv.get("extra-index-url") == ["https://download.pytorch.org/whl/cu126"]
    assert uv.get("find-links") == ["https://example.invalid/wheels"]


def test_shipped_env_files_cap_torch_version_for_hpc_drivers():
    """
    Both shipped env files (esm2_env, boltz_env) must cap torch below 2.9.

    PyPI's torch>=2.9 wheels are built against CUDA 12.9, which fails at
    runtime on Polaris (driver 12.8) with "NVIDIA driver too old". Took a
    full GPU-job triage cycle to surface — regression guard for the cap.

    Note: we don't use an extra-index-url to force a specific CUDA wheel
    because uv's default `first-index` strategy still resolves torch from
    PyPI when both PyPI and a PyTorch index list it. A version cap is the
    only mechanism that reliably constrains which wheel uv selects.
    """
    here = Path(__file__).resolve().parent.parent / "environments"
    for env_file in ("esm2_env.py", "boltz_env.py"):
        meta = parse_pep723_metadata((here / env_file).read_text())
        assert meta is not None, env_file
        deps = meta.get("dependencies", [])
        torch_spec = next((d for d in deps if d.startswith("torch")), None)
        assert torch_spec is not None, f"{env_file} missing torch dep"
        assert "<2.9" in torch_spec or "<= 2.8" in torch_spec, (
            f"{env_file}'s torch pin {torch_spec!r} is missing the <2.9 "
            f"cap that keeps PyPI wheels compatible with HPC NVIDIA drivers."
        )
