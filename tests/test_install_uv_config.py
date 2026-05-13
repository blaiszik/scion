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


def test_shipped_env_files_pin_cuda_torch_index():
    """
    Both shipped env files (esm2_env, boltz_env) must declare a torch-
    compatible extra-index-url. Pinned because we already debugged the
    "modern torch wheel needs newer CUDA driver than Polaris has" failure
    mode the hard way — losing this would silently regress it.
    """
    here = Path(__file__).resolve().parent.parent / "environments"
    for env_file in ("esm2_env.py", "boltz_env.py"):
        meta = parse_pep723_metadata((here / env_file).read_text())
        assert meta is not None, env_file
        urls = meta.get("tool", {}).get("uv", {}).get("extra-index-url", [])
        assert any("pytorch.org" in u for u in urls), (
            f"{env_file} is missing a pytorch.org extra-index-url; "
            f"that pin keeps the worker venv compatible with HPC NVIDIA "
            f"drivers that lag behind the latest torch wheels on PyPI."
        )
