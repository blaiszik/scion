"""
Tests for scion.cuda_compat.

These pin the small torch→CUDA table the project uses to surface
driver/wheel mismatches before they fail inside a GPU job.
"""

from __future__ import annotations

from scion.cuda_compat import (
    check_torch_spec_against_driver,
    cuda_target_for_torch,
    find_torch_spec,
    parse_dep_name_and_spec,
    torch_upper_bound_from_spec,
)


def test_parse_dep_name_and_spec_handles_extras_and_operators():
    assert parse_dep_name_and_spec("torch>=2.6,<2.9") == ("torch", ">=2.6,<2.9")
    assert parse_dep_name_and_spec("Numpy==1.24") == ("numpy", "==1.24")
    assert parse_dep_name_and_spec("torch[cuda]>=2.0") == ("torch", ">=2.0")
    assert parse_dep_name_and_spec("boltz") == ("boltz", "")


def test_find_torch_spec_picks_torch_out_of_deps():
    deps = ["numpy>=1.24", "torch>=2.6,<2.9", "fair-esm"]
    assert find_torch_spec(deps) == ">=2.6,<2.9"
    assert find_torch_spec(["numpy"]) is None


def test_cuda_target_for_torch_table():
    assert cuda_target_for_torch("2.6.0") == "12.6"
    assert cuda_target_for_torch("2.8.5") == "12.8"
    assert cuda_target_for_torch("2.9.0") == "12.9"


def test_torch_upper_bound_from_spec():
    assert torch_upper_bound_from_spec(">=2.6,<2.9") == "2.8"
    assert torch_upper_bound_from_spec("<=2.7") == "2.7"
    assert torch_upper_bound_from_spec(">=2.0") == "2.10"
    # Empty / invalid
    assert torch_upper_bound_from_spec("") is None


def test_check_torch_spec_against_driver_polaris_capped_torch_ok():
    # Polaris: driver maxes at CUDA 12.8; env caps torch at <2.9 (the v0
    # workaround). Should pass cleanly.
    deps = ["torch>=2.6,<2.9"]
    result = check_torch_spec_against_driver(deps, "12.8")
    assert result.ok, result.message
    assert result.torch_upper_minor == "2.8"
    assert result.required_cuda == "12.8"


def test_check_torch_spec_against_driver_polaris_loose_torch_fails():
    # Same Polaris driver, but env doesn't cap torch. Latest wheel (2.9.x)
    # wants CUDA 12.9 → fail.
    deps = ["torch>=2.6"]
    result = check_torch_spec_against_driver(deps, "12.8")
    assert not result.ok
    assert "12.9" in result.message or "12.8" in result.message


def test_check_torch_spec_against_driver_no_torch_passes():
    result = check_torch_spec_against_driver(["numpy>=1.24"], "12.8")
    assert result.ok


def test_check_torch_spec_against_driver_no_cap_passes():
    # Della-like: no cuda_driver_max declared. Don't block install.
    result = check_torch_spec_against_driver(["torch>=2.6"], None)
    assert result.ok


