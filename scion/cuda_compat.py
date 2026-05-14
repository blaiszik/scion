"""
Map torch versions to the CUDA toolkit their PyPI wheels ship against,
and check whether a cluster's NVIDIA driver can actually run them.

Used to catch the most expensive late-failure mode: torch wheels built
against a CUDA version newer than the cluster driver supports, which
imports fine but RuntimeErrors deep inside the inference path —
usually after a GPU-queue wait.

The CUDA-target table is intentionally short and covers the slice of
torch releases relevant on HPC clusters today. Add a row when a new
torch release lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from packaging.specifiers import SpecifierSet
from packaging.version import Version

# PyPI's default torch wheel CUDA target per release line. A driver
# advertising CUDA X can load any wheel built for CUDA ≤ X, so this
# table is the floor for compatibility on a given driver.
TORCH_CUDA_TABLE: list[tuple[str, str]] = [
    ("2.4", "12.4"),
    ("2.5", "12.4"),
    ("2.6", "12.6"),
    ("2.7", "12.6"),
    ("2.8", "12.8"),
    ("2.9", "12.9"),
    ("2.10", "13.0"),
]


def cuda_target_for_torch(torch_version: str) -> str | None:
    """Return the CUDA version PyPI's torch wheel ships against, or None."""
    try:
        v = Version(torch_version)
    except Exception:
        return None
    for key, cuda in TORCH_CUDA_TABLE:
        kv = Version(key)
        if v.major == kv.major and v.minor == kv.minor:
            return cuda
    # Newer than the table's last entry — assume it's at least as new.
    if v >= Version(TORCH_CUDA_TABLE[-1][0]):
        return TORCH_CUDA_TABLE[-1][1]
    return None


def torch_upper_bound_from_spec(spec_str: str) -> str | None:
    """
    Highest torch X.Y line from the table that satisfies the spec.

    Returns ``None`` when no entry matches. Probes both ``X.Y.0`` and
    ``X.Y.999`` so we admit both ``<=X.Y`` (which excludes ``X.Y.999``)
    and ``>=X.Y.Z`` (which excludes ``X.Y.0``).
    """
    if not spec_str:
        return None
    try:
        spec = SpecifierSet(spec_str)
    except Exception:
        return None

    def line_admissible(key: str) -> bool:
        return Version(f"{key}.0") in spec or Version(f"{key}.999") in spec

    matching = [key for key, _ in TORCH_CUDA_TABLE if line_admissible(key)]
    if not matching:
        return None
    return max(matching, key=Version)


def parse_dep_name_and_spec(dep_line: str) -> tuple[str, str]:
    """Split ``'torch>=2.6,<2.9'`` into ``('torch', '>=2.6,<2.9')``. Ignores extras."""
    s = dep_line
    if "[" in s:
        head, _, rest = s.partition("[")
        _, _, tail = rest.partition("]")
        s = head + tail
    for op in (">=", "<=", "==", "~=", "!=", ">", "<"):
        idx = s.find(op)
        if idx >= 0:
            return s[:idx].strip().lower(), s[idx:].strip()
    return s.strip().lower(), ""


def find_torch_spec(dependencies: list[str]) -> str | None:
    """Return the torch version spec string from a PEP 723 dependency list."""
    for dep in dependencies:
        name, spec = parse_dep_name_and_spec(dep)
        if name == "torch":
            return spec
    return None


def driver_supports(driver_version: str, required_cuda: str) -> bool:
    """True iff a driver's max-supported CUDA covers a required CUDA version."""
    try:
        return Version(driver_version) >= Version(required_cuda)
    except Exception:
        # Unparseable — be permissive rather than blocking the user.
        return True


@dataclass(frozen=True)
class CudaCheck:
    """Result of a torch-spec / driver-cap cross-check."""

    ok: bool
    message: str
    torch_spec: str | None = None
    torch_upper_minor: str | None = None
    required_cuda: str | None = None
    driver_cuda: str | None = None


def check_torch_spec_against_driver(
    dependencies: list[str],
    driver_cuda_max: str | None,
) -> CudaCheck:
    """
    Cross-check an env's torch pin against a cluster driver's CUDA cap.

    Used at install time, before invoking uv to build the venv: surfaces
    a too-loose upper bound (or no upper bound) on torch before the
    wheel even lands on disk.
    """
    torch_spec = find_torch_spec(dependencies)
    if torch_spec is None:
        return CudaCheck(ok=True, message="No torch dependency declared.")

    if driver_cuda_max is None:
        return CudaCheck(
            ok=True,
            message="Cluster profile has no cuda_driver_max; skipping torch/CUDA check.",
            torch_spec=torch_spec,
        )

    upper = torch_upper_bound_from_spec(torch_spec)
    if upper is None:
        # Unbounded: assume latest known torch.
        latest_key, latest_cuda = TORCH_CUDA_TABLE[-1]
        if not driver_supports(driver_cuda_max, latest_cuda):
            return CudaCheck(
                ok=False,
                message=(
                    f"torch is effectively unbounded; the latest PyPI wheel "
                    f"(torch {latest_key}.x) requires CUDA {latest_cuda}, but "
                    f"cluster driver caps at CUDA {driver_cuda_max}. Add an "
                    f"upper bound to your env file (e.g. 'torch<{latest_key}')."
                ),
                torch_spec=torch_spec,
                torch_upper_minor=latest_key,
                required_cuda=latest_cuda,
                driver_cuda=driver_cuda_max,
            )
        return CudaCheck(
            ok=True,
            message="torch upper bound is open; latest known wheel is compatible.",
            torch_spec=torch_spec,
            driver_cuda=driver_cuda_max,
        )

    required = cuda_target_for_torch(upper)
    if required is None:
        return CudaCheck(
            ok=True,
            message=f"torch upper bound {upper} has unknown CUDA target; skipping.",
            torch_spec=torch_spec,
            torch_upper_minor=upper,
            driver_cuda=driver_cuda_max,
        )

    if not driver_supports(driver_cuda_max, required):
        return CudaCheck(
            ok=False,
            message=(
                f"Env allows up to torch {upper}.x (PyPI wheel CUDA {required}), "
                f"but cluster driver caps at CUDA {driver_cuda_max}. Lower the "
                f"upper bound in your env file."
            ),
            torch_spec=torch_spec,
            torch_upper_minor=upper,
            required_cuda=required,
            driver_cuda=driver_cuda_max,
        )

    return CudaCheck(
        ok=True,
        message=(
            f"Env torch <={upper}.x (CUDA {required}) ≤ driver CUDA "
            f"{driver_cuda_max}."
        ),
        torch_spec=torch_spec,
        torch_upper_minor=upper,
        required_cuda=required,
        driver_cuda=driver_cuda_max,
    )


