"""
Cluster registry and profiles for Scion.

The package ships a small set of built-in cluster roots, then overlays
optional user/site profiles from TOML. This keeps the public
``get_root_for_cluster()`` API stable while allowing new HPC systems to be
added without changing Scion source code.
"""

from __future__ import annotations

import fnmatch
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_DIR

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


SCION_CLUSTERS_FILE_ENV = "SCION_CLUSTERS_FILE"
DEFAULT_CLUSTERS_FILE = DEFAULT_CONFIG_DIR / "clusters.toml"
DEFAULT_JOB_ENV_VARS: tuple[str, ...] = ("PBS_JOBID", "SLURM_JOB_ID")


@dataclass(frozen=True)
class ClusterProfile:
    """Resolved profile for one HPC system."""

    name: str
    root: str
    scheduler: str | None = None
    job_env_vars: tuple[str, ...] = DEFAULT_JOB_ENV_VARS
    hostname_patterns: tuple[str, ...] = field(default_factory=tuple)
    gpu_arch: str | None = None
    cuda_driver_max: str | None = None
    runtime_dir: str | None = None
    description: str | None = None
    source: str = "builtin"

    @property
    def root_path(self) -> Path:
        return Path(os.path.expandvars(self.root)).expanduser()


BUILTIN_CLUSTER_PROFILES: dict[str, ClusterProfile] = {
    "della": ClusterProfile(
        name="della",
        root="/scratch/gpfs/ROSENGROUP/common/scion",
        scheduler="slurm",
        job_env_vars=("SLURM_JOB_ID",),
        hostname_patterns=("*della*",),
    ),
    "sophia": ClusterProfile(
        name="sophia",
        root="/lus/eagle/projects/Garden-Ai/scion",
        scheduler="slurm",
        job_env_vars=("SLURM_JOB_ID",),
        hostname_patterns=("*sophia*",),
    ),
    "polaris": ClusterProfile(
        name="polaris",
        root="/lus/eagle/projects/Garden-Ai/scion",
        scheduler="pbs",
        job_env_vars=("PBS_JOBID",),
        hostname_patterns=("*polaris*",),
        cuda_driver_max="12.8",
    ),
}

# Backward-compatible view used by older code/docs. Dynamic user overrides are
# available through list_cluster_profiles()/get_cluster_profile().
CLUSTER_REGISTRY: dict[str, str] = {
    name: profile.root for name, profile in BUILTIN_CLUSTER_PROFILES.items()
}

KNOWN_ENVIRONMENTS = ["boltz", "esmfold", "chai", "esm2", "esmc"]


def _tuple_of_str(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return default


def _coerce_profile(name: str, data: Any, source: str) -> ClusterProfile | None:
    if not isinstance(data, dict):
        return None
    root = data.get("root")
    if not root:
        return None

    return ClusterProfile(
        name=name,
        root=str(root),
        scheduler=str(data["scheduler"]) if data.get("scheduler") else None,
        job_env_vars=_tuple_of_str(data.get("job_env_vars"), DEFAULT_JOB_ENV_VARS),
        hostname_patterns=_tuple_of_str(data.get("hostname_patterns")),
        gpu_arch=str(data["gpu_arch"]) if data.get("gpu_arch") else None,
        cuda_driver_max=(
            str(data["cuda_driver_max"]) if data.get("cuda_driver_max") else None
        ),
        runtime_dir=str(data["runtime_dir"]) if data.get("runtime_dir") else None,
        description=str(data["description"]) if data.get("description") else None,
        source=source,
    )


def _load_profiles_file(path: Path) -> dict[str, ClusterProfile]:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Warning: {path} is malformed TOML — ignoring: {e}", file=sys.stderr)
        return {}

    raw_clusters = data.get("clusters", data)
    if not isinstance(raw_clusters, dict):
        return {}

    profiles: dict[str, ClusterProfile] = {}
    for name, value in raw_clusters.items():
        profile = _coerce_profile(str(name), value, str(path))
        if profile is not None:
            profiles[profile.name] = profile
    return profiles


def _default_profile_paths() -> list[Path]:
    paths = [DEFAULT_CLUSTERS_FILE]
    env_path = os.environ.get(SCION_CLUSTERS_FILE_ENV)
    if env_path:
        paths.append(Path(env_path).expanduser())
    return paths


def list_cluster_profiles(
    config_paths: list[Path] | tuple[Path, ...] | None = None,
) -> dict[str, ClusterProfile]:
    """
    Return the layered cluster registry.

    Layering order is built-ins, then ``~/.config/scion/clusters.toml``, then
    the optional ``SCION_CLUSTERS_FILE`` override. Later layers replace earlier
    profiles with the same name.
    """
    profiles = dict(BUILTIN_CLUSTER_PROFILES)
    paths = _default_profile_paths() if config_paths is None else list(config_paths)
    for path in paths:
        profiles.update(_load_profiles_file(path))
    return profiles


def get_cluster_profile(cluster: str) -> ClusterProfile:
    """Get a cluster profile by name."""
    if cluster == "current":
        detected = detect_current_cluster()
        if detected is None:
            raise ValueError(
                "Could not detect current cluster from hostname. "
                "Use --cluster <name> or root='/path/to/scion'."
            )
        return detected

    profiles = list_cluster_profiles()
    if cluster not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(
            f"Unknown cluster '{cluster}'. Known clusters: {available}. "
            f"Use root='/path/to/scion' for custom locations or add "
            f"{DEFAULT_CLUSTERS_FILE}."
        )
    return profiles[cluster]


def get_root_for_cluster(cluster: str) -> Path:
    """Get the scion root directory for a known cluster."""
    return get_cluster_profile(cluster).root_path


def _normalized_path(path: Path | str) -> str:
    return str(Path(os.path.expandvars(str(path))).expanduser())


def get_cluster_for_root(root: Path | str) -> str | None:
    """Get cluster name for a given root path (reverse lookup)."""
    root_str = _normalized_path(root)
    for cluster, profile in list_cluster_profiles().items():
        if root_str == _normalized_path(profile.root):
            return cluster
    return None


def get_profile_for_root(root: Path | str) -> ClusterProfile | None:
    """Get cluster profile for a root path, if one is registered."""
    name = get_cluster_for_root(root)
    return get_cluster_profile(name) if name else None


def get_profile_for_root_or_host(root: Path | str) -> ClusterProfile | None:
    """
    Resolve a cluster profile from a root path with hostname fallback.

    A user with their own per-project root (e.g. ``$HOME/scion`` instead
    of the maintainer-shared path) still benefits from the active
    cluster's metadata: ``cuda_driver_max``, ``scheduler``,
    ``runtime_dir``. This lookup returns the matching profile by root
    first, then falls back to hostname-based detection so site-specific
    checks fire regardless of where the user keeps their install.
    """
    profile = get_profile_for_root(root)
    if profile is not None:
        return profile
    return detect_current_cluster()


def detect_current_cluster(hostname: str | None = None) -> ClusterProfile | None:
    """Best-effort cluster detection from hostname patterns."""
    host = (hostname or socket.getfqdn() or socket.gethostname()).lower()
    short = host.split(".", 1)[0]

    for profile in list_cluster_profiles().values():
        patterns = profile.hostname_patterns or (f"*{profile.name.lower()}*",)
        for pattern in patterns:
            pattern = pattern.lower()
            if fnmatch.fnmatch(host, pattern) or fnmatch.fnmatch(short, pattern):
                return profile
    return None
