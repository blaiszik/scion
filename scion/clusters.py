"""
Cluster configuration for Scion.

Maps cluster names to root directories where Scion environments and caches live.
"""

from __future__ import annotations

from pathlib import Path

CLUSTER_REGISTRY: dict[str, str] = {
    "della": "/scratch/gpfs/ROSENGROUP/common/scion",
    "sophia": "/eagle/Garden-Ai/scion",
}

KNOWN_ENVIRONMENTS = ["boltz", "esmfold", "chai", "esm2", "esmc"]


def get_root_for_cluster(cluster: str) -> Path:
    """Get the scion root directory for a known cluster."""
    if cluster not in CLUSTER_REGISTRY:
        available = ", ".join(CLUSTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown cluster '{cluster}'. Known clusters: {available}. "
            f"Use root='/path/to/scion' for custom locations."
        )
    return Path(CLUSTER_REGISTRY[cluster])


def get_cluster_for_root(root: Path | str) -> str | None:
    """Get cluster name for a given root path (reverse lookup)."""
    root_str = str(root)
    for cluster, path in CLUSTER_REGISTRY.items():
        if root_str == path:
            return cluster
    return None
