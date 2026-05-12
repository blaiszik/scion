"""
Capability registry and result dataclasses.

Each capability defines:
  * a stable string id (used as the RPC method name and in env files'
    ``CAPABILITIES`` lists)
  * a request signature (kwargs forwarded to the provider method)
  * a result dataclass returned to the user

The capability layer is intentionally thin: the wire protocol carries
generic JSON + blobs, and the worker dispatches by method name onto the
provider. These dataclasses only document the v0 contracts and shape
the user-facing return objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

CAPABILITIES: tuple[str, ...] = (
    "fold",
    "embed",
    # Reserved for future versions:
    # "design_sequence",
    # "generate",
    # "dock",
    # "score",
)


@dataclass
class FoldResult:
    """Output of a ``fold`` call."""

    mmcif: str
    confidence: dict[str, Any] = field(default_factory=dict)
    plddt: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> FoldResult:
        return cls(
            mmcif=data.get("mmcif", ""),
            confidence=data.get("confidence", {}) or {},
            plddt=data.get("plddt"),
            extras={
                k: v
                for k, v in data.items()
                if k not in {"mmcif", "confidence", "plddt"}
            },
        )


@dataclass
class EmbedResult:
    """Output of an ``embed`` call."""

    per_residue: np.ndarray
    per_sequence: np.ndarray | None = None
    contacts: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> EmbedResult:
        return cls(
            per_residue=data["per_residue"],
            per_sequence=data.get("per_sequence"),
            contacts=data.get("contacts"),
            extras={
                k: v
                for k, v in data.items()
                if k not in {"per_residue", "per_sequence", "contacts"}
            },
        )
