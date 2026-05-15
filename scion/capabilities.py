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
    "design_sequence",
    "dock",
    # Reserved for future versions:
    # "generate",
    # "score",
)


@dataclass
class FoldResult:
    """
    Output of a ``fold`` call.

    ``affinity`` is populated when the provider was asked to run a binding-
    affinity head alongside structure prediction (e.g. Boltz-2 with a ligand
    + ``predict_affinity=True``). Typical fields: ``kd``, ``log_kd``,
    ``binding_probability``, plus model-specific variants. ``None`` when
    affinity wasn't requested or the model doesn't support it.
    """

    mmcif: str
    confidence: dict[str, Any] = field(default_factory=dict)
    plddt: np.ndarray | None = None
    affinity: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> FoldResult:
        reserved = {"mmcif", "confidence", "plddt", "affinity"}
        return cls(
            mmcif=data.get("mmcif", ""),
            confidence=data.get("confidence", {}) or {},
            plddt=data.get("plddt"),
            affinity=data.get("affinity"),
            extras={k: v for k, v in data.items() if k not in reserved},
        )


@dataclass
class DesignResult:
    """
    Output of a ``design_sequence`` call.

    A provider may return one or more candidate sequences. The first is
    convenience-exposed as ``sequence``; the full list is on ``sequences``
    alongside per-sequence scores when the provider supplies them (e.g.
    LigandMPNN reports a sequence-recovery proxy as ``score``).
    """

    sequence: str
    sequences: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    masked_positions: list[int] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> DesignResult:
        seqs = data.get("sequences") or ([data["sequence"]] if data.get("sequence") else [])
        first = data.get("sequence") or (seqs[0] if seqs else "")
        reserved = {"sequence", "sequences", "scores", "masked_positions"}
        return cls(
            sequence=first,
            sequences=list(seqs),
            scores=list(data.get("scores") or []),
            masked_positions=list(data.get("masked_positions") or []),
            extras={k: v for k, v in data.items() if k not in reserved},
        )


@dataclass
class DockResult:
    """
    Output of a ``dock`` call.

    A docking provider returns one or more ligand poses with confidence
    scores. ``poses`` is a list of SDF strings (one per candidate pose);
    ``scores`` is the parallel list of confidence values (provider-defined
    semantics, usually higher = better). ``mmcif`` is the receptor used,
    handy when the docker conditioned on a freshly-predicted structure.
    """

    poses: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    mmcif: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> DockResult:
        reserved = {"poses", "scores", "mmcif"}
        return cls(
            poses=list(data.get("poses") or []),
            scores=list(data.get("scores") or []),
            mmcif=data.get("mmcif"),
            extras={k: v for k, v in data.items() if k not in reserved},
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
