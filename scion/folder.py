"""
Folder — capability client for ``fold`` (sequence -> structure).

Wraps a ScionSession that talks to a worker advertising ``fold``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import FoldResult
from .session import ScionSession, decode_result


class Folder:
    """
    Protein structure-prediction client.

    Example
    -------
        with Folder(cluster="della", model="boltz", checkpoint="boltz2") as folder:
            result = folder.fold("MKTAYIAKQ...")
            print(result.mmcif[:200])
            print(result.confidence["plddt_mean"])
    """

    def __init__(
        self,
        model: str,
        checkpoint: str | None = None,
        device: str = "cuda",
        cluster: str | None = None,
        root: Path | str | None = None,
        log=None,
        timeout: float = 600.0,
    ):
        env_name = f"{model}_env"
        self._session = ScionSession(
            env_name=env_name,
            required_capability="fold",
            model=model,
            checkpoint=checkpoint,
            device=device,
            cluster=cluster,
            root=root,
            log=log,
            timeout=timeout,
        )

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self._session.start()

    def stop(self) -> None:
        self._session.stop()

    def __enter__(self) -> Folder:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # --- capability ------------------------------------------------------

    def fold(
        self,
        sequence: str | list[str],
        msa: bytes | None = None,
        templates: bytes | None = None,
        num_recycles: int = 3,
        ligands: list[dict] | None = None,
        nucleic_acids: list[dict] | None = None,
        predict_affinity: bool = False,
        affinity_binder: str | None = None,
        **extra: Any,
    ) -> FoldResult:
        """
        Predict a structure (and optionally a binding affinity).

        Args:
            sequence: Protein sequence (str) or list of chains (multimer).
            msa: Optional A3M MSA blob (bytes). If ``None``, single-sequence
                prediction is used; no external MSA server is contacted.
            templates: Optional CIF templates blob (provider-dependent).
            num_recycles: Recycling steps. Defaults to 3.
            ligands: Optional list of ligand dicts, each containing either
                ``{"smiles": "..."}`` or ``{"ccd": "ATP"}``. An ``"id"``
                key is honored if given, otherwise chain IDs are auto-
                assigned. Required for protein-ligand co-folding.
            nucleic_acids: Optional list of nucleic-acid dicts, each
                ``{"type": "dna" | "rna", "sequence": "ATGC..."}``.
            predict_affinity: If True, ask the provider to run its
                binding-affinity head alongside structure prediction
                (Boltz-2 supports this). Result is on
                ``FoldResult.affinity``.
            affinity_binder: Chain ID of the binder when predicting
                affinity. Defaults to the first ligand if any are given.

        Extra kwargs pass through to the provider's ``fold(...)`` method
        and let provider-specific options (e.g. ``diffusion_samples``)
        be set without a client-side bump.
        """
        args: dict[str, Any] = {
            "sequence": sequence,
            "num_recycles": int(num_recycles),
            "predict_affinity": bool(predict_affinity),
            **extra,
        }
        if msa is not None:
            args["msa"] = msa
        if templates is not None:
            args["templates"] = templates
        if ligands:
            args["ligands"] = list(ligands)
        if nucleic_acids:
            args["nucleic_acids"] = list(nucleic_acids)
        if affinity_binder is not None:
            args["affinity_binder"] = affinity_binder

        reply = self._session.call("fold", args)
        payload = decode_result(reply)
        if not isinstance(payload, dict):
            raise RuntimeError(f"fold() expected a dict result, got {type(payload).__name__}")
        return FoldResult.from_dict(payload)
