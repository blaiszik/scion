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
        **extra: Any,
    ) -> FoldResult:
        """
        Predict a structure for ``sequence`` (FASTA-style, single chain
        or list of chains for a multimer).

        ``msa`` and ``templates`` are optional binary blobs (A3M, CIF)
        forwarded to the provider. Extra kwargs flow through to the
        provider's ``fold(...)`` method.
        """
        args: dict[str, Any] = {
            "sequence": sequence,
            "num_recycles": int(num_recycles),
            **extra,
        }
        if msa is not None:
            args["msa"] = msa
        if templates is not None:
            args["templates"] = templates

        reply = self._session.call("fold", args)
        payload = decode_result(reply)
        if not isinstance(payload, dict):
            raise RuntimeError(f"fold() expected a dict result, got {type(payload).__name__}")
        return FoldResult.from_dict(payload)
