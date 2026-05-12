"""
Embedder — capability client for ``embed`` (sequence -> embeddings).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import EmbedResult
from .session import ScionSession, decode_result


class Embedder:
    """
    Protein language-model embedding client.

    Example
    -------
        with Embedder(cluster="della", model="esm2",
                      checkpoint="esm2_t33_650M_UR50D") as e:
            result = e.embed(["MKTAYIAKQ..."])
            print(result.per_residue.shape)   # (1, L, D)
            print(result.per_sequence.shape)  # (1, D)
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
            required_capability="embed",
            model=model,
            checkpoint=checkpoint,
            device=device,
            cluster=cluster,
            root=root,
            log=log,
            timeout=timeout,
        )

    def start(self) -> None:
        self._session.start()

    def stop(self) -> None:
        self._session.stop()

    def __enter__(self) -> Embedder:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def embed(
        self,
        sequences: list[str],
        repr_layers: tuple[int, ...] = (33,),
        return_contacts: bool = False,
        **extra: Any,
    ) -> EmbedResult:
        """
        Compute embeddings for a list of sequences.

        ``repr_layers`` selects which transformer layers to return;
        the default of ``(33,)`` is the last layer of ESM2-650M.
        """
        args: dict[str, Any] = {
            "sequences": list(sequences),
            "repr_layers": list(repr_layers),
            "return_contacts": bool(return_contacts),
            **extra,
        }

        reply = self._session.call("embed", args)
        payload = decode_result(reply)
        if not isinstance(payload, dict):
            raise RuntimeError(f"embed() expected a dict result, got {type(payload).__name__}")
        return EmbedResult.from_dict(payload)
