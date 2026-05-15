"""
Designer — capability client for ``design_sequence``.

Wraps a ScionSession that talks to a worker advertising
``design_sequence``. Used for inverse-folding and ligand-aware
redesign workflows (e.g. LigandMPNN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import DesignResult
from .session import ScionSession, decode_result


class Designer:
    """
    Inverse-folding / sequence-design client.

    Example
    -------
        with Designer(cluster="polaris", model="ligandmpnn",
                      device="cuda") as d:
            r = d.design_sequence(
                mmcif=open("complex.cif").read(),
                positions=[42, 43, 46, 89, 92],  # pocket residues to redesign
                ligands=[{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}],
                temperature=0.1,
                num_sequences=4,
            )
            print(r.sequence)
            print(r.scores)
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
            required_capability="design_sequence",
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

    def __enter__(self) -> Designer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def design_sequence(
        self,
        mmcif: str,
        positions: list[int] | None = None,
        ligands: list[dict] | None = None,
        num_sequences: int = 1,
        temperature: float = 0.1,
        **extra: Any,
    ) -> DesignResult:
        """
        Redesign residues of a structure, optionally ligand-aware.

        Args:
            mmcif: Input structure as an mmCIF string. The provider
                infers chain/residue identity from this — pass the
                output of ``Folder.fold(...).mmcif`` directly.
            positions: 1-indexed residue numbers to redesign. ``None``
                means "redesign all" (rarely what you want for binder
                optimization; pass a focused pocket list instead).
            ligands: Ligand specs in the same shape as ``Folder.fold``
                accepts: ``[{"smiles": "..."} | {"ccd": "ATP"}]``.
                When supplied, the design is conditioned on the ligand
                pose (LigandMPNN specifically — plain ProteinMPNN
                ignores ligands).
            num_sequences: How many candidate sequences to sample.
            temperature: Sampling temperature. Lower = more conservative
                (closer to the WT sequence); higher = more exploration.
        """
        args: dict[str, Any] = {
            "mmcif": mmcif,
            "num_sequences": int(num_sequences),
            "temperature": float(temperature),
            **extra,
        }
        if positions is not None:
            args["positions"] = [int(p) for p in positions]
        if ligands:
            args["ligands"] = list(ligands)

        reply = self._session.call("design_sequence", args)
        payload = decode_result(reply)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"design_sequence() expected a dict result, "
                f"got {type(payload).__name__}"
            )
        return DesignResult.from_dict(payload)
