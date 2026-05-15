"""
Docker — capability client for ``dock``.

Wraps a ScionSession that talks to a worker advertising ``dock``. Used
for fast pose prediction / virtual screening (e.g. DiffDock-L) ahead
of the slower-but-more-accurate ``fold``+affinity pass via Boltz-2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import DockResult
from .session import ScionSession, decode_result


class Docker:
    """
    Ligand docking client.

    Example
    -------
        with Docker(cluster="polaris", model="diffdock",
                    device="cuda") as d:
            r = d.dock(
                receptor_mmcif=open("target.cif").read(),
                ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
                num_poses=10,
            )
            print(r.scores)            # confidence per pose, ranked
            print(len(r.poses))        # SDF strings, one per pose
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
            required_capability="dock",
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

    def __enter__(self) -> Docker:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def dock(
        self,
        receptor_mmcif: str,
        ligand_smiles: str,
        num_poses: int = 10,
        **extra: Any,
    ) -> DockResult:
        """
        Predict ligand poses against a receptor structure.

        Args:
            receptor_mmcif: Receptor as an mmCIF string. For a hierarchical
                screen, pass the output of ``Folder.fold(target).mmcif``.
            ligand_smiles: Single ligand SMILES. For multi-ligand screens,
                call ``dock`` in a loop — the worker stays resident, so
                only the ligand changes per call.
            num_poses: How many candidate poses to return, ranked by
                provider confidence score.
        """
        args: dict[str, Any] = {
            "receptor_mmcif": receptor_mmcif,
            "ligand_smiles": ligand_smiles,
            "num_poses": int(num_poses),
            **extra,
        }
        reply = self._session.call("dock", args)
        payload = decode_result(reply)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"dock() expected a dict result, got {type(payload).__name__}"
            )
        return DockResult.from_dict(payload)
