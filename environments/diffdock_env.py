# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # DiffDock-L's repo is the source of truth; not on PyPI yet.
#     "diffdock @ git+https://github.com/gcorso/DiffDock@main",
#     # Same torch cap as the other shipped envs.
#     "torch>=2.6,<2.9",
#     "numpy>=1.24",
#     "rdkit",
#     "biopython>=1.83",
#     # Geometric deep learning stack DiffDock builds on.
#     "torch-geometric>=2.5",
# ]
# ///
"""
DiffDock-L environment for Scion.

Capability: ``dock`` — fast pose prediction for a small molecule against
a protein receptor. Use this to rank a 1000-ligand library in minutes,
then take the top hits to Boltz-2 for precise co-fold + affinity.

Status: scaffold. Provider wires the call surface but the inference
path raises NotImplementedError pending end-to-end testing on Polaris.

Reference: https://github.com/gcorso/DiffDock
"""

from __future__ import annotations

CAPABILITIES = ["dock"]

DEFAULT_CHECKPOINT = "diffdock_l"


def setup(model: str, device: str = "cuda"):
    """
    Load a DiffDock-L provider.

    Args:
        model: Checkpoint name; defaults to the DiffDock-L weights.
        device: PyTorch device. GPU strongly recommended — CPU dock is
            slow enough that you'd prefer Boltz directly.

    Returns:
        Provider with a ``dock(...)`` method.
    """
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "DiffDock's deep-learning stack is not installed in this "
            "environment. Build with `scion install diffdock_env.py`."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT

    class DiffDockProvider:
        def __init__(self):
            self.checkpoint = checkpoint
            self.device = device

        def preload(self) -> None:
            """Validate that the geometric-DL stack is importable."""
            # DiffDock's import name is the source of truth — confirm
            # at first Polaris build and pin here.
            import torch_geometric  # noqa: F401
            # When the wire-up below is done, add a 5-atom ligand
            # dummy-dock against a tiny pseudo-receptor as a deeper check.

        def dock(
            self,
            receptor_mmcif: str,
            ligand_smiles: str,
            num_poses: int = 10,
            **kwargs,
        ) -> dict:
            """
            Wire-up checklist:

            1. Write receptor_mmcif to a tempfile; DiffDock's CLI takes
               a path. (Or use its Python API if exposed; check upstream.)
            2. Pass ``ligand_smiles`` directly — DiffDock accepts SMILES.
            3. Run inference (``num_poses`` samples).
            4. Read back the SDF poses and confidence scores.
            5. Return a dict matching capabilities.DockResult:
                  {
                    "poses":   [<SDF string per pose>],
                    "scores":  [<confidence per pose, higher = better>],
                    "mmcif":   receptor_mmcif,  # echo for the caller
                  }
            """
            raise NotImplementedError(
                "diffdock_env.dock is scaffolded but the inference "
                "path is not yet wired. See the docstring in this "
                "method for the five-step plan."
            )

    return DiffDockProvider()
