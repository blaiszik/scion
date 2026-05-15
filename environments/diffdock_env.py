# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # NOTE: this dep list is INTENTIONALLY a placeholder. DiffDock-L's
#     # upstream requirements.txt pins torch==1.13.1+cu117 plus exact
#     # versions of torch-cluster/scatter/sparse from a torch-1.13-specific
#     # wheel index, and pulls openfold (which has CUDA-kernel compile
#     # steps) and fair-esm[esmfold]. None of that resolves cleanly on
#     # Polaris's CUDA-12.8 driver out of the box. Wiring this needs a
#     # multi-hour Polaris session; see the "wire-up checklist" at the
#     # bottom of this file. Until then this env is intentionally not
#     # buildable — `scion install diffdock_env.py` will fail, by design.
#     "torch>=2.6,<2.9",
#     "numpy>=1.23,<2",
#     "rdkit",
#     "biopython>=1.83",
#     "prody>=2.4",
# ]
# ///
"""
DiffDock-L environment for Scion — SCAFFOLD ONLY (not yet wireable).

Capability: ``dock`` — fast pose prediction for a small molecule
against a protein receptor. Intended to power the hierarchical
"screen 1000 ligands quickly, refine top-K with Boltz" workflow.

Why this isn't wired yet (read before touching):

* Upstream requirements.txt (https://github.com/gcorso/DiffDock)
  pins ``torch==1.13.1+cu117``, which is incompatible with Polaris's
  CUDA-12.8 driver via the default PyPI wheels. To run on Polaris we
  need to use modern torch (>=2.6) and pick matching pyg companion
  wheels (``torch-cluster``, ``torch-scatter``, ``torch-sparse``)
  from https://data.pyg.org/whl/torch-<X.Y>.0+cu<NN>.html .
* ``openfold @ git+...`` is in upstream's requirements but does CUDA
  kernel compilation at install time; it needs ``nvcc`` matching the
  cluster's CUDA, plus glibc compatibility on the build node.
* DiffDock-L pulls receptor/ligand weights from HF Hub on first run
  (``gcorso/DiffDock-L``); needs HF Hub reachable at install / preload
  time.

For now, ``scion install diffdock_env.py`` is expected to fail and
the provider's ``dock`` method raises ``NotImplementedError``. Wiring
this cleanly is its own deliverable; the LigandMPNN env covers the
multi-model demo (fold -> design -> fold).

Reference: https://github.com/gcorso/DiffDock
"""

from __future__ import annotations

CAPABILITIES = ["dock"]

DEFAULT_CHECKPOINT = "diffdock_l"


def setup(model: str, device: str = "cuda"):
    """Scaffolded entry point. See the wire-up checklist at the bottom."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "torch not installed in this environment. DiffDock-L's dep "
            "tree is not yet pinned for Polaris — see this file's "
            "docstring for the known sticky points."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT

    class DiffDockProvider:
        def __init__(self):
            self.checkpoint = checkpoint
            self.device = device

        def preload(self) -> None:
            raise NotImplementedError(
                "diffdock_env is scaffold-only. The dep tree (torch + "
                "pyg companion wheels + openfold + fair-esm) needs a "
                "dedicated Polaris session to wire. See this file's "
                "module docstring for the known sticky points."
            )

        def dock(self, receptor_mmcif, ligand_smiles, num_poses=10, **kwargs):
            raise NotImplementedError(
                "diffdock_env.dock is not wired yet. Wire-up checklist:\n"
                "  1. Pin pyg companion wheels (torch-cluster/scatter/\n"
                "     sparse) from data.pyg.org's torch+cu index in\n"
                "     [tool.uv] extra-index-url in the PEP 723 block.\n"
                "  2. Decide: vendor openfold (clone in setup()) or pip\n"
                "     install it. Vendoring sidesteps the CUDA kernel\n"
                "     compile, like ligandmpnn_env does for openfold.\n"
                "  3. Clone DiffDock repo to $HOME/diffdock-install/src,\n"
                "     fetch DiffDock-L weights from HF Hub.\n"
                "  4. From dock(): write receptor_mmcif to a tempfile\n"
                "     (.pdb form via Bio.PDB), pass --protein_path and\n"
                "     --ligand_description=<SMILES> to inference.py,\n"
                "     read back SDF poses + confidence scores.\n"
                "  5. Return dict matching capabilities.DockResult."
            )

    return DiffDockProvider()
