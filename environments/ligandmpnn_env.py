# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # LigandMPNN's official repo (https://github.com/dauparas/LigandMPNN)
#     # isn't published on PyPI yet. Once it is, replace this with a
#     # version-pinned name. For now the env builder pulls it from a tag.
#     "ligandmpnn @ git+https://github.com/dauparas/LigandMPNN@main",
#     # Same torch cap as the other shipped envs — keeps wheels CUDA-12.8-
#     # compatible on Polaris-shape clusters.
#     "torch>=2.6,<2.9",
#     "numpy>=1.24",
#     "biopython>=1.83",  # mmCIF parsing
# ]
# ///
"""
LigandMPNN environment for Scion.

Capability: ``design_sequence`` — inverse folding with optional ligand
context. Given a structure (mmCIF) and a set of positions to redesign,
proposes new amino acids that fit the local environment, conditioned
on any bound ligand poses.

This is the ligand-aware sibling of ProteinMPNN; the only difference at
the call surface is that ``design_sequence`` honors a ``ligands=`` arg.
On a no-ligand call it falls back to plain MPNN behavior.

Status: scaffold. The Provider class wires the call surface but the
inference path raises NotImplementedError pending end-to-end testing
on Polaris. See the bottom of this file for the wire-up checklist.

Reference: https://github.com/dauparas/LigandMPNN
"""

from __future__ import annotations

CAPABILITIES = ["design_sequence"]

DEFAULT_CHECKPOINT = "ligandmpnn_v_32_010_25"


def setup(model: str, device: str = "cuda"):
    """
    Load a LigandMPNN design provider.

    Args:
        model: Checkpoint name; defaults to LigandMPNN's standard v32
            weights at noise level 0.10. Other recommended choices are
            documented in the upstream README.
        device: PyTorch device. CPU works but is slow on >100-residue
            inputs; GPU is the default.

    Returns:
        Provider with a ``design_sequence(...)`` method.
    """
    # Imports are inside setup() so the env file is import-cheap during
    # `scion check` (without --thorough) and tests can import the module
    # for pure-function unit testing without dragging torch in.
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "torch is not installed in this environment. "
            "Build with `scion install ligandmpnn_env.py`."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT

    class LigandMPNNProvider:
        def __init__(self):
            self.checkpoint = checkpoint
            self.device = device

        def preload(self) -> None:
            """
            Import every dep LigandMPNN touches during model
            construction. Catches the structural paper cuts (missing
            CUDA kernel packages, torch/driver mismatch) at install
            time. Once the inference path is wired below, extend this
            to do a 1-residue dummy design as a deeper check.
            """
            # The upstream package's import name is the source of truth.
            # Likely ``mpnn`` or ``ligand_mpnn`` — confirm on the first
            # Polaris build and pin here.
            import ligandmpnn  # noqa: F401  (TODO: confirm import name)

        def design_sequence(
            self,
            mmcif: str,
            positions: list[int] | None = None,
            ligands: list[dict] | None = None,
            num_sequences: int = 1,
            temperature: float = 0.1,
            **kwargs,
        ) -> dict:
            """
            Wire-up checklist for the next agent / next session:

            1. Parse mmCIF into the structure repr LigandMPNN expects.
               Upstream provides a parser; if it wants PDB instead,
               use Biopython to convert.
            2. Convert ``ligands`` (smiles / ccd) into the ligand
               coordinate format LigandMPNN expects. SMILES -> 3D via
               RDKit ETKDG, OR pull coords out of the input mmCIF if
               the ligand is already there (the common case for our
               lead-optimization demo: Boltz already placed the ligand
               in the input mmCIF).
            3. Build the position mask: ``positions`` (1-indexed) are
               redesigned, everything else is fixed.
            4. Sample ``num_sequences`` designs at ``temperature``.
            5. Return a dict matching capabilities.DesignResult:
                  {
                    "sequence":          <first sampled seq>,
                    "sequences":         [<all sampled seqs>],
                    "scores":            [<per-seq score, higher is better>],
                    "masked_positions":  list(positions),
                  }
            """
            raise NotImplementedError(
                "ligandmpnn_env.design_sequence is scaffolded but the "
                "inference path is not yet wired. See the docstring "
                "in this method for the four-step plan, or check "
                "README-AGENT.md section 8 for the broader checklist."
            )

    return LigandMPNNProvider()
