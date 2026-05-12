# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boltz>=2.0",
#     "torch>=2.2,<2.10",
#     "numpy>=1.24",
# ]
# ///
"""
Boltz-2 environment for Scion.

Capability: ``fold`` (sequence -> structure).

Boltz-2 is an MIT-licensed AF3-class structure prediction model that
also supports protein-ligand co-folding and binding-affinity scoring.
This v0 wires up monomer folding only; multimers, ligands, and MSA
inputs are reachable through the same provider but require additional
plumbing on the Boltz side.

Reference: https://github.com/jwohlwend/boltz
"""

CAPABILITIES = ["fold"]


def setup(model: str, device: str = "cuda"):
    """
    Load a Boltz fold provider.

    Args:
        model: Checkpoint name (e.g., "boltz2"). The Boltz CLI handles
            weight download via HF Hub on first run.
        device: PyTorch device string ("cuda", "cuda:0", "cpu").

    Returns:
        Provider object with a ``fold(sequence, ...)`` method.
    """

    # Import inside setup() so that simply importing the env file (for
    # PEP 723 validation) does not require Boltz to be installed.
    import tempfile
    from pathlib import Path

    try:
        from boltz.main import predict  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "boltz is not installed in this environment. "
            "This env file is meant to be built with `scion install boltz_env.py`."
        ) from e

    class BoltzProvider:
        def __init__(self, checkpoint: str, device: str):
            self.checkpoint = checkpoint
            self.device = device

        def fold(
            self,
            sequence,
            msa: bytes | None = None,
            templates: bytes | None = None,
            num_recycles: int = 3,
            **kwargs,
        ) -> dict:
            """
            Fold a single sequence (or list of sequences for a multimer).

            Returns:
                {
                    "mmcif": <str>,             # predicted structure
                    "confidence": {...},         # pLDDT mean, pTM, ipTM, ...
                    "plddt": <np.ndarray>,       # per-residue, float32
                }
            """
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                input_yaml = tmp_path / "input.yaml"
                output_dir = tmp_path / "out"
                output_dir.mkdir()

                # Minimal YAML schema for Boltz; expand for multimer/ligand.
                sequences = sequence if isinstance(sequence, list) else [sequence]
                chains = "\n".join(
                    f"  - protein:\n      id: {chr(ord('A') + i)}\n      sequence: {s}"
                    for i, s in enumerate(sequences)
                )
                input_yaml.write_text(f"sequences:\n{chains}\n")

                # The actual Boltz CLI call would go here; the structure
                # below shows the contract Scion expects this provider to
                # honor. Implementers should replace this with a real
                # boltz.main.predict(...) invocation and parse outputs.
                raise NotImplementedError(
                    "boltz_env: real Boltz invocation pending. "
                    "This is a v0 skeleton — wire up boltz.main.predict here."
                )

    return BoltzProvider(checkpoint=model or "boltz2", device=device)
