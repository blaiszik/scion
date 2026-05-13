# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fair-esm>=2.0",
#     # torch>=2.9 ships PyPI wheels built against CUDA 12.9, which fails at
#     # runtime on clusters whose NVIDIA driver caps at 12.8 (Polaris).
#     # Cap at <2.9 so PyPI's default wheel (CUDA 12.6 or 12.8 build) works.
#     "torch>=2.6,<2.9",
#     "numpy>=1.24",
# ]
# ///
"""
ESM2 environment for Scion.

Capability: ``embed`` (sequence -> per-residue and per-sequence embeddings).

Default checkpoint: ``esm2_t33_650M_UR50D`` (33 layers, 650M params,
1280-dim representation). Other checkpoints available via fair-esm:
``esm2_t6_8M_UR50D``, ``esm2_t12_35M_UR50D``, ``esm2_t30_150M_UR50D``,
``esm2_t33_650M_UR50D``, ``esm2_t36_3B_UR50D``, ``esm2_t48_15B_UR50D``.

Reference: https://github.com/facebookresearch/esm
"""

CAPABILITIES = ["embed"]

DEFAULT_CHECKPOINT = "esm2_t33_650M_UR50D"


def setup(model: str, device: str = "cuda"):
    """
    Load an ESM2 embed provider.

    Args:
        model: ESM2 checkpoint name (see module docstring).
        device: PyTorch device string ("cuda", "cuda:0", "cpu").

    Returns:
        Provider with an ``embed(sequences, ...)`` method.
    """
    try:
        import esm
        import numpy as np
        import torch
    except ImportError as e:
        raise ImportError(
            "fair-esm/torch not installed in this environment. "
            "This env file is meant to be built with `scion install esm2_env.py`."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT

    loader = getattr(esm.pretrained, checkpoint, None)
    if loader is None:
        raise ValueError(
            f"Unknown ESM2 checkpoint {checkpoint!r}. "
            f"See esm.pretrained for available loaders."
        )

    model_obj, alphabet = loader()
    batch_converter = alphabet.get_batch_converter()
    model_obj = model_obj.to(device).eval()

    # ESM2-t33 has 33 transformer layers; the final layer index is 33.
    default_repr_layer = model_obj.num_layers

    class ESM2Provider:
        def __init__(self):
            self.model = model_obj
            self.alphabet = alphabet
            self.batch_converter = batch_converter
            self.device = device

        @torch.no_grad()
        def embed(
            self,
            sequences,
            repr_layers=None,
            return_contacts: bool = False,
            **kwargs,
        ) -> dict:
            """
            Compute embeddings for a batch of sequences.

            Returns:
                {
                    "per_residue":  np.ndarray, shape (B, L_max, D), float32
                    "per_sequence": np.ndarray, shape (B, D), float32 (mean-pooled
                                    over real residues, excluding BOS/EOS/pad)
                    "contacts":     np.ndarray | None, shape (B, L, L) if requested
                }
            """
            if repr_layers is None:
                repr_layers = [default_repr_layer]
            else:
                repr_layers = [int(x) for x in repr_layers]

            data = [(f"seq{i}", s) for i, s in enumerate(sequences)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)

            out = self.model(
                tokens,
                repr_layers=repr_layers,
                return_contacts=return_contacts,
            )

            target_layer = repr_layers[-1]
            per_residue_t = out["representations"][target_layer]
            # Drop BOS at index 0 and EOS at index L+1 for each sequence;
            # tokens >= 1 are real residues. For mean-pooling, use a mask
            # over non-pad tokens.
            mask = (tokens != self.alphabet.padding_idx).float().unsqueeze(-1)
            # Exclude BOS/EOS by zeroing the first column; EOS is variable
            # per sequence but the mask above already excludes pads.
            mask[:, 0, :] = 0.0
            for i, (_, s) in enumerate(data):
                eos_idx = len(s) + 1
                if eos_idx < mask.shape[1]:
                    mask[i, eos_idx, :] = 0.0
            denom = mask.sum(dim=1).clamp(min=1.0)
            per_sequence_t = (per_residue_t * mask).sum(dim=1) / denom

            result: dict = {
                "per_residue": per_residue_t.float().cpu().numpy().astype(np.float32),
                "per_sequence": per_sequence_t.float().cpu().numpy().astype(np.float32),
            }
            if return_contacts and "contacts" in out:
                result["contacts"] = out["contacts"].float().cpu().numpy().astype(np.float32)
            return result

    return ESM2Provider()
