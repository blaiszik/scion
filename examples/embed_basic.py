#!/usr/bin/env python
"""
Minimal ESM2 embedding via Scion.

Computes per-residue and per-sequence (mean-pooled) embeddings for one
or more sequences using ESM2 (default checkpoint: ``esm2_t33_650M_UR50D``,
1280-dim representation). Useful for sanity-checking that the ``embed``
path works end-to-end and as a starting point for downstream analyses
(clustering, similarity search, function prediction, ...).

Usage::

    python examples/embed_basic.py --cluster polaris
    python examples/embed_basic.py --sequences MKTAYI GSHMASMTGGQ
    python examples/embed_basic.py --fasta my_proteins.fasta --out ./embeddings.npz
    python examples/embed_basic.py --checkpoint esm2_t30_150M_UR50D  # smaller/faster

Output ``embeddings.npz`` contains ``per_residue`` (B, L_max, D) and
``per_sequence`` (B, D) arrays, plus ``ids`` (list of sequence labels).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from scion import Embedder

# Two short, structurally distinct sequences — a quick way to sanity check
# that per-sequence embeddings differ meaningfully between inputs.
DEFAULT_SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "GSHMASMTGGQQMGRGSEFELRRQACGRINRHL",
]


def load_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Tiny FASTA reader — returns (ids, sequences)."""
    ids: list[str] = []
    seqs: list[str] = []
    current: list[str] = []
    current_id: str | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                ids.append(current_id)
                seqs.append("".join(current))
            current_id = line[1:].split()[0] or f"seq{len(ids)}"
            current = []
        else:
            current.append(line)
    if current_id is not None:
        ids.append(current_id)
        seqs.append("".join(current))
    return ids, seqs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])

    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--sequences", nargs="+", default=None,
        help=f"Whitespace-separated sequences (default: {len(DEFAULT_SEQUENCES)} demo sequences)",
    )
    inputs.add_argument(
        "--fasta", type=Path, default=None,
        help="FASTA file with one or more sequences",
    )

    parser.add_argument("--cluster", default="polaris",
                        help="Cluster name (default: polaris)")
    parser.add_argument("--root", default=None,
                        help="Explicit Scion root (overrides --cluster)")
    parser.add_argument("--checkpoint", default="esm2_t33_650M_UR50D",
                        help="ESM2 checkpoint (default: esm2_t33_650M_UR50D)")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--return-contacts", action="store_true",
                        help="Also return predicted residue-residue contacts")
    parser.add_argument("--out", type=Path, default=Path("./embeddings.npz"),
                        help="Output .npz path (default: ./embeddings.npz)")
    args = parser.parse_args()

    if args.fasta is not None:
        ids, sequences = load_fasta(args.fasta)
        if not sequences:
            print(f"No sequences found in {args.fasta}", file=sys.stderr)
            return 1
    elif args.sequences:
        sequences = list(args.sequences)
        ids = [f"seq{i}" for i in range(len(sequences))]
    else:
        sequences = list(DEFAULT_SEQUENCES)
        ids = [f"seq{i}" for i in range(len(sequences))]

    kwargs = {"model": "esm2", "checkpoint": args.checkpoint,
              "device": args.device, "log": sys.stderr}
    if args.root is not None:
        kwargs["root"] = args.root
    else:
        kwargs["cluster"] = args.cluster

    lens = [len(s) for s in sequences]
    print(f"Embedding {len(sequences)} sequence(s), lengths {lens}, on {args.device}")
    print(f"  checkpoint: {args.checkpoint}")
    print()

    t0 = time.perf_counter()
    with Embedder(**kwargs) as embedder:
        result = embedder.embed(sequences, return_contacts=args.return_contacts)
    elapsed = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "ids": np.array(ids),
        "per_residue": result.per_residue,
        "per_sequence": result.per_sequence,
    }
    if result.contacts is not None:
        save_kwargs["contacts"] = result.contacts
    np.savez(args.out, **save_kwargs)

    print(f"Done in {elapsed:.1f}s")
    print(f"  per_residue:  shape={result.per_residue.shape}  dtype={result.per_residue.dtype}")
    print(f"  per_sequence: shape={result.per_sequence.shape}  dtype={result.per_sequence.dtype}")
    if result.contacts is not None:
        print(f"  contacts:     shape={result.contacts.shape}  dtype={result.contacts.dtype}")
    print(f"  saved:        {args.out}")
    print()

    # Show a tiny snippet so the user can eyeball that embeddings differ.
    print("First 5 dims of each per-sequence embedding:")
    for label, vec in zip(ids, result.per_sequence):
        print(f"  {label:<10} {vec[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
