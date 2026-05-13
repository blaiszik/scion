#!/usr/bin/env python
"""
Minimal Boltz monomer fold via Scion.

The simplest end-to-end test: spawn a Boltz worker, fold one short
sequence on GPU, write the predicted mmCIF and confidence JSON to
disk. Use this to confirm the `fold` path works before adding ligands
and affinity (see ``fold_drug_discovery.py`` for that).

Usage::

    python examples/fold_basic.py
    python examples/fold_basic.py --cluster polaris
    python examples/fold_basic.py --root /lus/eagle/projects/Garden-Ai/scion
    python examples/fold_basic.py --sequence MKTAYIAKQ... --out ./out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scion import Folder

# A short protein (Trp-cage variant, 20 residues) — folds quickly and has
# clean expected structure. Replace with anything you care about.
DEFAULT_SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE,
                        help=f"Protein sequence (default: {DEFAULT_SEQUENCE!r})")
    parser.add_argument("--cluster", default="polaris",
                        help="Cluster name (default: polaris)")
    parser.add_argument("--root", default=None,
                        help="Explicit Scion root (overrides --cluster)")
    parser.add_argument("--checkpoint", default="boltz2",
                        help="Boltz checkpoint (default: boltz2)")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--num-recycles", type=int, default=3,
                        help="Recycling steps (default: 3)")
    parser.add_argument("--out", type=Path, default=Path("./out"),
                        help="Output directory (default: ./out)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    kwargs = {"model": "boltz", "checkpoint": args.checkpoint,
              "device": args.device, "log": sys.stderr}
    if args.root is not None:
        kwargs["root"] = args.root
    else:
        kwargs["cluster"] = args.cluster

    print(f"Folding {len(args.sequence)}-residue sequence on {args.device} ...")
    t0 = time.perf_counter()
    with Folder(**kwargs) as folder:
        result = folder.fold(args.sequence, num_recycles=args.num_recycles)
    elapsed = time.perf_counter() - t0

    cif_path = args.out / "structure.cif"
    conf_path = args.out / "confidence.json"
    cif_path.write_text(result.mmcif)
    conf_path.write_text(json.dumps(result.confidence, indent=2))

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  mmCIF:      {cif_path}  ({len(result.mmcif):,} bytes)")
    print(f"  confidence: {conf_path}")
    print(f"  fields:     {sorted(result.confidence.keys())}")
    for key in ("confidence_score", "ptm", "iptm", "complex_plddt"):
        if key in result.confidence:
            print(f"  {key:<22} {result.confidence[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
