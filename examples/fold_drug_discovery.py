#!/usr/bin/env python
"""
Protein-ligand co-fold + binding-affinity prediction via Scion + Boltz-2.

The drug-discovery flagship workflow: given a target sequence and a
ligand (SMILES or PDB CCD code), predict the complex structure and an
estimated binding affinity in a single call. Writes the predicted
complex mmCIF, the structure confidence JSON, and (if requested) the
affinity JSON to disk.

Run ``fold_basic.py`` first to confirm Boltz works end-to-end; this
script adds the ligand and affinity head on top.

Usage::

    # Default: target + aspirin, predict affinity
    python examples/fold_drug_discovery.py

    # Custom target and ligand
    python examples/fold_drug_discovery.py \\
        --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \\
        --smiles 'CC(=O)Oc1ccccc1C(=O)O' \\
        --out ./aspirin_complex

    # CCD code instead of SMILES (e.g., ATP, NAD)
    python examples/fold_drug_discovery.py --ccd ATP

    # Skip affinity (structure only — faster)
    python examples/fold_drug_discovery.py --no-affinity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scion import Folder

# Default target: human carbonic anhydrase II (CA2) — well-studied
# drug target. Default ligand: aspirin (acetylsalicylic acid).
DEFAULT_SEQUENCE = (
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPL"
    "DGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGK"
    "SADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
)
DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE,
                        help="Receptor protein sequence")

    ligand_group = parser.add_mutually_exclusive_group()
    ligand_group.add_argument(
        "--smiles", default=None,
        help=f"Ligand SMILES (default if neither flag given: {DEFAULT_SMILES})",
    )
    ligand_group.add_argument("--ccd", default=None,
                              help="Ligand by PDB CCD code (e.g., ATP, NAD, HEM)")

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
    parser.add_argument("--no-affinity", action="store_true",
                        help="Skip the affinity head (structure only)")
    parser.add_argument("--out", type=Path, default=Path("./out_complex"),
                        help="Output directory (default: ./out_complex)")
    args = parser.parse_args()

    if args.smiles is None and args.ccd is None:
        args.smiles = DEFAULT_SMILES
    ligand = {"smiles": args.smiles} if args.smiles else {"ccd": args.ccd}
    ligand_label = args.smiles or f"CCD:{args.ccd}"

    args.out.mkdir(parents=True, exist_ok=True)

    kwargs = {"model": "boltz", "checkpoint": args.checkpoint,
              "device": args.device, "log": sys.stderr}
    if args.root is not None:
        kwargs["root"] = args.root
    else:
        kwargs["cluster"] = args.cluster

    print(f"Co-folding {len(args.sequence)}-residue target + ligand {ligand_label}")
    print(f"  predict_affinity = {not args.no_affinity}")
    print(f"  device           = {args.device}")
    print()

    t0 = time.perf_counter()
    with Folder(**kwargs) as folder:
        result = folder.fold(
            sequence=args.sequence,
            ligands=[ligand],
            predict_affinity=not args.no_affinity,
            num_recycles=args.num_recycles,
        )
    elapsed = time.perf_counter() - t0

    cif_path = args.out / "complex.cif"
    conf_path = args.out / "confidence.json"
    cif_path.write_text(result.mmcif)
    conf_path.write_text(json.dumps(result.confidence, indent=2))
    print(f"Done in {elapsed:.1f}s")
    print(f"  complex mmCIF: {cif_path}  ({len(result.mmcif):,} bytes)")
    print(f"  confidence:    {conf_path}")

    if result.affinity is not None:
        aff_path = args.out / "affinity.json"
        aff_path.write_text(json.dumps(result.affinity, indent=2))
        print(f"  affinity:      {aff_path}")
        print()
        print("Affinity prediction:")
        for k, v in result.affinity.items():
            print(f"  {k:<28} {v}")
    elif not args.no_affinity:
        print("  (note: affinity was requested but the model returned none — "
              "check Boltz output for missing affinity_*.json)")

    print()
    print("Structure scores:")
    for key in ("confidence_score", "ptm", "iptm", "complex_plddt",
                "complex_iplddt", "interface_plddt"):
        if key in result.confidence:
            print(f"  {key:<28} {result.confidence[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
