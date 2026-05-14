"""
Polaris demo: fold a ~30-residue peptide with Boltz-2 and print metrics.

Submitted via submit_demo.sh on a Polaris GPU node. Writes the predicted
mmCIF to the submission directory and prints confidence scores. Designed
to finish in under a minute on a single A100, so a failed setup surfaces
quickly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from scion import Folder

# Ubiquitin N-terminus (residues 1-30). Well-folded β-grasp; Boltz-2
# routinely returns pLDDT > 80 on this. Short enough to fold in seconds
# on an A100, long enough that "the model actually ran" is unambiguous.
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ"

SCION_ROOT = os.environ.get("SCION_ROOT")
if not SCION_ROOT:
    sys.exit(
        "Error: $SCION_ROOT is not set. The PBS script (submit_demo.sh) sets "
        "this; run it via `qsub -v SCION_ROOT ...` or export the var first."
    )

# Polaris compute nodes always expose /dev/nvidia0; login nodes don't.
# Fall back to CPU so the script is also runnable interactively for
# smoke-testing the wiring without a job.
device = "cuda" if Path("/dev/nvidia0").exists() else "cpu"

print("=== Scion / Boltz-2 fold demo on Polaris ===")
print(f"  SCION_ROOT: {SCION_ROOT}")
print(f"  hostname:   {os.uname().nodename}")
print(f"  device:     {device}")
print(f"  sequence:   {SEQUENCE}  ({len(SEQUENCE)} residues)")
print(flush=True)

with Folder(root=SCION_ROOT, model="boltz", checkpoint="boltz2", device=device) as folder:
    result = folder.fold(SEQUENCE, num_recycles=3)

out_cif = Path("polaris_demo.cif").resolve()
out_cif.write_text(result.mmcif)

confidence = result.confidence or {}

def fmt(key: str) -> str:
    v = confidence.get(key)
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

print()
print("Fold complete.")
print(f"  mmCIF:               {out_cif}  ({out_cif.stat().st_size} bytes)")
print(f"  confidence_score:    {fmt('confidence_score')}")
print(f"  pTM:                 {fmt('ptm')}")
print(f"  iPTM:                {fmt('iptm')}")
print(f"  complex_plddt:       {fmt('complex_plddt')}")
print()
print("If complex_plddt is > 70 you have a successful end-to-end install.")
