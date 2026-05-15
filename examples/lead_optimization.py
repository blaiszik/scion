"""
Lead optimization in one screen: Boltz baseline -> LigandMPNN redesign
-> Boltz validation.

Demonstrates three Scion capabilities composed end-to-end:

    1. fold (Boltz-2)               -> co-fold target + ligand,
                                       baseline complex_plddt + log_kd
    2. design_sequence (LigandMPNN) -> rewrite pocket residues,
                                       ligand-aware
    3. fold (Boltz-2)               -> re-fold the designed target
                                       with the same ligand, get a
                                       new predicted affinity

Output: the design diff (mutations relative to wild-type), the two
predicted log_kd values, and the optimized mmCIF file.

Status: runnable once ligandmpnn_env has its inference path wired (the
current scaffold raises a NotImplementedError with a clear message
when step 2 executes). Steps 1 and 3 work today against a built
boltz_env.

Submit on Polaris:
    cd ~/scion-deploy
    qsub -A <YOUR_PROJECT> -v SCION_ROOT \\
        -- python examples/lead_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from scion import Designer, Folder

# A kinase-inhibitor test case. Real targets and ligands; values used
# only as a smoke test, not as a benchmark claim.
TARGET_SEQUENCE = (
    # Truncated ABL1 SH1 kinase domain (~75 residues, fits comfortably
    # on a single A100). Wild-type sequence.
    "MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNS"
)
LIGAND_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"  # imatinib

# Pocket residues to redesign. In a real study these come from a binding-
# site analysis; here we pick a small contiguous patch as a placeholder.
POCKET_POSITIONS = [12, 13, 14, 15, 16]


def diff_sequences(wt: str, mut: str) -> str:
    """Pretty-print mutations as ``W12L,Y14F`` strings."""
    pairs = []
    for i, (a, b) in enumerate(zip(wt, mut), start=1):
        if a != b:
            pairs.append(f"{a}{i}{b}")
    return ",".join(pairs) if pairs else "(no changes)"


def main() -> int:
    print("=== Lead optimization demo: Boltz -> LigandMPNN -> Boltz ===")
    print(f"  target:  {TARGET_SEQUENCE}")
    print(f"  ligand:  {LIGAND_SMILES}")
    print(f"  pocket:  {POCKET_POSITIONS}")
    print()

    # ---- Step 1: baseline ----------------------------------------------
    print("[1/3] Baseline co-fold with Boltz-2...")
    with Folder(cluster="polaris", model="boltz", device="cuda") as f:
        wt = f.fold(
            sequence=TARGET_SEQUENCE,
            ligands=[{"smiles": LIGAND_SMILES}],
            predict_affinity=True,
        )
    Path("wt_complex.cif").write_text(wt.mmcif)
    print(f"  WT complex_plddt:  {wt.confidence.get('complex_plddt', 'n/a')}")
    print(f"  WT log_kd:         {wt.affinity.get('log_kd', 'n/a') if wt.affinity else 'n/a'}")
    print()

    # ---- Step 2: design ------------------------------------------------
    print("[2/3] Ligand-aware redesign with LigandMPNN...")
    try:
        with Designer(cluster="polaris", model="ligandmpnn", device="cuda") as d:
            design = d.design_sequence(
                mmcif=wt.mmcif,
                positions=POCKET_POSITIONS,
                ligands=[{"smiles": LIGAND_SMILES}],
                num_sequences=4,
                temperature=0.1,
            )
    except NotImplementedError as e:
        print(f"  Skipping design (env not yet wired): {e}", file=sys.stderr)
        print("  When ligandmpnn_env is ready, this step yields a "
              "designed sequence; the next step folds it.", file=sys.stderr)
        return 1
    designed_seq = design.sequence
    print(f"  Mutations:         {diff_sequences(TARGET_SEQUENCE, designed_seq)}")
    print(f"  Sampled scores:    {design.scores}")
    print()

    # ---- Step 3: validate ----------------------------------------------
    print("[3/3] Validation co-fold of the designed target...")
    with Folder(cluster="polaris", model="boltz", device="cuda") as f:
        mut = f.fold(
            sequence=designed_seq,
            ligands=[{"smiles": LIGAND_SMILES}],
            predict_affinity=True,
        )
    Path("designed_complex.cif").write_text(mut.mmcif)
    mut_lk_val = (mut.affinity or {}).get("log_kd", "n/a")
    print(f"  Designed complex_plddt: {mut.confidence.get('complex_plddt', 'n/a')}")
    print(f"  Designed log_kd:        {mut_lk_val}")
    print()

    # ---- Summary -------------------------------------------------------
    wt_lk = (wt.affinity or {}).get("log_kd")
    mut_lk = (mut.affinity or {}).get("log_kd")
    print("=== Summary ===")
    print(f"  WT       log_kd: {wt_lk}")
    print(f"  designed log_kd: {mut_lk}")
    if isinstance(wt_lk, (int, float)) and isinstance(mut_lk, (int, float)):
        delta = mut_lk - wt_lk
        direction = "tighter" if delta < 0 else "weaker"
        print(f"  delta: {delta:+.2f}  ({direction} binding predicted)")
    print("  Structures: wt_complex.cif, designed_complex.cif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
