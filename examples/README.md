# Scion examples

Runnable smoke tests. Each script is self-contained — `pip install scion`
and a built env on a cluster are the only prerequisites.

| Script | What it does | When to run it |
|---|---|---|
| `embed_basic.py` | ESM2 per-residue and per-sequence embeddings for one or more sequences; saves as ``.npz``. | First thing after `scion install esm2_env.py` succeeds. Smallest model in Scion — fastest sanity check. |
| `fold_basic.py` | Boltz monomer fold of a short sequence; writes mmCIF + confidence JSON. | First thing after `scion install boltz_env.py` succeeds. Validates that the fold path works end-to-end. |
| `fold_drug_discovery.py` | Protein + ligand (SMILES or CCD) co-fold with binding-affinity prediction. | After `fold_basic.py` succeeds. The flagship drug-discovery workflow. |
| `showcase.py` | Same drug-discovery call as above but stripped to the minimum (no CLI, no helpers). | Tweet-sized screenshot of what Scion does. |
| `lead_optimization.py` | Boltz baseline → LigandMPNN redesign → Boltz validation. Three Scion capabilities composed end-to-end (fold + design + fold). | Once `ligandmpnn_env` is built and wired. Steps 1 & 3 run today against `boltz_env`; step 2 is scaffolded. |

## Quick start

On a GPU node with a Scion cluster registered:

```bash
# Smallest, fastest smoke test: ESM2 embeddings for two short sequences
python examples/embed_basic.py --cluster polaris

# Basic monomer fold (Trp-cage by default, ~20 residues, fast)
python examples/fold_basic.py --cluster polaris

# Drug-discovery flagship: CA2 + aspirin + affinity
python examples/fold_drug_discovery.py --cluster polaris

# Custom inputs
python examples/embed_basic.py --fasta my_proteins.fasta --out ./embeddings.npz
python examples/fold_drug_discovery.py \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
    --smiles 'CC(=O)Oc1ccccc1C(=O)O' \
    --out ./my_complex
```

Both scripts use `cluster="polaris"` by default; pass `--root <path>` to
override, or `--device cpu` to test (slowly) without a GPU.

Output goes to `./out/` (basic) or `./out_complex/` (drug discovery).
