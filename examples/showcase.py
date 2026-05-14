"""
Scion + Boltz-2: protein-ligand co-fold + binding-affinity in one call.

This is the screenshot-ready showcase script — runnable on a Polaris
GPU node in under a minute, no environment management visible.

Submit on Polaris:
    cd ~/scion-deploy
    qsub -A <YOUR_PROJECT> -v SCION_ROOT -- python examples/showcase.py
    # ...or wrap in scripts/polaris/submit_demo.sh if you prefer.

The headline value: no torch wheels to debug, no HuggingFace cache to
seed, no conda env to activate. `cluster="polaris"` resolves to the
maintainer-installed shared environment under /lus/eagle/...; Scion
brokers the call into that env's worker over a Unix socket. The same
script runs on any cluster with a Scion install — swap the cluster
string or pass `root=`.
"""

from scion import Folder

with Folder(cluster="polaris", model="boltz", device="cuda") as f:
    r = f.fold(
        sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ",     # ubiquitin 1-30
        ligands=[{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}],  # aspirin
        predict_affinity=True,
    )

print(f"complex pLDDT:       {r.confidence['complex_plddt']:.3f}")
print(f"binding probability: {r.affinity['binding_probability']:.3f}")
print(f"log Kd:              {r.affinity['log_kd']:+.2f}")
open("complex.cif", "w").write(r.mmcif)
