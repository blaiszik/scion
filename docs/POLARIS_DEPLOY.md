# Deploying Scion on Polaris (ALCF)

End-to-end recipe for installing Scion on the Argonne Polaris cluster and running a Boltz-2 fold on a single A100 node. Designed so the steps you run on the cluster are scripts checked into this repo — copy-paste the URLs into your terminal, not the commands.

Roughly 30 minutes wall-clock end to end, dominated by the one-time download of Boltz-2 weights (~5 GB) during the auto-preload step.

## What you need before you start

- An ALCF Polaris account and an allocation (the `-A <project>` you'll pass to `qsub`).
- A directory you can write to with at least ~10 GB free. **Recommended:** `/lus/eagle/projects/<YOUR_PROJECT>/scion` (project allocation, fast, large). **Fallback:** `$HOME/scion` (small home quota; weights will eat it).
- This repo, cloned anywhere on Polaris. The scripts are stand-alone — they don't need the rest of the repo at runtime, only when you invoke them.

```bash
ssh polaris.alcf.anl.gov
git clone https://github.com/blaiszik/scion.git ~/scion-deploy
cd ~/scion-deploy
```

## Step 0 — Clean up a previous install (optional)

If a previous Scion install exists at the same `$SCION_ROOT`, wipe it before reinstalling. Cache and downloaded weights are preserved by default; pass `--wipe-cache` if you want to start from zero.

```bash
export SCION_ROOT=/lus/eagle/projects/<YOUR_PROJECT>/scion
bash scripts/polaris/cleanup.sh
# OR, to also drop the ~5 GB Boltz weight cache:
bash scripts/polaris/cleanup.sh --wipe-cache
```

The script prompts before deleting. It removes `envs/`, `environments/`, `.python/`, `manifest.json`, and `cluster.toml`. Anything else under `$SCION_ROOT` (e.g. files you put there yourself) is untouched.

> [!NOTE]
> If you also want to remove the scion CLI itself: `rm -rf ~/.venvs/scion-cli`. The `uv` installer at `~/.local/bin/uv` is harmless to leave in place.

## Step 1 — Install scion and build the worker envs

Run this from a **login node** (compute nodes can't reach the public internet on Polaris without proxy setup, and the install needs HF Hub for weights).

```bash
export SCION_ROOT=/lus/eagle/projects/<YOUR_PROJECT>/scion
bash scripts/polaris/install.sh
```

What the script does:

1. `module load conda` to get a recent Python.
2. Installs `uv` to `~/.local/bin` if missing (Scion uses uv to build isolated worker venvs).
3. Creates `~/.venvs/scion-cli` and installs the `scion` CLI from GitHub into it.
4. Creates `$SCION_ROOT/{environments,envs,cache,home,.python}`.
5. Writes a starter `$SCION_ROOT/cluster.toml` with `OMP_NUM_THREADS=1` etc. for login-node use — avoids the `libgomp: Thread creation failed` crash on Polaris's shared login tier.
6. Builds `esm2_env` (sequence → embeddings; ~3 min, ~2.5 GB weights).
7. Builds `boltz_env` (sequence/ligand → structure + affinity; ~20 min, ~5 GB weights downloaded by the Boltz CLI during the auto-preload).

The install streams uv's output and Boltz's weight-download progress live, so a quiet terminal means something is wrong, not stuck. Expected end state: `scion status` shows two envs built and ready.

### Re-running the install

`install.sh` is idempotent: it skips envs already present. To force a rebuild of a single env:

```bash
source ~/.venvs/scion-cli/bin/activate
export SCION_ROOT=...
scion install boltz_env --force
```

## Step 2 — Submit the demo job

The demo folds residues 1–30 of human ubiquitin (a well-characterized β-grasp domain that finishes in under a minute on a single A100). Tight enough that a broken install surfaces immediately, real enough that the result is meaningful.

```bash
qsub -A <YOUR_PROJECT> -v SCION_ROOT scripts/polaris/submit_demo.sh
```

`-v SCION_ROOT` forwards your current `$SCION_ROOT` into the job's environment. `-A` is required by Polaris's PBS.

The job script (`submit_demo.sh`) is a standard Polaris debug-queue PBS file: 1 node, 4 GPUs, 20 min walltime, joined stdout/stderr. It activates the same `~/.venvs/scion-cli` and runs `demo_fold.py`.

Watch the queue:

```bash
qstat -u $USER
# pbsnodes -l reservation   # only if you booked something
```

When the job finishes, look at the output file (`scion_demo.o<jobid>` in your submit directory):

```bash
cat scion_demo.o<jobid>
```

Expected output ends with something like:

```
Fold complete.
  mmCIF:               /lus/.../polaris_demo.cif  (12345 bytes)
  confidence_score:    0.842
  pTM:                 0.612
  iPTM:                n/a
  complex_plddt:       82.4

If complex_plddt is > 70 you have a successful end-to-end install.
```

The `polaris_demo.cif` file is a real predicted structure you can open in PyMOL / ChimeraX.

## Step 3 — Use Scion from your own code

The same call works anywhere on Polaris once Step 1 succeeded:

```python
from scion import Folder

with Folder(root="/lus/eagle/projects/<YOUR_PROJECT>/scion",
            model="boltz", device="cuda") as f:
    r = f.fold("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ")
    print(r.confidence)
    open("out.cif", "w").write(r.mmcif)
```

For the headline drug-discovery use case (co-fold with a ligand + binding affinity), pass `ligands=[{"smiles": "..."}], predict_affinity=True`. See the project root README for full call shapes.

## Troubleshooting

The `scion check --thorough` command runs the same setup-then-preload path the install does, in isolation, and tags known failure patterns with structural fixes. Always reach for this first:

```bash
source ~/.venvs/scion-cli/bin/activate
export SCION_ROOT=...
scion doctor --cluster polaris
scion check boltz_env --thorough --device cpu
```

Common Polaris-specific failure modes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: NVIDIA driver on your system is too old` | torch wheel newer than Polaris driver supports (>= 12.9 needed). | The shipped env files already pin `torch<2.9`. If you've edited them, re-pin and `scion install <env> --force`. |
| `ModuleNotFoundError: cuequivariance_*` | Edit to `boltz_env.py` dropped a CUDA-kernel dep. | Restore the two `cuequivariance-*` lines in the env's PEP 723 block and `scion install boltz_env --force`. |
| `libgomp: Thread creation failed` on a login-node call | `$SCION_ROOT/cluster.toml` missing or `[login_env]` block dropped. | Re-seed: `bash scripts/polaris/install.sh` (skips already-built envs but rewrites the cluster.toml if absent). |
| Boltz weight download fails inside the job | Compute node has no direct outbound. | Re-run `scion preload boltz_env` on a login node; the cache at `$SCION_ROOT/home/.boltz` is reused inside jobs. |
| `qsub: account ... not active` | Wrong `-A` value or expired allocation. | `accounts` to list, pick a live one. |

Hints from `scion check` already point at the relevant fix when they fire — you should rarely need this table.
