# Scion

Scion makes it easy to use protein foundation models for **small-molecule drug discovery** on national lab and academic HPC clusters. The primary focus is workflows that combine structure, ligand co-folding, binding affinity, and (eventually) docking and ligand-aware design — from a single Python interface, without managing the conflicting environments each model requires.

Scion is the protein sibling of [Rootstock](https://github.com/Garden-AI/rootstock). Rootstock does this for machine-learned interatomic potentials; Scion does it for protein foundation models. A scion is the upper graft on a rootstock — same trunk, different fruit.

## Status

Scion is **early-stage software**. The v0 skeleton ships:

- **`fold`** capability — sequence → structure, with protein-ligand co-folding, multimers, nucleic acids, and binding-affinity prediction. `boltz_env` is **wired** against the full Boltz-2 capability surface (CLI in-process from the worker venv; returns mmCIF + confidence JSON + affinity JSON).
- **`embed`** capability — sequence → embeddings. `esm2_env` is **wired** against `fair-esm` (per-residue + mean-pooled per-sequence, optional contacts).

Next on the drug-discovery roadmap: `dock` (DiffDock-L, NeuralPLexer) for fast virtual screening; `design_sequence` (LigandMPNN) for ligand-aware inverse folding / binder optimization. `esmfold_env` and `chai_env` are deferred — Boltz-2 covers the same ground with strictly more capabilities. The wire protocol is method-name-dispatched, so adding a new capability is a new env file + a new client class.

## Showcase

Three workflows that Scion makes one-screen-of-code, and that would take a week of conda-wrangling without it.

### 1. Single-call drug discovery

Co-fold a target with a candidate ligand and request the Boltz-2 affinity head in the same call. No separate docking program, no separate scoring step, no separate environment.

```python
from scion import Folder

with Folder(cluster="polaris", model="boltz", device="cuda") as f:
    r = f.fold(
        sequence=ABL1_KINASE_DOMAIN,
        ligands=[{"smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"}],
        predict_affinity=True,
    )
    print(f"log Kd:              {r.affinity['log_kd']:+.2f}")
    print(f"binding probability: {r.affinity['binding_probability']:.3f}")
    print(f"complex pLDDT:       {r.confidence['complex_plddt']:.1f}")
    open("imatinib_complex.cif", "w").write(r.mmcif)
```

### 2. Two models with conflicting Python envs in one script

ESM2 needs `fair-esm` and one torch build; Boltz-2 needs `cuequivariance` and NVIDIA's CUDA-ops packages on a different torch build. Pip can't install both in one env. Scion gives each its own pre-built venv and brokers calls over a Unix socket — from the user's perspective, they're two context managers in the same file. Compose them freely:

```python
import numpy as np, pandas as pd
from scion import Embedder, Folder

# Embed a homolog family, pick the most divergent candidates
with Embedder(cluster="polaris", model="esm2", device="cuda") as e:
    embs = e.embed([TARGET] + HOMOLOG_LIBRARY)
    cos = embs.per_sequence @ embs.per_sequence[0]
    cos /= np.linalg.norm(embs.per_sequence, axis=1) * np.linalg.norm(embs.per_sequence[0])
    diverse_idx = cos[1:].argsort()[:20]

# Fold each with the same ligand, rank by predicted affinity
with Folder(cluster="polaris", model="boltz", device="cuda") as f:
    rows = []
    for i in diverse_idx:
        r = f.fold(sequence=HOMOLOG_LIBRARY[i],
                   ligands=[{"smiles": LIGAND_SMILES}],
                   predict_affinity=True)
        rows.append({"homolog": i, "log_kd": r.affinity["log_kd"],
                     "iptm": r.confidence.get("iptm", 0.0)})

print(pd.DataFrame(rows).sort_values("log_kd"))
```

### 3. Virtual screen a ligand library against one target

Boltz weights stay resident across the loop — one subprocess, one model load, per-ligand cost is just inference. No env activate/deactivate, no per-call spin-up, no docking-software babysitting.

```python
import pandas as pd
from pathlib import Path
from scion import Folder

library = pd.read_csv("ligand_library.csv")    # columns: name, smiles
poses_dir = Path("./poses"); poses_dir.mkdir(exist_ok=True)

with Folder(cluster="polaris", model="boltz", device="cuda") as f:
    rows = []
    for name, smiles in library.itertuples(index=False):
        r = f.fold(sequence=TARGET, ligands=[{"smiles": smiles}],
                   predict_affinity=True)
        rows.append({"ligand": name,
                     "log_kd": r.affinity["log_kd"],
                     "p_bind": r.affinity["binding_probability"],
                     "iptm":   r.confidence.get("iptm", 0.0)})
        (poses_dir / f"{name}.cif").write_text(r.mmcif)

pd.DataFrame(rows).sort_values("log_kd").to_csv("screen_results.csv", index=False)
```

## Quick start

```python
from scion import Folder, Embedder

# Embed (esm2_env is wired in v0)
with Embedder(cluster="della", model="esm2", checkpoint="esm2_t33_650M_UR50D") as embedder:
    result = embedder.embed(["MKTAYIAKQRQISFVKSHFSRQ"])
    print(result.per_residue.shape)   # (1, 22, 1280)
    print(result.per_sequence.shape)  # (1, 1280)

# Fold a monomer via Boltz-2.
with Folder(cluster="polaris", model="boltz", checkpoint="boltz2", device="cuda") as folder:
    result = folder.fold("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
    print(result.mmcif[:200])
    print(result.confidence)             # {"confidence_score": 0.83, "ptm": 0.45, ...}

# Protein-ligand co-fold + binding-affinity prediction (the drug-discovery path).
with Folder(cluster="polaris", model="boltz", checkpoint="boltz2", device="cuda") as folder:
    result = folder.fold(
        sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        ligands=[{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}],   # aspirin
        predict_affinity=True,
    )
    print(result.confidence)             # structure-quality scores
    print(result.affinity)               # {"kd": ..., "log_kd": ..., "binding_probability": ...}

# Multimer + cofactor + DNA (Boltz-2 handles all chain types in one call).
with Folder(cluster="polaris", model="boltz", device="cuda") as folder:
    result = folder.fold(
        sequence=["MKTAYI...", "GSHMAS..."],            # two protein chains
        ligands=[{"ccd": "ATP"}],                       # cofactor by PDB code
        nucleic_acids=[{"type": "dna", "sequence": "ATGCATGC"}],
    )
```

`cluster="della"` resolves to a maintainer-installed shared directory (see `CLUSTER_REGISTRY`). For a custom path use `root="/path/to/scion"` instead of `cluster=...`. Swapping `model="boltz"` to `model="esmfold"` will swap the underlying fold model with no other code changes once those envs are added.

## Installation

The lightweight `scion` package is what users install. The heavy ML dependencies (Boltz, ESM, PyTorch, JAX, ...) live in pre-built environments on the cluster.

```bash
pip install scion
```

## Architecture

When you create a `Folder` or `Embedder`, Scion spawns a subprocess that runs the model in its own pre-built virtual environment. The main process and the worker communicate over a Unix domain socket using a small request/response RPC (length-prefixed JSON header + binary blobs). This happens on a single node (no remote network calls).

```
Your script (on cluster node)            Worker subprocess
+-----------------------------+        +-------------------------------+
| Folder / Embedder           |        | Pre-built model environment   |
| (capability client)         |        |                               |
|                             |        |                               |
| session.py (RPC server)     |<------>| worker.py (RPC dispatch)      |
| - sends method + args       |  Unix  | - receives method + args      |
| - receives result + blobs   | socket | - calls provider.fold/.embed  |
+-----------------------------+        +-------------------------------+
```

Each environment file declares which capabilities it provides (`CAPABILITIES = ["fold"]`) and exposes a `setup(model, device)` function that returns a *provider* — an object with `.fold(...)` and/or `.embed(...)` methods matching the capability contracts.

The single-node design means models don't have to know about distributed inference. The IPC overhead is negligible relative to model forward-passes.

## Capability contracts

### `fold`

```python
provider.fold(
    sequence: str | list[str],          # FASTA sequence or list (multimer)
    msa: bytes | None = None,           # optional A3M MSA blob
    templates: bytes | None = None,     # optional CIF templates blob
    num_recycles: int = 3,
    ...
) -> {
    "mmcif": str,                       # predicted structure
    "confidence": dict,                 # pLDDT mean, pTM, ipTM, ...
    "plddt": ndarray,                   # per-residue
}
```

### `embed`

```python
provider.embed(
    sequences: list[str],
    repr_layers: tuple[int, ...] = (33,),
    return_contacts: bool = False,
) -> {
    "per_residue": ndarray,             # (B, L, D)
    "per_sequence": ndarray,            # (B, D), mean-pooled
    "contacts": ndarray | None,         # (B, L, L) if requested
}
```

## Available models

| Capability | Model | Environment | Default checkpoint | v0 status |
|------------|-------|-------------|--------------------|-----------|
| fold | `boltz` | `boltz_env` | `boltz2` | wired (subprocess to Boltz CLI; reloads weights per call) |
| fold | `esmfold` | `esmfold_env` | `esmfold_v1` | planned |
| fold | `chai` | `chai_env` | `chai1` | planned |
| embed | `esm2` | `esm2_env` | `esm2_t33_650M_UR50D` | wired |
| embed | `esmc` | `esmc_env` | `esmc_300m` | planned |
| design_sequence | `ligandmpnn` | `ligandmpnn_env` | `ligandmpnn_v_32_010_25` | scaffolded (env builds; provider raises until Polaris-wired) |
| dock | `diffdock` | `diffdock_env` | `diffdock_l` | scaffolded (env builds; provider raises until Polaris-wired) |

`environments/` ships `boltz_env.py` (skeleton showing the fold contract) and `esm2_env.py` (working `embed` against `fair-esm`). Adding another model is a small PEP 723 file in the same directory.

## CLI

```bash
scion init                                    # interactive setup of config + dirs + manifest
scion install environments/esm2_env.py        # build an env from a file
scion install environments/                   # build every *.py env in a directory
scion install esm2_env --force                # rebuild a registered env
scion sync environments/boltz_env.py          # refresh env_source.py in the worker venv
                                              #   without rebuilding (fast path for env-only edits)
scion sync                                    # refresh env_source.py across every built env
scion status                                  # show envs, capabilities, cache sizes
scion list                                    # list registered envs
scion doctor                                  # check cluster/root/env readiness
scion check boltz_env --device cpu            # diagnose a built env (calls setup() directly)
scion preload boltz_env                       # pre-warm model cache on a login node before
                                              #   submitting a GPU job (calls provider.preload())
scion resolve --cluster polaris               # look up a cluster's root path
scion serve esm2 --socket /tmp/s --checkpoint esm2_t33_650M_UR50D --device cuda
scion manifest show                           # dump installation manifest
scion manifest push                           # push manifest to dashboard (maintainer only)
```

`--root` can be passed to any command, or set via `SCION_ROOT`, or stored in `~/.config/scion/config.toml`.

## Cluster registry and profiles

Scion ships built-in profiles for known systems (`della`, `sophia`, `polaris`) and overlays user/site profiles from `~/.config/scion/clusters.toml`. Set `SCION_CLUSTERS_FILE=/path/to/clusters.toml` to add another overlay without editing the default file.

```toml
[clusters.perlmutter]
root = "/global/common/software/scion"
scheduler = "slurm"
job_env_vars = ["SLURM_JOB_ID"]
hostname_patterns = ["perlmutter*", "*.nersc.gov"]
gpu_arch = "a100"
cuda_driver_max = "12.4"
runtime_dir = "$SCRATCH/.scion/run"
```

Later layers replace earlier profiles with the same name, so a site maintainer can override the built-in root or scheduler metadata without changing user code. `cluster="current"` and `scion resolve --cluster current` use hostname patterns for best-effort auto-detection.

## Per-cluster environment overrides

A `cluster.toml` placed at the install root (`{root}/cluster.toml`) lets the maintainer set environment variables Scion applies to every subprocess it spawns — worker startup, `scion check`, and the `scion install --models` prebuild. Three optional tables:

```toml
[env]          # always applied
[login_env]    # only outside a PBS/SLURM batch job
[compute_env]  # only inside a PBS/SLURM batch job
```

Job detection defaults to `PBS_JOBID` (Polaris) and `SLURM_JOB_ID` (Della, Perlmutter), and cluster profiles can override the job env vars for other schedulers. See `cluster.toml.example` for a Polaris-ready starter that caps `OMP_NUM_THREADS=1` on login nodes (avoids the `libgomp: Thread creation failed` crash when loading large models on the shared login tier).

`scion doctor` reports which profile and `cluster.toml` overlays are active, whether Scion can write to its cache/home directories, whether login-node thread caps are in place, whether `uv` is available for installs, whether GPUs are visible via `nvidia-smi`, and whether registered/built envs look complete. It is a lightweight preflight; use `scion check <env> --thorough` for model-specific inference-path validation.

## Setting up a new cluster

Maintainer flow:

```bash
pip install scion
scion init                                    # configure root + maintainer + (optional) API
scion doctor                                  # validate root/profile/cluster.toml basics
scion install environments/esm2_env.py        # builds an isolated venv under {root}/envs/
scion install environments/boltz_env.py       # likewise for fold (once provider is wired)
scion doctor --env boltz_env                  # confirm env layout and cache readiness
scion status                                  # confirm
```

Each `scion install` creates an isolated venv at `{root}/envs/{env_name}/` from the PEP 723 deps in the source file, copies the source in as `env_source.py`, installs Scion itself, and refreshes the manifest.

## Local development

```bash
git clone https://github.com/blaiszik/scion.git
cd scion
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check scion/
pytest tests/
```

The protocol round-trip tests (4 cases) verify framing, named blob attachment, numpy-array blob payloads, and large header round-trips. They run without GPU.

## Lessons from the field

Bringing Boltz-2 up on Polaris surfaced five distinct dep/build paper cuts. Each had a clean structural fix; together they shape how new envs should be authored. Pinning them here so future env work avoids the same potholes.

| Paper cut | What surfaced | Structural fix |
|---|---|---|
| `libgomp: Thread creation failed` on a login node | Shared HPC login nodes apply tight `RLIMIT_NPROC`; torch/MKL/OpenBLAS spin up one thread per CPU core and the kernel says no. | Per-cluster `cluster.toml` with `[login_env] OMP_NUM_THREADS=1` (see `cluster.toml.example`). `scion check` also applies the cap as a diagnostic fallback. |
| `ModuleNotFoundError: cuequivariance_torch` during the first fold | Boltz imports `cuequivariance_torch` during model graph construction, *after* `import boltz` succeeds. The dep isn't in Boltz's pip metadata. | Declare it explicitly in the env file's PEP 723 deps. Don't trust upstream's transitive resolution for ML libraries. |
| `ModuleNotFoundError: cuequivariance_ops_torch` after the previous fix | NVIDIA splits cuequivariance into bindings (`cuequivariance-torch`) and CUDA-version-specific kernels (`cuequivariance-ops-torch-cu12`). The bindings import the kernels unconditionally. | Pin both packages explicitly. CUDA-suffixed packages need a cluster-appropriate suffix (`-cu12` for modern A100/H100, `-cu11` for older clusters). |
| `RuntimeError: NVIDIA driver too old (found version 12080)` | `pip install torch>=2.0` resolves to torch 2.9+ which is built against CUDA 12.9. Polaris's driver caps at 12.8. | Cap torch at `<2.9` in the env file. `[tool.uv] extra-index-url` is **not** sufficient — uv's default `first-index` strategy still prefers PyPI when both list `torch`. Version-pinning is the only mechanism that reliably constrains wheel selection. |
| `undefined symbol: ncclGetLsaMultimemDevicePointer` after surgical `--no-deps` reinstall | Replacing torch alone left an incompatible `nvidia-nccl-cu12` resident. Torch and NVIDIA libs have tight ABI coupling. | Don't use `--no-deps` for torch surgery — let uv resolve the matched NVIDIA libs. For env-level fixes, prefer `scion install --force` over per-package patches once the env file is correct. |

**Meta-lesson:** ML library dependency declarations don't reliably express what's needed to *run*. The default `scion check` only catches import-level breakage, but most of these surfaced during model construction (after `import` returned, before the first real call). That's why `provider.preload()` and `scion check --thorough` exist — they shift discovery from "after a GPU job's queue time" to "in 30 seconds on a login node." **Every new env should implement `provider.preload()`** so users can validate it before burning allocation.

## Relationship to Rootstock

Scion shares Rootstock's pre-built-environment + subprocess + Unix-socket architecture. The protocol is different — Rootstock uses i-PI (a tight numerical loop for energies/forces); Scion uses a generic capability RPC (one-shot calls that ship strings and blobs of arbitrary size). The environment management, CLI, manifest, and cluster registry are direct ports.
