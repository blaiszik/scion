# Scion

Scion makes it easy to use protein foundation models for **small-molecule drug discovery** on national lab and academic HPC clusters. The primary focus is workflows that combine structure, ligand co-folding, binding affinity, and (eventually) docking and ligand-aware design — from a single Python interface, without managing the conflicting environments each model requires.

Scion is the protein sibling of [Rootstock](https://github.com/Garden-AI/rootstock). Rootstock does this for machine-learned interatomic potentials; Scion does it for protein foundation models. A scion is the upper graft on a rootstock — same trunk, different fruit.

## Status

Scion is **early-stage software**. The v0 skeleton ships:

- **`fold`** capability — sequence → structure, with protein-ligand co-folding, multimers, nucleic acids, and binding-affinity prediction. `boltz_env` is **wired** against the full Boltz-2 capability surface (CLI in-process from the worker venv; returns mmCIF + confidence JSON + affinity JSON).
- **`embed`** capability — sequence → embeddings. `esm2_env` is **wired** against `fair-esm` (per-residue + mean-pooled per-sequence, optional contacts).

Next on the drug-discovery roadmap: `dock` (DiffDock-L, NeuralPLexer) for fast virtual screening; `design_sequence` (LigandMPNN) for ligand-aware inverse folding / binder optimization. `esmfold_env` and `chai_env` are deferred — Boltz-2 covers the same ground with strictly more capabilities. The wire protocol is method-name-dispatched, so adding a new capability is a new env file + a new client class.

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
scion check boltz_env --device cpu            # diagnose a built env (calls setup() directly)
scion preload boltz_env                       # pre-warm model cache on a login node before
                                              #   submitting a GPU job (calls provider.preload())
scion resolve --cluster polaris               # look up a cluster's root path
scion serve esm2 --socket /tmp/s --checkpoint esm2_t33_650M_UR50D --device cuda
scion manifest show                           # dump installation manifest
scion manifest push                           # push manifest to dashboard (maintainer only)
```

`--root` can be passed to any command, or set via `SCION_ROOT`, or stored in `~/.config/scion/config.toml`.

## Per-cluster environment overrides

A `cluster.toml` placed at the install root (`{root}/cluster.toml`) lets the maintainer set environment variables Scion applies to every subprocess it spawns — worker startup, `scion check`, and the `scion install --models` prebuild. Three optional tables:

```toml
[env]          # always applied
[login_env]    # only outside a PBS/SLURM batch job
[compute_env]  # only inside a PBS/SLURM batch job
```

Job detection looks at `PBS_JOBID` (Polaris) and `SLURM_JOB_ID` (Della, Perlmutter). See `cluster.toml.example` for a Polaris-ready starter that caps `OMP_NUM_THREADS=1` on login nodes (avoids the `libgomp: Thread creation failed` crash when loading large models on the shared login tier).

## Setting up a new cluster

Maintainer flow:

```bash
pip install scion
scion init                                    # configure root + maintainer + (optional) API
scion install environments/esm2_env.py        # builds an isolated venv under {root}/envs/
scion install environments/boltz_env.py       # likewise for fold (once provider is wired)
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

## Relationship to Rootstock

Scion shares Rootstock's pre-built-environment + subprocess + Unix-socket architecture. The protocol is different — Rootstock uses i-PI (a tight numerical loop for energies/forces); Scion uses a generic capability RPC (one-shot calls that ship strings and blobs of arbitrary size). The environment management, CLI, manifest, and cluster registry are direct ports.
