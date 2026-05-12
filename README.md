# Scion

Scion makes it easy to use protein foundation models on national lab and academic HPC clusters. Researchers can use multiple models (Boltz-2, ESMFold, Chai, ESM2, ESM-C, ProteinMPNN, RFDiffusion, ...) from a single Python interface without managing the conflicting Python environments that each model requires.

Scion is the protein sibling of [Rootstock](https://github.com/Garden-AI/rootstock). Rootstock does this for machine-learned interatomic potentials; Scion does it for protein foundation models. A scion is the upper graft on a rootstock — same trunk, different fruit.

## Status

Scion is **early-stage software**. The v0 skeleton covers two capabilities:

- **`fold`** — sequence → structure (Boltz-2, ESMFold, Chai)
- **`embed`** — sequence → embeddings (ESM2, ESM-C)

Planned: `design_sequence` (ProteinMPNN, LigandMPNN), `generate` (RFDiffusion, Chroma), `dock` (DiffDock, NeuralPLexer), `score` (ThermoMPNN, AF-Multimer iptm).

## Quick start

```python
from scion import Folder, Embedder

with Folder(cluster="della", model="boltz", checkpoint="boltz2", device="cuda") as folder:
    result = folder.fold("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
    print(result.mmcif[:200])
    print(result.confidence["plddt_mean"])

with Embedder(cluster="della", model="esm2", checkpoint="esm2_t33_650M_UR50D") as embedder:
    result = embedder.embed(["MKTAYIAKQRQISFVKSHFSRQ"])
    print(result.per_residue.shape)   # (1, 22, 1280)
    print(result.per_sequence.shape)  # (1, 1280)
```

Swapping `model="boltz"` to `model="esmfold"` swaps the underlying fold model with no other changes.

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

| Capability | Model | Environment | Default checkpoint |
|------------|-------|-------------|--------------------|
| fold | `boltz` | `boltz_env` | `boltz2` |
| fold | `esmfold` | `esmfold_env` | `esmfold_v1` |
| fold | `chai` | `chai_env` | `chai1` |
| embed | `esm2` | `esm2_env` | `esm2_t33_650M_UR50D` |
| embed | `esmc` | `esmc_env` | `esmc_300m` |

Only `boltz_env` and `esm2_env` ship as v0 reference environments. Adding a new one is a small PEP 723 file in `environments/`.

## Setting up a new cluster

Maintainer flow mirrors Rootstock: `scion init`, then `scion install environments/boltz_env.py`, etc. See `scion --help`.

## Local development

```bash
git clone https://github.com/Garden-AI/scion.git
cd scion
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check scion/
pytest tests/
```

## Relationship to Rootstock

Scion shares Rootstock's pre-built-environment + subprocess + Unix-socket architecture. The protocol is different — Rootstock uses i-PI (a tight numerical loop for energies/forces); Scion uses a generic capability RPC (one-shot calls that ship strings and blobs of arbitrary size). The environment management, CLI, manifest, and cluster registry are direct ports.
