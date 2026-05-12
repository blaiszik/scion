# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scion is a proof-of-concept for running protein foundation models (Boltz, ESMFold, Chai, ESM2, ESM-C, ProteinMPNN, RFDiffusion, ...) in isolated pre-built Python environments, communicating via a small capability RPC over Unix sockets.

Scion is the protein sibling of [Rootstock](https://github.com/Garden-AI/rootstock). Rootstock applies the same architecture to ML interatomic potentials.

**Current version: v0.1** — skeleton. Two capabilities: `fold` and `embed`. Two reference environments: `boltz_env`, `esm2_env`.

## Architecture

```
Main Process                              Worker Process (subprocess)
+-----------------------------+          +--------------------------------+
| Folder / Embedder           |          | Pre-built venv Python          |
| (capability client)         |          | (boltz_env/bin/python)         |
|                             |          |                                |
| session.py (RPC server)     |<-------->| worker.py (RPC dispatch)       |
| - sends {method, args, blobs}|   Unix   | - decodes frame                |
| - receives {result, blobs}  |  socket  | - calls provider.fold/.embed   |
+-----------------------------+          +--------------------------------+
```

**Wire protocol** (`scion/protocol.py`):
```
Frame:  [4-byte big-endian length][JSON header][optional binary blobs concatenated]
Header: {"id": int, "method": str, "args": {...}, "blobs": [{"name": str, "size": int}, ...]}
Reply:  same framing; header = {"id": int, "ok": bool, "result": {...}, "blobs": [...], "error": str|null}
```

Methods: `setup_done`, `health`, `shutdown`, `fold`, `embed`. Methods are dispatched onto a *capability provider* — an object returned from each environment's `setup(model, device)` that implements the relevant capability methods.

### Core files

- `scion/cli.py` — CLI entry point
- `scion/folder.py` — `Folder` capability client (fold)
- `scion/embedder.py` — `Embedder` capability client (embed)
- `scion/session.py` — shared spawn / connect / RPC / shutdown
- `scion/server.py` — RPC server side (lives in user's process)
- `scion/worker.py` — RPC dispatch (runs inside pre-built venv)
- `scion/protocol.py` — length-prefixed JSON+blob framing
- `scion/capabilities.py` — `FoldResult`, `EmbedResult` dataclasses; capability registry
- `scion/environment.py` — pre-built venv management, wrapper-script generation
- `scion/clusters.py` — cluster registry
- `scion/manifest.py` — installation manifest (extended with `capabilities` per env)
- `scion/pep723.py` — PEP 723 inline metadata parser

### Directory structure

```
{root}/
├── .python/                # uv-managed Python interpreters (portable)
├── environments/           # Environment SOURCE files (*.py with PEP 723)
│   ├── boltz_env.py
│   └── esm2_env.py
├── envs/                   # Pre-built virtual environments
│   ├── boltz_env/
│   │   ├── bin/python
│   │   ├── lib/python*/site-packages/
│   │   └── env_source.py   # copy of source for imports
│   └── esm2_env/
└── cache/                  # XDG_CACHE_HOME for model weights
    ├── huggingface/
    └── torch/
```

### Capability contract

Every env file defines:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [...]
# ///

CAPABILITIES = ["fold"]   # or ["embed"], or both

def setup(model: str, device: str = "cuda"):
    # load weights, return a provider object
    class Provider:
        def fold(self, sequence, **kwargs):
            ...  # returns dict {"mmcif": str, "confidence": dict, "plddt": ndarray}
    return Provider()
```

The worker imports `setup`, calls it once, then dispatches RPC method calls onto the returned object.

## Commands

### Local development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check scion/
pytest tests/
```

### CLI

```bash
scion init                                   # interactive setup
scion install environments/boltz_env.py      # build env on cluster
scion install environments/                  # build everything in dir
scion status
scion list
scion resolve --cluster della
scion serve <model> --socket <path> --checkpoint <name>
scion manifest show
```

## Differences from Rootstock

| Aspect | Rootstock | Scion |
|--------|-----------|-------|
| Standard interface | ASE Calculator (E/F/stress) | Capability RPC (fold/embed/...) |
| Wire protocol | i-PI (12-byte opcodes, atomic units) | length-prefixed JSON + blobs |
| Call shape | Tight loop, small arrays | One-shot, large strings/blobs |
| Entry-point class | `RootstockCalculator` (one) | `Folder`, `Embedder`, ... (one per capability) |
| Env contract | `setup(model, device) -> Calculator` | `setup(model, device) -> Provider` + `CAPABILITIES = [...]` |

Everything else (pre-built venvs, CLI shape, manifest, cluster registry, HOME/cache redirection, PEP 723 env files) is the same.
