# README-AGENT.md

Guidance for future AI agents (and humans!) working on Scion. Captures lessons learned during the bring-up phase that aren't visible in the code or git log alone.

This is a companion to `CLAUDE.md` (which describes *what* Scion is) — this file is about *how to extend it well* and *what mistakes to avoid*.

---

## 1. The Scion model in one paragraph

Scion is a thin user-process library that brokers calls into pre-built virtual environments on a shared HPC filesystem. Each env is a PEP 723 Python file declaring its deps, a `CAPABILITIES` list, and a `setup(model, device) → provider` function. The provider is a duck-typed object whose methods (named after the capabilities) do the actual ML work. At call time, Scion spawns the env's python as a subprocess, brokers RPC over a Unix socket using a length-prefixed JSON+blob framing, and reuses the worker across many calls in the same session.

The user-facing API is one capability client per capability (`Folder`, `Embedder`, ...) wrapping a shared `ScionSession`. Everything else (CLI, manifest, cluster registry, cluster.toml, HOME redirection) is plumbing.

## 2. Adding a new env: the checklist

When wiring a new model, do this in order. Each step has a reason — most were learned the hard way.

1. **Write the PEP 723 metadata first**, not the code. Include:
   - `requires-python` (most ML libs need `>=3.10` or `>=3.11`)
   - `dependencies` — list every import the provider will use. *Especially* CUDA-related packages that the model imports during graph construction (e.g. `cuequivariance-torch`, `cuequivariance-ops-torch-cu12`). Don't trust upstream's pip metadata to pull them in.
   - **Cap torch upper-bound** when the cluster's NVIDIA driver might be older than the latest PyPI wheel's CUDA target. Polaris-shape clusters (driver ≤ 12.8) need `torch<2.9`. Better: ask the cluster maintainer for the driver version before pinning.
2. **Declare `CAPABILITIES` at module scope.** A list of capability ids (`["fold"]`, `["embed"]`, etc.). Validated by `scion install`; required by the worker dispatch.
3. **Write `setup(model, device) → provider`.** Keep it minimal: import the library, load weights, return a provider. *Don't run inference here* — weight loading is fine, but anything that touches the inference path belongs in `preload()` or capability methods.
4. **Implement `provider.preload()` — this is mandatory in practice.** It's optional in the protocol, but without it `scion check --thorough` is a no-op and missing CUDA-runtime deps will surface inside GPU jobs instead of on a login node. The cheapest correct preload is a minimal real capability call: `self.fold("MKTA", num_recycles=1)` or `self.embed(["MK"])`. Cost: minutes once per cache warm-up. Saved cost: hours of GPU-time-debugging-by-trial-and-error.
5. **Factor non-trivial logic into pure functions at module scope** so it's unit-testable without installing the heavy deps. The Boltz YAML construction (`build_boltz_yaml`) is the model — 13 tests cover schema correctness with zero `boltz` installation needed.
6. **Run `scion install <env_file>.py`** (slow, one-time).
7. **Run `scion check <env> --thorough --device cpu`** on a login node. If this fails, fix the env file and run `scion sync <env_file>` (sub-second). Iterate until `--thorough` passes.
8. **Only then** submit a GPU job and try a real call.

## 3. The five dep paper cuts

When something works in `import` but fails later, suspect these (in order of frequency observed):

| Symptom | Likely cause | Fix |
|---|---|---|
| `libgomp: Thread creation failed: Resource temporarily unavailable` | Shared login node, `RLIMIT_NPROC` capped | `cluster.toml [login_env] OMP_NUM_THREADS=1` (and friends); `scion check` does this as a fallback |
| `ModuleNotFoundError: cuequivariance_torch` / `cuequivariance_ops_torch_*` | Upstream library imports unconditionally but doesn't list it in pip metadata | Add to PEP 723 deps explicitly. For CUDA-version-suffixed packages, the env needs a cluster-appropriate suffix |
| `RuntimeError: NVIDIA driver on your system is too old` | torch wheel built against CUDA newer than the cluster supports | Cap torch version. `extra-index-url` is **not** reliable — uv's default `first-index` strategy still picks PyPI |
| `undefined symbol: ncclGet...` or similar dynamic-link error | torch + nvidia-* package version skew (probably from `--no-deps` reinstall) | `scion install --force` rebuilds from scratch; surgical `--no-deps` reinstalls of torch are an anti-pattern |
| `Killed` / OOM during setup on a login node | Loading model weights on a memory-constrained login tier | Use `scion preload` from a login node (CPU mode pre-caches weights without holding them in RAM long); or move the load step into `setup()` only — never into module top-level |

## 4. The CLI surface in order of usage

```
scion init                                  # interactive setup of {root} + config + manifest
scion install <env.py> [--models X]         # build a worker venv from PEP 723; slow (minutes)
scion sync <env.py>                         # refresh env_source.py without rebuilding (fast)
scion doctor                                # preflight cluster/root/cache/env readiness
scion check <env> [--thorough]              # diagnose: setup() (default) + preload (--thorough)
scion preload <env>                         # warm caches on a login node (downloads weights)
scion status                                # show built envs + capabilities + cache sizes
scion serve <model> --socket ...            # worker entry for external RPC clients (rare)
```

Triage decision tree when something breaks:

- "I edited the env file": `scion sync <env.py>` (don't rebuild)
- "I changed the deps": `scion install <env.py> --force` (rebuild)
- "I'm on a new cluster / something basic feels off": `scion doctor`
- "I'm not sure if the env works at all": `scion check <env> --thorough`
- "I want weights cached before a GPU job": `scion preload <env>` on a login node
- "scion CLI behavior changed": `pip install --upgrade --force-reinstall git+...`

### Doctor and layered cluster profiles

`scion doctor` is the lightweight HPC preflight. It does **not** call model `setup()` or run inference; keep that boundary clear so the command stays cheap on login nodes. It reports:

- cluster/root resolution, including layered profile source
- root layout (`environments/`, `envs/`, `cache/`, `home/`, `.python/`)
- cache/home writability
- `{root}/cluster.toml` parsing and active env overlay
- login-node thread caps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`)
- runtime socket directory and Unix socket path-length risk
- `uv` availability, Python version, and `nvidia-smi` GPU visibility
- registered/built env completeness, optionally narrowed with `--env`

Useful forms:

```
scion doctor
scion doctor --cluster polaris
scion doctor --cluster current
scion doctor --root /path/to/scion --env boltz_env
scion doctor --json
```

Cluster lookup is now layered. Built-ins live in `scion/clusters.py`, then user/site TOML overlays are read from `~/.config/scion/clusters.toml`, then `SCION_CLUSTERS_FILE` if set. Later layers replace earlier profiles with the same name. Use profiles for site metadata:

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

Important split: cluster profiles describe the site; `{root}/cluster.toml` only overlays environment variables for subprocesses. Do not put model args, checkpoint choices, or dependency pins in either one. Dependency truth remains PEP 723 in the env file.

Runtime sockets default to the platform temp dir, but `SCION_RUNTIME_DIR` wins, then profile `runtime_dir`. Use this when a site has long temp paths, unusual `/tmp` behavior, or a policy requiring per-user runtime directories.

## 5. Things that look reasonable but aren't

- **`pip install --force-reinstall --no-deps torch`** — surgical reinstall sounds clean but creates ABI mismatches with the already-installed `nvidia-*` packages. Use `scion install --force` instead.
- **`[tool.uv] extra-index-url = ["https://download.pytorch.org/whl/cu126"]`** in PEP 723 — looks like it'd force torch to come from PyTorch's CUDA wheel index, but uv's default `first-index` strategy still picks PyPI when both list the package. Version-pinning is the only mechanism that reliably constrains wheel selection.
- **Trusting upstream `pip install <ml-lib>` to pull every transitive dep** — many ML libraries import optional CUDA packages unconditionally. Always declare CUDA packages in the env file explicitly.
- **Activating the worker venv at the shell level** — Scion's worker runs `{root}/envs/<env>/bin/python` directly; the venv is never "activated." Don't add activation steps to deployment recipes.
- **Putting weight-load code in module top-level** of the env file — runs during `import env_source` in `scion check` and other diagnostics, blowing up things that should be cheap. Put it inside `setup()`.
- **Using `/tmp` for anything that needs to outlive a single node** — on most HPC systems `/tmp` is node-local. Use the user's home or a project directory under the mounted shared filesystem.

## 6. Testing pattern

Tests live in `tests/` and run with `pytest tests/` (no GPU needed). Three useful patterns observed:

1. **Pure-function extraction from env files.** `environments/boltz_env.py` defines `build_boltz_yaml(...)` at module scope. Tests import the env file *as a regular Python module* (no env build needed) and pin the YAML schema against silent drift. Replicate this pattern for any non-trivial logic in an env file.
2. **Fake worker venvs.** `tests/test_sync.py` and `tests/test_check_thorough.py` build a fake `{root}/envs/<name>/` layout with a symlinked python and a hand-written `env_source.py`. Lets us exercise CLI commands end-to-end against synthetic envs with zero dependency installs.
3. **Socketpair round-trips.** `tests/test_protocol_roundtrip.py` tests the wire protocol with `socket.socketpair()` rather than real subprocesses. Fast, deterministic, no cleanup needed.

When adding a new CLI command or env feature, prefer to factor the testable bit out and pin it with one of these patterns.

## 7. Architectural invariants — don't break these

- **The protocol is method-name dispatched.** Adding a new capability is: (a) add it to `CAPABILITIES` of an env, (b) implement the method on the provider, (c) maybe add a client class. The wire protocol doesn't change.
- **Workers reuse model state across calls in a session.** Don't write capability methods that assume fresh state per call (e.g. caching globals based on the first call's args).
- **The user-process and the worker-venv-process have different scion versions.** They communicate by protocol, not by shared in-process state. Wire-protocol changes need to be backward-compatible or behind a feature flag.
- **`cluster.toml` overlays env vars only.** It is not a place to put model-tuning knobs. Those belong in env files or capability method args.
- **Cluster profiles are separate from `cluster.toml`.** Add roots, scheduler metadata, job env vars, hostname patterns, CUDA driver notes, and runtime-dir preferences in `~/.config/scion/clusters.toml` or `SCION_CLUSTERS_FILE`; keep `{root}/cluster.toml` for env vars only.
- **PEP 723 is the source of truth for env deps.** Don't add a parallel `requirements.txt` or `pyproject.toml` per env — `scion install` reads PEP 723 and forwards to uv.

## 8. When in doubt

- `scion check --thorough` first. If it passes, the env is healthy.
- `scion doctor` first when root resolution, scheduler context, cache writability, thread caps, `uv`, or GPU visibility is suspect.
- If a fix takes more than two surgical `uv pip install` invocations, stop and `scion install --force` from a corrected env file instead.
- Run pure logic locally (in the dev `.venv`) with `pytest tests/` before committing — most regressions surface there in 50 ms.
- Read `README.md`'s "Lessons from the field" before debugging anything torch- or CUDA-related on a new cluster.
