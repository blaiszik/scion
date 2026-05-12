"""
Diagnostic check command.

Spawns the env's Python directly and runs ``setup(model, device)``, printing
everything stdout/stderr produces. This bypasses Scion's RPC layer entirely
so failures can be attributed cleanly to:
  * the worker venv (missing deps, wrong torch wheel)
  * the model checkpoint (network/proxy, SSL, OOM)
  * the provider's setup() body (import errors, init bugs)
rather than blamed on the protocol / spawn / socket layer.

Examples
--------
    scion check esm2_env
    scion check esm2_env --model esm2_t33_650M_UR50D --device cpu
    scion check esm2_env --model esm2_t33_650M_UR50D --device cuda

Exit codes match the subprocess: 0 on success, nonzero on failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from .common import get_root_or_exit


def cmd_check(args) -> int:
    from ..environment import EnvironmentManager, get_model_cache_env

    root = get_root_or_exit(args)
    env_name = args.env_name
    model = args.model or ""
    device = args.device

    env_mgr = EnvironmentManager(root=root)
    try:
        env_python = env_mgr.get_env_python(env_name)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    env_dir = root / "envs" / env_name
    source_file = env_dir / "env_source.py"
    if not source_file.exists():
        print(f"Error: env_source.py missing in {env_dir}", file=sys.stderr)
        return 1

    env = {**os.environ, **get_model_cache_env(root)}

    # Login nodes on shared HPC systems (Polaris, Perlmutter, ...) apply tight
    # per-user RLIMIT_NPROC. Torch/MKL/OpenBLAS otherwise default to one thread
    # per CPU core and fail with "libgomp: Thread creation failed: Resource
    # temporarily unavailable". Cap to 1 thread by default for diagnostic
    # runs — the user can override by exporting the variable explicitly.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(var, "1")

    script = textwrap.dedent(
        f"""\
        import sys, traceback
        sys.path.insert(0, {str(env_dir)!r})
        try:
            from env_source import setup, CAPABILITIES
        except Exception as e:
            traceback.print_exc()
            sys.exit(f"[check] Failed to import env_source: {{e!r}}")

        print(f"[check] CAPABILITIES = {{CAPABILITIES}}", flush=True)
        print(f"[check] Calling setup({model!r}, {device!r}) ...", flush=True)
        try:
            provider = setup({model!r}, {device!r})
        except Exception as e:
            traceback.print_exc()
            sys.exit(f"[check] setup() raised: {{e!r}}")

        print(f"[check] OK: provider = {{type(provider).__name__}}", flush=True)
        methods = [c for c in CAPABILITIES if hasattr(provider, c)]
        missing = [c for c in CAPABILITIES if not hasattr(provider, c)]
        print(f"[check] provider implements: {{methods}}", flush=True)
        if missing:
            sys.exit(
                f"[check] WARN: declared CAPABILITIES not on provider: {{missing}}"
            )
        """
    )

    print(f"Checking env: {env_name}")
    print(f"  Python:     {env_python}")
    print(f"  Source:     {source_file}")
    print(f"  Model:      {model or '(env default)'}")
    print(f"  Device:     {device}")
    print(f"  HOME:       {env.get('HOME')}")
    print(f"  TORCH_HOME: {env.get('TORCH_HOME')}")
    print(f"  HF_HOME:    {env.get('HF_HOME')}")
    print(f"  Threads:    OMP={env.get('OMP_NUM_THREADS')} "
          f"MKL={env.get('MKL_NUM_THREADS')} "
          f"OPENBLAS={env.get('OPENBLAS_NUM_THREADS')}")
    print()

    proc = subprocess.run([str(env_python), "-c", script], env=env)
    if proc.returncode == 0:
        print(f"\nOK: {env_name} setup() succeeded")
    else:
        print(f"\nFAILED: {env_name} setup() exited {proc.returncode}", file=sys.stderr)
    return proc.returncode
