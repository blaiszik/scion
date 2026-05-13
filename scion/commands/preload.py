"""
Pre-warm an environment's model/data caches.

Spawns the env's pre-built python, calls ``setup(model, device)``, and
then calls ``provider.preload()`` if the provider defines it. Useful for
pulling model weights on a login node before submitting a GPU job — the
GPU node often can't reach the public internet (Polaris, Perlmutter)
without proxy plumbing, so warming caches on a login node where outbound
traffic is unrestricted is the cleanest pattern.

Provider contract
-----------------
Optional. If a provider exposes a ``preload()`` method, this command
calls it; otherwise relying on ``setup()`` to have populated whatever
the env needs.

Examples of which envs need an explicit preload():
  * ``boltz_env``    — setup() only imports boltz; weights download
                       lazily on the first ``predict`` call. Defines
                       preload() that runs a minimal fold to warm them.
  * ``esm2_env``     — setup() already loads ESM2 weights via torch.hub
                       during ``load_model_and_alphabet``. No preload()
                       needed; this command is effectively a no-op
                       beyond what ``scion check`` already does.

Examples
--------
    scion preload boltz_env
    scion preload boltz_env --device cpu   # default; safe on a login node
    scion preload esm2_env --model esm2_t33_650M_UR50D

Exit code matches the subprocess: 0 on success, nonzero on failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from .common import get_root_or_exit


def cmd_preload(args) -> int:
    from ..cluster_config import get_cluster_env
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
    env.update(get_cluster_env(root))
    # Same diagnostic fallback as `scion check`: login nodes' tight
    # RLIMIT_NPROC kills torch before it can download weights.
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
            sys.exit(f"[preload] Failed to import env_source: {{e!r}}")

        print(f"[preload] CAPABILITIES = {{CAPABILITIES}}", flush=True)
        print(f"[preload] Calling setup({model!r}, {device!r}) ...", flush=True)
        try:
            provider = setup({model!r}, {device!r})
        except Exception as e:
            traceback.print_exc()
            sys.exit(f"[preload] setup() raised: {{e!r}}")

        print(f"[preload] Provider loaded: {{type(provider).__name__}}", flush=True)

        if hasattr(provider, "preload"):
            print(f"[preload] Calling provider.preload() ...", flush=True)
            try:
                provider.preload()
            except Exception as e:
                traceback.print_exc()
                sys.exit(f"[preload] provider.preload() raised: {{e!r}}")
            print(f"[preload] preload() complete", flush=True)
        else:
            print(
                "[preload] (no provider.preload() defined — setup() should "
                "have already warmed any caches this env needs)",
                flush=True,
            )
        """
    )

    print(f"Preloading env: {env_name}")
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
        print(f"\nOK: {env_name} cache is warm")
    else:
        print(f"\nFAILED: {env_name} preload exited {proc.returncode}", file=sys.stderr)
    return proc.returncode
