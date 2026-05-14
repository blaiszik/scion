"""
Diagnostic check command.

Spawns the env's Python directly and runs ``setup(model, device)``, printing
everything stdout/stderr produces. This bypasses Scion's RPC layer entirely
so failures can be attributed cleanly to:
  * the worker venv (missing deps, wrong torch wheel)
  * the model checkpoint (network/proxy, SSL, OOM)
  * the provider's setup() body (import errors, init bugs)
rather than blamed on the protocol / spawn / socket layer.

By default ``scion check`` only runs ``setup()`` — fast (seconds) and
catches import-level breakage but misses problems that surface later,
e.g. NVIDIA cuequivariance ops that Boltz imports during model graph
construction, not during ``import boltz``. Pass ``--thorough`` to also
call ``provider.preload()`` if defined, which exercises the inference
path and surfaces those deeper bugs.

Examples
--------
    scion check esm2_env                           # fast: setup() only
    scion check boltz_env --device cpu --thorough  # also run provider.preload()
    scion check esm2_env --model esm2_t33_650M_UR50D --device cuda

Exit codes match the subprocess: 0 on success, nonzero on failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from .common import get_root_or_exit

# Known failure-mode patterns and the structural fix for each. Keep these
# tight: the goal is to save users a full triage cycle on the same paper
# cuts the README documents in "Lessons from the field".
HINT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "undefined symbol: nccl",
        "torch / nvidia-* ABI skew (often from a stray --no-deps reinstall). "
        "Rebuild with: scion install <env> --force",
    ),
    (
        "NVIDIA driver on your system is too old",
        "torch wheel was built for a newer CUDA than this cluster's driver "
        "supports. Cap torch in PEP 723 deps (e.g. 'torch<2.9' for clusters "
        "stuck at CUDA 12.8), then rebuild with: scion install <env> --force",
    ),
    (
        "ModuleNotFoundError: No module named 'cuequivariance",
        "Boltz imports cuequivariance unconditionally but doesn't list it. "
        "Add 'cuequivariance-torch' and 'cuequivariance-ops-torch-cu12' "
        "(or -cu11 on older clusters) to PEP 723 deps and rebuild.",
    ),
    (
        "libgomp: Thread creation failed",
        "Shared HPC login nodes apply tight RLIMIT_NPROC. Set "
        "OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1 via "
        "<root>/cluster.toml [login_env]. See cluster.toml.example.",
    ),
    (
        "huggingface_hub.utils._errors.LocalEntryNotFoundError",
        "HuggingFace Hub download failed — usually compute-node outbound "
        "blocked. Run `scion preload <env>` from a login node first, or "
        "set HTTPS_PROXY in <root>/cluster.toml [compute_env].",
    ),
    (
        "ConnectionError",
        "Network connection failed during model download. On most HPC "
        "compute nodes outbound HTTP is blocked; run `scion preload <env>` "
        "from a login node to populate the cache before the GPU job, or "
        "configure HTTPS_PROXY in <root>/cluster.toml [compute_env].",
    ),
    (
        "OSError: [Errno 28] No space left on device",
        "Out of disk in cache or home. Check `du -sh <root>/cache "
        "<root>/home` and either clear old weights or move the root to a "
        "larger volume (project allocation, $SCRATCH).",
    ),
)


def _suggest_hints(output: str) -> list[str]:
    """Return one-line tips matching known patterns in the worker output."""
    return [tip for needle, tip in HINT_PATTERNS if needle in output]


def cmd_check(args) -> int:
    """Diagnose one or all built environments."""
    if getattr(args, "all_envs", False):
        return _cmd_check_all(args)
    if not args.env_name:
        print(
            "Error: env_name is required unless --all-envs is passed.",
            file=sys.stderr,
        )
        return 1
    return _check_single_env(args, env_name=args.env_name)


def _cmd_check_all(args) -> int:
    """Run check across every built env in series; one summary line per env."""
    from ..environment import list_built_environments

    root = get_root_or_exit(args)
    envs = [name for name, _ in list_built_environments(root)]
    if not envs:
        print("No built environments found.", file=sys.stderr)
        return 1

    print(f"Checking {len(envs)} env(s): {', '.join(envs)}\n")
    failed: list[str] = []
    for name in envs:
        print(f"{'=' * 60}\n{name}\n{'=' * 60}")
        rc = _check_single_env(args, env_name=name)
        if rc != 0:
            failed.append(name)
        print()

    print(f"{'=' * 60}\nSummary: {len(envs) - len(failed)}/{len(envs)} OK")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return 1 if failed else 0


def _check_single_env(args, env_name: str) -> int:
    from ..environment import EnvironmentManager, get_model_cache_env

    root = get_root_or_exit(args)
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

    from ..cluster_config import get_cluster_env

    env = {**os.environ, **get_model_cache_env(root)}
    # Per-cluster overlay (e.g. Polaris login_env caps OMP/MKL/OPENBLAS to 1).
    env.update(get_cluster_env(root))

    # Diagnostic-only fallback: if no cluster.toml exists and the user hasn't
    # set them, cap thread libs to 1. Login nodes on shared HPC systems apply
    # tight per-user RLIMIT_NPROC; PyTorch/MKL/OpenBLAS otherwise spawn one
    # thread per CPU core and crash with "libgomp: Thread creation failed".
    # This is only baked into `check` because it's the triage path users hit
    # before cluster.toml is even in place.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(var, "1")

    base_script = textwrap.dedent(
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

    # --thorough: exercise the inference path by calling provider.preload()
    # if the env defines it. Surfaces errors that only happen during model
    # construction or a real call — e.g. missing CUDA kernel packages that
    # import-time setup() wouldn't reveal. If the env doesn't define
    # preload(), we say so but don't fail the check; env authors can add
    # one to enable deeper diagnostics.
    thorough_script = textwrap.dedent(
        """\
        if hasattr(provider, "preload"):
            print(
                "[check] --thorough: calling provider.preload() to exercise "
                "the inference path ...",
                flush=True,
            )
            try:
                provider.preload()
            except Exception as e:
                traceback.print_exc()
                sys.exit(f"[check] provider.preload() raised: {e!r}")
            print("[check] --thorough: provider.preload() OK", flush=True)
        else:
            print(
                "[check] --thorough: env has no provider.preload(); thorough "
                "mode can only run setup(). Add a preload() method to your "
                "env's provider to enable deeper diagnostic coverage (e.g. a "
                "minimal capability call that exercises model construction).",
                flush=True,
            )
        """
    )

    script = base_script + ("\n" + thorough_script if args.thorough else "")

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

    proc = subprocess.run(
        [str(env_python), "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    # Preserve the prior behavior of streaming worker output to the user's
    # terminal, then post-scan it for known failure-mode patterns.
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode == 0:
        print(f"\nOK: {env_name} setup() succeeded")
    else:
        print(f"\nFAILED: {env_name} setup() exited {proc.returncode}", file=sys.stderr)
        hints = _suggest_hints(proc.stdout + proc.stderr)
        if hints:
            print("\nHints:", file=sys.stderr)
            for h in hints:
                print(f"  - {h}", file=sys.stderr)
    return proc.returncode
