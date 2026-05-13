"""
Refresh env_source.py inside built worker venvs without rebuilding them.

The motivating case: an env file (e.g. ``environments/boltz_env.py``)
has changed locally, but its full dep stack hasn't, so the worker
venv at ``{root}/envs/<env>/`` is fine — only its embedded copy of
the source needs to update. ``scion install --force`` would work but
takes 15-25 min for a venv like Boltz's; ``scion sync`` does the same
practical refresh in under a second.

Three modes mirror ``scion install``:

  scion sync <file.py>     File mode. Validate the file, copy it to
                           {root}/environments/<name>.py (replacing the
                           registered source) and to
                           {root}/envs/<name>/env_source.py (the worker
                           venv's copy). Refresh the manifest.

  scion sync <env_name>    Name mode. Validate the registered source at
                           {root}/environments/<name>.py and copy it
                           into the worker venv. Useful when a maintainer
                           edits the registered source directly.

  scion sync               All envs: every *.py under {root}/environments
                           is synced to its built venv.

``--with-scion`` additionally reinstalls scion itself into the worker
venv (uv pip install --upgrade), in case the worker-side scion code
(protocol, worker dispatch) has changed too. Slow (~10s per env);
default off.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .common import get_root_or_exit


def cmd_sync(args) -> int:
    from ..environment import list_environments
    from ..pep723 import validate_environment_file
    from .install import _resolve_scion_install_spec
    from .manifest import update_and_push_manifest

    root = get_root_or_exit(args)
    source = args.source

    # Resolve which envs to sync into (name, registered_source_path).
    pending: list[tuple[str, Path]] = []

    if source is None:
        # All envs.
        for name, path in list_environments(root):
            pending.append((name, path))
        if not pending:
            print(f"No registered envs in {root / 'environments'}", file=sys.stderr)
            return 1

    else:
        source_path = Path(source)

        if source_path.is_file():
            # File mode: validate, register, then sync.
            is_valid, err = validate_environment_file(source_path)
            if not is_valid:
                print(f"Error: {err}", file=sys.stderr)
                return 1

            env_name = source_path.stem
            registered = root / "environments" / f"{env_name}.py"
            registered.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, registered)
            print(f"Registered: {source_path} -> {registered}")
            pending.append((env_name, registered))

        else:
            # Name mode.
            env_name = source
            registered = root / "environments" / f"{env_name}.py"
            if not registered.exists():
                print(
                    f"Error: env not registered at {registered}. "
                    f"Run `scion install {source}` to register it first.",
                    file=sys.stderr,
                )
                return 1
            is_valid, err = validate_environment_file(registered)
            if not is_valid:
                print(
                    f"Error in registered source {registered}: {err}",
                    file=sys.stderr,
                )
                return 1
            pending.append((env_name, registered))

    # Apply each sync.
    failed: list[str] = []
    for env_name, registered in pending:
        env_target = root / "envs" / env_name
        worker_source = env_target / "env_source.py"

        if not env_target.exists():
            print(
                f"Skipping {env_name}: worker venv missing at {env_target}. "
                f"Run `scion install {env_name}` to build it.",
                file=sys.stderr,
            )
            failed.append(env_name)
            continue

        shutil.copy2(registered, worker_source)
        print(f"Synced: {registered.name} -> {worker_source}")

        if args.with_scion:
            env_python = env_target / "bin" / "python"
            spec = _resolve_scion_install_spec()
            print(f"  Reinstalling scion in worker venv from {spec} ...")
            proc = subprocess.run(
                ["uv", "pip", "install", "--python", str(env_python), "--upgrade", spec],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(
                    f"  Error reinstalling scion: {proc.stderr}",
                    file=sys.stderr,
                )
                failed.append(env_name)
                continue
            print("  Reinstalled scion in worker venv")

    # Refresh manifest so source_hash / capabilities reflect the new source.
    update_and_push_manifest(root, quiet=False)

    if failed:
        print(
            f"\n{len(failed)} env(s) failed to sync: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0
