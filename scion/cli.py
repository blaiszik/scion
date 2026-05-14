"""
Scion CLI.

The --root flag specifies the scion root directory. If not provided,
the SCION_ROOT environment variable or ~/.config/scion/config.toml is used.

Commands:
    scion init
        Interactive setup of configuration and directory structure.

    scion install <source> [--root <path>] [--models m1,m2] [--force]
        Install from file (validates, registers, builds):
            scion install ./boltz_env.py --root /path/to/scion
        Install all environments from a directory:
            scion install ./environments/ --root /path/to/scion
        Rebuild existing environment by name:
            scion install boltz_env --root /path/to/scion --force

    scion status [--root <path>]
    scion list [--root <path>]
    scion doctor [--cluster <name>] [--root <path>] [--json]
    scion serve <model> [--root <path>] --socket <path> --checkpoint <name> [--device <dev>]
    scion resolve --cluster <name> [--json]
    scion manifest {show,push,init}
"""

import argparse
import os
import sys

from .commands import (
    cmd_check,
    cmd_doctor,
    cmd_init,
    cmd_install,
    cmd_list,
    cmd_manifest,
    cmd_preload,
    cmd_resolve,
    cmd_serve,
    cmd_status,
    cmd_sync,
)
from .commands.common import SCION_ROOT_ENV
from .config import DEFAULT_CONFIG_FILE


def main():
    parser = argparse.ArgumentParser(
        prog="scion",
        description="Scion protein foundation-model environment manager",
        epilog=f"Config file: {DEFAULT_CONFIG_FILE}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    init_parser = subparsers.add_parser(
        "init",
        help="Interactive setup of scion configuration",
        description="Guided setup for root directory, maintainer info, and API credentials.",
    )
    init_parser.add_argument("--skip-dirs", action="store_true",
                             help="Skip creating directory structure")
    init_parser.add_argument("--skip-manifest", action="store_true",
                             help="Skip initializing manifest")
    init_parser.set_defaults(func=cmd_init)

    # install
    install_parser = subparsers.add_parser(
        "install",
        help="Install environment(s) from file, directory, or rebuild by name",
    )
    install_parser.add_argument(
        "source",
        help="File path, directory, or env name (e.g., ./boltz_env.py, ./environments/, boltz_env)",
    )
    install_parser.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                                help=f"Root directory (default: ${SCION_ROOT_ENV})")
    install_parser.add_argument("--models", help="Comma-separated list of models to pre-download")
    install_parser.add_argument("--force", action="store_true",
                                help="Update registration and/or rebuild if exists")
    install_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    install_parser.add_argument(
        "--no-preload",
        action="store_true",
        dest="no_preload",
        help="Skip the auto-preload validation step after install. By default "
             "scion install runs provider.preload() on CPU to surface dep "
             "errors that only appear during model construction.",
    )
    install_parser.set_defaults(func=cmd_install)

    # sync — refresh env_source.py inside built venvs without rebuilding them
    sync_parser = subparsers.add_parser(
        "sync",
        help="Refresh env_source.py in worker venvs (no rebuild)",
        description=(
            "Fast alternative to `scion install --force` when only the env "
            "source file has changed (deps haven't). Three modes: file path "
            "(registers + syncs), env name (syncs registered source into "
            "worker venv), or no argument (syncs every registered env)."
        ),
    )
    sync_parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="File path, env name, or omit to sync all envs",
    )
    sync_parser.add_argument(
        "--root",
        default=os.environ.get(SCION_ROOT_ENV),
        help=f"Root directory (default: ${SCION_ROOT_ENV})",
    )
    sync_parser.add_argument(
        "--with-scion",
        action="store_true",
        help="Also reinstall scion into the worker venv (slow; needed only if "
             "worker-side scion code has changed, not just env source).",
    )
    sync_parser.set_defaults(func=cmd_sync)

    # status
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of scion installation",
    )
    status_parser.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                               help=f"Root directory (default: ${SCION_ROOT_ENV})")
    status_parser.set_defaults(func=cmd_status)

    # list
    list_parser = subparsers.add_parser("list", help="List registered environments")
    list_parser.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                             help=f"Root directory (default: ${SCION_ROOT_ENV})")
    list_parser.set_defaults(func=cmd_list)

    # resolve
    resolve_parser = subparsers.add_parser("resolve", help="Resolve cluster configuration")
    resolve_parser.add_argument("--cluster", required=True, help="Cluster name")
    resolve_parser.add_argument("--json", action="store_true", help="Output as JSON")
    resolve_parser.set_defaults(func=cmd_resolve)

    # check
    check_parser = subparsers.add_parser(
        "check",
        help="Diagnose a built env by calling its setup() directly (bypasses RPC)",
        description=(
            "Run the env's setup(model, device) inside its pre-built venv and "
            "report success/failure with full stdout/stderr. Useful for "
            "isolating model-download or import errors from the RPC layer."
        ),
    )
    check_parser.add_argument(
        "env_name",
        nargs="?",
        default=None,
        help="Environment name (e.g., esm2_env). Omit with --all-envs.",
    )
    check_parser.add_argument(
        "--all-envs",
        action="store_true",
        dest="all_envs",
        help="Run check against every built env in series.",
    )
    check_parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint name to pass to setup() (default: env's own default)",
    )
    check_parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to setup() (default: cpu — safest for login nodes)",
    )
    check_parser.add_argument(
        "--root",
        default=os.environ.get(SCION_ROOT_ENV),
        help=f"Root directory (default: ${SCION_ROOT_ENV})",
    )
    check_parser.add_argument(
        "--thorough",
        action="store_true",
        help="Also call provider.preload() to exercise the inference path. "
             "Catches errors that surface during model construction (e.g. "
             "missing CUDA kernel packages) rather than only at import time.",
    )
    check_parser.set_defaults(func=cmd_check)

    # doctor
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run HPC readiness diagnostics for this Scion installation",
        description=(
            "Check cluster/root resolution, cluster profile metadata, root "
            "directory layout, cluster.toml overlays, login-node thread caps, "
            "runtime socket directory, uv availability, GPU visibility, and "
            "registered/built environment state. This does not run model setup; "
            "use `scion check --thorough` for model inference-path validation."
        ),
    )
    doctor_parser.add_argument(
        "--cluster",
        default=None,
        help="Cluster profile name, or 'current' for hostname-based detection",
    )
    doctor_parser.add_argument(
        "--root",
        default=os.environ.get(SCION_ROOT_ENV),
        help=f"Root directory (default: ${SCION_ROOT_ENV}, config, or detected cluster)",
    )
    doctor_parser.add_argument(
        "--env",
        dest="env_name",
        default=None,
        help="Optional built environment name to check (e.g., boltz_env)",
    )
    doctor_parser.add_argument(
        "--all-envs",
        action="store_true",
        dest="all_envs",
        help="Run doctor checks against every built env in series.",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Output as JSON")
    doctor_parser.set_defaults(func=cmd_doctor)

    # preload
    preload_parser = subparsers.add_parser(
        "preload",
        help="Pre-warm an env's model/data cache (download weights on a login node)",
        description=(
            "Run the env's setup(model, device) and then provider.preload() "
            "if defined, so weight downloads happen on a login node with "
            "internet access rather than inside a GPU job. For envs whose "
            "setup() already pulls weights (e.g. esm2_env) this is effectively "
            "the same as `scion check`; for envs whose setup() only imports "
            "the library (e.g. boltz_env), provider.preload() runs a minimal "
            "real call to trigger the download."
        ),
    )
    preload_parser.add_argument("env_name", help="Environment name (e.g., boltz_env)")
    preload_parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint name to pass to setup() (default: env's own default)",
    )
    preload_parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to setup() (default: cpu — safest for login nodes)",
    )
    preload_parser.add_argument(
        "--root",
        default=os.environ.get(SCION_ROOT_ENV),
        help=f"Root directory (default: ${SCION_ROOT_ENV})",
    )
    preload_parser.set_defaults(func=cmd_preload)

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a worker for an external RPC client",
    )
    serve_parser.add_argument("model", help="Model family (e.g., boltz, esm2)")
    serve_parser.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                              help=f"Root directory (default: ${SCION_ROOT_ENV})")
    serve_parser.add_argument("--socket", required=True, help="Unix socket path to connect to")
    serve_parser.add_argument("--checkpoint", required=True, help="Checkpoint/weights name")
    serve_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    serve_parser.set_defaults(func=cmd_serve)

    # manifest
    manifest_parser = subparsers.add_parser("manifest", help="Manage installation manifest")
    manifest_subparsers = manifest_parser.add_subparsers(dest="manifest_action", required=True)

    show_p = manifest_subparsers.add_parser("show", help="Show current manifest")
    show_p.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                        help=f"Root directory (default: ${SCION_ROOT_ENV})")
    show_p.add_argument("--json", action="store_true", help="Output as JSON")
    show_p.set_defaults(func=cmd_manifest)

    push_p = manifest_subparsers.add_parser("push", help="Push manifest to backend")
    push_p.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                        help=f"Root directory (default: ${SCION_ROOT_ENV})")
    push_p.set_defaults(func=cmd_manifest)

    init_p = manifest_subparsers.add_parser("init", help="Initialize manifest for a cluster")
    init_p.add_argument("--root", default=os.environ.get(SCION_ROOT_ENV),
                        help=f"Root directory (default: ${SCION_ROOT_ENV})")
    init_p.add_argument("--cluster", required=True, help="Cluster name (e.g., della, sophia)")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    init_p.set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
