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
    scion serve <model> [--root <path>] --socket <path> --checkpoint <name> [--device <dev>]
    scion resolve --cluster <name> [--json]
    scion manifest {show,push,init}
"""

import argparse
import os
import sys

from .commands import (
    cmd_check,
    cmd_init,
    cmd_install,
    cmd_list,
    cmd_manifest,
    cmd_resolve,
    cmd_serve,
    cmd_status,
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
    install_parser.set_defaults(func=cmd_install)

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
    check_parser.add_argument("env_name", help="Environment name (e.g., esm2_env)")
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
    check_parser.set_defaults(func=cmd_check)

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
