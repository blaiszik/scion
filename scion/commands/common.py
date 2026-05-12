"""Common utilities for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import load_config

SCION_ROOT_ENV = "SCION_ROOT"


def get_root_or_exit(args) -> Path:
    """
    Resolve the root directory from args, env var, or config file.

    Priority:
    1. --root CLI flag
    2. SCION_ROOT environment variable
    3. root in ~/.config/scion/config.toml
    """
    if args.root:
        return Path(args.root)

    config = load_config()
    if config.root:
        return Path(config.root)

    print(
        f"Error: --root is required (or set {SCION_ROOT_ENV} environment variable, "
        "or configure root in ~/.config/scion/config.toml)",
        file=sys.stderr,
    )
    sys.exit(1)
