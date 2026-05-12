"""Interactive initialization command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..clusters import CLUSTER_REGISTRY, get_cluster_for_root
from ..config import DEFAULT_CONFIG_FILE, load_config, save_config
from ..manifest import create_manifest, save_manifest
from .common import SCION_ROOT_ENV
from .manifest import _refresh_manifest_environments


def prompt_with_default(prompt: str, default: str | None = None) -> str | None:
    full_prompt = f"{prompt} [{default}]: " if default else f"{prompt}: "
    value = input(full_prompt).strip()
    if not value and default:
        return default
    return value if value else None


def prompt_secret(prompt: str, existing: str | None = None) -> str | None:
    full_prompt = f"{prompt} [configured]: " if existing else f"{prompt}: "
    value = input(full_prompt).strip()
    if not value and existing:
        return existing
    return value if value else None


def cmd_init(args) -> int:
    """Interactive initialization of Scion configuration."""
    print("Welcome to Scion!")
    print("This will help you set up your configuration.\n")

    config = load_config()

    print("Root directory is where environments and caches are stored.")
    print(f"Known clusters: {', '.join(CLUSTER_REGISTRY.keys())}")
    print("You can enter a cluster name or a custom path.\n")

    root_default = config.root or os.environ.get(SCION_ROOT_ENV)
    root_input = prompt_with_default("Root directory", root_default)

    if not root_input:
        print("Error: Root directory is required.", file=sys.stderr)
        return 1

    if root_input in CLUSTER_REGISTRY:
        cluster = root_input
        root = Path(CLUSTER_REGISTRY[root_input])
        print(f"  -> Using cluster '{cluster}' root: {root}")
    else:
        root = Path(root_input).expanduser().resolve()
        cluster = get_cluster_for_root(root)
        if cluster:
            print(f"  -> Detected cluster: {cluster}")

    config.root = str(root)
    print()

    print("Are you the maintainer of this scion installation?")
    is_maintainer_input = prompt_with_default("Maintainer (y/n)", "n")
    config.is_maintainer = is_maintainer_input.lower() in ("y", "yes")
    print()

    if config.is_maintainer:
        print("Maintainer information (shown in manifests):")
        config.name = prompt_with_default("  Name", config.name)
        config.email = prompt_with_default("  Email", config.email)
        print()

        print("API credentials for pushing manifests (optional, press Enter to skip):")
        api_key = prompt_secret("  API Key", config.api_key)
        if api_key:
            config.api_key = api_key
            config.api_secret = prompt_secret("  API Secret", config.api_secret)
            config.api_url = prompt_with_default("  API URL", config.api_url)
        print()
    else:
        print("Skipping maintainer and API configuration.")

    save_config(config)
    print(f"Configuration saved to {DEFAULT_CONFIG_FILE}")

    if not args.skip_dirs:
        print("\nCreating directory structure...")
        dirs_to_create = [
            root / "environments",
            root / "envs",
            root / "cache",
            root / "home",
            root / ".python",
        ]

        for dir_path in dirs_to_create:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"  Created: {dir_path}")
                except PermissionError:
                    print(f"  Skipped (no permission): {dir_path}")
            else:
                print(f"  Exists:  {dir_path}")

    if cluster and not args.skip_manifest:
        print("\nInitializing manifest...")
        manifest = create_manifest(root, cluster, config)
        manifest = _refresh_manifest_environments(manifest, root)
        save_manifest(manifest, root)
        print(f"  Created: {root}/manifest.json")
        if manifest.environments:
            print(f"  Found {len(manifest.environments)} existing environment(s)")

        if config.is_maintainer and config.is_push_enabled():
            from ..client import ScionClient

            client = ScionClient(config)
            success, message = client.push_manifest(manifest)
            if success:
                print(f"  Pushed manifest: {message}")
            else:
                print(f"  Warning: Failed to push: {message}", file=sys.stderr)

    print("\nSetup complete!")
    print("\nNext steps:")
    print("  1. Install environments: scion install <env_file.py>")
    print("  2. Check status: scion status")

    return 0
