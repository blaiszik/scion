"""Manifest management commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..client import ScionClient
from ..clusters import get_cluster_for_root
from ..config import load_config
from ..manifest import (
    EnvironmentInfo,
    Manifest,
    compute_source_hash,
    create_manifest,
    get_installed_versions,
    load_manifest,
    now_iso,
    save_manifest,
)
from ..pep723 import get_capabilities, get_dependencies, get_requires_python
from .common import get_root_or_exit


def update_and_push_manifest(
    root: Path,
    cluster: str | None = None,
    quiet: bool = False,
) -> bool:
    """Update manifest with current state and push to backend if configured."""
    config = load_config()

    manifest = load_manifest(root)

    if cluster is None:
        if manifest is not None:
            cluster = manifest.cluster
        else:
            cluster = get_cluster_for_root(root)

    if cluster is None:
        if not quiet:
            print(
                "Warning: Cannot update manifest - cluster not specified and "
                "root doesn't match any known cluster. "
                "Run 'scion manifest init --cluster <name>' first.",
                file=sys.stderr,
            )
        return False

    if manifest is None:
        manifest = create_manifest(root, cluster, config)

    manifest = _refresh_manifest_environments(manifest, root)
    save_manifest(manifest, root)

    if config.is_maintainer and config.is_push_enabled():
        client = ScionClient(config)
        success, message = client.push_manifest(manifest)
        if not quiet:
            if success:
                print(f"Manifest pushed: {message}")
            else:
                print(f"Warning: Failed to push manifest: {message}", file=sys.stderr)
                print(
                    "Manifest saved locally. Run 'scion manifest push' to retry.",
                    file=sys.stderr,
                )
        return success
    elif not config.is_maintainer and not quiet:
        print("Manifest saved locally (not pushing - you are not the maintainer).")

    return True


def _refresh_manifest_environments(manifest: Manifest, root: Path) -> Manifest:
    """Update manifest with current built-environment state."""
    from .. import __version__
    from ..environment import list_built_environments

    manifest.scion_version = __version__

    built = list_built_environments(root)

    for env_name, env_path in built:
        source_file = env_path / "env_source.py"
        if not source_file.exists():
            continue

        source_hash = compute_source_hash(source_file)
        source_content = source_file.read_text()
        python_requires = get_requires_python(source_file) or ">=3.10"

        direct_deps = get_dependencies(source_file)
        if "scion" not in [d.lower() for d in direct_deps]:
            direct_deps.append("scion")

        dependencies = get_installed_versions(env_path, only_packages=direct_deps)
        capabilities = get_capabilities(source_file)

        existing_env = manifest.environments.get(env_name)
        checkpoints = existing_env.checkpoints if existing_env else []

        manifest.environments[env_name] = EnvironmentInfo(
            status="ready",
            built_at=existing_env.built_at if existing_env else now_iso(),
            source_hash=source_hash,
            source=source_content,
            python_requires=python_requires,
            dependencies=dependencies,
            capabilities=capabilities,
            checkpoints=checkpoints,
        )

    return manifest


def cmd_manifest(args) -> int:
    """Handle manifest subcommands."""
    if args.manifest_action == "show":
        return cmd_manifest_show(args)
    elif args.manifest_action == "push":
        return cmd_manifest_push(args)
    elif args.manifest_action == "init":
        return cmd_manifest_init(args)
    return 0


def cmd_manifest_show(args) -> int:
    """Show current manifest."""
    root = get_root_or_exit(args)
    manifest = load_manifest(root)

    if manifest is None:
        print(f"No manifest found at {root}/manifest.json", file=sys.stderr)
        print("Run 'scion manifest init --cluster <name>' to create one.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2))
    else:
        print(f"Manifest: {root}/manifest.json")
        print(f"  Schema version: {manifest.schema_version}")
        print(f"  Cluster:        {manifest.cluster}")
        print(f"  Root:           {manifest.root}")
        print(f"  Scion version:  {manifest.scion_version}")
        print(f"  Python version: {manifest.python_version}")
        print(f"  Last updated:   {manifest.last_updated}")
        print()
        print("  Maintainer:")
        print(f"    Name:  {manifest.maintainer.name}")
        print(f"    Email: {manifest.maintainer.email}")
        print()
        print(f"  Environments ({len(manifest.environments)}):")
        for name, env in manifest.environments.items():
            print(f"    {name}:")
            print(f"      Status:       {env.status}")
            caps_str = ", ".join(env.capabilities) if env.capabilities else "(none)"
            print(f"      Capabilities: {caps_str}")
            print(f"      Built at:     {env.built_at}")
            print(f"      Source hash:  {env.source_hash[:20]}...")
            print(f"      Dependencies: {len(env.dependencies)} packages")
            if env.checkpoints:
                print(f"      Checkpoints:  {', '.join(env.checkpoints)}")

    return 0


def cmd_manifest_push(args) -> int:
    """Push manifest to backend."""
    root = get_root_or_exit(args)
    config = load_config()

    valid, error = config.validate()
    if not valid:
        print(f"Error: {error}", file=sys.stderr)
        print("Configure API credentials in ~/.config/scion/config.toml", file=sys.stderr)
        return 1

    manifest = load_manifest(root)
    if manifest is None:
        print(f"No manifest found at {root}/manifest.json", file=sys.stderr)
        return 1

    valid, error = manifest.validate()
    if not valid:
        print(f"Error: Invalid manifest: {error}", file=sys.stderr)
        return 1

    client = ScionClient(config)
    success, message = client.push_manifest(manifest)

    if success:
        print(message)
        return 0
    else:
        print(f"Error: {message}", file=sys.stderr)
        return 1


def cmd_manifest_init(args) -> int:
    """Initialize manifest for a cluster."""
    root = get_root_or_exit(args)
    cluster = args.cluster
    config = load_config()

    existing = load_manifest(root)
    if existing and not args.force:
        print(f"Error: Manifest already exists at {root}/manifest.json", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        return 1

    if not config.name or not config.email:
        print("Warning: Maintainer info not configured.", file=sys.stderr)

    manifest = create_manifest(root, cluster, config)
    manifest = _refresh_manifest_environments(manifest, root)
    save_manifest(manifest, root)

    print(f"Manifest initialized: {root}/manifest.json")
    print(f"  Cluster: {cluster}")
    print(f"  Environments: {len(manifest.environments)}")

    if config.is_maintainer and config.is_push_enabled():
        client = ScionClient(config)
        success, message = client.push_manifest(manifest)
        if success:
            print(f"Manifest pushed: {message}")
        else:
            print(f"Warning: Failed to push manifest: {message}", file=sys.stderr)
    elif not config.is_maintainer:
        print("Manifest saved locally (not pushing - you are not the maintainer).")

    return 0
