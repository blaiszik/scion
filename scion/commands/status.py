"""Status and list commands."""

from __future__ import annotations

from ..config import DEFAULT_CONFIG_FILE
from .common import get_root_or_exit


def cmd_status(args) -> int:
    """Show status of scion installation."""
    from ..environment import list_built_environments, list_environments
    from ..pep723 import get_capabilities

    root = get_root_or_exit(args)
    print(f"Scion root: {root}")

    print("\nEnvironment sources:")
    sources = list_environments(root)
    if not sources:
        print("  (none)")
    else:
        for name, path in sources:
            caps = get_capabilities(path)
            caps_str = ", ".join(caps) if caps else "?"
            print(f"  {name:<20} [{caps_str}]")

    print("\nBuilt environments:")
    built = list_built_environments(root)
    if not built:
        print("  (none)")
    else:
        for name, path in built:
            has_source = (path / "env_source.py").exists()
            status = "ready" if has_source else "incomplete"
            caps = get_capabilities(path / "env_source.py") if has_source else []
            caps_str = ", ".join(caps) if caps else "?"
            print(f"  {name:<20} [{status}] [{caps_str}]")

    print("\nCache:")
    cache_dir = root / "cache"
    if cache_dir.exists():
        for subdir in sorted(cache_dir.iterdir()):
            if subdir.is_dir():
                total_size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"  {subdir.name + '/':<20} {size_mb:.1f} MB")
    else:
        print("  (no cache directory)")

    print(f"\nConfig file: {DEFAULT_CONFIG_FILE}")
    return 0


def cmd_list(args) -> int:
    """List registered environments."""
    from ..environment import list_built_environments, list_environments

    root = get_root_or_exit(args)
    sources = list_environments(root)
    built = list_built_environments(root)
    built_names = {name for name, _ in built}

    if not sources and not built:
        print(f"No environments in {root}")
        return 0

    print(f"Environments in {root}:")
    for name, path in sources:
        status = "built" if name in built_names else "source only"
        print(f"  {name:<20} [{status}]")

    return 0
