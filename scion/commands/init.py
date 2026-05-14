"""Interactive initialization command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..clusters import get_cluster_for_root, get_cluster_profile, list_cluster_profiles
from ..config import DEFAULT_CONFIG_FILE, load_config, save_config
from ..manifest import create_manifest, save_manifest
from ..resources import CLUSTER_TOML_TEMPLATE, find_bundled_environments_dir
from .common import SCION_ROOT_ENV
from .install import _install_single_environment
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


def _check_root_writable(root: Path) -> tuple[bool, str]:
    """
    Return (writable, reason). Walks up to the first existing ancestor
    and tests whether the user can create a tempfile there. Robust to
    the root itself not existing yet.
    """
    import tempfile

    probe = root
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return False, f"no existing ancestor of {root}"
        probe = parent
    if not os.access(probe, os.W_OK):
        return False, f"{probe} is not writable"
    try:
        with tempfile.NamedTemporaryFile(prefix=".scion_init_", dir=probe, delete=True):
            pass
    except OSError as e:
        return False, str(e)
    return True, "ok"


def cmd_init(args) -> int:
    """Interactive initialization of Scion configuration."""
    print("Welcome to Scion!")
    print("This will help you set up your configuration.\n")

    config = load_config()

    print("Root directory is where environments and caches are stored.")
    profiles = list_cluster_profiles()
    print(f"Known clusters: {', '.join(sorted(profiles.keys()))}")
    print("You can enter a cluster name or a custom path.\n")

    root_default = config.root or os.environ.get(SCION_ROOT_ENV)
    root_input = prompt_with_default("Root directory", root_default)

    if not root_input:
        print("Error: Root directory is required.", file=sys.stderr)
        return 1

    if root_input in profiles:
        cluster = root_input
        root = get_cluster_profile(root_input).root_path
        print(f"  -> Using cluster '{cluster}' root: {root}")
    else:
        root = Path(root_input).expanduser().resolve()
        cluster = get_cluster_for_root(root)
        if cluster:
            print(f"  -> Detected cluster: {cluster}")

    config.root = str(root)

    # Writability gate: the most common Polaris first-run failure mode is
    # selecting a maintainer-owned shared root (e.g. another project's
    # allocation) that this user can't write to. Detect early and offer
    # an alternative rather than silently no-op'ing every mkdir below.
    writable, why = _check_root_writable(root)
    if not writable:
        print(f"\nWarning: {root} is not writable by this user.")
        print(f"  ({why})")
        print(
            "  Shared / maintainer roots can only be initialized by the "
            "user who owns the directory. For your own install, pick a "
            "path under your home or project allocation, e.g. ~/scion or "
            "$SCRATCH/scion."
        )
        alt = prompt_with_default("Alternative root path (or Enter to abort)", "")
        if not alt:
            print("Aborting init.", file=sys.stderr)
            return 1
        root = Path(alt).expanduser().resolve()
        config.root = str(root)
        writable2, why2 = _check_root_writable(root)
        if not writable2:
            print(f"Error: {root} is also not writable: {why2}", file=sys.stderr)
            return 1
        cluster = get_cluster_for_root(root)
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

    _maybe_seed_cluster_toml(root)
    installed_envs = _maybe_install_bundled_envs(root)

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
    if installed_envs:
        first = installed_envs[0]
        names = ", ".join(installed_envs)
        print(f"  1. Validate: scion check {first} --thorough --device cpu")
        print(f"     (installed: {names})")
        print("  2. Call from Python:")
        if "esm2_env" in installed_envs:
            print(
                f"       from scion import Embedder\n"
                f"       with Embedder(root={str(root)!r}, model='esm2') as e:\n"
                f"           print(e.embed(['MKTAYIAKQRQ']).per_residue.shape)"
            )
        elif "boltz_env" in installed_envs:
            print(
                f"       from scion import Folder\n"
                f"       with Folder(root={str(root)!r}, model='boltz') as f:\n"
                f"           print(f.fold('MKTAYIAKQRQ').confidence)"
            )
    else:
        print("  1. Install an environment: scion install <env_file.py>")
        print(
            "     (bundled envs ship in this package — see "
            "`scion install` with the path your maintainer provides)"
        )
        print("  2. Validate: scion check <env> --thorough --device cpu")
        print("  3. Call from Python:")
        print(
            f"       from scion import Folder\n"
            f"       with Folder(root={str(root)!r}, model='boltz') as f:\n"
            f"           print(f.fold('MKTAYIAKQRQ').confidence)"
        )

    return 0


def _maybe_seed_cluster_toml(root: Path) -> None:
    """Prompt to seed ``{root}/cluster.toml`` with the bundled template."""
    target = root / "cluster.toml"
    if target.exists():
        print(f"\n{target} already exists; leaving it untouched.")
        return

    print(
        "\nSeed a starter cluster.toml at the install root? It caps "
        "OMP/MKL/OPENBLAS to 1 on login nodes (avoids `libgomp: Thread "
        "creation failed` on shared HPC login tiers)."
    )
    choice = prompt_with_default("Seed cluster.toml (y/n)", "y") or "n"
    if choice.lower() in ("y", "yes"):
        try:
            target.write_text(CLUSTER_TOML_TEMPLATE)
            print(f"  Wrote {target}")
        except OSError as e:
            print(f"  Warning: could not write {target}: {e}", file=sys.stderr)


def _maybe_install_bundled_envs(root: Path) -> list[str]:
    """
    Prompt to install the bundled env files. Returns the list of env names
    actually built (for the "next steps" hint). Skipped silently when the
    bundled envs can't be located.
    """
    bundle = find_bundled_environments_dir()
    if bundle is None:
        return []

    env_files = sorted(bundle.glob("*.py"))
    if not env_files:
        return []

    names = [p.stem for p in env_files]
    print(
        f"\nFound bundled env files in {bundle}:\n  - "
        + "\n  - ".join(names)
    )
    print(
        "Install them now? `scion install` builds an isolated venv per env; "
        "this can take several minutes per env on first run."
    )
    choice = prompt_with_default("Install bundled envs (y/n)", "n") or "n"
    if choice.lower() not in ("y", "yes"):
        return []

    installed: list[str] = []
    for env_file in env_files:
        print(f"\n--- installing {env_file.name} ---")
        rc = _install_single_environment(
            root=root,
            source=str(env_file),
            force=False,
            models=None,
            verbose=False,
            skip_preload=True,
        )
        if rc == 0:
            installed.append(env_file.stem)
        else:
            print(f"  (install failed for {env_file.stem}; continuing)", file=sys.stderr)
    return installed
