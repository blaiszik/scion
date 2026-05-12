"""Install command for building environments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import get_root_or_exit

SCION_GIT_URL = "git+https://github.com/blaiszik/scion.git"


def _resolve_scion_install_spec() -> str:
    """
    Return a pip-installable spec for installing scion into a worker venv.

    If the current scion package is an editable install pointing at a real
    source tree (i.e. ``<parent>/pyproject.toml`` exists), use that local
    path so the worker venv runs the same code as the user's process.
    Otherwise (a regular site-packages install via ``pip install git+...``
    or eventually PyPI), fall back to the git URL.
    """
    import scion

    candidate = Path(scion.__file__).resolve().parent.parent
    if (candidate / "pyproject.toml").exists():
        return str(candidate)
    return SCION_GIT_URL


def extract_minimum_python_version(requires_python: str) -> str:
    """Extract minimum Python version from a requires-python specifier."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec_set = SpecifierSet(requires_python)
    min_version = None

    for spec in spec_set:
        if spec.operator in (">=", "~=", "==", ">"):
            version = Version(spec.version)
            if min_version is None or version < min_version:
                min_version = version

    if min_version is None:
        raise ValueError(
            f"Cannot determine minimum Python version from '{requires_python}'. "
            "Specifier must include >=, ~=, ==, or > constraint."
        )

    return f"{min_version.major}.{min_version.minor}"


def _install_single_environment(
    root: Path,
    source: str,
    force: bool,
    models: str | None,
    verbose: bool,
) -> int:
    """Install a single environment from a file path or environment name."""
    from ..environment import get_model_cache_env
    from ..pep723 import parse_pep723_metadata, validate_environment_file
    from .manifest import update_and_push_manifest

    source_path = Path(source)

    if source_path.is_file():
        env_name = source_path.stem
        env_source = root / "environments" / f"{env_name}.py"

        print(f"Validating {source_path}...")
        is_valid, error = validate_environment_file(source_path)
        if not is_valid:
            print(f"Error: {error}", file=sys.stderr)
            return 1

        if env_source.exists() and not force:
            print(
                f"Error: Environment '{env_name}' already registered at {env_source}",
                file=sys.stderr,
            )
            print("Use --force to update and rebuild", file=sys.stderr)
            return 1

        env_dir = root / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, env_source)
        print(f"Registered: {source_path} -> {env_source}")

    else:
        env_name = source
        env_source = root / "environments" / f"{env_name}.py"

        if not env_source.exists():
            print(f"Error: Environment not found: {env_name}", file=sys.stderr)
            available = (
                list((root / "environments").glob("*.py"))
                if (root / "environments").exists()
                else []
            )
            if available:
                print(f"Available: {[p.stem for p in available]}", file=sys.stderr)
            return 1

    env_target = root / "envs" / env_name

    if env_target.exists():
        if force:
            print(f"Removing existing environment: {env_target}")
            shutil.rmtree(env_target)
        else:
            print(f"Error: Environment already built: {env_target}", file=sys.stderr)
            print("Use --force to rebuild", file=sys.stderr)
            return 1

    print(f"Building environment: {env_name}")
    print(f"  Source: {env_source}")
    print(f"  Target: {env_target}")

    content = env_source.read_text()
    metadata = parse_pep723_metadata(content)
    if metadata is None:
        print(f"Error: No PEP 723 metadata in {env_source}", file=sys.stderr)
        return 1

    dependencies = metadata.get("dependencies", [])
    requires_python = metadata.get("requires-python", ">=3.10")
    uv_config = metadata.get("tool", {}).get("uv", {})
    find_links = uv_config.get("find-links", [])

    try:
        python_version = extract_minimum_python_version(requires_python)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  Python: {requires_python} -> {python_version}")
    print(f"  Dependencies: {dependencies}")
    if find_links:
        print(f"  Find-links: {find_links}")

    home_dir = root / "home"
    home_dir.mkdir(parents=True, exist_ok=True)

    python_install_dir = root / ".python"
    python_install_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Creating virtual environment...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_python_dir = Path(tmp_dir) / ".python"

        download_env = os.environ.copy()
        download_env["UV_PYTHON_INSTALL_DIR"] = str(tmp_python_dir)

        result = subprocess.run(
            ["uv", "python", "install", python_version],
            capture_output=True,
            text=True,
            env=download_env,
        )
        if result.returncode != 0:
            print(f"Error downloading Python: {result.stderr}", file=sys.stderr)
            return 1

        if tmp_python_dir.exists():
            for item in tmp_python_dir.iterdir():
                dest = python_install_dir / item.name
                if not dest.exists():
                    if item.is_dir():
                        print(f"  Copying Python to {dest}")
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

    uv_env = os.environ.copy()
    uv_env["UV_PYTHON_INSTALL_DIR"] = str(python_install_dir)

    result = subprocess.run(
        ["uv", "venv", str(env_target), "--python", python_version],
        capture_output=True,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        print(f"Error creating venv: {result.stderr}", file=sys.stderr)
        return 1

    env_python = env_target / "bin" / "python"

    print("2. Installing dependencies...")
    if dependencies:
        pip_cmd = ["uv", "pip", "install", "--python", str(env_python)]
        for link in find_links:
            pip_cmd.extend(["--find-links", link])
        pip_cmd.extend(dependencies)

        result = subprocess.run(
            pip_cmd,
            capture_output=not verbose,
            text=True,
            env=uv_env,
        )
        if result.returncode != 0:
            print(
                f"Error installing dependencies: {result.stderr if not verbose else ''}",
                file=sys.stderr,
            )
            return 1

    print("3. Installing scion...")
    scion_install_spec = _resolve_scion_install_spec()
    print(f"   Source: {scion_install_spec}")

    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(env_python), scion_install_spec],
        capture_output=not verbose,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        print(
            f"Error installing scion: {result.stderr if not verbose else ''}",
            file=sys.stderr,
        )
        return 1

    print("4. Copying environment source...")
    shutil.copy(env_source, env_target / "env_source.py")

    if models:
        model_list = [m.strip() for m in models.split(",")]
        print(f"5. Pre-downloading models: {model_list}")

        cache_env = get_model_cache_env(root)
        env = {**os.environ, **cache_env}

        for model in model_list:
            print(f"   Downloading: {model}")
            script = f'''
import sys
sys.path.insert(0, "{env_target}")
from env_source import setup
provider = setup("{model}", "cpu")
print(f"Downloaded model: {model}")
'''
            result = subprocess.run(
                [str(env_python), "-c", script],
                env=env,
                capture_output=not verbose,
                text=True,
            )
            if result.returncode != 0:
                print(f"   Warning: Failed to download {model}", file=sys.stderr)
                if verbose:
                    print(result.stderr, file=sys.stderr)

    print(f"\nBuilt environment: {env_target}")

    update_and_push_manifest(root, quiet=False)

    return 0


def cmd_install(args) -> int:
    """Install environment(s) from a file, directory, or rebuild by name."""
    from ..environment import check_uv_available

    root = get_root_or_exit(args)
    source = args.source
    source_path = Path(source)

    if not check_uv_available():
        print(
            "Error: uv not found in PATH. Install uv: "
            "https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    if source_path.is_dir():
        env_files = sorted(source_path.glob("*.py"))
        if not env_files:
            print(f"Error: No *.py files found in {source_path}", file=sys.stderr)
            return 1

        print(f"Installing {len(env_files)} environment(s) from {source_path}:")
        for f in env_files:
            print(f"  - {f.name}")
        print()

        succeeded, failed = [], []

        for env_file in env_files:
            print(f"{'=' * 60}")
            print(f"Installing: {env_file.name}")
            print(f"{'=' * 60}")

            result = _install_single_environment(
                root=root,
                source=str(env_file),
                force=args.force,
                models=args.models,
                verbose=args.verbose,
            )

            (succeeded if result == 0 else failed).append(env_file.stem)
            print()

        print(f"{'=' * 60}")
        print("Summary:")
        print(f"  Succeeded: {len(succeeded)}")
        if succeeded:
            print(f"    {', '.join(succeeded)}")
        print(f"  Failed: {len(failed)}")
        if failed:
            print(f"    {', '.join(failed)}")

        return 1 if failed else 0

    return _install_single_environment(
        root=root,
        source=source,
        force=args.force,
        models=args.models,
        verbose=args.verbose,
    )
