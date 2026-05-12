"""
Manifest management for Scion.

The manifest tracks the state of a scion installation:
- Which environments are built
- What capabilities each provides (fold/embed/...)
- Dependency versions
- Checkpoints available
- Last update time

Schema is bumped to version "2" relative to Rootstock to carry the
per-environment `capabilities` list.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import UserConfig


@dataclass
class Maintainer:
    """Maintainer contact information."""

    name: str
    email: str

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data: dict) -> Maintainer:
        return cls(name=data["name"], email=data["email"])


@dataclass
class EnvironmentInfo:
    """Metadata for a single built environment."""

    status: str  # "ready", "building", "error"
    built_at: str  # ISO 8601 timestamp
    source_hash: str  # "sha256:abc123..."
    source: str  # Full source code of the environment file
    python_requires: str  # ">=3.10"
    dependencies: dict[str, str]  # {"boltz": "2.0.0"}
    capabilities: list[str] = field(default_factory=list)  # ["fold"], ["embed"], ...
    checkpoints: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "built_at": self.built_at,
            "source_hash": self.source_hash,
            "source": self.source,
            "python_requires": self.python_requires,
            "dependencies": self.dependencies,
            "capabilities": self.capabilities,
            "checkpoints": self.checkpoints,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        return d

    @classmethod
    def from_dict(cls, data: dict) -> EnvironmentInfo:
        return cls(
            status=data["status"],
            built_at=data["built_at"],
            source_hash=data["source_hash"],
            source=data.get("source", ""),
            python_requires=data["python_requires"],
            dependencies=data["dependencies"],
            capabilities=data.get("capabilities", []),
            checkpoints=data.get("checkpoints", []),
            error_message=data.get("error_message"),
        )


@dataclass
class Manifest:
    """Root manifest for a Scion installation."""

    schema_version: str
    cluster: str
    root: str
    maintainer: Maintainer
    scion_version: str
    python_version: str
    last_updated: str  # ISO 8601 timestamp
    environments: dict[str, EnvironmentInfo] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "cluster": self.cluster,
            "root": self.root,
            "maintainer": self.maintainer.to_dict(),
            "scion_version": self.scion_version,
            "python_version": self.python_version,
            "last_updated": self.last_updated,
            "environments": {name: env.to_dict() for name, env in self.environments.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        environments = {}
        for name, env_data in data.get("environments", {}).items():
            environments[name] = EnvironmentInfo.from_dict(env_data)

        return cls(
            schema_version=data["schema_version"],
            cluster=data["cluster"],
            root=data["root"],
            maintainer=Maintainer.from_dict(data["maintainer"]),
            scion_version=data["scion_version"],
            python_version=data["python_version"],
            last_updated=data["last_updated"],
            environments=environments,
        )

    def validate(self) -> tuple[bool, str]:
        if not self.schema_version:
            return False, "Missing schema_version"
        if not self.cluster:
            return False, "Missing cluster"
        if not self.root:
            return False, "Missing root"
        if not self.maintainer.name:
            return False, "Missing maintainer name"
        if not self.maintainer.email:
            return False, "Missing maintainer email"

        valid_statuses = {"ready", "building", "error"}
        for env_name, env_info in self.environments.items():
            if env_info.status not in valid_statuses:
                return False, f"Invalid status '{env_info.status}' for {env_name}"

        return True, "OK"


def compute_source_hash(source_path: Path) -> str:
    """Compute SHA256 hash of environment source file."""
    content = source_path.read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def get_installed_versions(
    env_path: Path, only_packages: list[str] | None = None
) -> dict[str, str]:
    """Get installed package versions from a venv (filtered to direct deps)."""
    python_path = env_path / "bin" / "python"
    if not python_path.exists():
        return {}

    def normalize(name: str) -> str:
        return name.lower().replace("-", "_").replace(".", "_")

    filter_set = (
        {
            normalize(
                p.split("==")[0]
                .split(">")[0]
                .split("<")[0]
                .split("~")[0]
                .split("[")[0]
                .strip()
            )
            for p in only_packages
        }
        if only_packages
        else None
    )

    packages_data = []

    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "pip", "list", "--python", str(python_path), "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                packages_data = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            pass

    if not packages_data:
        try:
            result = subprocess.run(
                [str(python_path), "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                packages_data = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            pass

    if not packages_data:
        return {}

    installed = {}
    for pkg in packages_data:
        name = pkg["name"]
        version = pkg["version"]
        if filter_set is None or normalize(name) in filter_set:
            installed[name] = version

    return installed


def detect_python_version(root: Path) -> str:
    """Detect Python version from installed interpreter."""
    python_dir = root / ".python"
    if python_dir.exists():
        for item in python_dir.iterdir():
            if item.is_dir() and "cpython" in item.name:
                parts = item.name.split("-")
                if len(parts) >= 2:
                    return parts[1]
    return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(root: Path) -> Manifest | None:
    """Load manifest from root directory."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            data = json.load(f)
        return Manifest.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def save_manifest(manifest: Manifest, root: Path) -> None:
    """Save manifest to root directory atomically."""
    manifest.last_updated = now_iso()
    manifest_path = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=root, suffix=".json")
    try:
        with open(fd, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        Path(temp_path).rename(manifest_path)
    except Exception:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass
        raise


def create_manifest(
    root: Path,
    cluster: str,
    config: UserConfig,
) -> Manifest:
    """Create a new manifest for a Scion installation."""
    from . import __version__

    return Manifest(
        schema_version="2",
        cluster=cluster,
        root=str(root),
        maintainer=Maintainer(
            name=config.name or "Unknown",
            email=config.email or "unknown@example.com",
        ),
        scion_version=__version__,
        python_version=detect_python_version(root),
        last_updated=now_iso(),
        environments={},
    )
