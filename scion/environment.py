"""
Environment management for Scion.

Manages pre-built virtual environments, generates wrapper scripts for worker
processes, and produces spawn commands. Mirrors Rootstock's environment.py
with model-cache redirection adjusted for protein FM stacks (HuggingFace,
Torch hub, ColabFold).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def get_model_cache_env(root: Path) -> dict[str, str]:
    """
    Environment variables to redirect model downloads to the shared cache.

    HOME redirect catches libraries that hardcode ~/.cache/... (ColabFold,
    MatGL-style libs). XDG_CACHE_HOME catches well-behaved libraries.
    HF_HOME explicitly catches HuggingFace, which most protein FMs route
    through (ESM, Boltz weights on HF Hub, etc.). TORCH_HOME catches
    torch.hub.
    """
    cache_dir = root / "cache"
    home_dir = root / "home"
    return {
        "HOME": str(home_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
        "TORCH_HOME": str(cache_dir / "torch"),
    }


WRAPPER_TEMPLATE = """
import sys
sys.path.insert(0, "{env_dir}")
from env_source import setup, CAPABILITIES
from scion.worker import run_worker

run_worker(
    setup_fn=setup,
    capabilities=CAPABILITIES,
    model={model!r},
    device={device!r},
    socket_path={socket_path!r},
)
"""


class EnvironmentManager:
    """
    Manages pre-built Scion environments and worker spawning.

    Environments live at {root}/envs/{env_name}/. The source file is
    copied into the venv as env_source.py during build, and exposes both
    a `setup(model, device)` function and a `CAPABILITIES` list.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._temp_files: list[Path] = []

    def get_env_python(self, env_name: str) -> Path:
        """Path to Python executable for a pre-built environment."""
        env_python = self.root / "envs" / env_name / "bin" / "python"

        if not env_python.exists():
            envs_dir = self.root / "envs"
            if envs_dir.exists():
                available = [p.name for p in envs_dir.iterdir() if p.is_dir()]
            else:
                available = []

            raise RuntimeError(
                f"Environment '{env_name}' not built. "
                f"Run: scion install {env_name} --root {self.root}\n"
                f"Available environments: {available}"
            )

        return env_python

    def generate_wrapper(
        self,
        env_name: str,
        model: str,
        device: str,
        socket_path: str,
    ) -> Path:
        """Generate a wrapper script for the given environment."""
        env_dir = self.root / "envs" / env_name

        wrapper_content = WRAPPER_TEMPLATE.format(
            env_dir=str(env_dir),
            model=model,
            device=device,
            socket_path=socket_path,
        )

        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="scion_wrapper_")
        with open(fd, "w") as f:
            f.write(wrapper_content)

        tmp_path = Path(tmp_path)
        self._temp_files.append(tmp_path)
        return tmp_path

    def get_spawn_command(self, env_name: str, wrapper_path: Path) -> list[str]:
        """Command to spawn a worker using the pre-built environment."""
        env_python = self.get_env_python(env_name)
        return [str(env_python), str(wrapper_path)]

    def get_environment_variables(self) -> dict[str, str]:
        """
        Environment variables to set for the worker process.

        Layering (later wins):
          1. inherited os.environ
          2. cache redirection (HOME, XDG_CACHE_HOME, HF_*, TORCH_HOME)
          3. {root}/cluster.toml overlay (login_env or compute_env)
        """
        from .cluster_config import get_cluster_env
        from .clusters import get_profile_for_root

        env = os.environ.copy()
        env.update(get_model_cache_env(self.root))
        env.update(get_cluster_env(self.root))
        profile = get_profile_for_root(self.root)
        if profile is not None and profile.runtime_dir:
            env.setdefault(
                "SCION_RUNTIME_DIR",
                str(Path(os.path.expandvars(profile.runtime_dir)).expanduser()),
            )
        return env

    def cleanup(self):
        """Clean up temporary wrapper files."""
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_files.clear()

    def __del__(self):
        self.cleanup()


def check_uv_available() -> bool:
    """Check if uv is available in PATH."""
    return shutil.which("uv") is not None


def list_environments(root: Path | str) -> list[tuple[str, Path]]:
    """List registered environment source files."""
    root = Path(root)
    env_dir = root / "environments"

    if not env_dir.exists():
        return []

    result = []
    for path in sorted(env_dir.glob("*.py")):
        result.append((path.stem, path))

    return result


def list_built_environments(root: Path | str) -> list[tuple[str, Path]]:
    """List pre-built environments."""
    root = Path(root)
    envs_dir = root / "envs"

    if not envs_dir.exists():
        return []

    result = []
    for path in sorted(envs_dir.iterdir()):
        if path.is_dir() and (path / "bin" / "python").exists():
            result.append((path.name, path))

    return result
