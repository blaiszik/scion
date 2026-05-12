"""
User configuration for Scion.

Loads/saves user-specific config from ~/.config/scion/config.toml with
environment variable fallback.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SCION_API_KEY_ENV = "SCION_API_KEY"
SCION_API_SECRET_ENV = "SCION_API_SECRET"
SCION_API_URL_ENV = "SCION_API_URL"

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "scion"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"


@dataclass
class UserConfig:
    """User configuration for Scion."""

    root: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_url: str | None = None
    name: str | None = None
    email: str | None = None
    is_maintainer: bool = False

    def is_push_enabled(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_url)

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "API key not configured"
        if not self.api_secret:
            return False, "API secret not configured"
        if not self.api_url:
            return False, "API URL not configured"
        return True, "OK"


def load_config(config_path: Path | None = None) -> UserConfig:
    """Load user configuration from TOML file with env var fallback."""
    config = UserConfig()

    path = config_path or DEFAULT_CONFIG_FILE
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        config.root = data.get("root")
        config.api_key = data.get("api_key")
        config.api_secret = data.get("api_secret")
        config.api_url = data.get("api_url")
        config.is_maintainer = data.get("is_maintainer", False)
        maintainer = data.get("maintainer", {})
        config.name = maintainer.get("name")
        config.email = maintainer.get("email")

    if not config.api_key:
        config.api_key = os.environ.get(SCION_API_KEY_ENV)
    if not config.api_secret:
        config.api_secret = os.environ.get(SCION_API_SECRET_ENV)
    if not config.api_url:
        config.api_url = os.environ.get(SCION_API_URL_ENV)

    return config


def save_config(config: UserConfig, config_path: Path | None = None) -> None:
    """Save user configuration to TOML file."""
    path = config_path or DEFAULT_CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if config.root:
        lines.append(f'root = "{config.root}"')
    lines.append(f"is_maintainer = {str(config.is_maintainer).lower()}")
    if config.api_key:
        lines.append(f'api_key = "{config.api_key}"')
    if config.api_secret:
        lines.append(f'api_secret = "{config.api_secret}"')
    if config.api_url:
        lines.append(f'api_url = "{config.api_url}"')

    if config.name or config.email:
        lines.append("")
        lines.append("[maintainer]")
        if config.name:
            lines.append(f'name = "{config.name}"')
        if config.email:
            lines.append(f'email = "{config.email}"')

    path.write_text("\n".join(lines) + "\n")
