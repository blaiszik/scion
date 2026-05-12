"""
HTTP client for Scion backend API.

Handles pushing manifests to the central registry.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import UserConfig
from .manifest import Manifest


class ScionClient:
    """Client for Scion backend API."""

    def __init__(self, config: UserConfig):
        self.config = config

    def push_manifest(self, manifest: Manifest) -> tuple[bool, str]:
        """Push manifest to backend API."""
        if not self.config.api_key:
            return False, "API key not configured"
        if not self.config.api_secret:
            return False, "API secret not configured"
        if not self.config.api_url:
            return False, "API URL not configured"

        url = self.config.api_url
        data = json.dumps(manifest.to_dict()).encode("utf-8")

        request = Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Modal-Key": self.config.api_key,
                "Modal-Secret": self.config.api_secret,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                if response.status in (200, 201, 204):
                    return True, "Manifest pushed successfully"
                return False, f"Unexpected status: {response.status}"
        except HTTPError as e:
            return False, f"HTTP error {e.code}: {e.reason}"
        except URLError as e:
            return False, f"Connection error: {e.reason}"
        except TimeoutError:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Error: {e}"
