"""Resolve command for cluster configuration lookup."""

from __future__ import annotations

import json
import sys


def cmd_resolve(args) -> int:
    """Resolve cluster configuration and print as JSON."""
    from ..clusters import get_cluster_profile

    try:
        profile = get_cluster_profile(args.cluster)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = {
        "root": str(profile.root_path),
        "cluster": profile.name,
        "scheduler": profile.scheduler,
        "job_env_vars": list(profile.job_env_vars),
        "runtime_dir": profile.runtime_dir,
        "source": profile.source,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Cluster:   {profile.name}")
        print(f"Root:      {profile.root_path}")
        print(f"Scheduler: {profile.scheduler or '(unknown)'}")
        print(f"Source:    {profile.source}")
    return 0
