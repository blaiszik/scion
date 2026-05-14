"""HPC readiness diagnostics for a Scion installation."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..cluster_config import get_cluster_env, is_in_batch_job, load_cluster_config
from ..clusters import (
    DEFAULT_JOB_ENV_VARS,
    ClusterProfile,
    detect_current_cluster,
    get_cluster_profile,
    get_profile_for_root_or_host,
)
from ..config import load_config
from ..environment import get_model_cache_env, list_built_environments, list_environments
from ..pep723 import get_capabilities
from .common import SCION_ROOT_ENV


@dataclass
class DoctorCheck:
    status: str
    name: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def _expand_runtime_dir(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(os.path.expandvars(value)).expanduser())


def _write_probe(path: Path) -> tuple[bool, str | None]:
    try:
        with tempfile.NamedTemporaryFile(prefix=".scion_doctor_", dir=path, delete=True):
            pass
    except OSError as e:
        return False, str(e)
    return True, None


def _add(
    checks: list[DoctorCheck],
    status: str,
    name: str,
    message: str,
    **detail: Any,
) -> None:
    checks.append(DoctorCheck(status=status, name=name, message=message, detail=detail))


def _resolve_target(args) -> tuple[Path | None, ClusterProfile | None, list[DoctorCheck]]:
    checks: list[DoctorCheck] = []

    if getattr(args, "cluster", None):
        try:
            profile = get_cluster_profile(args.cluster)
        except ValueError as e:
            _add(checks, "error", "cluster", str(e))
            return None, None, checks
        return profile.root_path, profile, checks

    if getattr(args, "root", None):
        root = Path(args.root).expanduser()
        return root, get_profile_for_root_or_host(root), checks

    env_root = os.environ.get(SCION_ROOT_ENV)
    if env_root:
        root = Path(env_root).expanduser()
        return root, get_profile_for_root_or_host(root), checks

    config = load_config()
    if config.root:
        root = Path(config.root).expanduser()
        return root, get_profile_for_root_or_host(root), checks

    profile = detect_current_cluster()
    if profile is not None:
        return profile.root_path, profile, checks

    _add(
        checks,
        "error",
        "root",
        f"No root configured. Pass --root, set {SCION_ROOT_ENV}, run scion init, "
        "or add a matching cluster profile.",
    )
    return None, None, checks


def _check_root(root: Path | None, checks: list[DoctorCheck]) -> None:
    if root is None:
        return

    if not root.exists():
        _add(checks, "error", "root", f"Scion root does not exist: {root}")
        return
    if not root.is_dir():
        _add(checks, "error", "root", f"Scion root is not a directory: {root}")
        return

    _add(checks, "ok", "root", f"Scion root exists: {root}")

    root_str = str(root)
    if root_str == "/tmp" or root_str.startswith("/tmp/"):
        _add(
            checks,
            "warn",
            "root persistence",
            "Root is under /tmp; most HPC /tmp filesystems are node-local.",
        )

    for name in ("environments", "envs", "cache", "home", ".python"):
        path = root / name
        if path.exists():
            _add(checks, "ok", f"dir:{name}", f"{name}/ exists")
        else:
            _add(checks, "warn", f"dir:{name}", f"{name}/ is missing")

    for name in ("cache", "home"):
        path = root / name
        if path.exists() and path.is_dir():
            ok, error = _write_probe(path)
            if ok:
                _add(checks, "ok", f"writable:{name}", f"{name}/ is writable")
            else:
                _add(
                    checks,
                    "warn",
                    f"writable:{name}",
                    f"{name}/ is not writable by this process",
                    error=error,
                )


def _check_profile(
    root: Path | None,
    profile: ClusterProfile | None,
    checks: list[DoctorCheck],
) -> tuple[bool, tuple[str, ...]]:
    if profile is None:
        if root is not None:
            _add(
                checks,
                "warn",
                "cluster profile",
                "Root does not match a registered cluster profile; using generic checks.",
            )
        return is_in_batch_job(), DEFAULT_JOB_ENV_VARS

    _add(
        checks,
        "ok",
        "cluster profile",
        f"{profile.name} profile loaded",
        source=profile.source,
        scheduler=profile.scheduler,
        root=str(profile.root_path),
    )

    in_job = is_in_batch_job(job_env_vars=profile.job_env_vars)
    context = "compute job" if in_job else "login/interactive shell"
    _add(
        checks,
        "ok",
        "scheduler context",
        f"Detected {context}",
        scheduler=profile.scheduler,
        job_env_vars=list(profile.job_env_vars),
    )
    return in_job, profile.job_env_vars


def _check_cluster_toml(
    root: Path | None,
    checks: list[DoctorCheck],
    in_job: bool,
) -> dict[str, str]:
    if root is None:
        return {}

    path = root / "cluster.toml"
    if path.exists():
        cfg = load_cluster_config(root)
        overlay = cfg.resolved_env(in_job=in_job)
        _add(
            checks,
            "ok",
            "cluster.toml",
            f"Loaded {path}",
            env=len(cfg.env),
            login_env=len(cfg.login_env),
            compute_env=len(cfg.compute_env),
        )
        if overlay:
            _add(
                checks,
                "ok",
                "env overlay",
                f"Resolved {len(overlay)} cluster environment override(s)",
                keys=sorted(overlay),
            )
        else:
            _add(checks, "info", "env overlay", "No cluster env overrides apply")
    else:
        _add(
            checks,
            "warn",
            "cluster.toml",
            "No cluster.toml found at the Scion root; login/compute env quirks "
            "will rely on defaults.",
        )

    return get_cluster_env(root, in_job=in_job)


def _check_threads(
    root: Path | None,
    checks: list[DoctorCheck],
    in_job: bool,
    cluster_env: dict[str, str],
) -> None:
    if root is None:
        return

    env = {**os.environ, **get_model_cache_env(root), **cluster_env}
    if in_job:
        _add(checks, "info", "thread caps", "Inside a batch job; login-node caps not required")
        return

    expected = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    values = {name: env.get(name) for name in expected}
    if all(values[name] == "1" for name in expected):
        _add(checks, "ok", "thread caps", "Login-node thread caps are set", values=values)
    else:
        _add(
            checks,
            "warn",
            "thread caps",
            "Login-node thread caps are not all set to 1; torch/MKL/OpenBLAS may "
            "hit process limits on shared login nodes.",
            values=values,
        )


def _check_runtime_dir(profile: ClusterProfile | None, checks: list[DoctorCheck]) -> None:
    runtime_dir = (
        os.environ.get("SCION_RUNTIME_DIR")
        or _expand_runtime_dir(profile.runtime_dir if profile else None)
        or tempfile.gettempdir()
    )
    path = Path(runtime_dir)
    if not path.exists():
        _add(checks, "warn", "runtime dir", f"Runtime dir does not exist: {path}")
        return
    if not path.is_dir():
        _add(checks, "warn", "runtime dir", f"Runtime path is not a directory: {path}")
        return

    sample = str(path / f"scion_doctor_{os.getpid()}_0123456789abcdef")
    detail = {"path": str(path), "sample_socket_length": len(sample)}
    if len(sample.encode()) > 100:
        _add(
            checks,
            "warn",
            "runtime dir",
            "Runtime dir is long enough to risk Unix socket path length limits",
            **detail,
        )
    else:
        _add(checks, "ok", "runtime dir", f"Runtime dir is usable: {path}", **detail)


def _check_tools(checks: list[DoctorCheck]) -> None:
    uv = shutil.which("uv")
    if uv:
        _add(checks, "ok", "uv", f"uv found: {uv}")
    else:
        _add(checks, "warn", "uv", "uv not found in PATH; scion install cannot build envs")

    _add(
        checks,
        "ok",
        "python",
        (
            f"Running Python {sys.version_info.major}."
            f"{sys.version_info.minor}.{sys.version_info.micro}"
        ),
    )


def _check_gpu(checks: list[DoctorCheck], in_job: bool) -> str | None:
    """
    Run nvidia-smi checks and return the highest driver_version observed
    (so subsequent checks can cross-reference torch/CUDA compat). Returns
    None when no usable GPU is detected.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        status = "warn" if in_job else "info"
        _add(checks, status, "gpu", "nvidia-smi not found in PATH")
        return None

    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _add(checks, "warn", "gpu", f"nvidia-smi could not be queried: {e}")
        return None

    if proc.returncode != 0:
        status = "warn" if in_job else "info"
        _add(checks, status, "gpu", "nvidia-smi returned a nonzero status", stderr=proc.stderr)
        return None

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    _add(checks, "ok", "gpu", f"nvidia-smi reports {len(lines)} GPU(s)", gpus=lines)

    # First column is driver_version (e.g. "535.230.02"). nvidia-smi's
    # driver_version is the *kernel-side* driver — separate from the CUDA
    # version it supports. Pull the CUDA version separately.
    try:
        proc2 = subprocess.run(
            [nvidia_smi, "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc2.returncode == 0:
            cuda_versions = [
                line.strip() for line in proc2.stdout.splitlines() if line.strip()
            ]
            if cuda_versions:
                return cuda_versions[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _check_torch_cuda_compat(
    profile: ClusterProfile | None,
    observed_cuda: str | None,
    root: Path | None,
    env_name: str | None,
    checks: list[DoctorCheck],
) -> None:
    """
    Cross-check each built env's torch pin against the driver's CUDA cap.

    Uses, in priority order: the cluster profile's ``cuda_driver_max`` (the
    maintainer's declared ceiling), or the observed cuda version from
    nvidia-smi. Compares against each env's pinned torch from its PEP 723
    deps. Surfaces a single ``error`` check per failing env so users learn
    about the mismatch before submitting a job.
    """
    from ..cuda_compat import check_torch_spec_against_driver
    from ..pep723 import get_dependencies

    cap = (profile.cuda_driver_max if profile else None) or observed_cuda
    if cap is None or root is None:
        return

    if env_name:
        env_sources = [(env_name, root / "envs" / env_name / "env_source.py")]
    else:
        env_sources = [
            (name, path / "env_source.py")
            for name, path in list_built_environments(root)
        ]

    for name, source in env_sources:
        if not source.exists():
            continue
        deps = get_dependencies(source)
        if not deps:
            continue
        result = check_torch_spec_against_driver(deps, cap)
        if result.torch_spec is None:
            # env has no torch dep
            continue
        check_name = f"cuda compat:{name}"
        if result.ok:
            _add(checks, "ok", check_name, result.message,
                 torch_spec=result.torch_spec, driver_cuda=cap)
        else:
            _add(checks, "error", check_name, result.message,
                 torch_spec=result.torch_spec,
                 required_cuda=result.required_cuda,
                 driver_cuda=cap)


def _check_environments(
    root: Path | None,
    env_name: str | None,
    checks: list[DoctorCheck],
) -> None:
    if root is None or not root.exists() or not root.is_dir():
        return

    if env_name:
        env_dir = root / "envs" / env_name
        python = env_dir / "bin" / "python"
        source = env_dir / "env_source.py"
        if python.exists() and source.exists():
            caps = get_capabilities(source)
            _add(
                checks,
                "ok",
                f"env:{env_name}",
                f"{env_name} is built",
                capabilities=caps,
                python=str(python),
            )
        else:
            _add(
                checks,
                "error",
                f"env:{env_name}",
                f"{env_name} is not built or is incomplete",
                python_exists=python.exists(),
                source_exists=source.exists(),
            )
        return

    sources = list_environments(root)
    built = list_built_environments(root)
    if not sources and not built:
        _add(checks, "warn", "environments", "No registered or built environments found")
        return

    _add(
        checks,
        "ok",
        "environments",
        f"{len(sources)} source env(s), {len(built)} built env(s)",
        sources=[name for name, _ in sources],
        built=[name for name, _ in built],
    )

    for name, path in built:
        source = path / "env_source.py"
        if not source.exists():
            _add(checks, "warn", f"env:{name}", "Built env is missing env_source.py")


def run_doctor(args) -> dict[str, Any]:
    root, profile, checks = _resolve_target(args)

    if profile is not None:
        cluster = profile.name
    else:
        cluster = None

    in_job, _ = _check_profile(root, profile, checks)
    _check_root(root, checks)
    cluster_env = _check_cluster_toml(root, checks, in_job=in_job)
    _check_threads(root, checks, in_job=in_job, cluster_env=cluster_env)
    _check_runtime_dir(profile, checks)
    _check_tools(checks)
    observed_cuda = _check_gpu(checks, in_job=in_job)
    env_arg = getattr(args, "env_name", None)
    _check_environments(root, env_arg, checks)
    _check_torch_cuda_compat(profile, observed_cuda, root, env_arg, checks)

    counts = {
        "ok": sum(c.status == "ok" for c in checks),
        "warn": sum(c.status == "warn" for c in checks),
        "error": sum(c.status == "error" for c in checks),
        "info": sum(c.status == "info" for c in checks),
    }
    return {
        "cluster": cluster,
        "root": str(root) if root is not None else None,
        "hostname": socket.getfqdn(),
        "in_job": in_job,
        "summary": counts,
        "checks": [asdict(c) for c in checks],
    }


def _print_text(report: dict[str, Any]) -> None:
    print("Scion doctor")
    print(f"  Host:    {report['hostname']}")
    print(f"  Cluster: {report['cluster'] or '(unregistered)'}")
    print(f"  Root:    {report['root'] or '(not resolved)'}")
    print(f"  Context: {'batch job' if report['in_job'] else 'login/interactive'}")
    print()

    for check in report["checks"]:
        status = check["status"].upper()
        print(f"[{status:<5}] {check['name']}: {check['message']}")

    s = report["summary"]
    print()
    print(
        f"Summary: {s['ok']} ok, {s['warn']} warning(s), "
        f"{s['error']} error(s), {s['info']} info"
    )


def cmd_doctor(args) -> int:
    """Run Scion HPC readiness checks."""
    if getattr(args, "all_envs", False):
        return _cmd_doctor_all_envs(args)
    report = run_doctor(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)
    return 1 if report["summary"]["error"] else 0


def _cmd_doctor_all_envs(args) -> int:
    """Doctor across every built env. JSON mode emits a list of per-env reports."""
    # Resolve root once so we can enumerate built envs.
    root, _profile, _checks = _resolve_target(args)
    if root is None or not root.exists():
        # Fall back to single-pass behaviour so the existing error surfaces.
        report = run_doctor(args)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_text(report)
        return 1 if report["summary"]["error"] else 0

    env_names = [name for name, _ in list_built_environments(root)]
    if not env_names:
        # Nothing to iterate; just run the global checks once.
        report = run_doctor(args)
        if args.json:
            print(json.dumps([report], indent=2))
        else:
            _print_text(report)
        return 1 if report["summary"]["error"] else 0

    reports = []
    overall_error = False
    for name in env_names:
        env_args = SimpleNamespace(
            **{**vars(args), "env_name": name, "all_envs": False}
        )
        report = run_doctor(env_args)
        reports.append({"env": name, "report": report})
        if report["summary"]["error"]:
            overall_error = True

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for entry in reports:
            print(f"\n{'=' * 60}\nenv: {entry['env']}\n{'=' * 60}")
            _print_text(entry["report"])
        print(
            f"\nOverall: {len(reports)} env(s) checked; "
            f"{'failures present' if overall_error else 'all clean'}."
        )
    return 1 if overall_error else 0
