"""
Scion: protein foundation models with isolated Python environments.

Provides capability clients (Folder, Embedder, ...) that broker calls
into pre-built virtual environments on HPC clusters via a small RPC
over a Unix domain socket.
"""

__version__ = "0.1.0"

from .capabilities import (
    CAPABILITIES,
    DesignResult,
    DockResult,
    EmbedResult,
    FoldResult,
)
from .client import ScionClient
from .cluster_config import (
    ClusterConfig,
    get_cluster_env,
    is_in_batch_job,
    load_cluster_config,
)
from .clusters import (
    CLUSTER_REGISTRY,
    KNOWN_ENVIRONMENTS,
    get_cluster_for_root,
    get_profile_for_root,
    get_profile_for_root_or_host,
    get_root_for_cluster,
)
from .config import UserConfig, load_config, save_config
from .designer import Designer
from .docker import Docker
from .embedder import Embedder
from .environment import (
    EnvironmentManager,
    get_model_cache_env,
    list_built_environments,
    list_environments,
)
from .errors import (
    ScionWorkerError,
    WorkerMethodError,
    WorkerProcessDied,
    WorkerSetupFailed,
)
from .folder import Folder
from .manifest import Manifest, load_manifest, save_manifest
from .pep723 import (
    get_capabilities,
    parse_pep723_metadata,
    validate_environment_file,
)
from .server import ScionServer
from .session import ScionSession
from .worker import run_worker

__all__ = [
    "__version__",
    "Folder",
    "Embedder",
    "Designer",
    "Docker",
    "FoldResult",
    "EmbedResult",
    "DesignResult",
    "DockResult",
    "CAPABILITIES",
    "ScionWorkerError",
    "WorkerSetupFailed",
    "WorkerProcessDied",
    "WorkerMethodError",
    "ScionSession",
    "ScionServer",
    "ScionClient",
    "ClusterConfig",
    "load_cluster_config",
    "get_cluster_env",
    "is_in_batch_job",
    "EnvironmentManager",
    "list_environments",
    "list_built_environments",
    "get_model_cache_env",
    "parse_pep723_metadata",
    "validate_environment_file",
    "get_capabilities",
    "run_worker",
    "CLUSTER_REGISTRY",
    "KNOWN_ENVIRONMENTS",
    "get_root_for_cluster",
    "get_cluster_for_root",
    "get_profile_for_root",
    "get_profile_for_root_or_host",
    "UserConfig",
    "load_config",
    "save_config",
    "Manifest",
    "load_manifest",
    "save_manifest",
]
