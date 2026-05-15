"""
Lint an environment file's imports against its PEP 723 dependency list.

Scope: catches *direct* imports the env file makes that aren't declared
in PEP 723. Does **not** catch transitive imports (e.g. Boltz importing
``cuequivariance_torch`` unconditionally) — for that, use
``scion check --thorough`` which actually loads the provider and watches
for ImportError at construction time.

Used as a non-fatal install-time warning. Cheap, static, and good for
catching the most common authoring mistake (forgot to add a dep).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from .pep723 import parse_pep723_metadata

# PEP 723 names → import root they expose. Only needed when the import
# name differs from a simple lowercase + ``-→_`` mapping. Add entries as
# real envs need them — speculative entries just rot.
PYPI_TO_IMPORT: dict[str, str] = {
    "fair-esm": "esm",
    "biopython": "Bio",
}


# Imports that come from packages Scion installs into every worker venv
# regardless of the env's own PEP 723 list (the user-facing scion package
# itself, plus numpy which boltz/fair-esm pull in transitively but env
# files commonly import directly).
ALWAYS_AVAILABLE: frozenset[str] = frozenset(
    {"scion", "numpy", "packaging", "tomli", "tomllib"}
)


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_").strip()


def _dep_name(dep_spec: str) -> str:
    """Extract the package name from a PEP 508 dep string."""
    s = dep_spec
    if "[" in s:
        head, _, rest = s.partition("[")
        _, _, tail = rest.partition("]")
        s = head + tail
    # PEP 508 direct URL reference: "pkgname @ url" — strip the URL half.
    if "@" in s:
        s = s.split("@", 1)[0]
    for op in (">=", "<=", "==", "~=", "!=", ">", "<", ";"):
        idx = s.find(op)
        if idx >= 0:
            s = s[:idx]
            break
    return s.strip().lower()


def _stdlib_modules() -> frozenset[str]:
    return frozenset(sys.stdlib_module_names)


def _declared_import_roots(deps: list[str]) -> set[str]:
    declared: set[str] = set()
    for dep in deps:
        name = _dep_name(dep)
        if not name:
            continue
        # Primary mapping: known-different name → its actual import root.
        mapped = PYPI_TO_IMPORT.get(name)
        if mapped:
            declared.add(_normalize(mapped.split(".")[0]))
        declared.add(_normalize(name))
    return declared


def _file_imports(content: str) -> set[str]:
    """All module-root names imported anywhere in the file (incl. inside functions)."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def lint_environment_imports(source_path: Path | str) -> list[str]:
    """
    Return human-readable warnings for imports the env file uses but
    didn't declare in its PEP 723 deps block. Empty list = no issues.
    """
    path = Path(source_path)
    if not path.exists():
        return [f"env file not found: {path}"]

    content = path.read_text()
    metadata = parse_pep723_metadata(content) or {}
    deps = metadata.get("dependencies", []) or []

    declared = _declared_import_roots(deps)
    stdlib = _stdlib_modules()
    imports = _file_imports(content)

    warnings: list[str] = []
    for mod in sorted(imports):
        norm = _normalize(mod)
        if norm in stdlib or norm in declared or norm in ALWAYS_AVAILABLE:
            continue
        if mod in stdlib or mod in ALWAYS_AVAILABLE:
            continue
        warnings.append(
            f"env imports {mod!r} but no matching PEP 723 dependency. "
            f"Add it to the # dependencies = [...] block."
        )
    return warnings
