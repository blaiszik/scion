"""
PEP 723 inline script metadata parser.

PEP 723 defines a standard format for embedding metadata in Python scripts
using a TOML block in comments:

    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["torch", "boltz"]
    # ///

Reference: https://peps.python.org/pep-0723/
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


PEP723_PATTERN = re.compile(
    r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///$",
    re.MULTILINE,
)


def parse_pep723_metadata(content: str) -> dict | None:
    """Extract PEP 723 metadata from script content."""
    match = PEP723_PATTERN.search(content)
    if match is None:
        return None

    toml_lines = []
    for line in match.group(1).splitlines():
        if line.startswith("# "):
            toml_lines.append(line[2:])
        elif line == "#":
            toml_lines.append("")
        else:
            toml_lines.append(line.lstrip("# "))

    try:
        return tomllib.loads("\n".join(toml_lines))
    except tomllib.TOMLDecodeError:
        return None


def validate_environment_file(path: Path | str) -> tuple[bool, str]:
    """
    Validate that a file is a valid Scion environment file.

    A valid environment file must:
    1. Exist and be readable
    2. Have valid PEP 723 metadata with dependencies
    3. Define a setup() function at module level
    4. Declare a module-level CAPABILITIES list
    """
    path = Path(path)

    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"

    try:
        content = path.read_text()
    except Exception as e:
        return False, f"Cannot read file: {e}"

    metadata = parse_pep723_metadata(content)
    if metadata is None:
        return False, "No valid PEP 723 metadata block found"
    if "dependencies" not in metadata:
        return False, "PEP 723 metadata missing 'dependencies' field"

    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as e:
        return False, f"Syntax error in file: {e}"

    setup_found = False
    capabilities_found = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            setup_found = True
            args = node.args
            if len(args.args) == 0 and len(args.posonlyargs) == 0:
                return False, "setup() function must accept at least 'model' parameter"
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CAPABILITIES":
                    capabilities_found = True

    if not setup_found:
        return False, "No setup() function found at module level"
    if not capabilities_found:
        return False, "No CAPABILITIES list found at module level"

    return True, "OK"


def get_dependencies(path: Path | str) -> list[str]:
    """Get the dependencies list from an environment file."""
    path = Path(path)
    try:
        content = path.read_text()
        metadata = parse_pep723_metadata(content)
        if metadata and "dependencies" in metadata:
            return metadata["dependencies"]
    except Exception:
        pass
    return []


def get_requires_python(path: Path | str) -> str | None:
    """Get the requires-python specifier from an environment file."""
    path = Path(path)
    try:
        content = path.read_text()
        metadata = parse_pep723_metadata(content)
        if metadata:
            return metadata.get("requires-python")
    except Exception:
        pass
    return None


def get_capabilities(path: Path | str) -> list[str]:
    """
    Parse the module-level CAPABILITIES list from an environment file using AST.

    Returns an empty list if not found or not a list of string literals.
    """
    path = Path(path)
    try:
        content = path.read_text()
    except Exception:
        return []

    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError:
        return []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CAPABILITIES":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        result = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                result.append(elt.value)
                        return result
    return []
