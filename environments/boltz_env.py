# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boltz>=2.0",
#     "torch>=2.2,<2.10",
#     "numpy>=1.24",
# ]
# ///
"""
Boltz-2 environment for Scion.

Capability: ``fold`` — sequence (or multi-chain complex) -> structure,
with optional ligands (SMILES / CCD), nucleic acids (DNA / RNA),
binding-affinity prediction, MSAs, and templates.

Boltz is invoked via its CLI in-process: the provider writes a Boltz
YAML schema to a tempdir, then spawns ``python -m boltz predict ...``
from within the worker venv. The resulting mmCIF, confidence JSON,
and (if requested) affinity JSON are read back and returned.

This wiring exposes the *full* Boltz-2 capability surface most useful
for drug discovery: protein-ligand co-folding, multimer folding,
nucleic-acid handling, and the affinity head. Per-call weight reload
is a v0 tradeoff; replacing with in-process inference is on the roadmap.

YAML schema construction is factored into ``build_boltz_yaml`` (pure
function, no boltz import required) so it can be unit-tested without
needing Boltz installed in the test environment.

Reference: https://github.com/jwohlwend/boltz
"""

from __future__ import annotations

CAPABILITIES = ["fold"]

DEFAULT_CHECKPOINT = "boltz2"


# ---------------------------------------------------------------------------
# YAML construction (pure — kept at module scope so tests can import it
# without boltz being installed).
# ---------------------------------------------------------------------------

def _next_id(used: set[str]) -> str:
    """Allocate the next free single-letter chain ID."""
    for code in range(ord("A"), ord("Z") + 1):
        c = chr(code)
        if c not in used:
            used.add(c)
            return c
    raise ValueError("Ran out of single-letter chain IDs (>26 chains)")


def build_boltz_yaml(
    proteins: list[str | dict],
    ligands: list[dict] | None = None,
    nucleic_acids: list[dict] | None = None,
    msa_path: str | None = None,
    predict_affinity: bool = False,
    affinity_binder: str | None = None,
) -> str:
    """
    Build a Boltz-2 input YAML string from typed inputs.

    Args:
        proteins: List of protein chains. Each item is either a raw
            sequence string or ``{"id": "A", "sequence": "..."}``.
        ligands: List of ``{"smiles": "..."}`` or ``{"ccd": "ATP"}``,
            optionally with ``"id"``.
        nucleic_acids: List of ``{"type": "dna" | "rna", "sequence": "..."}``,
            optionally with ``"id"``.
        msa_path: Path to an A3M MSA file. Attached to the first protein
            chain. ``None`` -> ``msa: empty`` (single-sequence prediction).
        predict_affinity: When True, emit a ``properties.affinity`` block.
            Boltz-2 will run its affinity head and write affinity_*.json.
        affinity_binder: Chain ID of the binder. Defaults to the first
            ligand's ID when ligands are present.

    Returns:
        YAML string ready to write to disk.
    """
    used: set[str] = set()

    # Collect user-specified IDs first so auto-assignment doesn't collide.
    for entry in proteins or []:
        if isinstance(entry, dict) and entry.get("id"):
            used.add(entry["id"])
    for entry in (ligands or []) + (nucleic_acids or []):
        if isinstance(entry, dict) and entry.get("id"):
            used.add(entry["id"])

    lines = ["version: 1", "sequences:"]

    # Proteins.
    first_protein = True
    for entry in proteins or []:
        if isinstance(entry, str):
            chain_id = _next_id(used)
            seq = entry
        else:
            chain_id = entry.get("id") or _next_id(used)
            seq = entry["sequence"]
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        if first_protein and msa_path is not None:
            lines.append(f"      msa: {msa_path}")
        else:
            # Hermetic: don't fall through to Boltz's MSA-server fetch.
            lines.append("      msa: empty")
        first_protein = False

    # Ligands.
    first_ligand_id: str | None = None
    for entry in ligands or []:
        chain_id = entry.get("id") or _next_id(used)
        if first_ligand_id is None:
            first_ligand_id = chain_id
        lines.append("  - ligand:")
        lines.append(f"      id: {chain_id}")
        if "smiles" in entry:
            # Quote SMILES to avoid YAML parsing surprises with chars
            # like '[', ']', '#', '@' that appear in valid SMILES.
            smiles = str(entry["smiles"]).replace('"', '\\"')
            lines.append(f'      smiles: "{smiles}"')
        elif "ccd" in entry:
            lines.append(f"      ccd: {entry['ccd']}")
        else:
            raise ValueError(
                f"Ligand entry must have 'smiles' or 'ccd' key: {entry!r}"
            )

    # Nucleic acids.
    for entry in nucleic_acids or []:
        chain_id = entry.get("id") or _next_id(used)
        kind = entry.get("type", "").lower()
        if kind not in ("dna", "rna"):
            raise ValueError(
                f"Nucleic-acid entry must have type='dna' or 'rna': {entry!r}"
            )
        lines.append(f"  - {kind}:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {entry['sequence']}")

    # Affinity head.
    if predict_affinity:
        binder = affinity_binder or first_ligand_id
        if binder is None:
            raise ValueError(
                "predict_affinity=True requires either a ligand or an "
                "explicit affinity_binder chain ID."
            )
        lines.append("properties:")
        lines.append("  - affinity:")
        lines.append(f"      binder: {binder}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Provider (calls into Boltz; imported only inside setup()).
# ---------------------------------------------------------------------------

def setup(model: str, device: str = "cuda"):
    """
    Load a Boltz fold provider.

    Args:
        model: Checkpoint name (informational — Boltz versions weights
            internally). Defaults to ``"boltz2"``.
        device: PyTorch device. ``"cuda"`` / ``"cuda:N"`` map to
            ``--accelerator gpu``; anything else maps to ``--accelerator cpu``.

    Returns:
        Provider with a ``fold(sequence, ...)`` method.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    try:
        import boltz  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "boltz is not installed in this environment. "
            "This env file is meant to be built with `scion install boltz_env.py`."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT
    accelerator = "gpu" if str(device).startswith("cuda") else "cpu"

    boltz_cache = Path.home() / ".boltz"
    boltz_cache.mkdir(parents=True, exist_ok=True)

    class BoltzProvider:
        def __init__(self):
            self.checkpoint = checkpoint
            self.device = device
            self.accelerator = accelerator
            self.cache = boltz_cache

        def fold(
            self,
            sequence,
            msa: bytes | None = None,
            templates: bytes | None = None,
            num_recycles: int = 3,
            ligands: list[dict] | None = None,
            nucleic_acids: list[dict] | None = None,
            predict_affinity: bool = False,
            affinity_binder: str | None = None,
            **kwargs,
        ) -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                input_yaml = tmp_path / "input.yaml"
                out_dir = tmp_path / "out"
                out_dir.mkdir()

                # Normalize proteins.
                proteins = sequence if isinstance(sequence, list) else [sequence]

                # Optional MSA: write to a file Boltz can reference.
                msa_path = None
                if msa is not None:
                    msa_path = str(tmp_path / "msa.a3m")
                    (tmp_path / "msa.a3m").write_bytes(msa)

                if templates is not None:
                    raise NotImplementedError(
                        "templates argument is reserved for a v0.1 follow-up."
                    )

                yaml_text = build_boltz_yaml(
                    proteins=proteins,
                    ligands=ligands,
                    nucleic_acids=nucleic_acids,
                    msa_path=msa_path,
                    predict_affinity=predict_affinity,
                    affinity_binder=affinity_binder,
                )
                input_yaml.write_text(yaml_text)

                cmd = [
                    sys.executable, "-m", "boltz", "predict",
                    str(input_yaml),
                    "--out_dir", str(out_dir),
                    "--cache", str(self.cache),
                    "--accelerator", self.accelerator,
                    "--devices", "1",
                    "--recycling_steps", str(int(num_recycles)),
                ]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"boltz predict failed (exit {proc.returncode}).\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"YAML:\n{yaml_text}\n"
                        f"--- stdout ---\n{proc.stdout}\n"
                        f"--- stderr ---\n{proc.stderr}"
                    )

                # Output layout varies by Boltz version; rglob a wide net.
                cif_candidates = sorted(out_dir.rglob("*_model_*.cif"))
                conf_candidates = sorted(out_dir.rglob("confidence_*.json"))
                affinity_candidates = sorted(out_dir.rglob("affinity_*.json"))

                if not cif_candidates:
                    contents = sorted(p.relative_to(out_dir) for p in out_dir.rglob("*"))
                    raise RuntimeError(
                        f"No mmCIF found under {out_dir}. Output tree:\n  "
                        + "\n  ".join(str(p) for p in contents)
                    )

                mmcif = cif_candidates[0].read_text()
                confidence: dict = {}
                if conf_candidates:
                    try:
                        confidence = json.loads(conf_candidates[0].read_text())
                    except json.JSONDecodeError:
                        confidence = {"_parse_error": conf_candidates[0].name}

                affinity = None
                if predict_affinity and affinity_candidates:
                    try:
                        affinity = json.loads(affinity_candidates[0].read_text())
                    except json.JSONDecodeError:
                        affinity = {"_parse_error": affinity_candidates[0].name}

                return {
                    "mmcif": mmcif,
                    "confidence": confidence,
                    "affinity": affinity,
                    "plddt": None,
                }

    return BoltzProvider()
