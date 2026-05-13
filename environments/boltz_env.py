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

Capability: ``fold`` — sequence (or multimer) -> structure, with optional
MSA and template inputs.

Boltz is invoked via its CLI in-process: the provider writes a Boltz
YAML schema (plus any MSA/template blobs) to a tempdir, then spawns
``python -m boltz predict ...`` from within the worker venv where
``boltz`` is installed. The resulting mmCIF and confidence JSON are
read back and returned. This is the maintainer-supported invocation
path, so it tracks upstream Boltz changes more reliably than calling
internal ``boltz.main`` functions.

**Performance note:** the CLI reloads model weights on every call,
so this v0 wiring is appropriate for one-off predictions and small
campaigns. For high-throughput use, a v0.1 follow-up will replace the
subprocess with in-process weight loading and per-call inference.

Reference: https://github.com/jwohlwend/boltz
"""

from __future__ import annotations

CAPABILITIES = ["fold"]

DEFAULT_CHECKPOINT = "boltz2"


def setup(model: str, device: str = "cuda"):
    """
    Load a Boltz fold provider.

    Args:
        model: Checkpoint name (currently informational — Boltz versions
            its weights internally; left here so manifests record what
            users requested). Defaults to ``"boltz2"``.
        device: PyTorch device. ``"cuda"`` / ``"cuda:N"`` map to
            ``--accelerator gpu``; anything starting with ``cpu`` maps to
            ``--accelerator cpu``.

    Returns:
        Provider with a ``fold(sequence, ...)`` method.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # Confirm boltz is importable in this venv before returning the
    # provider — otherwise the first fold() call would surface a
    # confusing subprocess error instead of an ImportError.
    try:
        import boltz  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "boltz is not installed in this environment. "
            "This env file is meant to be built with `scion install boltz_env.py`."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT
    accelerator = "gpu" if str(device).startswith("cuda") else "cpu"

    # Boltz uses ~/.boltz/ for weight caching. Scion has redirected HOME to
    # {root}/home/, so this lands inside the shared cache automatically.
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
            **kwargs,
        ) -> dict:
            """
            Predict a structure.

            Args:
                sequence: Single chain (str) or multimer (list[str]).
                msa: Optional A3M MSA bytes. If None, ``msa: empty`` is
                    written and Boltz does single-sequence prediction
                    (no external MSA server fetch).
                templates: Optional CIF templates bytes (not yet wired).
                num_recycles: Boltz recycling steps. Default 3.

            Returns:
                ``{"mmcif": str, "confidence": dict, "plddt": None}``.
                Per-residue pLDDT extraction from CIF B-factors is a
                v0.1 follow-up.
            """
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                input_yaml = tmp_path / "input.yaml"
                out_dir = tmp_path / "out"
                out_dir.mkdir()

                sequences = sequence if isinstance(sequence, list) else [sequence]

                # Optional MSA: write to a file Boltz can reference. Single
                # blob applies to chain A by convention; per-chain MSAs are a
                # v0.1 extension.
                msa_path = None
                if msa is not None:
                    msa_path = tmp_path / "msa.a3m"
                    msa_path.write_bytes(msa)

                if templates is not None:
                    # Reserved — Boltz template support varies by version
                    # and is left unwired until we verify the schema on
                    # the user's installed version.
                    raise NotImplementedError(
                        "templates argument is reserved for a v0.1 follow-up."
                    )

                yaml_lines = ["sequences:"]
                for i, seq in enumerate(sequences):
                    chain_id = chr(ord("A") + i)
                    yaml_lines.append("  - protein:")
                    yaml_lines.append(f"      id: {chain_id}")
                    yaml_lines.append(f"      sequence: {seq}")
                    if msa_path is not None and i == 0:
                        yaml_lines.append(f"      msa: {msa_path}")
                    else:
                        # Avoid Boltz's automatic MSA-server fetch so the
                        # call is hermetic and reproducible.
                        yaml_lines.append("      msa: empty")
                input_yaml.write_text("\n".join(yaml_lines) + "\n")

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
                        f"--- stdout ---\n{proc.stdout}\n"
                        f"--- stderr ---\n{proc.stderr}"
                    )

                # Boltz output layout varies by version (boltz-1 vs boltz-2);
                # rglob a wide net rather than hardcode subdirectory names.
                cif_candidates = sorted(out_dir.rglob("*_model_*.cif"))
                conf_candidates = sorted(out_dir.rglob("confidence_*.json"))

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

                return {
                    "mmcif": mmcif,
                    "confidence": confidence,
                    "plddt": None,
                }

    return BoltzProvider()
