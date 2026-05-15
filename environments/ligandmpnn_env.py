# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Same torch cap as the other shipped envs — keeps PyPI wheels
#     # CUDA-12.8-compatible on Polaris-shape clusters. Upstream pins
#     # torch==2.2.1 but the model itself isn't version-sensitive.
#     "torch>=2.6,<2.9",
#     # LigandMPNN's vendored openfold and ProDy break on numpy 2.x.
#     "numpy>=1.23,<2",
#     "biopython>=1.79",     # we use Bio.PDB to convert mmCIF -> PDB
#     "prody>=2.4",           # LigandMPNN's structure parser
#     "scipy>=1.10",
#     "networkx>=3.0",
#     "ml-collections>=0.1.1",
#     "dm-tree>=0.1.8",
# ]
# ///
"""
LigandMPNN environment for Scion.

Capability: ``design_sequence`` — ligand-aware inverse folding. Given
a structure (mmCIF) and a set of positions to redesign, proposes new
amino acids that fit the local environment, conditioned on any bound
ligand poses present in the input.

The upstream repository (https://github.com/dauparas/LigandMPNN) is
research code without a ``pyproject.toml``, so the env follows the
boltz_env pattern: declare the actual library deps in PEP 723, clone
the repo into ``{root}/home/ligandmpnn-install/src`` on first ``setup()``,
download the requested checkpoint, then invoke ``run.py`` as a
subprocess for each ``design_sequence`` call.

Sharing model across users: with HOME redirected to ``{root}/home``
(Scion default), the maintainer's first install populates the shared
``ligandmpnn-install/`` and subsequent users reuse it.

Reference: https://github.com/dauparas/LigandMPNN
"""

from __future__ import annotations

CAPABILITIES = ["design_sequence"]

DEFAULT_CHECKPOINT = "ligandmpnn_v_32_010_25"

LIGANDMPNN_GIT_URL = "https://github.com/dauparas/LigandMPNN.git"
LIGANDMPNN_WEIGHT_BASE = "https://files.ipd.uw.edu/pub/ligandmpnn"


def _ensure_ligandmpnn_install(checkpoint: str):
    """
    Clone the LigandMPNN repo and fetch the requested checkpoint if
    they aren't already present under ``$HOME/ligandmpnn-install``.

    Returns ``(repo_dir, weight_file)`` as ``pathlib.Path`` objects.
    """
    import subprocess
    import urllib.request
    from pathlib import Path

    install_root = Path.home() / "ligandmpnn-install"
    repo_dir = install_root / "src"
    params_dir = install_root / "model_params"
    weight_file = params_dir / f"{checkpoint}.pt"

    if not (repo_dir / "run.py").exists():
        install_root.mkdir(parents=True, exist_ok=True)
        # Idempotent: if repo_dir exists but is broken, blow it away.
        if repo_dir.exists():
            import shutil
            shutil.rmtree(repo_dir)
        subprocess.run(
            ["git", "clone", "--depth=1", LIGANDMPNN_GIT_URL, str(repo_dir)],
            check=True,
        )

    if not weight_file.exists():
        params_dir.mkdir(parents=True, exist_ok=True)
        url = f"{LIGANDMPNN_WEIGHT_BASE}/{checkpoint}.pt"
        # urllib has reasonable HTTPS defaults and no extra deps. Use a
        # temp filename + rename so a partial download doesn't get cached.
        tmp = weight_file.with_suffix(".pt.tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(weight_file)

    return repo_dir, weight_file


def _mmcif_to_pdb(mmcif_text: str, pdb_path):
    """Convert an mmCIF string to a PDB file via Biopython."""
    from io import StringIO

    from Bio.PDB import PDBIO, MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("x", StringIO(mmcif_text))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def setup(model: str, device: str = "cuda"):
    """
    Load a LigandMPNN design provider.

    Args:
        model: Checkpoint name (e.g. ``ligandmpnn_v_32_010_25``,
            ``proteinmpnn_v_48_020``). Lower noise level (``_010``) =
            more conservative designs.
        device: PyTorch device. ``"cuda"`` or ``"cpu"``. LigandMPNN
            picks this up via its own torch defaults; we don't need
            to pass an explicit ``--device`` flag.

    Returns:
        Provider with a ``design_sequence(...)`` method.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "torch not installed in this environment. "
            "Build with `scion install ligandmpnn_env.py`."
        ) from e

    checkpoint = model or DEFAULT_CHECKPOINT
    repo_dir, weight_file = _ensure_ligandmpnn_install(checkpoint)
    is_ligand_model = checkpoint.startswith("ligandmpnn")

    class LigandMPNNProvider:
        def __init__(self):
            self.checkpoint = checkpoint
            self.device = device
            self.repo_dir = repo_dir
            self.weight_file = weight_file
            self.is_ligand_model = is_ligand_model

        def preload(self) -> None:
            """
            Validate that LigandMPNN's imports work in this env.

            We don't run a dummy design here — the install-time clone +
            weight download already validated the heavy parts of the
            wire-up; running a real ``run.py`` call needs a PDB file,
            which a no-op preload can't supply.
            """
            # Import Bio.PDB to verify biopython landed; ProDy is what
            # LigandMPNN itself uses and is a good canary for the
            # broader dep tree.
            import Bio.PDB  # noqa: F401
            import prody  # noqa: F401

        def design_sequence(
            self,
            mmcif: str,
            positions: list[int] | None = None,
            ligands: list[dict] | None = None,  # noqa: ARG002  (reserved)
            num_sequences: int = 1,
            temperature: float = 0.1,
            chain_id: str = "A",
            **kwargs,
        ) -> dict:
            """
            Redesign residues of a structure.

            ``positions`` are 1-indexed residue numbers on chain
            ``chain_id`` (default ``"A"``). The input mmCIF is
            converted to PDB internally (LigandMPNN takes PDB), so any
            ligand HETATM records present in the mmCIF are preserved
            and the ``ligand_mpnn`` model conditions on them
            automatically. The ``ligands=`` kwarg is reserved for
            future symmetry with ``Folder.fold``; ignored today
            because the ligand pose is already in the input structure.
            """
            import os
            import subprocess
            import sys
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                pdb_path = tmpdir / "input.pdb"
                out_folder = tmpdir / "out"
                out_folder.mkdir()

                _mmcif_to_pdb(mmcif, pdb_path)

                redesigned = ""
                if positions:
                    redesigned = " ".join(f"{chain_id}{int(p)}" for p in positions)

                model_type = "ligand_mpnn" if self.is_ligand_model else "protein_mpnn"
                ckpt_flag = (
                    "--checkpoint_ligand_mpnn"
                    if self.is_ligand_model
                    else "--checkpoint_protein_mpnn"
                )

                cmd = [
                    sys.executable, str(self.repo_dir / "run.py"),
                    "--model_type", model_type,
                    ckpt_flag, str(self.weight_file),
                    "--pdb_path", str(pdb_path),
                    "--out_folder", str(out_folder),
                    "--number_of_batches", "1",
                    "--batch_size", str(int(num_sequences)),
                    "--temperature", str(float(temperature)),
                    "--verbose", "0",
                ]
                if redesigned:
                    cmd.extend(["--redesigned_residues", redesigned])

                # Pass extra LigandMPNN flags through unchanged. Caller
                # can opt into bias_AA, omit_AA, symmetry_residues etc.
                # by setting them as kwargs whose name matches the CLI.
                for k, v in kwargs.items():
                    cmd.extend([f"--{k}", str(v)])

                env = os.environ.copy()
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"LigandMPNN run.py failed (exit {proc.returncode}).\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"--- stdout ---\n{proc.stdout}\n"
                        f"--- stderr ---\n{proc.stderr}"
                    )

                # LigandMPNN writes <out_folder>/seqs/<pdb_stem>.fa with
                # one FASTA record per sample. The header lines carry
                # per-sample metadata; we parse `overall_confidence=`
                # as the score.
                seqs_dir = out_folder / "seqs"
                fasta_files = list(seqs_dir.glob("*.fa")) if seqs_dir.exists() else []
                if not fasta_files:
                    contents = sorted(p.relative_to(out_folder) for p in out_folder.rglob("*"))
                    raise RuntimeError(
                        f"No FASTA outputs under {out_folder}. Tree:\n  "
                        + "\n  ".join(str(p) for p in contents)
                    )

                sequences: list[str] = []
                scores: list[float] = []
                for fasta in fasta_files:
                    pending_score: float | None = None
                    for line in fasta.read_text().splitlines():
                        if line.startswith(">"):
                            pending_score = None
                            for tok in line.split(","):
                                tok = tok.strip()
                                if tok.startswith("overall_confidence="):
                                    try:
                                        pending_score = float(tok.split("=", 1)[1])
                                    except ValueError:
                                        pending_score = None
                        elif line.strip():
                            sequences.append(line.strip())
                            scores.append(pending_score if pending_score is not None else 0.0)

                # LigandMPNN's first record is usually the input
                # sequence itself; drop it if we have alternates.
                if len(sequences) > 1:
                    sequences = sequences[1:]
                    scores = scores[1:]

                if not sequences:
                    raise RuntimeError(
                        f"FASTA files were empty: {[str(f) for f in fasta_files]}"
                    )

                return {
                    "sequence": sequences[0],
                    "sequences": sequences,
                    "scores": scores,
                    "masked_positions": list(positions or []),
                }

    return LigandMPNNProvider()
