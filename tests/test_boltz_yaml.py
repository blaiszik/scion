"""
Tests for boltz_env.build_boltz_yaml.

The YAML builder is the part of Boltz wiring most likely to silently
drift if Boltz's schema changes, so we pin its output here. We import
it directly from the env file (no boltz install required).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_boltz_env_module():
    """Load environments/boltz_env.py without installing the boltz package."""
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "boltz_env_under_test", here / "environments" / "boltz_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def boltz_env():
    return _load_boltz_env_module()


def test_monomer_no_msa(boltz_env):
    out = boltz_env.build_boltz_yaml(proteins=["MKTAYIAKQ"])
    assert "version: 1" in out
    assert "- protein:" in out
    assert "id: A" in out
    assert "sequence: MKTAYIAKQ" in out
    assert "msa: empty" in out
    assert "ligand:" not in out
    assert "properties:" not in out


def test_monomer_with_msa(boltz_env, tmp_path):
    msa = tmp_path / "msa.a3m"
    msa.write_text(">q\nMKTA\n")
    out = boltz_env.build_boltz_yaml(proteins=["MKTAYI"], msa_path=str(msa))
    assert f"msa: {msa}" in out
    assert out.count("msa:") == 1  # only the one protein chain


def test_multimer_two_chains(boltz_env):
    out = boltz_env.build_boltz_yaml(proteins=["MKT", "GSH"])
    assert out.count("- protein:") == 2
    assert "id: A" in out
    assert "id: B" in out
    assert "sequence: MKT" in out
    assert "sequence: GSH" in out


def test_protein_ligand_cofold_smiles(boltz_env):
    out = boltz_env.build_boltz_yaml(
        proteins=["MKT"],
        ligands=[{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}],
    )
    assert "- ligand:" in out
    assert "id: B" in out  # protein took A, ligand gets B
    assert 'smiles: "CC(=O)Oc1ccccc1C(=O)O"' in out


def test_protein_ligand_cofold_ccd(boltz_env):
    out = boltz_env.build_boltz_yaml(
        proteins=["MKT"], ligands=[{"ccd": "ATP"}]
    )
    assert "- ligand:" in out
    assert "ccd: ATP" in out
    assert "smiles:" not in out


def test_affinity_requires_binder_or_ligand(boltz_env):
    with pytest.raises(ValueError, match="affinity_binder"):
        boltz_env.build_boltz_yaml(proteins=["MKT"], predict_affinity=True)


def test_affinity_defaults_to_first_ligand(boltz_env):
    out = boltz_env.build_boltz_yaml(
        proteins=["MKT"],
        ligands=[{"smiles": "CCO"}, {"ccd": "ATP"}],
        predict_affinity=True,
    )
    assert "properties:" in out
    assert "- affinity:" in out
    # First ligand was assigned id 'B' (protein took 'A')
    assert "binder: B" in out


def test_affinity_explicit_binder(boltz_env):
    out = boltz_env.build_boltz_yaml(
        proteins=["MKT"],
        ligands=[{"id": "LIG", "smiles": "CCO"}],
        predict_affinity=True,
        affinity_binder="LIG",
    )
    assert "binder: LIG" in out


def test_explicit_ids_avoided_in_auto_assignment(boltz_env):
    out = boltz_env.build_boltz_yaml(
        proteins=[{"id": "A", "sequence": "MKT"}, "GSH"],
    )
    # Second chain shouldn't take A (already used); should take B
    assert "id: A" in out
    assert "id: B" in out


def test_dna_rna_chains(boltz_env):
    out = boltz_env.build_boltz_yaml(
        proteins=["MKT"],
        nucleic_acids=[
            {"type": "dna", "sequence": "ATGC"},
            {"type": "rna", "sequence": "AUGC"},
        ],
    )
    assert "- dna:" in out
    assert "sequence: ATGC" in out
    assert "- rna:" in out
    assert "sequence: AUGC" in out


def test_invalid_nucleic_type_raises(boltz_env):
    with pytest.raises(ValueError, match="dna.*rna"):
        boltz_env.build_boltz_yaml(
            proteins=["M"], nucleic_acids=[{"type": "xxx", "sequence": "A"}]
        )


def test_ligand_missing_smiles_and_ccd_raises(boltz_env):
    with pytest.raises(ValueError, match="smiles.*ccd"):
        boltz_env.build_boltz_yaml(proteins=["M"], ligands=[{"id": "L"}])


def test_full_drug_discovery_complex(boltz_env):
    """Receptor + ligand + cofactor + DNA + affinity — exercises the lot."""
    out = boltz_env.build_boltz_yaml(
        proteins=[
            {"id": "REC", "sequence": "MKTAYIAKQ"},
            {"id": "HLP", "sequence": "GSHM"},
        ],
        ligands=[
            {"id": "DRG", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"id": "COF", "ccd": "ATP"},
        ],
        nucleic_acids=[{"id": "OPR", "type": "dna", "sequence": "ATGCATGC"}],
        predict_affinity=True,
        affinity_binder="DRG",
    )
    for tok in [
        "id: REC", "id: HLP", "id: DRG", "id: COF", "id: OPR",
        'smiles: "CC(=O)Oc1ccccc1C(=O)O"', "ccd: ATP",
        "- dna:", "sequence: ATGCATGC",
        "properties:", "- affinity:", "binder: DRG",
    ]:
        assert tok in out, f"missing: {tok!r}\nYAML:\n{out}"
