"""Ensure the Nim-backed MSA featurization is identical to the pure-Python path."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from alphafold3_pytorch.data import data_pipeline, msa_parsing

# a fairly complicated input: multi-line sequences, lowercase insertions,
# unknown residues, duplicate sequences, and various description formats
A3M_MIXED = """\
>101|tr|A0A146SKV9|A0A146SKV9_FUNHE
ttACDEFGHIKACx
>tr|A0A146SKV9|A0A146SKV9_FUNHE
ACDEFGHI
KacAC
>101|tr|A0A146SKV9|A0A146SKV9_FUNHE
ACDEXGtHIKAC
>sp|Q8JTF3|Q8JTF3_YEAST
ACDEFGHIKAC
>sp|Q8JTF3|Q8JTF3_YEAST
ACDEFGHIKAC
"""

# scenario: (chain_id -> a3m string, chain_id_to_residue, msa_type)
SCENARIOS = {
    "mixed_chemtypes_with_atomized_residue": (
        {"A": A3M_MIXED},
        {
            "A": {
                # 8 peptide residues, an atomized modified peptide residue
                # (duplicated residue index), one rna, one dna, and two ligands
                "chemtype": [0] * 8 + [0, 0] + [1, 2] + [3, 3],
                "residue_index": [1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13, 14],
            }
        },
        "protein",
    ),
    "crlf_dna_msa_with_leading_ligand": (
        {
            "A": (
                ">101|tr|A0A146SKV9|A0A146SKV9_FUNHE\r\n"
                "aACG\r\n"
                ">102\r\n"
                "ACgG\r\n"
            )
        },
        {"A": {"chemtype": [3, 2, 2, 2], "residue_index": [1, 2, 3, 4]}},
        "dna",
    ),
    "all_ligand_chain": (
        {"A": ">101\n>102\n"},
        {"A": {"chemtype": [3, 3, 3], "residue_index": [1, 2, 3]}},
        "protein",
    ),
    "ligand_in_middle_with_atomized_pairs": (
        {
            "A": (
                ">101\n"
                "ACDcEFG\n"
                ">102\n"
                "ACDEFgG\n"
            )
        },
        {
            "A": {
                "chemtype": [0, 0, 0, 3, 0, 0, 0, 0],
                "residue_index": [1, 2, 3, 4, 5, 5, 6, 7],
            }
        },
        "protein",
    ),
    "single_residue_with_leading_inserts": (
        {"A": ">101\nccccA\n"},
        {"A": {"chemtype": [0], "residue_index": [1]}},
        "protein",
    ),
    "pure_rna_chain": (
        {"A": ">101\naACGU\n"},
        {"A": {"chemtype": [1, 1, 1, 1], "residue_index": [1, 2, 3, 4]}},
        "rna",
    ),
    "shared_entity_across_chains": (
        {
            "A": ">101|sp|Q8JTF3|Q8JTF3_YEAST\nACD\n",
            "B": ">101|sp|Q8JTF3|Q8JTF3_YEAST\nACD\n",
        },
        {
            "A": {"chemtype": [0, 0, 0], "residue_index": [1, 2, 3]},
            "B": {"chemtype": [0, 0, 3, 0], "residue_index": [1, 2, 3, 4]},
        },
        "protein",
    ),
}


@pytest.fixture(scope = "module")
def nim_modules():
    pytest.importorskip("nimporter_plus")

    try:
        os.environ.setdefault("NIMPORTER_COMPILER_ARGS", "-d:release")
        from alphafold3_pytorch.nim import a3m_parse, msa_features

        return a3m_parse, msa_features
    except Exception:
        pytest.skip("Nim compilation is not available")


def _featurize(a3m_strings, chain_id_to_residue, msa_type):
    msas = {
        chain_id: msa_parsing.parse_a3m(a3m_string, msa_type = msa_type)
        for chain_id, a3m_string in a3m_strings.items()
    }
    return msas, data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)


@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_nim_matches_python_msa_featurization(nim_modules, scenario):
    """The Nim path must produce features identical to the Python path."""
    nim_a3m_parse, nim_msa_features = nim_modules
    a3m_strings, chain_id_to_residue, msa_type = SCENARIOS[scenario]

    py_msas, py_chains = _featurize(a3m_strings, chain_id_to_residue, msa_type)

    with patch.object(data_pipeline, "import_nim_msa_features", return_value = nim_msa_features), patch.object(
        msa_parsing, "import_nim_a3m_parse", return_value = nim_a3m_parse
    ):
        nim_msas, nim_chains = _featurize(a3m_strings, chain_id_to_residue, msa_type)

    for chain_id, py_msa in py_msas.items():
        nim_msa = nim_msas[chain_id]
        assert nim_msa.sequences == py_msa.sequences
        assert nim_msa.deletion_matrix == py_msa.deletion_matrix
        assert nim_msa.descriptions == py_msa.descriptions

    for py_chain, nim_chain in zip(py_chains, nim_chains):
        for key, value in py_chain.items():
            assert np.array_equal(nim_chain[key], value), key
