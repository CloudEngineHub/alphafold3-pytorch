"""Ensure the Nim-backed MSA featurization is identical to the pure-Python path."""

import os
from contextlib import contextmanager
from pathlib import Path
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
        from alphafold3_pytorch.nim import a3m_parse, msa_features, msa_stats, species_id

        return a3m_parse, msa_features, msa_stats, species_id
    except Exception:
        pytest.skip("Nim compilation is not available")


@contextmanager
def _nim_path_patches(nim_a3m, nim_msa_features, nim_msa_stats, nim_species_id):
    with patch.object(msa_parsing, "import_nim_a3m_parse", return_value = nim_a3m), patch.object(
        data_pipeline, "import_nim_msa_features", return_value = nim_msa_features
    ), patch.object(data_pipeline, "import_nim_msa_stats", return_value = nim_msa_stats), patch.object(
        data_pipeline, "import_nim_species_id", return_value = nim_species_id
    ):
        yield


def _featurize(a3m_strings, chain_id_to_residue, msa_type, nim_modules = None):
    """Featurize with the nim modules patched in, or the pure-Python path when
    `nim_modules` is `None`."""
    nim_a3m_parse, nim_msa_features, nim_msa_stats, nim_species_id = (
        nim_modules or (None, None, None, None)
    )

    msas = {
        chain_id: msa_parsing.parse_a3m(a3m_string, msa_type = msa_type)
        for chain_id, a3m_string in a3m_strings.items()
    }

    with _nim_path_patches(nim_a3m_parse, nim_msa_features, nim_msa_stats, nim_species_id):
        chains = data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    return msas, chains


@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_nim_matches_python_msa_featurization(nim_modules, scenario):
    """The Nim path must produce features identical to the Python path."""
    a3m_strings, chain_id_to_residue, msa_type = SCENARIOS[scenario]

    py_msas, py_chains = _featurize(a3m_strings, chain_id_to_residue, msa_type)
    nim_msas, nim_chains = _featurize(a3m_strings, chain_id_to_residue, msa_type, nim_modules)

    for chain_id, py_msa in py_msas.items():
        nim_msa = nim_msas[chain_id]
        assert nim_msa.sequences == py_msa.sequences
        assert nim_msa.deletion_matrix == py_msa.deletion_matrix
        assert nim_msa.descriptions == py_msa.descriptions

    for py_chain, nim_chain in zip(py_chains, nim_chains):
        for key, value in py_chain.items():
            assert np.array_equal(nim_chain[key], value), key


def _assert_nim_parse_equals_python(nim_a3m_parse, a3m_string, msa_type = "protein"):
    py_msa = msa_parsing.parse_a3m(a3m_string, msa_type = msa_type)
    with patch.object(msa_parsing, "import_nim_a3m_parse", return_value = nim_a3m_parse):
        nim_msa = msa_parsing.parse_a3m(a3m_string, msa_type = msa_type)
    assert nim_msa.sequences == py_msa.sequences
    assert nim_msa.deletion_matrix == py_msa.deletion_matrix
    assert nim_msa.descriptions == py_msa.descriptions


def test_nim_parse_exotic_line_separators(nim_modules):
    """Python `str.splitlines` boundaries beyond `\n`/`\r\n` must behave identically."""
    nim_a3m_parse, _, _, _ = nim_modules

    for sep in ("\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\u0085", "\u2028", "\u2029"):
        a3m = f">h1{sep}ACD{sep}EFG{sep}>h2{sep}ACDEFGH{sep}"
        _assert_nim_parse_equals_python(nim_a3m_parse, a3m)

    # consecutive separators and trailing separator produce empty lines
    a3m = ">h1\n\nACD\n\n>h2\n"
    _assert_nim_parse_equals_python(nim_a3m_parse, a3m)

    # mixed crlf and lf
    a3m = ">h1\r\nACD\nEFG\rACDE\r\nX"
    _assert_nim_parse_equals_python(nim_a3m_parse, a3m)


def test_nim_parse_unicode_lowercase(nim_modules):
    """Non-ASCII lowercase counts as a deletion but stays in the sequence (python quirk)."""
    nim_a3m_parse, _, _, _ = nim_modules

    _assert_nim_parse_equals_python(nim_a3m_parse, ">h1\nAβCdEFβ\n>h2\nACβDE\n")
    _assert_nim_parse_equals_python(nim_a3m_parse, ">h1\nβαAβX\n")
    _assert_nim_parse_equals_python(nim_a3m_parse, ">h1\nACβDEF\n>h2\nβββACDEF\n")


def test_nim_parse_unicode_whitespace(nim_modules):
    """Unicode whitespace is stripped from lines like python `str.strip`."""
    nim_a3m_parse, _, _, _ = nim_modules

    a3m = "\u00a0>h1\u00a0\n\u2003ACD\u2003\n>h2\nACDE"
    _assert_nim_parse_equals_python(nim_a3m_parse, a3m)

    a3m = "\u2028>h1\u2028\nACDE\n"
    _assert_nim_parse_equals_python(nim_a3m_parse, a3m)


def test_nim_parse_errors_match_python(nim_modules):
    """Malformed inputs must raise in both paths (not silently mis-featurize)."""
    nim_a3m_parse, _, _, _ = nim_modules

    with pytest.raises(Exception):
        msa_parsing.parse_a3m("ACDEF\n", msa_type = "protein")
    with patch.object(msa_parsing, "import_nim_a3m_parse", return_value = nim_a3m_parse), pytest.raises(Exception):
        msa_parsing.parse_a3m("ACDEF\n", msa_type = "protein")


def test_nim_featurization_truncated_sequence_raises(nim_modules):
    """A sequence shorter than the chain's polymer residue count must raise, not
    silently read out of bounds (the flat deletion buffer is unbounded)."""
    a3m_string = ">101\nACD\n>102\nACDE\n"
    chain_id_to_residue = {
        "A": {"chemtype": [0] * 5, "residue_index": [1, 2, 3, 4, 5]}
    }
    msas = {chain_id: msa_parsing.parse_a3m(a3m_string, msa_type = "protein") for chain_id in ["A"]}

    with _nim_path_patches(None, None, None, None):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    with _nim_path_patches(*nim_modules):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)


def test_nim_featurization_deletion_row_too_short_raises(nim_modules):
    """A deletion row shorter than the chain's polymer residue count must raise in
    both paths instead of crossing into the next row of the flat buffer."""
    a3m_string = ">101\nACDEFGHI\n>102\nACDE\n"
    chain_id_to_residue = {
        "A": {"chemtype": [0] * 8, "residue_index": list(range(1, 9))}
    }
    msas = {chain_id: msa_parsing.parse_a3m(a3m_string, msa_type = "protein") for chain_id in ["A"]}

    with _nim_path_patches(None, None, None, None):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    with _nim_path_patches(*nim_modules):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)


def _real_msa_a3m(max_seqs = None):
    """Real MSA content from the repo test data: gaps, insertions, uniprot
    descriptions. Truncated to the first `max_seqs` sequences to keep the test fast."""
    msa_path = Path("data", "test", "pdb_data", "data_caches", "msa", "msas", "721p-assembly1A_protein.a3m")
    if not msa_path.exists():
        pytest.skip("real MSA test data not available")

    text = msa_path.read_text()
    if max_seqs:
        lines = text.splitlines()
        header_idx = [i for i, line in enumerate(lines) if line.startswith(">")]
        text = "\n".join(lines[: header_idx[max_seqs]]) + "\n"
    return text


def test_nim_matches_python_on_real_msa(nim_modules):
    """Representative real MSA (gaps, insertions, uniprot descriptions) must
    produce identical parse + featurization in both paths."""
    a3m_string = _real_msa_a3m(max_seqs = 400)
    query_len = len(a3m_string.splitlines()[1])
    chain_id_to_residue = {
        "A": {"chemtype": [0] * query_len, "residue_index": list(range(1, query_len + 1))}
    }

    def run(nim_patch_vals):
        nim_a3m, nim_msa_features, nim_msa_stats, nim_species_id = nim_patch_vals
        with _nim_path_patches(nim_a3m, nim_msa_features, nim_msa_stats, nim_species_id):
            msas = {chain_id: msa_parsing.parse_a3m(a3m_string, msa_type = "protein") for chain_id in ["A"]}
            return msas, data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    py_msas, py_chains = run((None, None, None, None))
    nim_msas, nim_chains = run(nim_modules)

    for chain_id, py_msa in py_msas.items():
        nim_msa = nim_msas[chain_id]
        assert nim_msa.sequences == py_msa.sequences
        assert nim_msa.deletion_matrix == py_msa.deletion_matrix
        assert nim_msa.descriptions == py_msa.descriptions

    for py_chain, nim_chain in zip(py_chains, nim_chains):
        for key, value in py_chain.items():
            assert np.array_equal(nim_chain[key], value), key


def test_nim_parse_multiline_lowercase_carry(nim_modules):
    """Lowercase insertions spanning multiple lines of one sequence must carry
    into the deletion count (python joins lines before counting)."""
    nim_a3m_parse, _, _, _ = nim_modules

    for a3m in (
        ">h\nACDa\nXbc\n",
        ">h\nACD\naaa\nXbc\n",
        ">h\nACDa\nb\ncX\n",
        ">h\nACDa\nβX\n",
    ):
        _assert_nim_parse_equals_python(nim_a3m_parse, a3m)


def test_nim_featurization_empty_chain(nim_modules):
    """A chain with no residues must produce identical features in both paths."""
    a3m_string = ">101\n\n"
    chain_id_to_residue = {"A": {"chemtype": [], "residue_index": []}}

    py_chains = _featurize({"A": a3m_string}, chain_id_to_residue, "protein")[1]
    nim_chains = _featurize({"A": a3m_string}, chain_id_to_residue, "protein", nim_modules)[1]

    for py_chain, nim_chain in zip(py_chains, nim_chains):
        for key, value in py_chain.items():
            assert np.array_equal(nim_chain[key], value), key


def test_nim_featurization_sequence_longer_than_chain_raises(nim_modules):
    """A sequence with more residues than the chain must raise in both paths."""
    a3m_string = ">101\nACDEF\n"
    chain_id_to_residue = {"A": {"chemtype": [0] * 3, "residue_index": [1, 2, 3]}}
    msas = {"A": msa_parsing.parse_a3m(a3m_string, msa_type = "protein")}

    with _nim_path_patches(None, None, None, None):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    with _nim_path_patches(*nim_modules):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)


def test_nim_featurization_empty_msa_raises_identically(nim_modules):
    """An empty a3m (no sequences) must raise the same error in both paths."""
    a3m_string = ""
    chain_id_to_residue = {"A": {"chemtype": [0], "residue_index": [1]}}
    msas = {"A": msa_parsing.parse_a3m(a3m_string, msa_type = "protein")}

    with _nim_path_patches(None, None, None, None):
        with pytest.raises(Exception) as py_exc:
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    with _nim_path_patches(*nim_modules):
        with pytest.raises(Exception) as nim_exc:
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 32)

    assert type(py_exc.value) is type(nim_exc.value)


@pytest.mark.parametrize("num_msa_one_hot", (22, 24))
def test_nim_featurization_varied_one_hot_bins(nim_modules, num_msa_one_hot):
    """Non-default (but in-range) one-hot bin counts must produce identical
    features, including the Nim-computed profile."""
    a3m_string = ">101\nACDEFGHIK\n>102\nAXDEFGHIK\n>103\nACDEFGHIK\n"
    chain_id_to_residue = {"A": {"chemtype": [0] * 9, "residue_index": list(range(1, 10))}}
    msas = {"A": msa_parsing.parse_a3m(a3m_string, msa_type = "protein")}

    def run(nim_patch_vals):
        with _nim_path_patches(*nim_patch_vals):
            return data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = num_msa_one_hot)

    py_chains = run((None, None, None, None))
    nim_chains = run(nim_modules)

    for py_chain, nim_chain in zip(py_chains, nim_chains):
        for key, value in py_chain.items():
            assert np.array_equal(nim_chain[key], value), key


def test_nim_profile_out_of_range_raises(nim_modules):
    """MSA residue types beyond the one-hot bin count must raise in both paths,
    not silently drop counts."""
    a3m_string = ">101\nACD\n"
    chain_id_to_residue = {"A": {"chemtype": [0] * 3, "residue_index": [1, 2, 3]}}
    msas = {"A": msa_parsing.parse_a3m(a3m_string, msa_type = "protein")}

    with _nim_path_patches(None, None, None, None):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 4)

    with _nim_path_patches(*nim_modules):
        with pytest.raises(Exception):
            data_pipeline.make_msa_features(msas, chain_id_to_residue, num_msa_one_hot = 4)


def test_nim_species_ids_match_python(nim_modules):
    """Species id extraction must match `get_identifiers` on uniprot identifiers,
    tab-separated jackhmmer headers, and malformed descriptions."""
    _, _, _, nim_species_id = nim_modules

    descriptions = [
        "tr|A0A146SKV9|A0A146SKV9_FUNHE", "sp|P0C2L1|A3X1_LOXLA",
        "sp|Q8JTF3|Q8JTF3_YEAST", "tr|B3LYP1|X3X5_9PROT",
        "tr|A0A146SKV9_1|A0A146SKV9_FUNHE_5", "sp|P0C2L1|A3X1_LOXLA_5",
        "sp|ABCDEF|XYZ_SIM", "sp|ABCDE|XYZ_SIM", "sp|ABCDEFGHIJK|XYZ_SIM",
        "sp|ABC_DEF|XYZ_SIM", "sp|ABCDEF|XYZ_SIM_123", "tr|123456|XYZ_SIM",
        "101|tr|A0A146SKV9|A0A146SKV9_FUNHE", "x tr|A0A146SKV9|A0A146SKV9_FUNHE",
        "tr|A0A146SKV9|A0A146SKV9_FUNHE ", "  tr|A0A146SKV9|A0A146SKV9_FUNHE",
        ">tr|A0A146SKV9|A0A146SKV9_FUNHE", "tr|A0A146SKV9|A0A146SKV9_FUNHE/2",
        "UPI0000E11D81", "101", "", "   ", "sp|ABCDEF|AB_S", "sp|ABCDEF|ABCDEF",
        "UPI0000E11D81\t143\t0.692\t2.962E-36\t0\t116\t118\t4\t119\t247",
        "tr|A0A146SKV9|A0A146SKV9_FUNHE\t143\t0.5",
        "sp|P0C2L1|A3X1_LOXLA\t1\t2.0",
        "QFQGGGLVQPGGS\t143\t0.692",
        "a\tb\tc\t", "\t", "a\t", "\tb", "a b\tc d",
    ]

    for tab_separated in (False, True):
        py_species = [
            msa_parsing.get_identifiers(
                description = d, tab_separated_alignment_headers = tab_separated
            ).species_id
            for d in descriptions
        ]
        nim_species = nim_species_id.species_ids(descriptions, tab_separated)
        assert nim_species == py_species
