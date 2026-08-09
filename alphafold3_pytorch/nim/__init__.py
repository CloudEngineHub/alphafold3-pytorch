"""Lazy, USE_NIM-gated access to the Nim-backed data functions.

The Nim modules (`msa_features.nim`, `a3m_parse.nim`) are compiled on first
import by nimporter_plus, so importing this package never requires Nim to be
installed or triggers a compilation.
"""

import os

import numpy as np
from beartype.typing import Any, List, Optional

from alphafold3_pytorch.tensor_typing import USE_NIM

_nim_modules: Optional[Any] = None


def _load_nim_modules() -> dict:
    global _nim_modules

    if _nim_modules is not None:
        return _nim_modules

    _nim_modules = {}

    try:
        if not USE_NIM:
            return _nim_modules

        # NOTE: compile in release mode by default (unless overridden), as the
        # debug build runs ~50x slower on these hot loops.
        os.environ.setdefault('NIMPORTER_COMPILER_ARGS', '-d:release')

        import nimporter_plus  # registers the nim import hook

        from alphafold3_pytorch.nim import a3m_parse, msa_features  # compiles the nim modules

        _nim_modules.update(a3m_parse = a3m_parse, msa_features = msa_features)
    except Exception:
        pass

    return _nim_modules


def import_nim_msa_features() -> Optional[Any]:
    """Return the Nim MSA featurization module, or `None` if unavailable."""
    return _load_nim_modules().get('msa_features')


def import_nim_a3m_parse() -> Optional[Any]:
    """Return the Nim a3m parsing module, or `None` if unavailable."""
    return _load_nim_modules().get('a3m_parse')


def deletion_bytes_to_matrix(deletion_bytes: bytes, row_lengths: List[int]) -> List[List[int]]:
    """Decode a flat int32 buffer and row lengths into a list of deletion rows."""
    if not row_lengths:
        return []

    deletions_flat = np.frombuffer(deletion_bytes, dtype = np.int32)
    return [row.tolist() for row in np.split(deletions_flat, np.cumsum(row_lengths[:-1]))]
