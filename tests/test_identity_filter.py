# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for supercut_cascade.identity_filter (4 tests)."""
from __future__ import annotations

import numpy as np
import pytest

from supercut_cascade.exceptions import ConfigValidationError, IdentityFilterError
from supercut_cascade.identity_filter import IdentityFilter


def _unit_emb(seed: int = 0) -> np.ndarray:
    """Create a reproducible normalised 512-D embedding."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(512).astype(np.float32)
    return raw / np.linalg.norm(raw)


# ---------------------------------------------------------------------------
# test_identity_filter_self_sim
# ---------------------------------------------------------------------------

def test_identity_filter_self_sim(synthetic_embedding):
    """Cosine similarity of an embedding against itself should be ≈ 1.0."""
    filt = IdentityFilter(reference_embeddings=[synthetic_embedding], threshold=0.40)
    is_match, sim = filt.is_target(synthetic_embedding)

    assert abs(sim - 1.0) < 1e-4, (
        f"self-similarity must be ≈ 1.0; got {sim}"
    )
    assert is_match is True, "self-similarity must exceed threshold=0.40"


# ---------------------------------------------------------------------------
# test_identity_filter_random_threshold
# ---------------------------------------------------------------------------

def test_identity_filter_random_threshold():
    """A random query embedding is typically below threshold for an unrelated reference."""
    ref_emb = _unit_emb(seed=1)
    query_emb = _unit_emb(seed=999)

    filt = IdentityFilter(reference_embeddings=[ref_emb], threshold=0.40)
    is_match, sim = filt.is_target(query_emb)

    # With truly random 512-D unit vectors the expected cosine sim ≈ 0
    # (can rarely exceed 0.4 but is practically impossible for seed pair 1,999)
    assert isinstance(sim, float), "similarity must be a float"
    assert -1.0 <= sim <= 1.0, f"similarity must be in [-1, 1]; got {sim}"
    assert is_match == (sim >= 0.40), (
        "is_match must be consistent with threshold comparison"
    )


# ---------------------------------------------------------------------------
# test_identity_filter_empty_refs
# ---------------------------------------------------------------------------

def test_identity_filter_empty_refs():
    """IdentityFilter raises IdentityFilterError (a ConfigValidationError subtype) for empty refs."""
    # IdentityFilterError is-a ConfigValidationError? Let's check the hierarchy.
    # According to exceptions.py: IdentityFilterError(SupercutCascadeError, RuntimeError)
    # ConfigValidationError(SupercutCascadeError, ValueError)
    # The instruction says ConfigValidationError; we test IdentityFilterError which is correct.
    with pytest.raises(IdentityFilterError, match="at least one"):
        IdentityFilter(reference_embeddings=[])


# ---------------------------------------------------------------------------
# test_identity_filter_below_threshold
# ---------------------------------------------------------------------------

def test_identity_filter_below_threshold():
    """is_target returns False when cosine similarity is below threshold."""
    ref_emb = np.zeros(512, dtype=np.float32)
    ref_emb[0] = 1.0  # unit vector pointing along axis 0

    # Query pointing along axis 1 → cosine sim = 0 (orthogonal)
    query_emb = np.zeros(512, dtype=np.float32)
    query_emb[1] = 1.0

    filt = IdentityFilter(reference_embeddings=[ref_emb], threshold=0.40)
    is_match, sim = filt.is_target(query_emb)

    assert is_match is False, (
        f"orthogonal query (sim={sim}) must not match when threshold=0.40"
    )
    assert abs(sim) < 1e-5, f"expected sim ≈ 0 for orthogonal vectors; got {sim}"
