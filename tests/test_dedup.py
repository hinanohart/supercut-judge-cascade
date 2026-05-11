# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Tests for supercut_cascade.dedup (3 tests)."""
from __future__ import annotations

import numpy as np

from supercut_cascade.dedup import compute_phash, phash_dedup

# ---------------------------------------------------------------------------
# test_compute_phash_shape
# ---------------------------------------------------------------------------

def test_compute_phash_shape(synthetic_face_image):
    """compute_phash returns a boolean array of exactly 64 elements."""
    ph = compute_phash(synthetic_face_image)

    assert ph.shape == (64,), (
        f"pHash must have shape (64,); got {ph.shape}"
    )
    assert ph.dtype == bool or ph.dtype == np.bool_, (
        f"pHash must be a bool array; got dtype {ph.dtype}"
    )


# ---------------------------------------------------------------------------
# test_phash_dedup_identical
# ---------------------------------------------------------------------------

def test_phash_dedup_identical(synthetic_face_image):
    """phash_dedup reduces identical frames to a single representative."""
    frames = [synthetic_face_image] * 5  # five copies of the same image
    kept = phash_dedup(frames, threshold=5)

    assert len(kept) == 1, (
        f"five identical frames must deduplicate to 1; got {len(kept)} kept indices"
    )
    assert kept[0] == 0, "the kept index must be 0 (first frame)"


# ---------------------------------------------------------------------------
# test_phash_dedup_unique
# ---------------------------------------------------------------------------

def test_phash_dedup_unique():
    """phash_dedup preserves all frames when they are perceptually distinct."""
    rng = np.random.default_rng(7)
    # Generate 4 random noise images — highly distinct hashes
    frames = [rng.integers(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    kept = phash_dedup(frames, threshold=5)

    assert len(kept) == 4, (
        f"four distinct random frames must all be kept; got {len(kept)}"
    )
    assert kept == list(range(4)), (
        f"kept indices must be [0,1,2,3] in order; got {kept}"
    )
