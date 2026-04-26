# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Perceptual-hash deduplication of candidate frames/windows.

Uses a 64-bit DCT-based perceptual hash (pHash) computed with ``cv2.dct``
to detect near-duplicate images.  No external ``imagehash`` library is
required for the core path; ``cv2`` and ``numpy`` are sufficient.

Algorithm
---------
1. Resize input image to 32×32 greyscale.
2. Apply a full 32×32 2-D DCT via ``cv2.dct``.
3. Take the top-left 8×8 low-frequency sub-block (64 coefficients).
4. Binarise: each coefficient > block median → ``True``.
5. Hamming distance between two such hashes = number of differing bits.

Public API
----------
:func:`compute_phash` — compute a 64-bit hash for one image.
:func:`phash_dedup`   — keep the indices of non-duplicate images from a list.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------


def compute_phash(frame: np.ndarray) -> np.ndarray:
    """Compute a 64-bit DCT perceptual hash for a single image.

    Parameters
    ----------
    frame:
        BGR or grayscale uint8 image.

    Returns
    -------
    Boolean array of shape ``(64,)`` representing the pHash bits.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct_mat = cv2.dct(resized)
    low_freq = dct_mat[:8, :8].flatten()
    median = float(np.median(low_freq))
    return low_freq > median


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Return the Hamming distance between two bool[64] pHash arrays.

    Parameters
    ----------
    a, b:
        Boolean arrays of shape ``(64,)``.

    Returns
    -------
    Number of differing bits (integer in ``[0, 64]``).
    """
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def phash_dedup(
    frames: list[np.ndarray],
    threshold: int = 5,
) -> list[int]:
    """Return indices of frames that survive perceptual-hash deduplication.

    Applies greedy NMS: frames are processed in order; the first frame is
    always kept.  Each subsequent frame is kept only if its Hamming distance
    to **all** previously kept frames is greater than ``threshold``.

    Parameters
    ----------
    frames:
        List of BGR or grayscale uint8 images.
    threshold:
        Maximum Hamming distance (inclusive) to consider two frames
        duplicates.  Lower values are stricter (fewer kept).

    Returns
    -------
    List of integer indices (into ``frames``) that were kept.  Preserves
    input order.
    """
    if not frames:
        return []

    kept_indices: list[int] = []
    kept_hashes: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        ph = compute_phash(frame)
        is_dup = any(hamming_distance(ph, kh) <= threshold for kh in kept_hashes)
        if not is_dup:
            kept_indices.append(i)
            kept_hashes.append(ph)

    log.debug(
        "phash_dedup: in=%d kept=%d removed=%d threshold=%d",
        len(frames), len(kept_indices), len(frames) - len(kept_indices), threshold,
    )
    return kept_indices


__all__ = ["compute_phash", "hamming_distance", "phash_dedup"]
