# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""OpenCV-based pre-filter: reject blurry, blacked-out, or washed-out frames.

Requires ``opencv-python`` (``cv2``), which is a core dependency.

Three lightweight checks are applied in order:

1. **Blur** — Laplacian variance < ``min_sharpness`` (default 50).
2. **Blackout** — fraction of pixels in bins [0, 15] > ``blackout_max``
   (default 0.15).
3. **Whiteout** — fraction of pixels in bins [240, 255] > ``whiteout_max``
   (default 0.05).

Status values returned by :func:`cv_prefilter_status`:
    ``"pass"``, ``"blur"``, ``"blackout"``, ``"whiteout"``

The higher-level :func:`cv_prefilter` returns a boolean suitable for
pipeline use (``True`` = passes all checks).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _laplacian_var(gray: np.ndarray) -> float:
    """Return Laplacian variance of a grayscale image as a blur proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _histogram_ratios(gray: np.ndarray) -> tuple[float, float]:
    """Return ``(dark_ratio, bright_ratio)`` from the 256-bin histogram.

    Parameters
    ----------
    gray:
        Single-channel uint8 image.

    Returns
    -------
    Tuple of ``(dark_ratio, bright_ratio)`` where each is a fraction in
    ``[0, 1]``.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    if total == 0.0:
        return 0.0, 0.0
    dark_ratio = float(hist[:16].sum()) / total
    bright_ratio = float(hist[240:].sum()) / total
    return dark_ratio, bright_ratio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cv_prefilter_status(
    frame: np.ndarray,
    *,
    min_sharpness: float = 50.0,
    blackout_max: float = 0.15,
    whiteout_max: float = 0.05,
) -> str:
    """Run all CV checks and return a status string.

    Parameters
    ----------
    frame:
        BGR or grayscale uint8 image.
    min_sharpness:
        Minimum Laplacian variance.  Frames below this are ``"blur"``.
    blackout_max:
        Maximum dark-pixel fraction (bins 0–15).  Above this is
        ``"blackout"``.
    whiteout_max:
        Maximum bright-pixel fraction (bins 240–255).  Above this is
        ``"whiteout"``.

    Returns
    -------
    One of ``"pass"``, ``"blur"``, ``"blackout"``, ``"whiteout"``.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if _laplacian_var(gray) < min_sharpness:
        return "blur"

    dark_ratio, bright_ratio = _histogram_ratios(gray)
    if dark_ratio > blackout_max:
        return "blackout"
    if bright_ratio > whiteout_max:
        return "whiteout"

    return "pass"


def cv_prefilter(
    frame: np.ndarray,
    min_sharpness: float = 50.0,
    min_faces: int = 0,
) -> bool:
    """Return ``True`` if ``frame`` passes all CV pre-filter checks.

    This is the primary pipeline interface.  It runs sharpness, blackout,
    and whiteout checks.  The ``min_faces`` parameter is reserved for future
    use (face-count gating); it is currently ignored.

    Parameters
    ----------
    frame:
        BGR or grayscale uint8 image.
    min_sharpness:
        Minimum Laplacian variance threshold.
    min_faces:
        Reserved; currently not enforced.

    Returns
    -------
    ``True`` if the frame passes all checks, ``False`` otherwise.
    """
    status = cv_prefilter_status(frame, min_sharpness=min_sharpness)
    if status != "pass":
        log.debug("cv_prefilter rejected frame: %s", status)
        return False
    return True


__all__ = ["cv_prefilter", "cv_prefilter_status"]
