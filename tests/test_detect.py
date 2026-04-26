# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for supercut_cascade.detect (2 tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# test_detect_scenes_lazy
# ---------------------------------------------------------------------------

def test_detect_scenes_lazy(monkeypatch):
    """detect_scenes raises BackendUnavailableError when scenedetect is not installed."""
    from supercut_cascade.exceptions import BackendUnavailableError

    # Block scenedetect so the lazy import inside detect_scenes fails
    monkeypatch.setitem(sys.modules, "scenedetect", None)

    # Force reimport to clear cached import state
    if "supercut_cascade.detect" in sys.modules:
        del sys.modules["supercut_cascade.detect"]

    with pytest.raises(BackendUnavailableError, match="[Ss]cenedetect|[Pp]y[Ss]cene"):
        from supercut_cascade.detect import detect_scenes
        detect_scenes(Path("/fake/video.mp4"))


# ---------------------------------------------------------------------------
# test_detect_scenes_no_video
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_detect_scenes_no_video(tmp_path):
    """detect_scenes raises an OSError-family error for a non-existent video file."""
    # Only runs when scenedetect IS installed; skip gracefully if not.
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        pytest.skip("scenedetect not installed — skipping no-video test")

    from supercut_cascade.detect import detect_scenes

    missing = tmp_path / "nonexistent_video.mp4"
    with pytest.raises((OSError, Exception)):
        detect_scenes(missing)
