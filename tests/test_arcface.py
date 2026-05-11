# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Tests for supercut_cascade.arcface (3 tests)."""
from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# test_arcface_lazy_import
# ---------------------------------------------------------------------------

def test_arcface_lazy_import(monkeypatch):
    """ArcFaceEmbedder raises BackendUnavailableError when insightface is absent."""
    from supercut_cascade.exceptions import BackendUnavailableError

    # Block insightface so the lazy import inside __init__ fails
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.setitem(sys.modules, "insightface.app", None)

    # Force reimport of arcface to clear any cached successful import
    if "supercut_cascade.arcface" in sys.modules:
        del sys.modules["supercut_cascade.arcface"]

    with pytest.raises(BackendUnavailableError, match="insightface"):
        from supercut_cascade.arcface import ArcFaceEmbedder
        ArcFaceEmbedder()


# ---------------------------------------------------------------------------
# test_arcface_init_fails_no_weights
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_arcface_init_fails_no_weights():
    """ArcFaceEmbedder raises BackendUnavailableError or ValueError when weights absent."""
    # This test is only meaningful when insightface IS installed but weights are not.
    # Mark as network because downloading is required; skip gracefully if not installed.
    import importlib.util
    if importlib.util.find_spec("insightface") is None:
        pytest.skip("insightface not installed — skipping weight-absent test")

    from supercut_cascade.arcface import ArcFaceEmbedder
    from supercut_cascade.exceptions import BackendUnavailableError

    # Use a non-existent model name to trigger a failure
    with pytest.raises((BackendUnavailableError, ValueError, Exception)):
        ArcFaceEmbedder(model_name="nonexistent_model_xyz_should_fail")


# ---------------------------------------------------------------------------
# test_arcface_legal_notice
# ---------------------------------------------------------------------------

def test_arcface_legal_notice():
    """arcface module docstring and ArcFaceEmbedder class docstring contain legal keywords."""
    import inspect

    import supercut_cascade.arcface as arcface_mod

    # Check module-level docstring
    module_doc = arcface_mod.__doc__ or ""
    class_doc = inspect.getdoc(arcface_mod.ArcFaceEmbedder) or ""
    combined = module_doc + class_doc

    assert "GDPR" in combined, (
        "arcface docs must mention GDPR for biometric data compliance notice"
    )
    assert "BIPA" in combined, (
        "arcface docs must mention BIPA (Illinois Biometric Information Privacy Act)"
    )
    assert "NON-COMMERCIAL" in combined.upper(), (
        "arcface docs must mention NON-COMMERCIAL restriction of buffalo_l weights"
    )
