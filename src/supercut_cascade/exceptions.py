# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Custom exception types for supercut-judge-cascade."""


class SupercutCascadeError(Exception):
    """Base class for all supercut-judge-cascade errors."""


class BackendUnavailableError(SupercutCascadeError, ImportError):
    """Raised when an optional backend (e.g. scenedetect, insightface) is not installed."""


class JudgeError(SupercutCascadeError, RuntimeError):
    """Raised when a judge (LLM or CV) fails to produce a valid score."""


class IdentityFilterError(SupercutCascadeError, RuntimeError):
    """Raised when the identity filter cannot build a valid reference embedding."""


class ConfigValidationError(SupercutCascadeError, ValueError):
    """Raised when a configuration value is invalid or out of range."""
