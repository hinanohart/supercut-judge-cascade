# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""supercut-judge-cascade: pipeline for judge-cascade supercut generation."""

from importlib.metadata import PackageNotFoundError, version as _v
import logging

try:
    __version__ = _v("supercut-judge-cascade")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .exceptions import (  # noqa: E402
    SupercutCascadeError,
    BackendUnavailableError,
    JudgeError,
    IdentityFilterError,
    ConfigValidationError,
)
from .judge import VisionLLMJudge, JudgeResult  # noqa: E402
from .select import select_and_order  # noqa: E402
from .qa import final_qa  # noqa: E402
from .arcface import ArcFaceEmbedder  # noqa: E402
from .identity_filter import IdentityFilter  # noqa: E402
from .detect import detect_scenes  # noqa: E402
from .windows import Window, build_windows  # noqa: E402
from .io import extract_frame, cut_clip, concat_clips, probe_duration  # noqa: E402

__all__ = [
    "__version__",
    # exceptions
    "SupercutCascadeError",
    "BackendUnavailableError",
    "JudgeError",
    "IdentityFilterError",
    "ConfigValidationError",
    # judge
    "VisionLLMJudge",
    "JudgeResult",
    # select
    "select_and_order",
    # qa
    "final_qa",
    # arcface
    "ArcFaceEmbedder",
    # identity filter
    "IdentityFilter",
    # detect
    "detect_scenes",
    # windows
    "Window",
    "build_windows",
    # io
    "extract_frame",
    "cut_clip",
    "concat_clips",
    "probe_duration",
]
