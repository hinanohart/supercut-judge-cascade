# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""supercut-judge-cascade: pipeline for judge-cascade supercut generation."""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

try:
    __version__ = _v("supercut-judge-cascade")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .arcface import ArcFaceEmbedder  # noqa: E402
from .detect import detect_scenes  # noqa: E402
from .exceptions import (  # noqa: E402
    BackendUnavailableError,
    ConfigValidationError,
    IdentityFilterError,
    JudgeError,
    SupercutCascadeError,
)
from .identity_filter import IdentityFilter  # noqa: E402
from .io import concat_clips, cut_clip, extract_frame, probe_duration  # noqa: E402
from .judge import JudgeResult, VisionLLMJudge  # noqa: E402
from .qa import final_qa  # noqa: E402
from .select import select_and_order  # noqa: E402
from .windows import Window, build_windows  # noqa: E402

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
