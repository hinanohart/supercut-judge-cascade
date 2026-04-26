# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for supercut_cascade.windows (3 tests)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from supercut_cascade.windows import Window, build_windows, plan_windows_from_segment


# ---------------------------------------------------------------------------
# test_window_dataclass
# ---------------------------------------------------------------------------

def test_window_dataclass():
    """Window dataclass can be instantiated and converted to dict via dataclasses.asdict."""
    w = Window(
        video_id="mv0",
        idx=0,
        stable_id="mv0__s000__w00",
        shot_idx=0,
        window_idx_in_shot=0,
        start=0.0,
        end=1.0,
        duration=1.0,
        max_face_ratio_h=0.35,
        face_area_ratio=0.12,
        soft_score=0.8,
        n_excluded=0,
        edge_density=0.4,
        gate_pass=True,
        gate_reasons=["pass"],
    )
    d = dataclasses.asdict(w)
    assert d["video_id"] == "mv0", "video_id must survive asdict"
    assert d["duration"] == 1.0, "duration must survive asdict"
    assert d["gate_pass"] is True, "gate_pass must survive asdict"
    assert isinstance(d["gate_reasons"], list), "gate_reasons must be a list"


# ---------------------------------------------------------------------------
# test_build_windows_empty_scenes
# ---------------------------------------------------------------------------

def test_build_windows_empty_scenes(tmp_path):
    """build_windows returns an empty list when the JSONL has no segments."""
    jsonl = tmp_path / "empty_video.jsonl"
    jsonl.write_text(
        json.dumps({
            "meta": {"video_id": "mv0"},
            "shots": [],
            "segments": [],
        }),
        encoding="utf-8",
    )
    result = build_windows([jsonl])
    assert result == [], (
        "build_windows must return [] when segments list is empty; "
        f"got {len(result)} windows"
    )


# ---------------------------------------------------------------------------
# test_build_windows_basic
# ---------------------------------------------------------------------------

def test_build_windows_basic(tmp_path):
    """build_windows produces windows from a single 5-second closeup segment."""
    jsonl = tmp_path / "mv1.jsonl"
    # One shot with a strong face closeup
    shot = {
        "start_sec": 0.0,
        "end_sec": 5.0,
        "max_face_ratio_h": 0.40,   # above min_face_ratio=0.28
        "face_area_ratio": 0.20,
        "soft_score": 0.9,
        "edge_density": 0.3,
        "n_excluded": 0,
        "sampled": 10,
    }
    # One segment covering the same 5-second span
    segment = {"start": 0.0, "end": 5.0}

    jsonl.write_text(
        json.dumps({
            "meta": {"video_id": "mv1"},
            "shots": [shot],
            "segments": [segment],
        }),
        encoding="utf-8",
    )

    result = build_windows(
        [jsonl],
        min_face_ratio=0.28,
        target_min=0.5,
        target_max=1.5,
        stride=0.5,
        window_length=1.0,
    )

    assert len(result) > 0, (
        "build_windows must produce at least one window for a valid 5s closeup segment"
    )
    for w in result:
        assert isinstance(w, Window), "each result must be a Window instance"
        assert w.video_id == "mv1", "video_id must match the source record"
        assert w.gate_pass is True, "all returned windows must have gate_pass=True"
        assert w.duration > 0, f"window duration must be positive; got {w.duration}"
