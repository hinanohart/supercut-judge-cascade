# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for supercut-judge-cascade tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_face_image():
    """Synthetic 100x100 BGR image with gradient pattern."""
    h = w = 100
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) / h
    img = np.stack([
        ((x + y) * 128).astype(np.uint8),
        (x * 200 + 50).astype(np.uint8),
        (y * 200 + 50).astype(np.uint8),
    ], axis=-1)
    return img


@pytest.fixture
def synthetic_embedding():
    """Normalised 512-D float32 embedding from a fixed seed."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(512).astype(np.float32)
    return raw / np.linalg.norm(raw)


@pytest.fixture
def synthetic_clips():
    """List of 10 Clip objects for select_and_order tests."""
    from supercut_cascade.select import Clip
    return [
        Clip(
            stable_id=f"w{i}",
            video_id=f"mv{i % 3}",
            shot_idx=i,
            start=i * 2.0,
            end=i * 2.0 + 1.0,
            duration=1.0,
            judge_avg=5.0 + i * 0.2,
            face_area=0.15,
            face_ratio=0.30,
        )
        for i in range(10)
    ]
