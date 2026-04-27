# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for supercut_cascade.select (4 tests)."""
from __future__ import annotations

from supercut_cascade.select import Clip, per_mv_cap, temporal_nms


def _make_clip(
    stable_id: str,
    video_id: str,
    shot_idx: int,
    start: float,
    end: float,
    judge_avg: float = 7.0,
) -> Clip:
    """Helper to build a minimal Clip."""
    return Clip(
        stable_id=stable_id,
        video_id=video_id,
        shot_idx=shot_idx,
        start=start,
        end=end,
        duration=end - start,
        judge_avg=judge_avg,
        face_area=0.15,
        face_ratio=0.30,
    )


# ---------------------------------------------------------------------------
# test_temporal_nms_no_overlap
# ---------------------------------------------------------------------------

def test_temporal_nms_no_overlap():
    """Clips from different videos are all preserved by temporal NMS."""
    clips = [
        _make_clip("c0", "mv0", 0, 0.0, 1.0),
        _make_clip("c1", "mv1", 0, 0.0, 1.0),
        _make_clip("c2", "mv2", 0, 0.0, 1.0),
    ]
    kept = temporal_nms(clips, iou_max=0.3, min_gap_sec=5.0)
    assert len(kept) == 3, (
        "clips from different videos should all survive NMS; "
        f"got {len(kept)} instead of 3"
    )


# ---------------------------------------------------------------------------
# test_temporal_nms_overlap_drop
# ---------------------------------------------------------------------------

def test_temporal_nms_overlap_drop():
    """Within the same video, clips closer than min_gap_sec are suppressed."""
    # Two clips in mv0 with center gap < 5s → lower-scored one dropped
    clips = [
        _make_clip("c0", "mv0", 0, 0.0, 1.0, judge_avg=9.0),
        _make_clip("c1", "mv0", 1, 1.5, 2.5, judge_avg=6.0),  # center gap = 2s < 5s
    ]
    kept = temporal_nms(clips, iou_max=0.3, min_gap_sec=5.0)
    assert len(kept) == 1, (
        "lower-scoring clip within min_gap_sec of a higher-scoring clip must be dropped; "
        f"got {len(kept)} clips"
    )
    assert kept[0].stable_id == "c0", "the higher-scored clip must be retained"


# ---------------------------------------------------------------------------
# test_per_mv_cap
# ---------------------------------------------------------------------------

def test_per_mv_cap():
    """per_mv_cap limits each video to at most cap clips, keeping highest scores."""
    clips = [
        _make_clip(f"c{i}", "mv0", i, float(i * 3), float(i * 3 + 1), judge_avg=float(i))
        for i in range(6)
    ]
    kept = per_mv_cap(clips, cap=3)
    assert len(kept) == 3, f"expected 3 clips from cap=3, got {len(kept)}"
    scores = [c.judge_avg for c in kept]
    assert all(s >= 3.0 for s in scores), (
        "per_mv_cap should retain the 3 highest-scored clips; "
        f"retained scores: {scores}"
    )


# ---------------------------------------------------------------------------
# test_select_and_order_e2e
# ---------------------------------------------------------------------------

def test_select_and_order_e2e(synthetic_clips, tmp_path):
    """select_and_order pipeline returns a list no longer than the input clips."""
    import json

    from supercut_cascade.select import select_and_order

    # Write windows_seed.jsonl from synthetic_clips
    windows_path = tmp_path / "windows_seed.jsonl"
    judged_path = tmp_path / "judged.jsonl"
    out_path = tmp_path / "curated.jsonl"

    with windows_path.open("w") as wf:
        for c in synthetic_clips:
            wf.write(
                json.dumps({
                    "stable_id": c.stable_id,
                    "video_id": c.video_id,
                    "shot_idx": c.shot_idx,
                    "start": c.start,
                    "end": c.end,
                    "duration": c.duration,
                    "window_idx_in_shot": 0,
                    "face_area_ratio": c.face_area,
                    "max_face_ratio_h": c.face_ratio,
                }) + "\n"
            )

    # Write judged.jsonl with scores above threshold
    with judged_path.open("w") as jf:
        for c in synthetic_clips:
            jf.write(
                json.dumps({
                    "stable_id": c.stable_id,
                    "judge_c": {"verdict": "accept"},
                    "judge_a": {"score": c.judge_avg},
                    "judge_b": {"score": c.judge_avg},
                    "c_accepted": True,
                    "ab_avg": c.judge_avg,
                }) + "\n"
            )

    result = select_and_order(
        windows_path,
        judged_path,
        out_path,
        target_sec=30.0,
        tol_sec=10.0,
        per_mv_cap_n=5,
        judge_avg_min=6.0,
    )

    assert isinstance(result, list), "select_and_order must return a list"
    assert len(result) <= len(synthetic_clips), (
        "output must not exceed the number of input clips"
    )
    assert out_path.exists(), "curated.jsonl must be written to disk"
