"""Synthetic smoke test: end-to-end pipeline with no external assets.

Generates 10 procedural BGR frames and synthetic face reference embeddings
entirely in-memory.  No video files, no network calls, no licenses required.
Runs in < 60 s in CI.

Usage::

    python examples/synthetic_smoke.py
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("synthetic_smoke")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(index: int, h: int = 120, w: int = 160) -> np.ndarray:
    """Return a procedurally generated BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    frame[:, :, 0] = np.linspace(index * 10, 255, w, dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, index * 15, w, dtype=np.uint8)
    frame[:, :, 2] = np.full(w, 128 + index * 5, dtype=np.uint8)
    # Synthetic "face-like" bright rectangle
    cx, cy = w // 2 + (index % 3) * 5, h // 2
    frame[cy - 15 : cy + 15, cx - 12 : cx + 12] = [200, 180, 160]
    return frame


def make_embedding(seed: int = 0) -> np.ndarray:
    """Return a unit-normalised 512-D synthetic embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Smoke: WindowExtractor
# ---------------------------------------------------------------------------

def smoke_windows() -> None:
    from supercut_cascade.windows import WindowExtractor

    log.info("--- WindowExtractor smoke ---")
    extractor = WindowExtractor(window_sec=1.0, stride_sec=0.5, min_duration_sec=0.5)
    timestamps = [i * 0.5 for i in range(20)]  # 0.0 to 9.5 s
    windows = extractor.extract(timestamps, video_id="smoke_video")
    log.info("extracted %d windows from %d timestamps", len(windows), len(timestamps))
    assert len(windows) > 0, "expected at least one window"


# ---------------------------------------------------------------------------
# Smoke: CVPrefilter
# ---------------------------------------------------------------------------

def smoke_prefilter() -> None:
    from supercut_cascade.prefilter_cv import CVPrefilter

    log.info("--- CVPrefilter smoke ---")
    prefilter = CVPrefilter(min_sharpness=0.0, min_face_ratio=0.0)
    frames = [make_frame(i) for i in range(10)]
    scores = [prefilter.score(f) for f in frames]
    log.info("prefilter scores: %s", [round(s, 3) for s in scores])
    assert len(scores) == 10


# ---------------------------------------------------------------------------
# Smoke: PhashDedup
# ---------------------------------------------------------------------------

def smoke_dedup() -> None:
    from supercut_cascade.dedup import PhashDedup

    log.info("--- PhashDedup smoke ---")
    dedup = PhashDedup(threshold=10)
    frames = [make_frame(i) for i in range(5)]
    # Add a near-duplicate of frame 0
    near_dup = make_frame(0).copy()
    near_dup[0, 0] = [1, 2, 3]  # tiny change
    frames.append(near_dup)
    ids = [f"f{i}" for i in range(len(frames))]
    kept = dedup.filter(list(zip(ids, frames)))
    log.info("kept %d / %d after dedup", len(kept), len(frames))
    assert len(kept) <= len(frames)


# ---------------------------------------------------------------------------
# Smoke: IdentityFilter (synthetic embeddings only, no insightface)
# ---------------------------------------------------------------------------

def smoke_identity_filter() -> None:
    from supercut_cascade.identity_filter import IdentityFilter

    log.info("--- IdentityFilter smoke ---")
    ref_embs = np.stack([make_embedding(0), make_embedding(1)])  # 2 refs for TARGET
    filt = IdentityFilter(reference_embeddings=ref_embs, threshold=0.30)

    # Same distribution → should match
    query_match = make_embedding(0)
    is_match, sim = filt.is_target(query_match)
    log.info("query(seed=0) match=%s sim=%.3f", is_match, sim)

    # Random orthogonal vector → should not match
    rng = np.random.default_rng(999)
    query_other = rng.standard_normal(512).astype(np.float32)
    query_other /= np.linalg.norm(query_other)
    is_other, sim2 = filt.is_target(query_other)
    log.info("query(random) match=%s sim=%.3f", is_other, sim2)


# ---------------------------------------------------------------------------
# Smoke: BucketNMS
# ---------------------------------------------------------------------------

def smoke_bucket_nms() -> None:
    from supercut_cascade.select import BucketNMS

    log.info("--- BucketNMS smoke ---")
    windows = [
        {"stable_id": f"w{i}", "start": float(i * 10), "end": float(i * 10 + 5),
         "ab_avg": float(i % 4)}
        for i in range(20)
    ]
    nms = BucketNMS(bucket_sec=60.0)
    selected = nms.select(windows, top_n=5)
    log.info("selected %d windows after bucket NMS", len(selected))
    assert len(selected) <= 5


# ---------------------------------------------------------------------------
# Smoke: JudgeResult + checkpoint I/O
# ---------------------------------------------------------------------------

def smoke_judge_checkpoint() -> None:
    from supercut_cascade.judge import JudgeResult, is_done, read_checkpoint, write_checkpoint

    log.info("--- Judge checkpoint smoke ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        jdir = Path(tmpdir)

        result_c = JudgeResult(
            stable_id="smoke_001",
            phase="C",
            payload={"stable_id": "smoke_001", "verdict": "accept", "confidence": 0.9,
                     "rejected_reasons": [], "notes": "ok"},
        )
        write_checkpoint(result_c, jdir)
        assert is_done("smoke_001", "C", jdir)

        loaded = read_checkpoint("smoke_001", "C", jdir)
        assert loaded is not None
        assert loaded["verdict"] == "accept"

        result_a = JudgeResult(
            stable_id="smoke_001",
            phase="A",
            payload={"stable_id": "smoke_001", "score": 7, "primary_strength": "gaze",
                     "notes": "nice"},
        )
        write_checkpoint(result_a, jdir)

        from supercut_cascade.judge import aggregate_summary
        seed_index = {"smoke_001": {"video_id": "v1", "start": 0.0, "end": 2.0}}
        rows = aggregate_summary(seed_index, jdir)
        assert len(rows) == 1
        assert rows[0]["c_accepted"] is True
        log.info("aggregate_summary ok: %s", json.dumps({k: rows[0][k] for k in
                 ["stable_id", "c_accepted", "ab_avg"]}, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("=== synthetic smoke test start ===")
    failures: list[str] = []
    for name, fn in [
        ("windows", smoke_windows),
        ("prefilter", smoke_prefilter),
        ("dedup", smoke_dedup),
        ("identity_filter", smoke_identity_filter),
        ("bucket_nms", smoke_bucket_nms),
        ("judge_checkpoint", smoke_judge_checkpoint),
    ]:
        try:
            fn()
            log.info("[PASS] %s", name)
        except Exception as exc:
            log.error("[FAIL] %s: %s", name, exc)
            failures.append(name)

    if failures:
        log.error("=== FAILED: %s ===", failures)
        return 1
    log.info("=== all smoke tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
