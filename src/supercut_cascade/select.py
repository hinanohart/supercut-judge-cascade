# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Clip selection and ordering for the final supercut.

Given a pool of judged candidate windows (``judged.jsonl``), this module
applies a pipeline of filters and reordering steps to produce the final
curated clip list:

1. Phase-C defect reject gate.
2. Minimum judge-average threshold (``judge_avg_min``).
3. Shot-level temporal NMS + minimum-gap enforcement within the same source
   video (prevents near-duplicate visual repeats).
4. Per-video-clip cap.
5. Budget cut to target duration.
6. 3-act intensity reorder (hook → build → climax → coda).
7. No-consecutive same-video enforcement.

If ``judged.jsonl`` is absent, a hash-based deterministic fallback score is
used (draft mode, no quality signal).

Output
------
``curated.jsonl`` — one row per adopted clip, consumed by the encode/concat
stage.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Clip:
    """Candidate clip from a judged window.

    Parameters
    ----------
    stable_id:
        Unique window identifier.
    video_id:
        Source video identifier.
    shot_idx:
        Shot index within the source video.
    start:
        Clip start time in seconds.
    end:
        Clip end time in seconds.
    duration:
        Clip duration in seconds.
    judge_avg:
        Average judge score (1-10).
    face_area:
        Face-area ratio used for ordering diversity.
    face_ratio:
        Max face height ratio (used as gate metric upstream).
    role:
        Suggested role in the final cut
        (``"opener"``, ``"buildup"``, ``"climax"``, ``"breather"``,
        ``"closer"``, ``"filler"``).
    """

    stable_id: str
    video_id: str
    shot_idx: int
    start: float
    end: float
    duration: float
    judge_avg: float
    face_area: float
    face_ratio: float
    role: str = "filler"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file; return empty list if absent.

    Parameters
    ----------
    path:
        Path to the JSONL file.

    Returns
    -------
    list[dict[str, Any]]
    """
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def _temporal_iou(a: Clip, b: Clip) -> float:
    """Compute temporal IoU between two clips from the same video.

    Parameters
    ----------
    a:
        First clip.
    b:
        Second clip.

    Returns
    -------
    float
        Intersection-over-union in the time domain (0-1).
    """
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    inter = max(0.0, end - start)
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def temporal_nms(
    clips: list[Clip],
    iou_max: float,
    min_gap_sec: float = 5.0,
) -> list[Clip]:
    """Apply temporal NMS and minimum-gap enforcement within the same video.

    Algorithm
    ---------
    1. Keep only the highest-scoring window per ``(video_id, shot_idx)``.
    2. IoU NMS across remaining clips of the same video.
    3. Minimum center-to-center gap enforcement within the same video.

    Parameters
    ----------
    clips:
        Candidate clips, unsorted.
    iou_max:
        Clips with IoU above this threshold are suppressed.
    min_gap_sec:
        Minimum center-to-center gap in seconds between two clips from the
        same video.

    Returns
    -------
    list[Clip]
        Filtered clips, sorted by descending ``judge_avg``.
    """
    clips = sorted(clips, key=lambda c: -c.judge_avg)

    # Phase 1: best per (video_id, shot_idx)
    best_per_shot: dict[tuple[str, int], Clip] = {}
    for c in clips:
        key = (c.video_id, c.shot_idx)
        if key not in best_per_shot:
            best_per_shot[key] = c

    # Phase 2 + 3: IoU NMS + min-gap
    kept: list[Clip] = []
    for c in sorted(best_per_shot.values(), key=lambda x: -x.judge_avg):
        c_mid = (c.start + c.end) / 2
        conflict = False
        for k in kept:
            if k.video_id != c.video_id:
                continue
            if _temporal_iou(c, k) > iou_max:
                conflict = True
                break
            k_mid = (k.start + k.end) / 2
            if abs(c_mid - k_mid) < min_gap_sec:
                conflict = True
                break
        if not conflict:
            kept.append(c)
    return kept


def per_mv_cap(clips: list[Clip], cap: int) -> list[Clip]:
    """Limit the number of clips per source video.

    Parameters
    ----------
    clips:
        Candidate clips (will be sorted by descending ``judge_avg``).
    cap:
        Maximum clips per video.

    Returns
    -------
    list[Clip]
    """
    clips = sorted(clips, key=lambda c: -c.judge_avg)
    counts: dict[str, int] = {}
    kept: list[Clip] = []
    for c in clips:
        if counts.get(c.video_id, 0) >= cap:
            continue
        kept.append(c)
        counts[c.video_id] = counts.get(c.video_id, 0) + 1
    return kept


def interleave_no_consec(clips: list[Clip]) -> list[Clip]:
    """Reorder clips so no two consecutive clips share the same video.

    Uses a greedy strategy: at each step pick the next highest-ranked clip
    whose ``video_id`` differs from the last selected clip.  Falls back to
    the head of the remaining pool if no alternative exists.

    Parameters
    ----------
    clips:
        Ordered candidate clips.

    Returns
    -------
    list[Clip]
        Reordered clips.
    """
    pool = list(clips)
    out: list[Clip] = []
    last_vid: str | None = None
    while pool:
        idx = next((i for i, c in enumerate(pool) if c.video_id != last_vid), 0)
        pick = pool.pop(idx)
        out.append(pick)
        last_vid = pick.video_id
    return out


def three_act_reorder(clips: list[Clip]) -> list[Clip]:
    """Apply a simple 3-act structure: hook → build → coda.

    The top ``hook_n`` clips by score form the opening hook.  The next
    ``coda_n`` form the closing coda.  The remaining middle clips are
    sorted by ascending ``face_area`` for visual variety.

    Parameters
    ----------
    clips:
        Adopted clips.

    Returns
    -------
    list[Clip]
        Reordered clips.
    """
    n = len(clips)
    if n < 10:
        return clips
    ranked = sorted(clips, key=lambda c: -c.judge_avg)
    hook_n = max(5, n // 10)
    coda_n = max(3, n // 15)
    hook = ranked[:hook_n]
    coda = ranked[hook_n: hook_n + coda_n]
    mid = ranked[hook_n + coda_n:]
    mid.sort(key=lambda c: c.face_area)
    return hook + mid + coda


# ---------------------------------------------------------------------------
# Main selection pipeline
# ---------------------------------------------------------------------------

def select_and_order(
    windows_path: Path,
    judged_path: Path,
    out_path: Path,
    *,
    target_sec: float = 270.0,
    tol_sec: float = 30.0,
    per_mv_cap_n: int = 5,
    iou_max: float = 0.3,
    min_gap_sec: float = 5.0,
    judge_avg_min: float = 6.0,
) -> list[Clip]:
    """Full selection pipeline: filter → NMS → cap → budget → order.

    Parameters
    ----------
    windows_path:
        Path to ``windows_seed.jsonl`` (candidate windows).
    judged_path:
        Path to ``judged.jsonl`` (aggregated judge output).
    out_path:
        Path to write ``curated.jsonl``.
    target_sec:
        Target total duration in seconds.
    tol_sec:
        Tolerance beyond the target before stopping adoption.
    per_mv_cap_n:
        Maximum clips per source video.
    iou_max:
        Temporal IoU threshold for NMS.
    min_gap_sec:
        Minimum center-to-center gap (seconds) within the same video.
    judge_avg_min:
        Minimum average judge score to include a clip.

    Returns
    -------
    list[Clip]
        Final ordered clips (also written to ``out_path``).
    """
    windows = _load_jsonl(windows_path)
    if not windows:
        # Fallback: try legacy path
        legacy = windows_path.parent / "windows.jsonl"
        windows = _load_jsonl(legacy)

    judged_map: dict[str, dict[str, Any]] = {
        j["stable_id"]: j for j in _load_jsonl(judged_path)
    }

    by_sid: dict[str, dict[str, Any]] = {}
    for w in windows:
        sid = w.get("stable_id") or (
            f"{w['video_id']}__s{w['shot_idx']:03d}__w{w['window_idx_in_shot']:02d}"
        )
        by_sid[sid] = w

    clips: list[Clip] = []
    for sid, w in by_sid.items():
        j = judged_map.get(sid)
        if j:
            if (j.get("judge_c") or {}).get("verdict", "accept") == "reject":
                continue
            a = float((j.get("judge_a") or {}).get("score", 0))
            b = float((j.get("judge_b") or {}).get("score", 0))
            avg = (a + b) / 2.0
            if avg < judge_avg_min:
                continue
        else:
            # Draft fallback: deterministic hash score, no quality signal
            h = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
            avg = 5.0 + (h % 1000) / 1000.0 * 4.0
            if avg < judge_avg_min:
                continue

        clips.append(Clip(
            stable_id=sid,
            video_id=w["video_id"],
            shot_idx=int(w.get("shot_idx", 0)),
            start=float(w["start"]),
            end=float(w["end"]),
            duration=float(w["duration"]),
            judge_avg=avg,
            face_area=float(w.get("face_area_ratio", 0)),
            face_ratio=float(w.get("max_face_ratio_h", 0)),
        ))

    log.info("candidate clips after accept filter: %d", len(clips))
    clips = temporal_nms(clips, iou_max=iou_max, min_gap_sec=min_gap_sec)
    log.info("after temporal NMS: %d", len(clips))
    clips = per_mv_cap(clips, cap=per_mv_cap_n)
    log.info("after per-video cap=%d: %d", per_mv_cap_n, len(clips))

    clips.sort(key=lambda c: -c.judge_avg)
    adopted: list[Clip] = []
    cum = 0.0
    for c in clips:
        adopted.append(c)
        cum += c.duration
        if cum >= target_sec + tol_sec:
            break
    log.info("adopted: %d clips, %.1fs", len(adopted), cum)

    ordered = three_act_reorder(adopted)
    ordered = interleave_no_consec(ordered)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(ordered):
            fh.write(json.dumps({
                "global_idx": i,
                "stable_id": c.stable_id,
                "video_id": c.video_id,
                "start": c.start,
                "end": c.end,
                "duration": c.duration,
                "judge_avg": c.judge_avg,
                "face_area_ratio": c.face_area,
            }, ensure_ascii=False) + "\n")
    tmp.replace(out_path)

    total = sum(c.duration for c in ordered)
    log.info("final: %s | %d clips | %.1fs (%.1f min)", out_path, len(ordered), total, total / 60)
    return ordered
