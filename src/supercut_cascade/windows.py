# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Window builder: convert detection JSONL records into candidate time windows.

Two public helpers are provided:

* :func:`plan_windows_from_segment` — low-level sliding-window planner for a
  single ``[start, end]`` segment.
* :func:`build_windows` — high-level loader that reads detection JSONL files,
  applies a face-ratio gate, and returns :class:`Window` objects.

A relaxed variant ``build_windows_v2`` additionally accepts wide / no-face
shots and labels each window with an *intensity bucket*.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A candidate time window extracted from a single detected shot/segment.

    Attributes
    ----------
    video_id:
        Identifier of the source video (typically the file stem).
    idx:
        Global sequential index across all windows in the manifest.
    stable_id:
        Content-stable identifier: ``"{video_id}__s{shot:03d}__w{win:02d}"``.
    shot_idx:
        Index of the originating shot within the video.
    window_idx_in_shot:
        Index of this window within the shot (for sliding windows).
    start:
        Window start time in seconds.
    end:
        Window end time in seconds.
    duration:
        ``end - start`` in seconds.
    max_face_ratio_h:
        Maximum face-height-to-frame-height ratio sampled in this shot.
    face_area_ratio:
        Face bounding-box area as a fraction of frame area.
    soft_score:
        Shot-level closeup soft score (0–1).
    n_excluded:
        Number of frames excluded due to crowd (≥ N faces).
    edge_density:
        Canny edge density of the center crop (content proxy).
    gate_pass:
        Whether this window passed the face-ratio hard gate.
    gate_reasons:
        Human-readable list of gate decisions.
    """

    video_id: str
    idx: int
    stable_id: str
    shot_idx: int
    window_idx_in_shot: int
    start: float
    end: float
    duration: float
    max_face_ratio_h: float
    face_area_ratio: float
    soft_score: float
    n_excluded: int
    edge_density: float
    gate_pass: bool
    gate_reasons: list[str]


# ---------------------------------------------------------------------------
# Low-level planner
# ---------------------------------------------------------------------------


def plan_windows_from_segment(
    video_id: str,
    shot_idx: int,
    start: float,
    end: float,
    target_min: float,
    target_max: float,
    stride: float = 0.5,
    window_length: float = 1.0,
) -> list[tuple[float, float]]:
    """Generate sliding candidate windows for a single shot segment.

    For long segments (> ``window_length``), emits overlapping 1-second
    windows spaced ``stride`` seconds apart.  A final tail window is added
    when significant residual footage exists beyond the last regular window.

    For short segments (<= ``window_length``), emits at most one window
    covering the full segment, provided it is at least ``target_min * 0.5``
    seconds long.

    Parameters
    ----------
    video_id:
        Source video identifier (unused here; kept for API symmetry).
    shot_idx:
        Shot index within the video (unused here; kept for API symmetry).
    start:
        Segment start time in seconds.
    end:
        Segment end time in seconds.
    target_min:
        Minimum window duration for downstream judging (seconds).
    target_max:
        Maximum window duration (seconds).
    stride:
        Step between successive window start times (seconds).
    window_length:
        Nominal window length (seconds).

    Returns
    -------
    List of ``(start_sec, end_sec)`` tuples.
    """
    d = max(end - start, 0.0)
    if d <= 0.0:
        return []
    if d <= window_length:
        if d >= target_min * 0.5:
            return [(start, end)]
        return []

    windows: list[tuple[float, float]] = []
    t = start
    while t + window_length <= end + 1e-9:
        windows.append((t, t + window_length))
        t += stride

    if windows:
        last_end = windows[-1][1]
        if end - last_end >= target_min * 0.5:
            tail_start = max(end - window_length, start)
            tail_end = (
                end if end - tail_start <= target_max else tail_start + window_length
            )
            windows.append((tail_start, tail_end))

    return windows


# ---------------------------------------------------------------------------
# Gate helper
# ---------------------------------------------------------------------------


def _gate(
    max_face_ratio_h: float,
    face_area_ratio: float,
    edge_density: float,
    n_excluded: int,
    sampled: int,
    min_face_ratio: float,
    min_face_area: float,
    max_edge_density_for_blur: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    closeup_ok = (max_face_ratio_h >= min_face_ratio) or (face_area_ratio >= min_face_area)
    if not closeup_ok:
        reasons.append(
            f"closeup_fail face_h={max_face_ratio_h:.2f}<{min_face_ratio} "
            f"area={face_area_ratio:.2f}<{min_face_area}"
        )
        return False, reasons
    if sampled > 0 and (n_excluded / sampled) >= 0.5:
        reasons.append(f"crowd_shot n_excluded/sampled={n_excluded}/{sampled}")
        return False, reasons
    reasons.append("pass")
    return True, reasons


# ---------------------------------------------------------------------------
# High-level builders
# ---------------------------------------------------------------------------


def build_windows(
    jsonl_paths: Iterable[Path],
    *,
    min_face_ratio: float = 0.28,
    min_face_area: float = 0.10,
    target_min: float = 1.0,
    target_max: float = 1.5,
    stride: float = 0.5,
    window_length: float = 1.0,
    max_edge_density_for_blur: float = 0.95,
    log: logging.Logger | None = None,
) -> list[Window]:
    """Load detection JSONLs, apply face-ratio gate, split into windows.

    Reads the ``shots`` and ``segments`` arrays from each JSONL record.
    Per-segment metrics are taken from the nearest shot by midpoint.

    Parameters
    ----------
    jsonl_paths:
        Iterable of paths to per-video detection JSONL files.
    min_face_ratio:
        Minimum ``max_face_ratio_h`` for a segment to pass the gate.
    min_face_area:
        Minimum ``face_area_ratio`` (OR-gate with ``min_face_ratio``).
    target_min:
        Minimum window length for downstream judging (seconds).
    target_max:
        Maximum window length (seconds).
    stride:
        Sliding-window stride (seconds).
    window_length:
        Nominal window length (seconds).
    max_edge_density_for_blur:
        Reserved threshold; currently not used in gate logic.
    log:
        Optional logger; defaults to the module logger.

    Returns
    -------
    Flat list of :class:`Window` objects across all input JSONLs.
    """
    lg = log or logging.getLogger(__name__)
    out: list[Window] = []
    global_idx = 0

    for jp in sorted(jsonl_paths):
        try:
            rec = json.loads(Path(jp).read_text())
        except Exception as exc:
            lg.warning("bad jsonl %s: %s", jp, exc)
            continue

        vid = rec["meta"]["video_id"]
        shots = rec.get("shots", [])
        segments = rec.get("segments", [])

        for seg_idx, seg in enumerate(segments):
            s = float(seg["start"])
            e = float(seg["end"])

            nearest = None
            if shots:
                nearest = min(
                    shots,
                    key=lambda sh: abs(
                        (float(sh["start_sec"]) + float(sh["end_sec"])) / 2
                        - (s + e) / 2
                    ),
                )
            max_r = float(nearest.get("max_face_ratio_h", 0.0)) if nearest else 0.0
            area = float(nearest.get("face_area_ratio", 0.0)) if nearest else 0.0
            soft = float(nearest.get("soft_score", 0.0)) if nearest else 0.0
            edge = float(nearest.get("edge_density", 0.0)) if nearest else 0.0
            n_excluded = int(nearest.get("n_excluded", 0)) if nearest else 0
            sampled = int(nearest.get("sampled", 0)) if nearest else 0

            gate_pass, gate_reasons = _gate(
                max_r, area, edge, n_excluded, sampled,
                min_face_ratio, min_face_area, max_edge_density_for_blur,
            )
            if not gate_pass:
                continue

            for win_idx, (ws, we) in enumerate(
                plan_windows_from_segment(
                    vid, seg_idx, s, e, target_min, target_max,
                    stride=stride, window_length=window_length,
                )
            ):
                stable_id = f"{vid}__s{seg_idx:03d}__w{win_idx:02d}"
                out.append(
                    Window(
                        video_id=vid,
                        idx=global_idx,
                        stable_id=stable_id,
                        shot_idx=seg_idx,
                        window_idx_in_shot=win_idx,
                        start=ws,
                        end=we,
                        duration=we - ws,
                        max_face_ratio_h=max_r,
                        face_area_ratio=area,
                        soft_score=soft,
                        n_excluded=n_excluded,
                        edge_density=edge,
                        gate_pass=True,
                        gate_reasons=gate_reasons,
                    )
                )
                global_idx += 1

    return out


def _intensity_bucket(face_h: float, face_area: float) -> str:
    """Classify shot intensity based on face geometry."""
    if face_h >= 0.35 or face_area >= 0.18:
        return "strong"
    if face_h >= 0.20 or face_area >= 0.08:
        return "medium"
    if face_h >= 0.05 or face_area >= 0.01:
        return "wide"
    return "no_face"


def build_windows_v2(
    jsonl_paths: Iterable[Path],
    *,
    min_face_ratio: float = 0.05,
    min_face_area: float = 0.01,
    include_no_face: bool = False,
    max_crowd_ratio: float = 0.7,
    min_shot_dur: float = 0.5,
    max_shot_dur: float = 6.0,
    target_min: float = 1.0,
    target_max: float = 1.5,
    stride: float = 0.5,
    window_length: float = 1.0,
    log: logging.Logger | None = None,
) -> list[dict]:
    """Relaxed window builder that includes wide / no-face shots.

    Reads raw ``shots[]`` arrays (not post-merged ``segments[]``) from each
    JSONL record so that atmospheric and wide shots are retained.  Each
    output dict includes an ``intensity_bucket`` field (``"strong"``,
    ``"medium"``, ``"wide"``, or ``"no_face"``).

    Parameters
    ----------
    jsonl_paths:
        Iterable of paths to per-video detection JSONL files.
    min_face_ratio:
        Minimum face-height ratio when a face is present.
    min_face_area:
        Minimum face-area ratio (OR-gate) when a face is present.
    include_no_face:
        When ``True``, accept shots where no face was detected at all.
    max_crowd_ratio:
        Reject shots where ``n_excluded / sampled > max_crowd_ratio``.
    min_shot_dur:
        Skip shots shorter than this (seconds).
    max_shot_dur:
        Skip shots longer than this (seconds; rejects static/background cuts).
    target_min:
        Minimum window length (seconds).
    target_max:
        Maximum window length (seconds).
    stride:
        Sliding-window stride (seconds).
    window_length:
        Nominal window length (seconds).
    log:
        Optional logger.

    Returns
    -------
    List of window dicts with keys matching :class:`Window` plus
    ``intensity_bucket``.
    """
    lg = log or logging.getLogger(__name__)
    out: list[dict] = []
    global_idx = 0
    kept_buckets: dict[str, int] = {"strong": 0, "medium": 0, "wide": 0, "no_face": 0}
    rejected: dict[str, int] = {
        "too_short": 0, "too_long": 0, "crowd": 0, "closeup_fail": 0,
    }

    for jp in sorted(jsonl_paths):
        try:
            rec = json.loads(Path(jp).read_text())
        except Exception as exc:
            lg.warning("bad jsonl %s: %s", jp, exc)
            continue

        vid = rec["meta"]["video_id"]
        shots = rec.get("shots", [])

        for shot_idx, sh in enumerate(shots):
            s = float(sh.get("start_sec", 0))
            e = float(sh.get("end_sec", 0))
            dur = e - s

            if dur < min_shot_dur:
                rejected["too_short"] += 1
                continue
            if dur > max_shot_dur:
                rejected["too_long"] += 1
                continue

            face_h = float(sh.get("max_face_ratio_h", 0))
            face_area = float(sh.get("face_area_ratio", 0))
            soft = float(sh.get("soft_score", 0))
            edge = float(sh.get("edge_density", 0))
            n_excluded = int(sh.get("n_excluded", 0))
            sampled = int(sh.get("sampled", 1))

            if sampled > 0 and (n_excluded / sampled) > max_crowd_ratio:
                rejected["crowd"] += 1
                continue

            has_face = face_h >= 0.02 or face_area >= 0.005
            if has_face:
                if face_h < min_face_ratio and face_area < min_face_area:
                    rejected["closeup_fail"] += 1
                    continue
            else:
                if not include_no_face:
                    rejected["closeup_fail"] += 1
                    continue

            bucket = _intensity_bucket(face_h, face_area)
            kept_buckets[bucket] += 1

            windows = plan_windows_from_segment(
                vid, shot_idx, s, e, target_min, target_max,
                stride=stride, window_length=window_length,
            )
            for win_idx, (ws, we) in enumerate(windows):
                stable_id = f"{vid}__s{shot_idx:03d}__w{win_idx:02d}"
                out.append({
                    "video_id": vid,
                    "stable_id": stable_id,
                    "global_idx": global_idx,
                    "shot_idx": shot_idx,
                    "window_idx_in_shot": win_idx,
                    "start": ws,
                    "end": we,
                    "duration": we - ws,
                    "max_face_ratio_h": face_h,
                    "face_area_ratio": face_area,
                    "soft_score": soft,
                    "edge_density": edge,
                    "n_excluded": n_excluded,
                    "intensity_bucket": bucket,
                    "gate_reasons": ["pass_v2"],
                })
                global_idx += 1

    lg.info("kept shots by bucket: %s", kept_buckets)
    lg.info("rejected: %s", rejected)
    return out


__all__ = [
    "Window",
    "plan_windows_from_segment",
    "build_windows",
    "build_windows_v2",
]
