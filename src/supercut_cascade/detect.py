# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""PySceneDetect content-detector wrapper for scene boundary detection.

Requires the ``scenedetect`` extra::

    pip install "supercut-judge-cascade[scenedetect]"
"""
from __future__ import annotations

import logging
from pathlib import Path

from .exceptions import BackendUnavailableError

log = logging.getLogger(__name__)


def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect scene boundaries in a video file using PySceneDetect.

    Uses the ``ContentDetector`` algorithm (HSV histogram difference) to
    find cut points.  If no cuts are found, returns the entire video as a
    single scene.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    threshold:
        ContentDetector sensitivity threshold.  Lower values detect more
        cuts; higher values detect only hard cuts.  Typical range: 15-40.
    min_scene_len_sec:
        Minimum scene duration in seconds.  Scenes shorter than this are
        merged into their neighbour.

    Returns
    -------
    List of ``(start_sec, end_sec)`` tuples, one per detected scene.
    Empty list if the video cannot be opened or has zero duration.

    Raises
    ------
    BackendUnavailableError
        If ``scenedetect`` is not installed.
    OSError
        If the video file cannot be read.
    """
    try:
        from scenedetect import SceneManager, open_video  # type: ignore[import]
        from scenedetect.detectors import ContentDetector  # type: ignore[import]
    except ImportError as exc:
        raise BackendUnavailableError(
            "PySceneDetect is required for detect_scenes(). "
            "Install it with: pip install 'supercut-judge-cascade[scenedetect]'"
        ) from exc

    video = open_video(str(video_path))
    fps = max(float(video.frame_rate), 1.0)
    min_scene_len_frames = max(int(round(min_scene_len_sec * fps)), 1)

    sm = SceneManager()
    sm.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
    )
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()

    if not scenes:
        dur = float(video.duration.get_seconds()) if video.duration else 0.0
        if dur > 0:
            log.debug("no cuts detected; returning full video as single scene (%.2fs)", dur)
            return [(0.0, dur)]
        log.warning("no scenes detected and video duration is zero: %s", video_path)
        return []

    result = [(float(s.get_seconds()), float(e.get_seconds())) for s, e in scenes]
    log.debug("detected %d scenes in %s (threshold=%.1f)", len(result), video_path.name, threshold)
    return result


__all__ = ["detect_scenes"]
