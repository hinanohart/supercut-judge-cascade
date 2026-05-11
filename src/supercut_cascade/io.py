# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""FFmpeg subprocess wrapper for video I/O.

All functions use ``subprocess.run`` with ``shell=False`` to prevent shell
injection.  Arguments are always passed as a list.  FFmpeg / ffprobe absence
raises :exc:`FileNotFoundError` via the subprocess layer.

Functions
---------
extract_frame(video_path, timestamp)
    Decode a single frame at the given timestamp; return BGR ndarray.
cut_clip(video_path, start, end, output_path)
    Extract a time-delimited clip without re-encoding.
concat_clips(clips, output)
    Concatenate a list of clips using the ffmpeg concat demuxer.
probe_duration(video_path)
    Return video duration in seconds via ffprobe.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 600  # seconds
_FFPROBE_TIMEOUT = 60  # seconds


def _run(cmd: list[str], timeout: int, check: bool = True) -> subprocess.CompletedProcess:
    """Run *cmd* with ``shell=False``; raise on non-zero exit when *check=True*.

    Parameters
    ----------
    cmd:
        Command as a list of strings (never joined into a shell string).
    timeout:
        Timeout in seconds.
    check:
        If ``True``, raise :exc:`subprocess.CalledProcessError` on failure.

    Returns
    -------
    subprocess.CompletedProcess
    """
    log.debug("run: %s", cmd)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        check=check,
        timeout=timeout,
        shell=False,
    )


def probe_duration(video_path: Path | str) -> float:
    """Return the duration of *video_path* in seconds.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    FileNotFoundError
        If ``ffprobe`` is not on PATH.
    RuntimeError
        If ffprobe returns a non-zero exit code or the duration cannot be parsed.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = _run(cmd, timeout=_FFPROBE_TIMEOUT, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffprobe not found on PATH; install ffmpeg to use supercut_cascade.io"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse ffprobe duration output: {exc}") from exc


def extract_frame(video_path: Path | str, timestamp: float) -> np.ndarray:
    """Decode a single frame at *timestamp* seconds from *video_path*.

    Uses ``-ss`` before ``-i`` (fast seek) then ``-ss 0`` after ``-i`` for
    frame-accurate extraction.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    timestamp:
        Target timestamp in seconds.

    Returns
    -------
    np.ndarray
        BGR image array of shape ``(H, W, 3)``, dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``ffmpeg`` is not on PATH.
    RuntimeError
        If ffmpeg exits with an error or the output is empty.
    """
    cmd = [
        "ffmpeg",
        "-ss", f"{timestamp:.6f}",
        "-i", str(video_path),
        "-ss", "0",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    try:
        result = _run(cmd, timeout=_FFMPEG_TIMEOUT, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg not found on PATH; install ffmpeg to use supercut_cascade.io"
        ) from exc

    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            f"extract_frame failed at t={timestamp:.3f}s "
            f"(rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )

    # Determine H×W from ffprobe so we can reshape the raw bytes
    dur_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    probe = _run(dur_cmd, timeout=_FFPROBE_TIMEOUT, check=False)
    try:
        info = json.loads(probe.stdout)
        stream = info["streams"][0]
        h = int(stream["height"])
        w = int(stream["width"])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not determine video dimensions: {exc}") from exc

    expected = h * w * 3
    raw = result.stdout
    if len(raw) < expected:
        raise RuntimeError(
            f"extract_frame: received {len(raw)} bytes, expected {expected} "
            f"({w}x{h}x3) at t={timestamp:.3f}s"
        )

    frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(h, w, 3)
    return frame.copy()


def cut_clip(
    video_path: Path | str,
    start: float,
    end: float,
    output_path: Path | str,
) -> None:
    """Extract a clip from *video_path* using stream copy (no re-encode).

    Parameters
    ----------
    video_path:
        Path to the source video.
    start:
        Clip start time in seconds.
    end:
        Clip end time in seconds.
    output_path:
        Destination file path.  Parent directory must exist.

    Raises
    ------
    FileNotFoundError
        If ``ffmpeg`` is not on PATH.
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    ValueError
        If ``end <= start``.
    """
    if end <= start:
        raise ValueError(f"cut_clip: end ({end}) must be greater than start ({start})")

    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.6f}",
        "-i", str(video_path),
        "-t", f"{duration:.6f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    try:
        result = _run(cmd, timeout=_FFMPEG_TIMEOUT, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg not found on PATH; install ffmpeg to use supercut_cascade.io"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"cut_clip failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )

    log.debug("cut_clip: %s -> %s [%.3f-%.3f]", video_path, output_path, start, end)


def concat_clips(clips: list[Path], output: Path) -> None:
    """Concatenate *clips* into *output* using the ffmpeg concat demuxer.

    Writes a temporary concat list file, then runs ffmpeg with ``-f concat``
    and stream copy.  The temporary file is removed after the call.

    Parameters
    ----------
    clips:
        Ordered list of clip paths to concatenate.
    output:
        Destination file path.  Parent directory must exist.

    Raises
    ------
    FileNotFoundError
        If ``ffmpeg`` is not on PATH.
    ValueError
        If *clips* is empty.
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    """
    if not clips:
        raise ValueError("concat_clips: clips list must not be empty")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        list_path = Path(fh.name)
        for clip in clips:
            # ffmpeg concat list format requires escaping single quotes in path
            safe = str(clip).replace("'", "'\\''")
            fh.write(f"file '{safe}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output),
        ]
        try:
            result = _run(cmd, timeout=_FFMPEG_TIMEOUT, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "ffmpeg not found on PATH; install ffmpeg to use supercut_cascade.io"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"concat_clips failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:500]}"
            )
    finally:
        list_path.unlink(missing_ok=True)

    log.info("concat_clips: %d clips -> %s", len(clips), output)


__all__ = ["extract_frame", "cut_clip", "concat_clips", "probe_duration"]
