# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Final QA checks for the completed supercut video file.

Checks
------
1. Stream metadata: resolution, fps, pixel format via ``ffprobe``.
2. No unexpected audio streams.
3. Full decode scan: ``ffmpeg -v error -f null -`` (zero errors required).
4. Black-frame detection (center-crop to exclude letterbox pads).
5. Freeze-frame detection.
6. PTS monotonicity.
7. Duration bounds.

All results are aggregated into a report dict and optionally written to a
JSON file.

Usage
-----
>>> from supercut_cascade.qa import final_qa
>>> report = final_qa(Path("output/final.mp4"))
>>> assert report["pass"], report["failures"]
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FFPROBE_TIMEOUT: int = 120
FFMPEG_DETECT_TIMEOUT: int = 3600


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _ffprobe_streams(path: Path) -> dict[str, Any]:
    """Return parsed ffprobe JSON for the given file.

    Parameters
    ----------
    path:
        Path to the video file.

    Returns
    -------
    dict[str, Any]
        Parsed ffprobe output.

    Raises
    ------
    RuntimeError
        If ffprobe exits with a non-zero return code.
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False, timeout=FFPROBE_TIMEOUT,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:500]}")
    return json.loads(r.stdout)


def _run_filter(path: Path, vf: str) -> list[str]:
    """Run an ffmpeg filter pass and return lines containing event markers.

    Parameters
    ----------
    path:
        Path to the video file.
    vf:
        ``-vf`` filter string (e.g. ``"blackdetect=d=0.033:pix_th=0.10"``).

    Returns
    -------
    list[str]
        Matching stderr lines.
    """
    r = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-v", "info",
         "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=FFMPEG_DETECT_TIMEOUT,
    )
    return [
        line.strip()
        for line in r.stderr.splitlines()
        if "black_start" in line or "freeze_start" in line or "lavfi.freezedetect" in line
    ]


def _full_decode(path: Path) -> list[str]:
    """Full-decode scan; return error lines from stderr.

    Parameters
    ----------
    path:
        Path to the video file.

    Returns
    -------
    list[str]
        Non-empty stderr lines (each represents a decode error).
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=FFMPEG_DETECT_TIMEOUT,
    )
    return [line for line in r.stderr.splitlines() if line.strip()]


def _check_pts_monotonic(path: Path) -> list[str]:
    """Check that packet PTS values are strictly monotonic.

    Parameters
    ----------
    path:
        Path to the video file.

    Returns
    -------
    list[str]
        Descriptions of any non-monotonic PTS steps.
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_packets", "-show_entries", "packet=pts",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False, timeout=FFPROBE_TIMEOUT * 3,
    )
    issues: list[str] = []
    prev: int | None = None
    prev_idx: int = -1
    for idx, line in enumerate(r.stdout.splitlines()):
        line = line.strip()
        if not line or line == "N/A":
            continue
        try:
            pts = int(line)
        except ValueError:
            continue
        if prev is not None and pts <= prev:
            issues.append(
                f"pkt[{idx}] pts={pts} <= prev[{prev_idx}]={prev} (non-monotonic)"
            )
        prev, prev_idx = pts, idx
    return issues


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(report: dict[str, Any], path: Path) -> None:
    """Atomically write QA report JSON.

    Parameters
    ----------
    report:
        Report dict.
    path:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main QA function
# ---------------------------------------------------------------------------

def final_qa(
    video: Path,
    *,
    report_path: Path | None = None,
    expected_fps: int = 30,
    expected_width: int = 1920,
    expected_height: int = 1080,
    expected_pix_fmts: list[str] | None = None,
    max_black_hits: int = 0,
    max_freeze_hits: int = 0,
    max_decode_errors: int = 0,
    min_dur: float = 240.0,
    max_dur: float = 300.0,
    loose_duration: bool = False,
) -> dict[str, Any]:
    """Run all QA checks on the final supercut video.

    Parameters
    ----------
    video:
        Path to the final MP4 file.
    report_path:
        If provided, write the JSON report to this path atomically.
    expected_fps:
        Required frame rate.
    expected_width:
        Required video width in pixels.
    expected_height:
        Required video height in pixels.
    expected_pix_fmts:
        Acceptable pixel formats (defaults to ``["yuv420p", "yuv420p10le"]``).
    max_black_hits:
        Maximum allowed black-frame detection events (0 = strict).
    max_freeze_hits:
        Maximum allowed freeze-frame events (0 = strict).
    max_decode_errors:
        Maximum allowed decode errors from ffmpeg.
    min_dur:
        Minimum acceptable duration in seconds.
    max_dur:
        Maximum acceptable duration in seconds.
    loose_duration:
        If ``True``, relax duration bounds by ±60 seconds.

    Returns
    -------
    dict[str, Any]
        Report with keys ``pass`` (bool), ``failures`` (list[str]),
        ``warnings`` (list[str]), and ``checks`` (dict).
    """
    if expected_pix_fmts is None:
        expected_pix_fmts = ["yuv420p", "yuv420p10le"]

    report: dict[str, Any] = {
        "video": str(video),
        "checks": {},
        "pass": True,
        "warnings": [],
        "failures": [],
    }

    if not video.exists():
        log.error("video not found: %s", video)
        report["pass"] = False
        report["failures"].append(f"file not found: {video}")
        if report_path:
            _write_report(report, report_path)
        return report

    # ------------------------------------------------------------------
    # 1. Stream metadata
    # ------------------------------------------------------------------
    log.info("check 1: stream metadata ...")
    try:
        info = _ffprobe_streams(video)
    except RuntimeError as exc:
        log.error("ffprobe error: %s", exc)
        report["pass"] = False
        report["failures"].append(str(exc))
        if report_path:
            _write_report(report, report_path)
        return report

    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]

    if not vstreams:
        report["pass"] = False
        report["failures"].append("no video stream found")
    else:
        v = vstreams[0]
        w = int(v.get("width", 0))
        h = int(v.get("height", 0))
        pix_fmt = v.get("pix_fmt", "")
        codec = v.get("codec_name", "")
        num, den = v.get("r_frame_rate", "0/1").split("/")
        fps_val = float(num) / float(den) if float(den) else 0.0
        sar = v.get("sample_aspect_ratio", "1:1")
        dur = float(info.get("format", {}).get("duration", 0) or 0)
        nb_frames = v.get("nb_frames", "?")

        report["checks"]["stream"] = {
            "width": w, "height": h, "pix_fmt": pix_fmt, "codec": codec,
            "fps": fps_val, "sar": sar, "duration_sec": dur, "nb_frames": nb_frames,
        }
        log.info("stream: %dx%d %s/%s @ %.2ffps sar=%s dur=%.1fs nb_frames=%s",
                 w, h, codec, pix_fmt, fps_val, sar, dur, nb_frames)

        if w != expected_width or h != expected_height:
            report["failures"].append(
                f"resolution {w}x{h} != {expected_width}x{expected_height}"
            )
            report["pass"] = False
        if pix_fmt not in expected_pix_fmts:
            report["failures"].append(f"pix_fmt {pix_fmt!r} not in {expected_pix_fmts}")
            report["pass"] = False
        if abs(fps_val - expected_fps) > 0.01:
            report["failures"].append(f"fps {fps_val} != {expected_fps}")
            report["pass"] = False
        if sar not in ("1:1", "0:1", "1/1"):
            report["warnings"].append(f"non-square SAR: {sar}")

        lo = min_dur - (60 if loose_duration else 0)
        hi = max_dur + (60 if loose_duration else 0)
        report["checks"]["duration_sec"] = dur
        if not (lo <= dur <= hi):
            report["failures"].append(
                f"duration {dur:.1f}s outside [{lo:.0f}, {hi:.0f}]s"
            )
            report["pass"] = False
        else:
            log.info("duration %.1fs OK (bounds [%.0f, %.0f])", dur, lo, hi)

    # ------------------------------------------------------------------
    # 2. No audio
    # ------------------------------------------------------------------
    report["checks"]["audio_tracks"] = len(astreams)
    if astreams:
        report["failures"].append(f"unexpected audio streams: {len(astreams)}")
        report["pass"] = False
    else:
        log.info("no audio OK")

    # ------------------------------------------------------------------
    # 3. Full decode
    # ------------------------------------------------------------------
    log.info("check 3: full decode scan ...")
    decode_errors = _full_decode(video)
    report["checks"]["decode_errors"] = len(decode_errors)
    report["checks"]["decode_errors_sample"] = decode_errors[:20]
    if len(decode_errors) > max_decode_errors:
        report["failures"].append(f"decode errors: {len(decode_errors)}")
        report["pass"] = False
    else:
        log.info("full decode: %d errors OK", len(decode_errors))

    # ------------------------------------------------------------------
    # 4. Blackdetect (center crop)
    # ------------------------------------------------------------------
    log.info("check 4: blackdetect (center crop) ...")
    vf_black = "crop=iw*0.8:ih*0.8,blackdetect=d=0.033:pix_th=0.10"
    blacks = _run_filter(video, vf_black)
    report["checks"]["blacks_found"] = len(blacks)
    report["checks"]["blacks_sample"] = blacks[:50]
    if len(blacks) > max_black_hits:
        report["failures"].append(f"blackdetect hits: {len(blacks)}")
        report["pass"] = False
    else:
        log.info("blackdetect: %d hits OK", len(blacks))

    # ------------------------------------------------------------------
    # 5. Freezedetect
    # ------------------------------------------------------------------
    log.info("check 5: freezedetect ...")
    vf_freeze = "freezedetect=n=0.001:d=0.05"
    freezes = _run_filter(video, vf_freeze)
    freeze_events = sum(1 for x in freezes if "freeze_start" in x)
    report["checks"]["freeze_events"] = freeze_events
    report["checks"]["freeze_sample"] = freezes[:50]
    if freeze_events > max_freeze_hits:
        report["failures"].append(f"freeze events: {freeze_events}")
        report["pass"] = False
    else:
        log.info("freezedetect: %d events OK", freeze_events)

    # ------------------------------------------------------------------
    # 6. PTS monotonic
    # ------------------------------------------------------------------
    log.info("check 6: PTS monotonic ...")
    pts_issues = _check_pts_monotonic(video)
    report["checks"]["pts_issues"] = len(pts_issues)
    report["checks"]["pts_issues_sample"] = pts_issues[:20]
    if pts_issues:
        report["failures"].append(f"PTS non-monotonic: {len(pts_issues)} occurrences")
        report["pass"] = False
    else:
        log.info("PTS monotonic: OK")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if report_path:
        _write_report(report, report_path)
        log.info("qa report: %s", report_path)

    if report["failures"]:
        log.error("FAIL: %s", "; ".join(report["failures"]))
    if report["warnings"]:
        log.warning("WARN: %s", "; ".join(report["warnings"]))
    if report["pass"]:
        log.info("PASS all QA checks")

    return report
