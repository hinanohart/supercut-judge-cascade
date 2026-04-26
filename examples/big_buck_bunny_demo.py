"""Big Buck Bunny demo: end-to-end supercut generation from a CC-BY video.

Big Buck Bunny is licensed under the Creative Commons Attribution 3.0 license:
    https://creativecommons.org/licenses/by/3.0/

The video is NOT included in this repository.  Download it first:

    wget -c "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4" \
         -O /tmp/bbb_320x180.mp4

Or the 30-second clip used by the demo:

    ffmpeg -i /tmp/bbb_320x180.mp4 -t 30 -c copy /tmp/bbb_30s.mp4

Requirements
------------
    pip install supercut-judge-cascade[detect,vision]
    # Vision LLM: set ANTHROPIC_API_KEY or OPENAI_API_KEY
    ffmpeg  # must be on PATH

Usage::

    python examples/big_buck_bunny_demo.py \
        --video /tmp/bbb_30s.mp4 \
        --out /tmp/bbb_supercut.mp4 \
        --model claude-haiku-4-5 \
        --top-n 5

The script runs all three judge stages (C gate, A viewer, B editor) and
assembles the top-N windows into a supercut via ffmpeg concat.

Arguments
---------
--video     Path to the input video (download bbb_30s.mp4 first)
--out       Output supercut path (default /tmp/bbb_supercut.mp4)
--model     litellm model string for all judges (default claude-haiku-4-5)
--top-n     Number of clips in the final supercut (default 5)
--bucket-sec  Bucket size for temporal NMS (default 10.0 s)
--threshold-c  Minimum Judge C confidence to accept (default: accept on verdict)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bbb_demo")

BBB_URL = "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def detect_scenes(video_path: Path) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) scene intervals using PySceneDetect."""
    try:
        from scenedetect import detect, ContentDetector
    except ImportError as exc:
        raise SystemExit(
            "scenedetect required: pip install 'supercut-judge-cascade[detect]'"
        ) from exc

    scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
    intervals: list[tuple[float, float]] = []
    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        if end - start >= 0.5:
            intervals.append((start, end))
    log.info("detected %d scenes", len(intervals))
    return intervals


def extract_frame(video_path: Path, ts: float, tmpdir: Path) -> np.ndarray | None:
    """Extract a single frame at timestamp ts via ffmpeg."""
    try:
        import cv2
    except ImportError:
        return None
    out = tmpdir / f"frame_{ts:.3f}.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
         "-vframes", "1", "-q:v", "3", str(out)],
        capture_output=True, check=True,
    )
    if not out.exists():
        return None
    return cv2.imread(str(out))


def run_judge_c(judge, frames: list[np.ndarray], prompt_c: str, stable_id: str):
    """Run Stage C defect gate; return True if accepted."""
    from supercut_cascade.judge import JudgeResult
    try:
        result = judge.judge(frames, prompt_c, stable_id=stable_id, phase="C")
        return result.verdict == "accept"
    except Exception as exc:
        log.warning("judge C error for %s: %s", stable_id, exc)
        return True  # default accept on error


def run_judge_ab(judge_a, judge_b, frames: list[np.ndarray],
                 prompt_a: str, prompt_b: str, stable_id: str) -> float | None:
    """Run Stage A + B; return average score or None on error."""
    try:
        ra = judge_a.judge(frames, prompt_a, stable_id=stable_id, phase="A")
        rb = judge_b.judge(frames, prompt_b, stable_id=stable_id, phase="B")
        sa, sb = ra.score, rb.score
        if sa is not None and sb is not None:
            return round((sa + sb) / 2, 2)
    except Exception as exc:
        log.warning("judge A/B error for %s: %s", stable_id, exc)
    return None


def concat_clips(video_path: Path, windows: list[dict], out_path: Path) -> None:
    """Assemble selected windows into a supercut via ffmpeg concat."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        concat_list = tmpdir / "concat.txt"
        clips: list[Path] = []
        for i, w in enumerate(windows):
            clip = tmpdir / f"clip_{i:03d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(w["start"]), "-to", str(w["end"]),
                 "-i", str(video_path),
                 "-c", "copy", str(clip)],
                capture_output=True, check=True,
            )
            clips.append(clip)
        with concat_list.open("w") as fh:
            for c in clips:
                fh.write(f"file '{c}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(out_path)],
            capture_output=True, check=True,
        )
    log.info("supercut written: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Big Buck Bunny end-to-end demo")
    parser.add_argument("--video", type=Path, default=Path("/tmp/bbb_30s.mp4"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/bbb_supercut.mp4"))
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--bucket-sec", type=float, default=10.0)
    args = parser.parse_args()

    if not args.video.exists():
        log.error("Video not found: %s", args.video)
        log.error("Download with:")
        log.error("  wget -c '%s' -O /tmp/bbb_320x180.mp4", BBB_URL)
        log.error("  ffmpeg -i /tmp/bbb_320x180.mp4 -t 30 -c copy /tmp/bbb_30s.mp4")
        return 1

    if not check_ffmpeg():
        log.error("ffmpeg not found on PATH. Install ffmpeg first.")
        return 1

    # Load prompts
    prompt_dir = Path(__file__).parent.parent / "prompts"
    try:
        prompt_c = (prompt_dir / "judge_c.md").read_text(encoding="utf-8")
        prompt_a = (prompt_dir / "judge_a.md").read_text(encoding="utf-8")
        prompt_b = (prompt_dir / "judge_b.md").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        log.error("Prompt file not found: %s", exc)
        return 1

    try:
        from supercut_cascade.judge import VisionLLMJudge
        from supercut_cascade.select import BucketNMS
    except ImportError as exc:
        log.error("Import failed: %s", exc)
        return 1

    judge_c = VisionLLMJudge(model=args.model, max_tokens=256)
    judge_a = VisionLLMJudge(model=args.model, max_tokens=512)
    judge_b = VisionLLMJudge(model=args.model, max_tokens=512)

    scenes = detect_scenes(args.video)
    if not scenes:
        log.error("No scenes detected.")
        return 1

    candidates: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for idx, (start, end) in enumerate(scenes):
            mid = (start + end) / 2.0
            stable_id = f"bbb_{idx:04d}"

            frame = extract_frame(args.video, mid, tmpdir)
            if frame is None:
                log.warning("could not extract frame at %.2f s", mid)
                continue

            accepted = run_judge_c(judge_c, [frame], prompt_c, stable_id)
            if not accepted:
                log.info("  [C-reject] %s  t=%.2f", stable_id, mid)
                continue

            ab_avg = run_judge_ab(judge_a, judge_b, [frame], prompt_a, prompt_b, stable_id)
            log.info("  [C-accept] %s  t=%.2f  ab_avg=%s", stable_id, mid, ab_avg)
            candidates.append({
                "stable_id": stable_id,
                "start": start,
                "end": end,
                "ab_avg": ab_avg if ab_avg is not None else 0.0,
            })

    if not candidates:
        log.error("No candidates passed Stage C.")
        return 1

    nms = BucketNMS(bucket_sec=args.bucket_sec)
    selected = nms.select(candidates, top_n=args.top_n)
    log.info("selected %d windows for supercut", len(selected))

    if not selected:
        log.error("No windows selected after NMS.")
        return 1

    concat_clips(args.video, selected, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
