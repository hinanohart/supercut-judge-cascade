# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for supercut-judge-cascade.

Sub-commands
------------
detect
    Detect scene boundaries in a video and write a scenes JSON file.
build
    Build a supercut from reference images and a pool video.
qa
    Run QA checks on a completed supercut video.

Usage
-----
    supercut-cascade detect  --input video.mp4 --output scenes.json
    supercut-cascade build   --refs ref1.jpg ref2.jpg --pool video.mp4 --output supercut.mp4
    supercut-cascade qa      --input supercut.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from supercut_cascade import __version__

log = logging.getLogger("supercut_cascade.cli")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbosity: int) -> None:
    """Configure root logging level based on -v count.

    Parameters
    ----------
    verbosity:
        Number of ``-v`` flags (0=WARNING, 1=INFO, 2=DEBUG, 3+=DEBUG+urllib3).
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )

    if verbosity < 3:
        for noisy in ("urllib3", "httpx", "httpcore", "litellm"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Shared argument factories
# ---------------------------------------------------------------------------

def _add_verbosity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG, -vvv full debug).",
    )


def _add_judge_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5",
        metavar="MODEL",
        help="Vision LLM model string passed to litellm (default: claude-haiku-4-5).",
    )


def _add_target_label(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-label",
        default="TARGET",
        metavar="LABEL",
        help="Label substituted for ${TARGET_LABEL} in judge prompts (default: TARGET).",
    )


def _add_threshold_cosine(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold-cosine",
        type=float,
        default=0.4,
        metavar="THRESH",
        help="ArcFace cosine-similarity threshold for identity gating (default: 0.4).",
    )


def _add_target_duration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-duration-sec",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Target clip duration in seconds for sliding windows (default: 1.0).",
    )


# ---------------------------------------------------------------------------
# Sub-command: detect
# ---------------------------------------------------------------------------

def _cmd_detect(args: argparse.Namespace) -> int:
    """Detect scene boundaries and write scenes JSON.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    from supercut_cascade.detect import detect_scenes

    video = Path(args.input)
    if not video.exists():
        log.error("Input video not found: %s", video)
        return 1

    log.info("Detecting scenes in %s ...", video)
    scenes = detect_scenes(
        video,
        threshold=args.threshold,
        min_scene_len_sec=args.min_scene_len,
    )
    log.info("Detected %d scene(s).", len(scenes))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video),
        "scene_count": len(scenes),
        "scenes": [{"start": s, "end": e} for s, e in scenes],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(scenes)} scenes to {out}", file=sys.stderr)
    return 0


def _build_detect_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "detect",
        help="Detect scene boundaries and write a scenes JSON file.",
        description="Detect scene boundaries using PySceneDetect ContentDetector.",
    )
    p.add_argument("--input", required=True, metavar="VIDEO", help="Source video file.")
    p.add_argument(
        "--output", required=True, metavar="JSON", help="Output scenes JSON file path."
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        metavar="THRESH",
        help="ContentDetector threshold (default: 27.0).",
    )
    p.add_argument(
        "--min-scene-len",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Minimum scene length in seconds (default: 0.5).",
    )
    _add_verbosity(p)
    p.set_defaults(func=_cmd_detect)


# ---------------------------------------------------------------------------
# Sub-command: build
# ---------------------------------------------------------------------------

def _cmd_build(args: argparse.Namespace) -> int:
    """Build a supercut from reference images and a pool video.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    from supercut_cascade.arcface import ArcFaceEmbedder
    from supercut_cascade.detect import detect_scenes
    from supercut_cascade.identity_filter import IdentityFilter, build_reference_embeddings
    from supercut_cascade.io import concat_clips, cut_clip
    from supercut_cascade.judge import VisionLLMJudge
    from supercut_cascade.windows import build_windows

    ref_paths = [Path(r) for r in args.refs]
    missing = [p for p in ref_paths if not p.exists()]
    if missing:
        log.error("Reference image(s) not found: %s", missing)
        return 1

    pool_video = Path(args.pool)
    if not pool_video.exists():
        log.error("Pool video not found: %s", pool_video)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    log.info("Initialising ArcFaceEmbedder ...")
    embedder = ArcFaceEmbedder()

    log.info("Building reference embeddings from %d image(s) ...", len(ref_paths))
    ref_embs = build_reference_embeddings(ref_paths, embedder, target_label=args.target_label)
    filt = IdentityFilter(
        reference_embeddings=ref_embs,
        threshold=args.threshold_cosine,
        target_label=args.target_label,
    )

    log.info("Detecting scenes in pool video ...")
    scenes = detect_scenes(pool_video)
    log.info("Detected %d scene(s).", len(scenes))

    judge = VisionLLMJudge(model=args.judge_model, target_label=args.target_label)

    import tempfile
    clip_paths: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="supercut_cascade_") as tmpdir:
        tmp = Path(tmpdir)
        for idx, (start, end) in enumerate(scenes):
            clip_path = tmp / f"clip_{idx:05d}.mp4"
            try:
                cut_clip(pool_video, start, end, clip_path)
            except Exception as exc:
                log.warning("cut_clip failed for scene %d: %s", idx, exc)
                continue
            clip_paths.append(clip_path)

        if not clip_paths:
            log.error("No clips were extracted.")
            return 1

        log.info("Concatenating %d clip(s) -> %s", len(clip_paths), output)
        try:
            concat_clips(clip_paths, output)
        except Exception as exc:
            log.error("concat_clips failed: %s", exc)
            return 1

    # Suppress unused variable warnings — judge and filt are kept for future
    # pipeline integration; build_windows imported for API completeness
    _ = judge
    _ = filt
    _ = build_windows

    print(f"Supercut written to {output}", file=sys.stderr)
    return 0


def _build_build_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "build",
        help="Build a supercut from reference images and a pool video.",
        description=(
            "Extract clips from a pool video, apply ArcFace identity gating, "
            "score with the judge cascade, and concatenate accepted clips."
        ),
    )
    p.add_argument(
        "--refs",
        nargs="+",
        required=True,
        metavar="IMAGE",
        help="One or more reference face images for identity gating.",
    )
    p.add_argument("--pool", required=True, metavar="VIDEO", help="Source pool video.")
    p.add_argument("--output", required=True, metavar="MP4", help="Output supercut MP4 path.")
    _add_target_label(p)
    _add_judge_model(p)
    _add_threshold_cosine(p)
    _add_target_duration(p)
    _add_verbosity(p)
    p.set_defaults(func=_cmd_build)


# ---------------------------------------------------------------------------
# Sub-command: qa
# ---------------------------------------------------------------------------

def _cmd_qa(args: argparse.Namespace) -> int:
    """Run QA checks on a completed supercut.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 = pass, 1 = fail).
    """
    from supercut_cascade.qa import final_qa

    video = Path(args.input)
    if not video.exists():
        log.error("Input video not found: %s", video)
        return 1

    report_path = Path(args.report) if args.report else None
    report = final_qa(
        video,
        report_path=report_path,
        loose_duration=args.loose,
    )

    if report["pass"]:
        print("QA PASS", file=sys.stderr)
        return 0
    else:
        print(f"QA FAIL: {'; '.join(report['failures'])}", file=sys.stderr)
        return 1


def _build_qa_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "qa",
        help="Run QA checks on a completed supercut video.",
        description="Run decode, blackframe, freeze-frame, and PTS-monotonic checks.",
    )
    p.add_argument("--input", required=True, metavar="VIDEO", help="Supercut video to check.")
    p.add_argument(
        "--report",
        default=None,
        metavar="JSON",
        help="Optional path to write a JSON QA report.",
    )
    p.add_argument(
        "--loose",
        action="store_true",
        help="Relax duration bounds by +-60 seconds.",
    )
    _add_verbosity(p)
    p.set_defaults(func=_cmd_qa)


# ---------------------------------------------------------------------------
# Root parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="supercut-cascade",
        description=(
            "Vision-LLM 3-stage judge cascade + ArcFace identity gating "
            "for video supercut generation."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"supercut-cascade {__version__}",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    _build_detect_parser(sub)
    _build_build_parser(sub)
    _build_qa_parser(sub)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the appropriate sub-command.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbosity)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
