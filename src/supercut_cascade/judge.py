# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""3-stage Vision LLM judge cascade (Cheap → Mid → Expensive).

Stages
------
C (Cheap / Gate)
    Defect detector: fast, cheap model.  Binary accept/reject on technical
    quality.  Only windows that pass stage C are eligible for A and B.

A (Mid / Viewer)
    Scene quality score from a viewer perspective.  1-10 integer.

B (Expensive / Editor)
    Scene quality score from an editor perspective.  1-10 integer, with
    sub-axis breakdown (C/M/N).

All models are addressed via *litellm*, making the provider (Anthropic,
OpenAI, Gemini, …) swappable at runtime.

Checkpoint I/O
--------------
Results are written atomically to ``<judgements_dir>/<stable_id>_<phase>.json``
so that partial runs can be resumed.  A checkpoint with ``"error": "parse"``
flags a retry-eligible failure.

Usage
-----
>>> from supercut_cascade.judge import VisionLLMJudge, plan_batch, ingest_result, aggregate_summary
>>> judge = VisionLLMJudge(model="claude-haiku-4-5")
>>> result = judge.judge(frame_bgr, prompt_template, stable_id="abc123")
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from supercut_cascade.exceptions import BackendUnavailableError, JudgeError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Phase = Literal["C", "A", "B"]
PHASES: tuple[Phase, ...] = ("C", "A", "B")


@dataclass
class JudgeResult:
    """Structured output from a single judge call.

    Parameters
    ----------
    stable_id:
        Unique identifier for the candidate window.
    phase:
        Judge phase that produced this result.
    payload:
        Raw parsed JSON returned by the model.
    error:
        If set to ``"parse"``, the result is a retry-eligible failure.
    """

    stable_id: str
    phase: Phase
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def verdict(self) -> str | None:
        """Phase-C verdict (``"accept"`` or ``"reject"``), or ``None``."""
        return self.payload.get("verdict")

    @property
    def score(self) -> float | None:
        """Phase-A / Phase-B numeric score (1-10), or ``None``."""
        v = self.payload.get("score")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def is_valid(self) -> bool:
        """Return ``True`` if the result is not a parse-error placeholder."""
        return self.error != "parse"


# ---------------------------------------------------------------------------
# VisionLLMJudge
# ---------------------------------------------------------------------------

class VisionLLMJudge:
    """Vision LLM judge that scores a single candidate frame.

    Uses *litellm* so that any supported Vision LLM provider can be
    addressed with the same interface.  ``litellm`` is imported lazily so
    that the module can be imported even when the library is not installed.

    Parameters
    ----------
    model:
        Any model string accepted by litellm, e.g. ``"claude-haiku-4-5"``,
        ``"gpt-4o-mini"``, or ``"gemini/gemini-1.5-flash"``.
    target_label:
        Generic placeholder for the subject of interest.  Replaces
        ``${TARGET_LABEL}`` in prompt templates.  Defaults to ``"TARGET"``.
    max_tokens:
        Maximum tokens to request from the model.
    temperature:
        Sampling temperature.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        *,
        target_label: str = "TARGET",
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        try:
            import litellm  # noqa: F401  lazy import check
        except ImportError as exc:
            raise BackendUnavailableError(
                "litellm is required for VisionLLMJudge: "
                "pip install 'supercut-judge-cascade[vision]'"
            ) from exc
        self._litellm_module = None  # loaded on demand in _call
        self.model = model
        self.target_label = target_label
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_litellm(self) -> Any:
        if self._litellm_module is None:
            import litellm
            self._litellm_module = litellm
        return self._litellm_module

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode a BGR numpy frame to base-64 JPEG string.

        Parameters
        ----------
        frame:
            HxWx3 uint8 array in BGR colour order (OpenCV convention).

        Returns
        -------
        str
            Base-64 encoded JPEG bytes.
        """
        try:
            import cv2
        except ImportError as exc:
            raise BackendUnavailableError(
                "opencv-python is required to encode frames: pip install opencv-python"
            ) from exc
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise JudgeError("cv2.imencode failed on provided frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _render_prompt(self, template: str, extra: dict[str, Any] | None = None) -> str:
        """Substitute ``${TARGET_LABEL}`` and optional keys in a prompt template.

        Parameters
        ----------
        template:
            Raw prompt markdown with ``${...}`` placeholders.
        extra:
            Additional key→value substitutions.

        Returns
        -------
        str
            Prompt with all placeholders replaced.
        """
        result = template.replace("${TARGET_LABEL}", self.target_label)
        if extra:
            for k, v in extra.items():
                result = result.replace(f"${{{k}}}", str(v))
        return result

    def _call(self, prompt_text: str, b64_images: list[str]) -> str:
        """Send prompt + images to the model and return raw text response.

        Parameters
        ----------
        prompt_text:
            Final rendered system/user prompt text.
        b64_images:
            List of base-64 encoded JPEG strings.

        Returns
        -------
        str
            Raw model response text.

        Raises
        ------
        JudgeError
            If the litellm call fails.
        """
        litellm = self._get_litellm()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for b64 in b64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise JudgeError(f"litellm call failed for model={self.model}: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(
        self,
        frames: np.ndarray | list[np.ndarray],
        prompt_template: str,
        stable_id: str,
        phase: Phase,
        extra_vars: dict[str, Any] | None = None,
    ) -> JudgeResult:
        """Score a candidate window using the configured Vision LLM.

        Parameters
        ----------
        frames:
            Single frame (HxWx3 uint8 BGR) or list of frames.
        prompt_template:
            Prompt markdown with optional ``${TARGET_LABEL}`` placeholders.
        stable_id:
            Identifier for this candidate, echoed in the result.
        phase:
            Judge phase (``"C"``, ``"A"``, or ``"B"``).
        extra_vars:
            Additional ``${KEY}`` → value substitutions for the template.

        Returns
        -------
        JudgeResult
            Structured result.  Check ``result.is_valid()`` before using.
        """
        if isinstance(frames, np.ndarray):
            frames = [frames]

        b64s = [self._encode_frame(f) for f in frames]
        prompt = self._render_prompt(prompt_template, extra_vars)
        raw = self._call(prompt, b64s)

        # Extract JSON from response (model may wrap in markdown)
        payload = _extract_json(raw, stable_id, phase)
        if payload is None:
            log.warning("parse error for stable_id=%s phase=%s raw=%.200s", stable_id, phase, raw)
            return JudgeResult(
                stable_id=stable_id,
                phase=phase,
                payload={"stable_id": stable_id, "error": "parse", "raw_response": raw[:500]},
                error="parse",
            )

        payload.setdefault("stable_id", stable_id)
        return JudgeResult(stable_id=stable_id, phase=phase, payload=payload)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str, stable_id: str, phase: Phase) -> dict[str, Any] | None:
    """Try to extract a JSON object from model response text.

    Parameters
    ----------
    text:
        Raw model response (may contain markdown fences).
    stable_id:
        Used only for logging context.
    phase:
        Used only for logging context.

    Returns
    -------
    dict or None
        Parsed dict, or ``None`` on failure.
    """
    # Strip markdown code fences
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                stripped = candidate
                break

    # Find first { ... } block
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def checkpoint_path(stable_id: str, phase: Phase, judgements_dir: Path) -> Path:
    """Return the canonical checkpoint file path.

    Parameters
    ----------
    stable_id:
        Candidate identifier.
    phase:
        Judge phase.
    judgements_dir:
        Root directory for checkpoint files.

    Returns
    -------
    Path
    """
    return judgements_dir / f"{stable_id}_{phase}.json"


def write_checkpoint(result: JudgeResult, judgements_dir: Path) -> Path:
    """Write a JudgeResult atomically to disk.

    Parameters
    ----------
    result:
        Judge result to persist.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    Path
        Path of the written checkpoint.
    """
    judgements_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_path(result.stable_id, result.phase, judgements_dir)
    tmp = dest.with_suffix(".json.tmp")
    payload = dict(result.payload)
    payload["judge_phase"] = result.phase
    if result.error:
        payload["error"] = result.error
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)
    log.info("checkpoint written: %s", dest)
    return dest


def read_checkpoint(
    stable_id: str, phase: Phase, judgements_dir: Path
) -> dict[str, Any] | None:
    """Load a checkpoint; return ``None`` if absent or parse-error.

    Parameters
    ----------
    stable_id:
        Candidate identifier.
    phase:
        Judge phase.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    dict or None
    """
    p = checkpoint_path(stable_id, phase, judgements_dir)
    if not p.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        if data.get("error") == "parse":
            return None
        return data
    except Exception as exc:
        log.warning("failed to load checkpoint %s: %s", p, exc)
        return None


def is_done(stable_id: str, phase: Phase, judgements_dir: Path) -> bool:
    """Return ``True`` if a valid (non-error) checkpoint exists.

    Parameters
    ----------
    stable_id:
        Candidate identifier.
    phase:
        Judge phase.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    bool
    """
    p = checkpoint_path(stable_id, phase, judgements_dir)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("error") != "parse"
    except Exception:
        return False


def is_c_accepted(stable_id: str, judgements_dir: Path) -> bool:
    """Return ``True`` iff the phase-C checkpoint records ``verdict=accept``.

    Parameters
    ----------
    stable_id:
        Candidate identifier.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    bool
    """
    p = checkpoint_path(stable_id, "C", judgements_dir)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("verdict") == "accept"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Plan batch
# ---------------------------------------------------------------------------

def plan_batch(
    seed_index: dict[str, dict[str, Any]],
    thumb_index: dict[str, dict[str, Any]],
    phase: Phase,
    prompt_text: str,
    plan_dir: Path,
    judgements_dir: Path,
    batch_size: int = 10,
) -> Path:
    """Write the next batch of unscored windows to a JSONL plan file.

    For phase ``"C"`` all unscored seed windows are eligible.  For phases
    ``"A"`` and ``"B"`` only windows with a phase-C ``accept`` checkpoint
    are eligible.

    Parameters
    ----------
    seed_index:
        Mapping of ``stable_id`` → window metadata from ``windows_seed.jsonl``.
    thumb_index:
        Mapping of ``stable_id`` → thumbnail manifest row.
    phase:
        Judge phase to plan.
    prompt_text:
        Contents of the prompt template for this phase.
    plan_dir:
        Directory for plan batch files.
    judgements_dir:
        Directory for checkpoint files (used to skip already-done windows).
    batch_size:
        Maximum number of windows per batch file.

    Returns
    -------
    Path
        Path of the written batch file (may be empty if no pending windows).
    """
    plan_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for sid, win in seed_index.items():
        if is_done(sid, phase, judgements_dir):
            continue
        if phase in ("A", "B") and not is_c_accepted(sid, judgements_dir):
            continue
        manifest_row = thumb_index.get(sid, {})
        candidates.append({
            "stable_id": sid,
            "video_id": win["video_id"],
            "start": win["start"],
            "end": win["end"],
            "duration": win.get("duration", win["end"] - win["start"]),
            "thumb_paths": manifest_row.get("thumb_paths", []),
            "contact_sheet": manifest_row.get("contact_sheet", ""),
            "judge_phase": phase,
            "prompt_text": prompt_text,
        })

    existing = sorted(plan_dir.glob("batch_*.jsonl"))
    next_n = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 1
    out = plan_dir / f"batch_{next_n:03d}.jsonl"

    batch = candidates[:batch_size]
    with out.open("w", encoding="utf-8") as fh:
        for row in batch:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info(
        "plan written: %s  phase=%s  rows=%d  pending_total=%d",
        out, phase, len(batch), len(candidates),
    )
    return out


# ---------------------------------------------------------------------------
# Ingest result
# ---------------------------------------------------------------------------

def ingest_result(
    phase: Phase,
    json_str: str,
    judgements_dir: Path,
) -> JudgeResult:
    """Parse a sub-agent JSON result string and write a checkpoint.

    Parameters
    ----------
    phase:
        Judge phase.
    json_str:
        Raw JSON string returned by the sub-agent.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    JudgeResult
        Parsed result (may have ``error="parse"`` on failure).

    Raises
    ------
    JudgeError
        If the JSON cannot be parsed at all and no ``stable_id`` can be
        recovered.
    """
    try:
        payload: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"JSON parse failed: {exc} | input={json_str[:200]}") from exc

    stable_id = payload.get("stable_id")
    if not stable_id:
        raise JudgeError(f"ingest payload missing 'stable_id': {json_str[:200]}")

    error: str | None = None
    required = {"C": "verdict", "A": "score", "B": "score"}
    if required[phase] not in payload:
        log.warning("phase %s payload missing '%s', marking parse error: %s",
                    phase, required[phase], stable_id)
        payload = {"stable_id": stable_id, "judge_phase": phase,
                   "error": "parse", "raw": payload}
        error = "parse"

    result = JudgeResult(stable_id=stable_id, phase=phase, payload=payload, error=error)
    write_checkpoint(result, judgements_dir)
    return result


def ingest_file(phase: Phase, path: Path, judgements_dir: Path) -> JudgeResult:
    """Read a result JSON file and ingest it.

    Parameters
    ----------
    phase:
        Judge phase.
    path:
        Path to the result JSON file.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    JudgeResult
    """
    return ingest_result(phase, path.read_text(encoding="utf-8"), judgements_dir)


def ingest_batch_dir(phase: Phase, batch_dir: Path, judgements_dir: Path) -> list[JudgeResult]:
    """Ingest all ``*.json`` files in a directory.

    Parameters
    ----------
    phase:
        Judge phase.
    batch_dir:
        Directory containing result JSON files.
    judgements_dir:
        Directory for checkpoint files.

    Returns
    -------
    list[JudgeResult]
        All ingested results.
    """
    results: list[JudgeResult] = []
    for fp in sorted(batch_dir.glob("*.json")):
        try:
            results.append(ingest_file(phase, fp, judgements_dir))
        except JudgeError as exc:
            log.error("ingest failed for %s: %s", fp, exc)
    return results


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def aggregate_summary(
    seed_index: dict[str, dict[str, Any]],
    judgements_dir: Path,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Aggregate all checkpoints into a list of judged rows.

    Each row contains:
    ``stable_id``, ``video_id``, ``start``, ``end``, ``duration``,
    ``judge_c``, ``judge_a``, ``judge_b``, ``c_accepted``, ``ab_avg``.

    Parameters
    ----------
    seed_index:
        Mapping of ``stable_id`` → window metadata.
    judgements_dir:
        Directory for checkpoint files.
    out_path:
        If provided, write the rows as JSONL to this path (atomic).

    Returns
    -------
    list[dict[str, Any]]
        Sorted by ``stable_id``.
    """
    rows: list[dict[str, Any]] = []
    for sid, win in seed_index.items():
        judge_c = read_checkpoint(sid, "C", judgements_dir)
        judge_a = read_checkpoint(sid, "A", judgements_dir)
        judge_b = read_checkpoint(sid, "B", judgements_dir)

        c_accepted = judge_c is not None and judge_c.get("verdict") == "accept"

        ab_avg: float | None = None
        score_a = judge_a.get("score") if judge_a else None
        score_b = judge_b.get("score") if judge_b else None
        if score_a is not None and score_b is not None:
            try:
                ab_avg = round((float(score_a) + float(score_b)) / 2, 2)
            except (TypeError, ValueError):
                pass

        rows.append({
            "stable_id": sid,
            "video_id": win["video_id"],
            "start": win["start"],
            "end": win["end"],
            "duration": win.get("duration", win["end"] - win["start"]),
            "judge_c": judge_c,
            "judge_a": judge_a,
            "judge_b": judge_b,
            "c_accepted": c_accepted,
            "ab_avg": ab_avg,
        })

    rows.sort(key=lambda r: r["stable_id"])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out_path)
        log.info("judged.jsonl written: %s  rows=%d", out_path, len(rows))

    return rows


def progress_status(
    seed_index: dict[str, dict[str, Any]],
    judgements_dir: Path,
    plan_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a progress summary dict across all phases.

    Parameters
    ----------
    seed_index:
        Mapping of ``stable_id`` → window metadata.
    judgements_dir:
        Directory for checkpoint files.
    plan_dir:
        If provided, count existing batch plan files.

    Returns
    -------
    dict[str, Any]
        Summary with ``total_seed_windows`` and per-phase stats.
    """
    total = len(seed_index)
    summary: dict[str, Any] = {"total_seed_windows": total, "phases": {}}

    for phase in PHASES:
        done = errors = accepted = rejected = 0
        for sid in seed_index:
            p = checkpoint_path(sid, phase, judgements_dir)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if data.get("error") == "parse":
                        errors += 1
                    else:
                        done += 1
                        if phase == "C":
                            if data.get("verdict") == "accept":
                                accepted += 1
                            else:
                                rejected += 1
                except Exception:
                    errors += 1
        info: dict[str, Any] = {
            "done": done, "errors": errors, "pending": total - done - errors,
        }
        if phase == "C":
            info["accepted"] = accepted
            info["rejected"] = rejected
        summary["phases"][phase] = info

    if plan_dir is not None and plan_dir.exists():
        summary["plan_batches"] = len(sorted(plan_dir.glob("batch_*.jsonl")))

    return summary
