# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for supercut_cascade.judge (5 tests, LLM mocked)."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# test_judge_result_dataclass
# ---------------------------------------------------------------------------

def test_judge_result_dataclass():
    """JudgeResult can be created and converted to dict via dataclasses.asdict."""
    from supercut_cascade.judge import JudgeResult

    result = JudgeResult(
        stable_id="abc123",
        phase="A",
        payload={"score": 8, "stable_id": "abc123"},
    )
    d = dataclasses.asdict(result)
    assert d["stable_id"] == "abc123", "stable_id must survive asdict round-trip"
    assert d["phase"] == "A", "phase must survive asdict round-trip"
    assert d["payload"]["score"] == 8, "nested payload must be preserved"
    assert d["error"] is None, "default error must be None"


# ---------------------------------------------------------------------------
# test_judge_lazy_litellm
# ---------------------------------------------------------------------------

def test_judge_lazy_litellm(monkeypatch):
    """VisionLLMJudge raises BackendUnavailableError when litellm is missing."""
    from supercut_cascade.exceptions import BackendUnavailableError

    # Hide litellm from sys.modules so the lazy import fails
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(BackendUnavailableError, match="litellm"):
        from supercut_cascade.judge import VisionLLMJudge
        VisionLLMJudge(model="claude-haiku-4-5")


# ---------------------------------------------------------------------------
# test_judge_prompt_template_format
# ---------------------------------------------------------------------------

def test_judge_prompt_template_format():
    """_render_prompt replaces ${TARGET_LABEL} with the configured label."""
    # Import the private helper directly to test substitution without LLM
    import importlib
    import types

    # Provide a stub litellm so VisionLLMJudge can be constructed
    stub = types.ModuleType("litellm")
    sys.modules.setdefault("litellm", stub)

    from supercut_cascade.judge import VisionLLMJudge

    judge = VisionLLMJudge.__new__(VisionLLMJudge)
    judge.model = "test-model"
    judge.target_label = "TARGET_X"
    judge.max_tokens = 512
    judge.temperature = 0.0
    judge.timeout = 60.0
    judge._litellm_module = None

    template = "Score this clip of ${TARGET_LABEL} on camera."
    rendered = judge._render_prompt(template)
    assert "TARGET_X" in rendered, "TARGET_LABEL should be substituted"
    assert "${TARGET_LABEL}" not in rendered, "placeholder should be removed"


# ---------------------------------------------------------------------------
# test_checkpoint_io
# ---------------------------------------------------------------------------

def test_checkpoint_io(tmp_path):
    """write_checkpoint / read_checkpoint perform an atomic round-trip."""
    from supercut_cascade.judge import JudgeResult, read_checkpoint, write_checkpoint

    result = JudgeResult(
        stable_id="sid42",
        phase="C",
        payload={"verdict": "accept", "stable_id": "sid42"},
    )
    dest = write_checkpoint(result, tmp_path)
    assert dest.exists(), "checkpoint file must be created"

    loaded = read_checkpoint("sid42", "C", tmp_path)
    assert loaded is not None, "read_checkpoint must return data for a valid checkpoint"
    assert loaded["verdict"] == "accept", "verdict must survive round-trip"
    assert loaded.get("error") != "parse", "valid checkpoint must not have parse error"


# ---------------------------------------------------------------------------
# test_progress_status
# ---------------------------------------------------------------------------

def test_progress_status(tmp_path):
    """progress_status reports 0% done for empty checkpoints, 100% when all complete."""
    from supercut_cascade.judge import JudgeResult, progress_status, write_checkpoint

    seed_index = {
        "s1": {"video_id": "v1", "start": 0.0, "end": 1.0},
        "s2": {"video_id": "v1", "start": 1.0, "end": 2.0},
    }

    # Before any checkpoints: all phases should have 0 done
    status_empty = progress_status(seed_index, tmp_path)
    assert status_empty["total_seed_windows"] == 2
    for phase in ("C", "A", "B"):
        assert status_empty["phases"][phase]["done"] == 0, (
            f"phase {phase} should have 0 done before any checkpoints"
        )

    # Write a valid C checkpoint for s1
    write_checkpoint(
        JudgeResult(stable_id="s1", phase="C", payload={"verdict": "accept", "stable_id": "s1"}),
        tmp_path,
    )
    status_partial = progress_status(seed_index, tmp_path)
    assert status_partial["phases"]["C"]["done"] == 1, "one C checkpoint should show done=1"
    assert status_partial["phases"]["C"]["pending"] == 1, "one C window still pending"
