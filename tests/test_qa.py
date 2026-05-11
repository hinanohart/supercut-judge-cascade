# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Tests for supercut_cascade.qa (3 tests)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# test_final_qa_no_video
# ---------------------------------------------------------------------------

def test_final_qa_no_video(tmp_path):
    """final_qa returns pass=False with 'file not found' failure for missing video."""
    from supercut_cascade.qa import final_qa

    missing = tmp_path / "does_not_exist.mp4"
    report = final_qa(missing)

    assert report["pass"] is False, "missing video must cause pass=False"
    assert any("not found" in f for f in report["failures"]), (
        "failures must contain 'not found' message; got: " + str(report["failures"])
    )


# ---------------------------------------------------------------------------
# test_final_qa_report_atomic
# ---------------------------------------------------------------------------

def test_final_qa_report_atomic(tmp_path):
    """final_qa writes an atomic JSON report to report_path when video is missing."""
    from supercut_cascade.qa import final_qa

    missing = tmp_path / "missing.mp4"
    report_path = tmp_path / "qa_report.json"

    final_qa(missing, report_path=report_path)

    assert report_path.exists(), "report file must be written even for missing video"
    data = json.loads(report_path.read_text())
    assert "pass" in data, "report JSON must contain 'pass' key"
    assert "failures" in data, "report JSON must contain 'failures' key"
    assert isinstance(data["failures"], list), "'failures' must be a list"


# ---------------------------------------------------------------------------
# test_final_qa_pts_check
# ---------------------------------------------------------------------------

def test_final_qa_pts_check():
    """_check_pts_monotonic identifies non-monotonic PTS correctly."""
    from supercut_cascade import qa

    # Monotonic sequence: no issues
    monotonic_output = "100\n200\n300\n400\n"

    def _fake_run_monotonic(*args, **kwargs):
        m = MagicMock()
        m.stdout = monotonic_output
        return m

    with patch("supercut_cascade.qa.subprocess.run", side_effect=_fake_run_monotonic):
        issues = qa._check_pts_monotonic(Path("/fake/video.mp4"))
    assert issues == [], (
        "monotonic PTS sequence must produce no issues; got: " + str(issues)
    )

    # Non-monotonic: PTS goes backwards
    nonmono_output = "100\n200\n150\n300\n"

    def _fake_run_nonmono(*args, **kwargs):
        m = MagicMock()
        m.stdout = nonmono_output
        return m

    with patch("supercut_cascade.qa.subprocess.run", side_effect=_fake_run_nonmono):
        issues = qa._check_pts_monotonic(Path("/fake/video.mp4"))
    assert len(issues) > 0, (
        "non-monotonic PTS must produce at least one issue; got empty list"
    )
