# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: MIT
"""Tests for supercut_cascade.cli (3 tests)."""
from __future__ import annotations

import pytest

from supercut_cascade.cli import build_parser

# ---------------------------------------------------------------------------
# test_cli_version
# ---------------------------------------------------------------------------

def test_cli_version():
    """build_parser supports --version and exits with SystemExit(0)."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0, (
        f"--version must exit with code 0; got {exc_info.value.code}"
    )


# ---------------------------------------------------------------------------
# test_cli_help
# ---------------------------------------------------------------------------

def test_cli_help():
    """build_parser --help exits with SystemExit(0)."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0, (
        f"--help must exit with code 0; got {exc_info.value.code}"
    )


# ---------------------------------------------------------------------------
# test_cli_unknown_subcmd
# ---------------------------------------------------------------------------

def test_cli_unknown_subcmd():
    """An unknown subcommand causes a non-zero SystemExit."""
    from supercut_cascade.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["totally-unknown-subcommand-xyz"])
    assert exc_info.value.code != 0, (
        f"unknown subcommand must exit non-zero; got code {exc_info.value.code}"
    )
