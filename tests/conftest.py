"""Shared pytest configuration for audiblez-kokoro-pipeline tests."""

import importlib.util
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require runtime dependencies "
        "(kokoro, torch, espeak-ng, ffmpeg). Deselect with: pytest -m 'not integration'",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.integration tests when kokoro is not installed.

    The integration tests document themselves as "skipped by default in CI";
    this implements that contract so a kokoro-free environment reports them as
    skipped rather than failed.
    """
    if importlib.util.find_spec("kokoro") is not None:
        return
    skip_no_kokoro = pytest.mark.skip(reason="kokoro not installed (integration test)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_no_kokoro)


# Ensure the src directory is importable from all test modules
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
