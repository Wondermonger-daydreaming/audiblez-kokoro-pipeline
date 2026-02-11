"""Shared pytest configuration for audiblez-kokoro-pipeline tests."""

import sys
from pathlib import Path


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require runtime dependencies "
        "(kokoro, torch, espeak-ng, ffmpeg). Deselect with: pytest -m 'not integration'",
    )


# Ensure the src directory is importable from all test modules
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
