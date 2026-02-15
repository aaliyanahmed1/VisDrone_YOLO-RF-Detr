"""Pytest fixtures for PPE Demo tests.

Provides project_root for script tests. Tests are isolated and do not
depend on external state.
"""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory (parent of tests/).

    Returns:
        Path: Project root for resolving script and data paths.
    """
    return Path(__file__).resolve().parent.parent
