"""Tests for scripts.analyze_data: CLI and VISDRONE_CLASS_NAMES."""

import subprocess
from pathlib import Path

import pytest


class TestAnalyzeDataMain:
    """Tests for scripts/analyze_data.py CLI."""

    def test_main_does_not_raise_on_missing_labels_dir(
        self, tmp_path: Path
    ) -> None:
        """Running the script with an empty dir exits 0 and prints split info."""
        project_root = Path(__file__).resolve().parent.parent
        script = project_root / "scripts" / "analyze_data.py"
        result = subprocess.run(
            ["python", str(script), "--data", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=10,
        )
        assert result.returncode == 0
        # Should mention train/valid/test or "No labels dir"
        out = result.stdout + result.stderr
        assert "train" in out or "valid" in out or "test" in out or "No labels" in out

    def test_visdrone_class_names_length(self) -> None:
        """VISDRONE_CLASS_NAMES has 10 elements (VisDrone dataset)."""
        from scripts.analyze_data import VISDRONE_CLASS_NAMES

        assert len(VISDRONE_CLASS_NAMES) == 10
        assert "car" in VISDRONE_CLASS_NAMES
        assert "pedestrian" in VISDRONE_CLASS_NAMES
