"""Tests for artifact publisher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ratiocinator.synthesis.publisher import ArtifactPublisher, get_git_hash


class TestArtifactPublisher:
    def test_build_card(self):
        pub = ArtifactPublisher("user/repo")
        tags = {"git_hash": "abc123", "experiment": "test-1"}
        files = ["paper.pdf", "diffs.json"]
        card = pub._build_card(tags, files)

        assert "ratiocinator" in card
        assert "abc123" in card
        assert "`paper.pdf`" in card
        assert "`diffs.json`" in card

    def test_publish_missing_huggingface_hub(self):
        pub = ArtifactPublisher("user/repo")
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            pytest.raises(ImportError, match="huggingface_hub"),
        ):
            pub.publish({"test.txt": Path("/nonexistent")})


class TestGetGitHash:
    def test_returns_hash(self, tmp_path):
        """In our actual repo, should return a real hash."""
        h = get_git_hash(Path("."))
        assert len(h) == 40 or h == "unknown"

    def test_returns_unknown_for_bad_path(self, tmp_path):
        h = get_git_hash(tmp_path / "nonexistent")
        assert h == "unknown"
