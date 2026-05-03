"""Tests for `ratiocinator init` command."""

from __future__ import annotations

from click.testing import CliRunner

from ratiocinator.cli import main


class TestInitCommand:
    def test_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        assert (tmp_path / "research" / "specs").is_dir()
        assert (tmp_path / "research" / "results").is_dir()
        assert (tmp_path / ".ratiocinator").is_dir()

    def test_updates_gitignore(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        gitignore = (tmp_path / ".gitignore").read_text()
        assert ".ratiocinator/" in gitignore
        assert "research/results/" in gitignore

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        gitignore = (tmp_path / ".gitignore").read_text()
        assert gitignore.count(".ratiocinator/") == 1
        assert gitignore.count("research/results/") == 1

    def test_appends_to_existing_gitignore(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".gitignore").write_text("node_modules/\n")

        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        gitignore = (tmp_path / ".gitignore").read_text()
        assert "node_modules/" in gitignore
        assert ".ratiocinator/" in gitignore
        assert "research/results/" in gitignore

    def test_does_not_duplicate_existing_entries(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".gitignore").write_text(".ratiocinator/\n")

        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        gitignore = (tmp_path / ".gitignore").read_text()
        assert gitignore.count(".ratiocinator/") == 1
        assert "research/results/" in gitignore

    def test_gitignore_without_trailing_newline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".gitignore").write_text("node_modules/")  # no trailing \n

        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        lines = (tmp_path / ".gitignore").read_text().splitlines()
        assert "node_modules/" in lines
        assert ".ratiocinator/" in lines
        assert "research/results/" in lines
