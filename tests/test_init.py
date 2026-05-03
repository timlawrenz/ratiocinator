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

    def test_creates_agents_md(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        agents_md = (tmp_path / "AGENTS.md").read_text()
        assert "## Ratiocinator" in agents_md
        assert "ratiocinator fleet run" in agents_md

    def test_appends_to_existing_agents_md(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# My Project\n\nSome instructions.\n")

        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        agents_md = (tmp_path / "AGENTS.md").read_text()
        assert "# My Project" in agents_md
        assert "## Ratiocinator" in agents_md

    def test_agents_md_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        agents_md = (tmp_path / "AGENTS.md").read_text()
        assert agents_md.count("## Ratiocinator") == 1

    def test_agents_md_no_leading_blank_line_when_empty(self, tmp_path, monkeypatch):
        """When AGENTS.md is empty, the Ratiocinator section should start at line 1."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("")
        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        agents_md = (tmp_path / "AGENTS.md").read_text()
        assert agents_md.startswith("## Ratiocinator")

    def test_agents_md_exactly_one_blank_line_separator(self, tmp_path, monkeypatch):
        """When appending to existing content, exactly one blank line separates sections."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# My Project\n\nSome instructions.\n")
        runner = CliRunner()
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        agents_md = (tmp_path / "AGENTS.md").read_text()
        # Should not have double blank lines before the section
        assert "\n\n\n" not in agents_md
