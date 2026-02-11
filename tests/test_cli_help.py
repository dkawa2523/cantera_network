import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rxn_platform.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_help_top_level() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_help_subcommands() -> None:
    for subcommand in (
        "cfg",
        "sim",
        "task",
        "pipeline",
        "viz",
        "doctor",
        "artifacts",
    ):
        result = _run_cli(subcommand, "--help")
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
